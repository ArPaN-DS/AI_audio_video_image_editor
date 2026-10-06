"""
Small-model planning runtime: routing, budgeted prompts, structured output,
bounded repair, tier escalation and request decomposition.

Used by `agent_processor.query_agent_orchestrator`:

    session = PlanningSession(prompt, media_context, history, local_plan_fn)
    if session.skip_llm:                       # rule-based plan is confident -> no model call
        return session.local_plan
    payload = session.payload(media_text, memory_block, skills_hint, legacy_payload)
    content = session.complete(lambda body: _request_reasoning_content(url, body))

Routing (env LOCAL_REASONING_ROUTING):
    auto     (default) skip the model when the rule-based plan is confident and validated and a
             model is being served; otherwise ask the model.
    cascade  always skip the model for confident rule-based plans.
    always   always ask the model (legacy behaviour).
    local    never ask the model.

Cost-aware tier order: short requests start on the smallest served tier and escalate to
larger tiers only when the answer fails validation (after one repair); long multi-step
requests go to the largest tier first, or — when only a small tier is served — are split
into chunks the small model can handle.

Everything logged here is numeric/categorical (no model names, no user text).
"""

import collections
import json
import logging
import os
import threading
import time

import agent_planner
import slm_context
import slm_profiles

_log = logging.getLogger("slm_runtime")

ROUTING_MODES = ("auto", "cascade", "always", "local")
SIMPLE_MAX_CLAUSES = 2
COMPLEX_MIN_CLAUSES = 5
CHUNK_CLAUSES = 3
MAX_REPAIRS = 1
# Rule-based conversational answers that are curated (not generic) — see agent_nlu._question_reply.
CONFIDENT_CHAT_THOUGHTS = ("Greeting", "Capability overview", "Media knowledge")
_STATS = collections.deque(maxlen=200)
_STATS_LOCK = threading.Lock()


def routing_mode():
    mode = (os.environ.get("LOCAL_REASONING_ROUTING") or "auto").strip().lower()
    return mode if mode in ROUTING_MODES else "auto"


def record(event, **fields):
    """Numbers-only telemetry for evaluation (`recent_stats()`)."""
    entry = {"event": event, "t": round(time.time(), 3), **fields}
    with _STATS_LOCK:
        _STATS.append(entry)
    return entry


def recent_stats(clear=False):
    with _STATS_LOCK:
        items = list(_STATS)
        if clear:
            _STATS.clear()
    return items


def current_ladder():
    try:
        import reasoning_models
        return reasoning_models.reasoning_ladder()
    except Exception:
        return []


def _variant_tier(variant):
    return (getattr(variant, "options", None) or {}).get("tier", "lite")


def _variant_base(variant):
    url = (getattr(variant, "options", None) or {}).get("url") or ""
    return url[:-len("/chat/completions")] if url.endswith("/chat/completions") else None


def profile_for(variant):
    if variant is None:
        return slm_profiles.default_profile()
    return slm_profiles.registry.profile(variant.variant_id, _variant_base(variant), _variant_tier(variant))


def local_confident(plan):
    """
    A validated, unambiguous rule-based answer (the parser clarifies whenever unsure): either a
    non-empty plan, or an explained refusal (unsupported operation / wrong media type).
    """
    if not isinstance(plan, dict) or plan.get("clarification_needed") or not plan.get("validated", True):
        return False
    ladder = current_ladder()
    if not plan.get("tools"):
        if ladder:
            return False
        if str(plan.get("thought", "")).startswith(CONFIDENT_CHAT_THOUGHTS):
            return True
    return bool(plan.get("tools") or plan.get("plan_notes"))


def clauses_of(prompt):
    try:
        import agent_nlu
        return agent_nlu.split_clauses(agent_nlu.normalize_text(prompt))
    except Exception:
        return [prompt]


# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT DIAGNOSIS (drives the bounded repair retry)
# ═══════════════════════════════════════════════════════════════════════════

def diagnose(content, mode="plan", allowed=None):
    """Short, model-readable list of problems with the output ([] = usable)."""
    raw, leftover = agent_planner.extract_json_object(content if isinstance(content, str) else "")
    if raw is None:
        text = (leftover or "").strip()
        if mode == "chat" and text and "{" not in text:
            return []
        if "{" in (content or ""):
            return ["The JSON was cut off or malformed. Return complete valid JSON; keep thought under 15 words."]
        return ["Answer with one JSON object only, no prose."]
    problems = []
    tools = raw.get("tools")
    if tools is not None and not isinstance(tools, list):
        problems.append('"tools" must be a list of {"name", "args"} steps.')
        tools = []
    for step in tools or []:
        if isinstance(step, dict) and isinstance(step.get("skill"), str):
            continue
        name = step.get("name") if isinstance(step, dict) else step
        resolved = agent_planner.resolve_tool_name(name)
        if not resolved:
            names = ", ".join(sorted(allowed)) if allowed else "the listed tools"
            problems.append(f"Unknown tool {str(name)[:40]!r}; use only: {names}.")
            continue
        spec = agent_planner.TOOL_REGISTRY.get(resolved)
        args = step.get("args") if isinstance(step, dict) else {}
        if spec is None or not isinstance(args, dict):
            continue
        for key in args:
            canonical = key if key in spec.params else agent_planner.PARAM_ALIASES.get(str(key).lower(), "")
            if canonical is None:
                continue
            if canonical not in spec.params:
                valid = ", ".join(n for n, p in spec.params.items() if not p.internal) or "none"
                problems.append(f"{spec.name} has no arg {str(key)[:30]!r}; valid args: {valid}.")
        for pname, param in spec.params.items():
            if param.required and not param.internal and not any(
                    (k == pname or agent_planner.PARAM_ALIASES.get(str(k).lower()) == pname) for k in args):
                if not raw.get("clarification_needed"):
                    problems.append(f"{spec.name} needs {pname}; if the user did not say it, set "
                                    f"clarification_needed true and tools [].")
    return problems[:4]


def _score(content, mode, allowed):
    problems = diagnose(content, mode, allowed)
    return len(problems), problems


# ═══════════════════════════════════════════════════════════════════════════
#  REQUEST FIELDS (structured output, sampling, thinking switches, caching)
# ═══════════════════════════════════════════════════════════════════════════

def request_fields(profile, schema, mode="plan"):
    fields = {}
    if profile.structured == "json_schema" and schema:
        fields["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "edit_plan", "strict": True, "schema": schema}}
    elif profile.structured == "json_object":
        fields["response_format"] = {"type": "json_object"}
    if mode == "chat":
        sampling = dict(profile.chat_sampling)
        sampling["temperature"] = min(float(sampling.get("temperature", 0.6)), 0.7)
    else:
        sampling = profile.plan_sampling()
    if profile.server_kind == "generic":
        sampling.pop("top_k", None)
    fields.update(sampling)
    fields.update(profile.thinking_fields())
    if profile.server_kind == "llamacpp":
        fields["cache_prompt"] = True
    return fields


# ═══════════════════════════════════════════════════════════════════════════
#  PLANNING SESSION
# ═══════════════════════════════════════════════════════════════════════════

class PlanningSession:

    def __init__(self, prompt, media_context=None, history=None, local_plan_fn=None, ladder=None):
        self.prompt = prompt or ""
        self.media_context = media_context if isinstance(media_context, dict) else {}
        self.media_type = self.media_context.get("type") if self.media_context.get("type") in \
            agent_planner.MEDIA_TYPES else None
        self.history = history
        self.ladder = current_ladder() if ladder is None else list(ladder)
        self.local_plan = None
        if local_plan_fn is not None:
            try:
                self.local_plan = local_plan_fn()
            except Exception as err:
                _log.warning("Rule-based plan unavailable for routing (%s).", type(err).__name__)
        self.confident = local_confident(self.local_plan)
        self.mode = "plan"
        if isinstance(self.local_plan, dict) and not self.local_plan.get("tools") \
                and not self.local_plan.get("clarification_needed"):
            self.mode = "chat"                    # greeting / question / knowledge reply
        self.clauses = clauses_of(self.prompt)
        self.tiers = [_variant_tier(variant) for variant in self.ladder]
        self.skip_llm, self.route_reason = self._route()
        self.decompose = (not self.skip_llm and self.mode == "plan" and len(self.clauses) >= COMPLEX_MIN_CLAUSES
                          and set(self.tiers) <= {"lite"} and bool(self.tiers))
        self._rendered = {}
        self._parts = None
        self._legacy = None
        record("route", skip=self.skip_llm, reason=self.route_reason, mode=self.mode, clauses=len(self.clauses),
               tiers=len(self.ladder), decompose=self.decompose)

    def _route(self):
        mode = routing_mode()
        if mode == "local":
            return True, "local-only"
        if mode == "always":
            return False, "always"
        if not self.confident:
            return False, "needs-model"
        if mode == "cascade" or self.ladder:
            return True, "rules-confident"
        return False, "no-served-model"        # auto + discovery unknown: legacy behaviour

    # ── tier ordering (cost-aware cascade) ──
    def tier_groups(self):
        if not self.ladder:
            return [None]
        largest_first = list(self.tiers)            # ladder is largest first
        if len(self.clauses) <= SIMPLE_MAX_CLAUSES and len(largest_first) > 1:
            smallest = largest_first[-1]
            return [[smallest], [tier for tier in largest_first if tier != smallest]]
        return [largest_first]

    # ── prompt building ──
    def parts(self, media_text="", memory="", skills_hint="", request=None, media_type=None):
        hint_tools = [step.get("name") for step in (self.local_plan or {}).get("tools") or []
                      if isinstance(step, dict)]
        skill_ids = ()
        if skills_hint:
            try:
                import agent_skills
                skill_ids = tuple(skill.id for skill, _ in agent_skills.rank_skills(self.prompt, self.media_type, 3))
            except Exception:
                skill_ids = ()
        return slm_context.PromptParts(
            request if request is not None else self.prompt, media_type or self.media_type, media_text, memory,
            skills_hint, hint_tools, want_clarify=bool((self.local_plan or {}).get("clarification_needed")),
            mode=self.mode, skill_ids=skill_ids)

    def render(self, variant, parts=None, shrink=1.0):
        parts = parts or self._parts
        profile = profile_for(variant)
        tier = _variant_tier(variant) if variant is not None else "lite"
        counter = slm_profiles.TokenCounter(profile)
        original = profile.context_tokens
        profile_view = profile
        if shrink < 1.0:
            profile_view = slm_profiles.ModelProfile(profile.model_id, profile.base_url, profile.server_kind,
                                                     int(original * shrink), profile.train_context, profile.params_b,
                                                     profile.structured, profile.tokenize_url, profile.tokenize_style)
            profile_view.chars_per_token = profile.chars_per_token
            profile_view.system_role, profile_view.thinking = profile.system_role, profile.thinking
        messages, budget = slm_context.compose(parts, profile_view, tier, counter)
        total = slm_context.prompt_tokens(messages, counter)
        if total + budget.output > profile.context_tokens and shrink > 0.5:   # exact count disagreed: shrink
            return self.render(variant, parts, shrink * 0.8)
        allowed = budget.selected_tools
        schema = slm_context.plan_schema([agent_planner.TOOL_REGISTRY[name] for name in allowed], parts.skill_ids,
                                         strict=os.environ.get("LOCAL_REASONING_SCHEMA", "strict").lower() != "loose")
        body = {"messages": messages, "max_tokens": budget.output}
        body.update(request_fields(profile, schema, parts.mode))
        _log.info("Reasoning prompt budget: tier=%s %s", tier, budget.log_line())
        record("budget", tier=tier, prompt_tokens=total, exact=bool(profile.tokenize_url), **budget.as_dict())
        return body, budget, profile

    def payload(self, media_text="", memory="", skills_hint="", legacy_payload=None):
        """First-attempt payload (rendered for the first tier that will be tried) plus per-tier hooks."""
        self._legacy = legacy_payload
        self._parts = self.parts(media_text, memory, skills_hint)
        self._first_payload = self._payload_for(self.tier_groups()[0], self._parts)
        return self._first_payload

    def _payload_for(self, tiers, parts):
        first = next((v for v in self.ladder if tiers is None or _variant_tier(v) in tiers), None)
        body, budget, profile = self.render(first, parts)
        self._last_allowed = set(budget.selected_tools)
        payload = dict(body)
        payload["model"] = (self._legacy or {}).get("model", "local")
        cache = {first.variant_id: body} if first is not None else {}

        def render(variant):
            key = variant.variant_id
            if key not in cache:
                cache[key] = self.render(variant, parts)[0]
            return cache[key]

        payload["_render"] = render
        payload["_observe"] = self._observe
        payload["_recover"] = lambda variant, error: self._recover(variant, error, cache)
        if self.ladder:
            payload["_ladder"] = self.ladder
        if tiers:
            payload["_tiers"] = list(tiers)
        return payload

    # ── server feedback ──
    def _observe(self, variant, body_out, body, elapsed):
        usage = body.get("usage") if isinstance(body, dict) else None
        profile = profile_for(variant)
        prompt_chars = sum(len(m.get("content", "")) for m in body_out.get("messages", []))
        if isinstance(usage, dict) and usage.get("prompt_tokens"):
            overhead = slm_context.TEMPLATE_OVERHEAD_TOKENS + slm_context.MESSAGE_OVERHEAD_TOKENS * len(
                body_out.get("messages", []))
            slm_profiles.registry.calibrate(profile, prompt_chars, usage["prompt_tokens"], overhead)
        record("call", tier=_variant_tier(variant), latency=round(elapsed, 3),
               prompt_tokens=(usage or {}).get("prompt_tokens"), completion_tokens=(usage or {}).get(
                   "completion_tokens"), structured=profile.structured)

    def _recover(self, variant, error, cache):
        import reasoning_models
        profile = profile_for(variant)
        if isinstance(error, reasoning_models.StructuredOutputUnsupportedError) and profile.structured != "off":
            slm_profiles.registry.downgrade_structured(profile)
        elif isinstance(error, reasoning_models.ContextOverflowError) and profile.context_tokens > 512:
            limit = getattr(error, "limit", None)
            shrunk = limit if limit and 256 <= limit < profile.context_tokens else profile.context_tokens // 2
            profile.context_tokens = max(512, int(shrunk))
            profile.context_source = "overflow"
        else:
            return False
        cache.pop(variant.variant_id, None)
        record("recover", kind=type(error).__name__)
        return True

    # ── completion with repair / escalation / decomposition ──
    def complete(self, request_fn):
        """Model answer (JSON text) for `_plan_from_reasoning`, after grounding guards."""
        return self.guard(self._complete_raw(request_fn))

    def guard(self, content):
        """
        Small models over-commit: when the rule-based planner asked for a missing value, a model
        answer that *invents* that value (a speed or format the user never said) is replaced by the
        planner's clarification; conversational turns never start edits.
        """
        raw, _ = agent_planner.extract_json_object(content if isinstance(content, str) else "")
        local = self.local_plan or {}
        if raw is None:
            return content
        if self.mode == "chat" and (raw.get("tools") or raw.get("clarification_needed")):
            record("guard", kind="chat-edit")
            reply = raw.get("reply") if isinstance(raw.get("reply"), str) and raw.get("reply").strip() else                 local.get("reply", "")
            return json.dumps({"thought": "Conversational reply", "tools": [], "clarification_needed": False,
                               "reply": reply, "suggested_actions": raw.get("suggested_actions") or []})
        if local.get("clarification_needed") and raw.get("tools") and not ungrounded_values_ok(raw, self.prompt):
            record("guard", kind="ungrounded")
            return json.dumps({"thought": "Missing value", "tools": [], "clarification_needed": True,
                               "clarification_options": local.get("clarification_options") or [],
                               "reply": local.get("reply") or "Could you give the exact value?"})
        return content

    def _complete_raw(self, request_fn):
        if self.decompose:
            try:
                merged = self._complete_chunked(request_fn)
                if merged is not None:
                    return merged
            except Exception as err:
                _log.info("Chunked planning failed (%s); planning the whole request.", type(err).__name__)
        groups = self.tier_groups()
        best, best_score, last_error = None, None, None
        for index, tiers in enumerate(groups):
            payload = self.payload_for_group(index, tiers)
            try:
                content = request_fn(payload)
            except Exception as err:
                last_error = err
                record("attempt", group=index, ok=False, error=type(err).__name__)
                continue
            score, problems = _score(content, self.mode, self._last_allowed)
            record("attempt", group=index, ok=True, problems=score)
            if best is None or score < best_score:
                best, best_score = content, score
            if not problems or self.confident:
                return content
            for _ in range(MAX_REPAIRS):
                repaired = self._repair(request_fn, payload, content, problems)
                if repaired is None:
                    break
                score, problems = _score(repaired, self.mode, self._last_allowed)
                record("repair", group=index, problems=score)
                if score < best_score:
                    best, best_score = repaired, score
                if not problems:
                    return repaired
        if best is None:
            raise last_error or ConnectionError("No reasoning tier answered.")
        return best

    def payload_for_group(self, index, tiers):
        if index == 0 and self._parts is not None and hasattr(self, "_first_payload"):
            return self._first_payload
        payload = self._payload_for(tiers, self._parts or self.parts())
        if index == 0:
            self._first_payload = payload
        return payload

    def _repair(self, request_fn, payload, content, problems):
        snippet = (content or "")[:600]
        fix = ("Your answer had problems: " + " ".join(problems)
               + " Reply with the corrected JSON object only.")
        repair = {key: value for key, value in payload.items() if key != "_render"}
        base_render = payload.get("_render")

        def render(variant):
            body = dict(base_render(variant)) if callable(base_render) else dict(payload)
            body["messages"] = list(body["messages"]) + [{"role": "assistant", "content": snippet},
                                                         {"role": "user", "content": fix}]
            return body

        repair["messages"] = list(payload["messages"]) + [{"role": "assistant", "content": snippet},
                                                          {"role": "user", "content": fix}]
        repair["_render"] = render
        try:
            return request_fn(repair)
        except Exception as err:
            record("repair", ok=False, error=type(err).__name__)
            return None

    def _complete_chunked(self, request_fn):
        """Plan a long request in chunks of a few clauses; media type is carried across chunks."""
        chunks = [", then ".join(self.clauses[i:i + CHUNK_CLAUSES])
                  for i in range(0, len(self.clauses), CHUNK_CLAUSES)]
        media_type = self.media_type
        tools, replies = [], []
        for number, chunk in enumerate(chunks, 1):
            parts = self.parts(self._parts.media_text if self._parts else "", "", "",
                               request=f"(part {number} of {len(chunks)}) {chunk}", media_type=media_type)
            parts.hint_tools = ()
            content = request_fn(self._payload_for(None, parts))
            raw, _ = agent_planner.extract_json_object(content or "")
            if raw is None or raw.get("clarification_needed") or diagnose(content, "plan", self._last_allowed):
                return None
            for step in agent_planner.normalize_raw_plan(raw)["tools"]:
                tools.append({"name": step["name"], "args": step["args"]})
                spec = agent_planner.TOOL_REGISTRY.get(agent_planner.resolve_tool_name(step["name"]) or "")
                if spec is not None and spec.kind == "edit":
                    media_type = spec.output_type(step["args"], media_type)
            if raw.get("reply"):
                replies.append(str(raw["reply"])[:200])
        record("decompose", chunks=len(chunks), steps=len(tools))
        return json.dumps({"thought": f"Planned in {len(chunks)} parts", "tools": tools,
                           "clarification_needed": False, "reply": " ".join(replies)})


def _prompt_numbers(prompt):
    import re
    try:
        import agent_nlu
        text = agent_nlu.normalize_text(prompt)
    except Exception:
        text = (prompt or "").lower()
    numbers = set()
    for token in re.findall(r"\d+(?:\.\d+)?", text):
        value = float(token)
        numbers.update({round(value, 3), round(1 + value / 100, 3), round(1 - value / 100, 3)})
        if value:
            numbers.add(round(1 / value, 3))
    return numbers, text


def ungrounded_values_ok(raw, prompt):
    """True when every user-facing *required* choice (speed, output format) the model filled in
    is actually stated in the request (numbers, 1/x for 'slower', percentages, format names)."""
    numbers, text = _prompt_numbers(prompt)
    for step in raw.get("tools") or []:
        if not isinstance(step, dict):
            continue
        spec = agent_planner.TOOL_REGISTRY.get(agent_planner.resolve_tool_name(step.get("name")) or "")
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        if spec is None:
            continue
        for name, param in spec.params.items():
            if not param.required or param.internal or name not in args:
                continue
            value = args[name]
            if param.enum and isinstance(value, str):
                aliases = {value.lower()} | {alias for alias, target in agent_planner.FORMAT_ALIASES.items()
                                             if target == value.lower()}
                if not any(alias in text for alias in aliases):
                    return False
            elif name == "speed":
                try:
                    if round(float(value), 3) not in numbers:
                        return False
                except (TypeError, ValueError):
                    return False
    return True


def memory_budget(media_text, user_prompt, ladder=None):
    """Memory token budget for the smallest tier that may answer (used by the orchestrator)."""
    ladder = current_ladder() if ladder is None else ladder
    variant = ladder[-1] if ladder else None
    profile = profile_for(variant)
    estimate = slm_profiles.TokenCounter(profile).estimate
    allowance = slm_context.memory_allowance(_variant_tier(variant) if variant else "lite", profile,
                                             estimate(user_prompt or "") + 8, estimate(media_text or ""), estimate)
    return max(96, allowance)
