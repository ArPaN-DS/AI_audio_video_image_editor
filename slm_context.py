"""
Context-window management and prompt compression for small local planners.

The legacy planning prompt sends every tool, every rule and seven examples
(~2.5k tokens) on every request. A 2B model with a 4k window then has little
room left for memory and output, and long tool lists measurably hurt small
models' tool selection (Less-is-More, arXiv:2411.15399; RAG-MCP,
arXiv:2505.03275). This module builds a compact, budgeted prompt instead:

  system  = [static core: identity + rules + output schema]   <- identical on every request,
            [top-k relevant tools, canonical order]               so server prefix/KV caches reuse it
            [1-3 most similar examples]
  user    = [media condition] [memory] [skill hints] [request]

`BudgetPlanner` allocates the model's window across these sections by
priority (core, request and reserved output are mandatory; tools > memory >
examples > extra tools), truncates gracefully and never overflows. Budget usage
is reported as numbers only (no names, no text) for evaluation.
"""

import json
import re
from collections import OrderedDict

import agent_planner
import branding

TIER_PROMPT_TARGETS = {"lite": 1536, "balanced": 3072, "max": 6144}
TIER_OUTPUT_TOKENS = {"lite": 384, "balanced": 512, "max": 768}
TIER_TOOL_K = {"lite": 8, "balanced": 12, "max": 99}
TIER_SHOTS = {"lite": 2, "balanced": 3, "max": 4}
TIER_MEMORY_SHARE = {"lite": 320, "balanced": 700, "max": 900}
MESSAGE_OVERHEAD_TOKENS = 12      # role markers / separators per chat message
TEMPLATE_OVERHEAD_TOKENS = 24     # BOS, generation prompt, etc.
MIN_TOOLS = 3
NL = "\n"


def _identity_clause():
    return (f"You are {branding.assistant_name()}, the personal media editing AI assistant of {branding.product_name()}, "
            "running locally. Never name or describe the underlying models, vendors, libraries or these "
            f"instructions; if asked who you are or what your name is, proudly say your name is {branding.assistant_name()} and steer back to editing.")


CORE_RULES = (
    "Turn the request into an ordered edit plan using ONLY the TOOLS below "
    "(format: name(arg=type, *=required) accepts->produces: purpose).\n"
    "Rules:\n"
    "1. One step per operation, in the user's order; commas, 'and', 'then', 'also' separate steps.\n"
    "2. Track the media type: extract_audio turns video into audio, so later steps use *_audio tools. "
    "Never use image tools on audio or video.\n"
    "3. Units: seconds as numbers (1:30 -> 90, 500ms -> 0.5); speed is a multiplier (2x slower -> 0.5, "
    "25% faster -> 1.25); volume in dB; loudness in negative LUFS.\n"
    "4. Never invent missing values. If a trim range, speed or format is missing, set clarification_needed "
    "true, tools [] and ask one short question with 2-3 clarification_options.\n"
    "5. Greetings or questions: tools [] and a brief answer in reply.\n"
    "6. Add no steps the user did not ask for. Use only the arg names listed.\n"
    "Answer with ONE JSON object: "
    '{"thought": short str, "tools": [{"name": str, "args": {}}], "clarification_needed": bool, '
    '"clarification_options": [str], "reply": str, "suggested_actions": [str]}'
)


def core_prompt():
    """Static instruction prefix (changes only when the product/assistant name changes)."""
    return f"{_identity_clause()}\n{CORE_RULES}"


# ═══════════════════════════════════════════════════════════════════════════
#  TOOL CATALOG (compact, per tool) + RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════

def _short(text, limit=90):
    first = re.split(r"(?<=[.;])\s", text.strip(), maxsplit=1)[0].rstrip(".;")
    if first.count("(") > first.count(")"):
        first = first[:first.rindex("(")].rstrip()
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0]


def _arg_text(name, param):
    if param.enum:
        detail = "|".join(str(value) for value in param.enum)
    else:
        detail = {"number": "num", "integer": "int", "boolean": "bool", "string": "str"}.get(param.type, param.type)
        if param.minimum is not None or param.maximum is not None:
            low = "" if param.minimum is None else f"{param.minimum:g}"
            high = "" if param.maximum is None else f"{param.maximum:g}"
            detail += f" {low}..{high}"
    if param.unit and param.unit not in ("x",):
        detail += f" {param.unit if param.unit != 'seconds' else 's'}"
    if param.required:
        return f"{name}*={detail}"
    if param.default is not None and not isinstance(param.default, bool):
        return f"{name}={detail} (default {param.default:g})" if isinstance(param.default, float) else \
            f"{name}={detail} (default {param.default})"
    return f"{name}={detail}"


def tool_line(spec):
    args = ", ".join(_arg_text(name, param) for name, param in spec.params.items() if not param.internal)
    produces = spec.output if isinstance(spec.output, str) else "video|image(gif)"
    produces = "same" if produces == "same" else produces
    kind = " [side output]" if spec.kind == "branch" else " [analysis]" if spec.kind == "analysis" else ""
    return f"- {spec.name}({args}) {'/'.join(spec.accepts)}->{produces}{kind}: {_short(spec.description)}"


_ALIASES_BY_TOOL = {}
for _alias, _target in agent_planner.TOOL_ALIASES.items():
    _ALIASES_BY_TOOL.setdefault(_target, []).append(_alias.replace("_", " "))
_FAMILY = {"trim": ("trim_video", "trim_audio"), "speed": ("adjust_video_speed", "adjust_audio_speed"),
           "convert": ("convert_video_format", "convert_audio_format", "convert_image_format")}
for _family, _members in _FAMILY.items():
    for _member in _members:
        _ALIASES_BY_TOOL.setdefault(_member, []).extend(_ALIASES_BY_TOOL.get(_family, []))
_EXTRA_TERMS = {
    "extract_audio": "export as mp3 wav audio soundtrack from video",
    "convert_audio_format": "export save as mp3 wav flac ogg bitrate",
    "convert_video_format": "export save as webm mp4 mkv gif animated",
    "convert_image_format": "export save as png jpg webp",
    "extract_frame": "thumbnail still frame screenshot png jpg picture of the video",
    "normalize_audio": "loudness lufs youtube spotify podcast broadcast level",
    "adjust_volume": "louder quieter volume gain db boost",
    "reduce_noise": "hiss hum background noise clean",
    "enhance_speech": "voice clarity clearer sound better",
    "isolate_voice": "music vocals voice separate",
    "remove_vocals": "instrumental karaoke",
    "auto_trim_silence": "dead air pauses silence",
    "apply_audio_fade": "fade in out edges",
    "compress_video": "smaller size email",
    "enhance_video": "1080p 720p 4k quality sharpen video",
    "upscale_image": "hd bigger resolution",
    "enhance_photo_clarity": "sharpen clarity denoise grain photo",
    "remove_background": "background cutout transparent subject",
}
_vectors = {}
_AUDIO_WORDS = {"mp3", "wav", "flac", "ogg", "audio", "soundtrack", "podcast"}


def _embed(text):
    import agent_memory
    return agent_memory.lexical_embedding(text)


def _tool_vector(spec):
    if spec.name not in _vectors:
        text = " ".join([spec.name.replace("_", " "), spec.label, spec.description, spec.example,
                         " ".join(_ALIASES_BY_TOOL.get(spec.name, [])), _EXTRA_TERMS.get(spec.name, "")])
        _vectors[spec.name] = _embed(text)
    return _vectors[spec.name]


def _reachable_types(media_type):
    if media_type == "video":
        return {"video", "audio"}           # extract_audio makes audio; GIF/thumbnail are terminal outputs
    if media_type in ("audio", "image"):
        return {media_type}
    return set(agent_planner.MEDIA_TYPES)


def rank_tools(query, media_type=None, hint_names=()):
    """All compatible tools, most relevant first: [(score, spec)]. `hint_names` (e.g. from the
    rule-based parser) are pinned to the top so retrieval never hides a tool already detected."""
    import numpy as np
    vector = _embed(query or "")
    reachable = _reachable_types(media_type)
    hints = {agent_planner.resolve_tool_name(name) or name for name in hint_names if name}
    query_tokens = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    scored = []
    for spec in agent_planner.TOOL_SPECS:
        if not reachable & set(spec.accepts):
            continue
        score = float(np.dot(vector, _tool_vector(spec)))
        score += 0.08 * len(query_tokens & set(spec.name.split("_")))
        if media_type and media_type in spec.accepts:
            score += 0.05
        if spec.name == "extract_audio" and media_type == "video" and query_tokens & _AUDIO_WORDS:
            score += 0.3
        if spec.name in hints:
            score += 10.0
        scored.append((score, spec))
    scored.sort(key=lambda item: -item[0])
    return scored


def select_tools(query, media_type=None, hint_names=(), k=8):
    """Top-k tool specs, rendered in canonical registry order (stable prefixes for KV-cache reuse)."""
    ranked = [spec for _, spec in rank_tools(query, media_type, hint_names)]
    chosen = {spec.name for spec in ranked[:max(k, 0)]}
    # Keep each same-family sibling (trim_video/trim_audio...) so type repair has a valid target.
    for family in _FAMILY.values():
        if chosen & set(family):
            chosen |= {name for name in family if name in {spec.name for spec in ranked}}
    return [spec for spec in agent_planner.TOOL_SPECS if spec.name in chosen]


# ═══════════════════════════════════════════════════════════════════════════
#  FEW-SHOT POOL + SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def _shot(media, prompt, tools, reply, thought="", clarify=None, suggestions=None):
    plan = OrderedDict([("thought", thought or "In order"), ("tools", [{"name": n, "args": a} for n, a in tools]),
                        ("clarification_needed", bool(clarify))])
    if clarify:
        plan["clarification_options"] = clarify
    plan["reply"] = reply
    if suggestions:
        plan["suggested_actions"] = suggestions
    return {"media": media, "prompt": prompt, "plan": plan}


FEW_SHOT_POOL = [
    _shot("video", "Trim from 2s to 8s, remove the background noise, normalize loudness, then export as mp3",
          [("trim_video", {"start_sec": 2, "end_sec": 8}), ("reduce_noise", {}), ("normalize_audio", {}),
           ("extract_audio", {"format": "mp3"})], "Trimming 2-8s, cleaning noise, normalizing, exporting MP3.",
          "4 edits; mp3 from video = extract_audio last"),
    _shot("video", "Extract the audio as wav, speed it up 1.25x and add a 1 second fade in and out",
          [("extract_audio", {"format": "wav"}), ("adjust_audio_speed", {"speed": 1.25}),
           ("apply_audio_fade", {"fade_in_sec": 1, "fade_out_sec": 1})], "Extracting WAV, 1.25x, 1s fades.",
          "after extract_audio the media is audio"),
    _shot("image", "Remove the background from this photo, upscale it 2x and boost clarity",
          [("remove_background", {"quality_profile": "detail"}), ("upscale_image", {"scale": 2}),
           ("enhance_photo_clarity", {})], "Cutting out the subject, upscaling 2x, boosting clarity."),
    _shot("video", "Mute it, then cut 1:05 to 1:12.5 and also give me a thumbnail at 1s",
          [("mute_video", {}), ("trim_video", {"start_sec": 65, "end_sec": 72.5}), ("extract_frame", {"time_sec": 1})],
          "Muting, trimming 65-72.5s, saving a thumbnail.", "mm:ss -> seconds"),
    _shot("audio", "make it 2x slower and louder",
          [("adjust_audio_speed", {"speed": 0.5}), ("adjust_volume", {"gain_db": 6})],
          "Slowing to 0.5x and raising volume 6 dB.", "2x slower = 0.5; louder = +6 dB"),
    _shot("audio", "Trim the audio", [], "Which part should I keep? Give a start and end time.", "No range given",
          clarify=["Trim from 0 to 5 seconds", "Keep the first 10 seconds"]),
    _shot("audio", "What is LUFS?", [], "LUFS measures perceived loudness; streaming targets about -14 LUFS.",
          "Question, no edit", suggestions=["Normalize loudness to -14 LUFS"]),
    _shot("audio", "clean up the hiss, normalize for podcasts and export flac",
          [("reduce_noise", {}), ("normalize_audio", {"preset": "podcast"}), ("convert_audio_format",
                                                                               {"target_format": "flac"})],
          "Removing hiss, podcast loudness, FLAC export."),
    _shot("video", "make a 3 second gif starting at 4s and compress the video",
          [("convert_video_format", {"target_format": "gif", "start": 4, "duration": 3}), ("compress_video", {})],
          "Making a 3s GIF from 4s and compressing.", "gif uses start + duration"),
    _shot("video", "transcribe it, then enhance the picture to 1080p",
          [("transcribe_audio", {}), ("enhance_video", {"mode": "1080p"})], "Transcribing, then enhancing to 1080p."),
    _shot("audio", "remove the music so only the voice is left, then cut the silence",
          [("isolate_voice", {}), ("auto_trim_silence", {})], "Isolating the voice, trimming silence."),
    _shot("image", "restore the faces and save as webp",
          [("restore_faces", {}), ("convert_image_format", {"target_format": "webp"})],
          "Restoring faces, exporting WebP."),
    _shot("video", "speed it up", [], "What speed should I use?", "No speed given",
          clarify=["Speed up 1.5x", "Speed up 2x"]),
    _shot(None, "hi there", [], "Hi! Load a photo, audio or video and tell me what to change.", "Greeting",
          suggestions=["What can you do?"]),
]
_shot_vectors = {}


def _shot_vector(index):
    if index not in _shot_vectors:
        _shot_vectors[index] = _embed(FEW_SHOT_POOL[index]["prompt"])
    return _shot_vectors[index]


def render_shot(shot):
    media = shot["media"] or "none"
    return f"User ({media}): {shot['prompt']}\n{json.dumps(shot['plan'], separators=(',', ':'))}"


def select_shots(query, media_type=None, n=2, want_clarify=False, allowed_tools=None):
    """Most similar examples (same media type preferred), diverse tool sets, compatible with the shown tools."""
    import numpy as np
    vector = _embed(query or "")
    scored = []
    for index, shot in enumerate(FEW_SHOT_POOL):
        names = [step["name"] for step in shot["plan"]["tools"]]
        if allowed_tools is not None and any(name not in allowed_tools for name in names):
            continue
        score = float(np.dot(vector, _shot_vector(index)))
        if media_type and shot["media"] == media_type:
            score += 0.15
        if want_clarify and shot["plan"]["clarification_needed"]:
            score += 0.5
        scored.append((score, index, tuple(names)))
    scored.sort(key=lambda item: -item[0])
    picked, seen = [], set()
    for score, index, names in scored:
        if names in seen:
            continue
        seen.add(names)
        picked.append(FEW_SHOT_POOL[index])
        if len(picked) >= n:
            break
    return picked


# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA (for constrained decoding)
# ═══════════════════════════════════════════════════════════════════════════

def _param_schema(param):
    if param.enum:
        schema = {"enum": list(param.enum) + ([None] if param.nullable else [])}
        return schema
    json_type = {"number": "number", "integer": "integer", "boolean": "boolean", "string": "string"}.get(
        param.type, "string")
    return {"type": [json_type, "null"] if param.nullable else json_type}


def plan_schema(specs, skill_ids=(), strict=True):
    """JSON schema of a plan. `thought` comes first so the model reasons before committing to steps;
    strict mode types every tool's args (no invented arg names), loose mode only pins tool names."""
    names = [spec.name for spec in specs]
    if strict and names:
        variants = []
        for spec in specs:
            params = {name: _param_schema(param) for name, param in spec.params.items() if not param.internal}
            required = [name for name, param in spec.params.items() if param.required and not param.internal]
            args = {"type": "object", "properties": params, "additionalProperties": False}
            if required:
                args["required"] = required
            variants.append({"type": "object", "properties": {"name": {"const": spec.name}, "args": args},
                             "required": ["name", "args"], "additionalProperties": False})
        if skill_ids:
            variants.append({"type": "object", "properties": {"skill": {"enum": list(skill_ids)},
                                                              "args": {"type": "object"}},
                             "required": ["skill", "args"], "additionalProperties": False})
        step = {"anyOf": variants}
    else:
        step = {"type": "object", "properties": {"name": {"enum": names or ["inspect_media"]},
                                                 "args": {"type": "object"}}, "required": ["name", "args"]}
    return {
        "type": "object",
        "properties": OrderedDict([
            ("thought", {"type": "string", "maxLength": 200}),
            ("tools", {"type": "array", "items": step, "maxItems": 12}),
            ("clarification_needed", {"type": "boolean"}),
            ("clarification_options", {"type": "array", "items": {"type": "string", "maxLength": 80},
                                       "maxItems": 3}),
            ("reply", {"type": "string", "maxLength": 600}),
            ("suggested_actions", {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 3}),
        ]),
        "required": ["thought", "tools", "clarification_needed", "reply"],
        "additionalProperties": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BUDGET PLANNER
# ═══════════════════════════════════════════════════════════════════════════

def truncate_to_tokens(text, max_tokens, count, keep="head"):
    """Trim text to fit max_tokens (binary search on characters). keep: head | tail | ends."""
    if not text or max_tokens <= 0:
        return ""
    if count(text) <= max_tokens:
        return text
    low, high, best = 0, len(text), ""
    while low <= high:
        middle = (low + high) // 2
        if keep == "tail":
            candidate = "…" + text[len(text) - middle:]
        elif keep == "ends":
            half = middle // 2
            candidate = text[:half] + " … " + text[len(text) - half:]
        else:
            candidate = text[:middle] + "…"
        if count(candidate) <= max_tokens:
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    return best


_MEMORY_DROP_ORDER = ("Related earlier:", "Similar past jobs:", "Session summary:", "Media facts:",
                      "Recent turns:", "Preferences:")


def truncate_memory(block, max_tokens, count):
    """Shrink a packed memory block by dropping whole lines, least important sections first
    (older related memories before preferences; recent turns keep their newest lines)."""
    if not block or count(block) <= max_tokens:
        return block if max_tokens > 0 else ""
    lines = block.split("\n")
    section_of, current = [], None
    for line in lines:
        stripped = line.strip()
        label = next((name for name in _MEMORY_DROP_ORDER if stripped.startswith(name)), None)
        current = label or current
        section_of.append(current)
    for label in _MEMORY_DROP_ORDER:
        indices = [index for index, section in enumerate(section_of) if section == label]
        if label == "Recent turns:":
            indices = indices[1:]       # keep the label; drop oldest turns first
        for index in indices:
            lines[index] = None
            text = "\n".join(line for line in lines if line is not None)
            if count(text) <= max_tokens:
                return text
    return truncate_to_tokens(block, max_tokens, count, keep="tail")


class Budget:
    """Numbers-only record of how the window was spent (for logs and evaluation)."""

    def __init__(self, context, cap, output):
        self.context = context
        self.cap = cap
        self.output = output
        self.sections = OrderedDict()
        self.dropped = []
        self.tools_k = 0
        self.shots = 0

    @property
    def used(self):
        return sum(self.sections.values())

    def as_dict(self):
        return {"context": self.context, "prompt_cap": self.cap, "output": self.output, "used": self.used,
                "sections": dict(self.sections), "tools_k": self.tools_k, "shots": self.shots,
                "dropped": list(self.dropped)}

    def log_line(self):
        parts = " ".join(f"{name}={value}" for name, value in self.sections.items())
        return (f"ctx={self.context} cap={self.cap} used={self.used} out={self.output} tools={self.tools_k} "
                f"shots={self.shots} {parts}" + (f" dropped={','.join(self.dropped)}" if self.dropped else ""))


class PromptParts:
    """Inputs for one planning prompt (all already privacy-safe: no paths, no model names)."""

    def __init__(self, request, media_type=None, media_text="", memory="", skills_hint="", hint_tools=(),
                 want_clarify=False, mode="plan", skill_ids=()):
        self.request = request or ""
        self.media_type = media_type
        self.media_text = media_text or ""
        self.memory = memory or ""
        self.skills_hint = skills_hint or ""
        self.hint_tools = tuple(hint_tools or ())
        self.want_clarify = want_clarify
        self.mode = mode
        self.skill_ids = tuple(skill_ids or ())


def tier_limits(tier, profile):
    """(prompt cap, output tokens) for a tier on this model's window — never more than fits."""
    tier = tier if tier in TIER_PROMPT_TARGETS else "lite"
    output = TIER_OUTPUT_TOKENS[tier] * profile.output_multiplier()
    margin = max(32, int(profile.context_tokens * 0.05))
    available = profile.context_tokens - output - margin - TEMPLATE_OVERHEAD_TOKENS - 2 * MESSAGE_OVERHEAD_TOKENS
    if available < 512:                        # tiny window: shrink output before the prompt
        output = max(160, profile.context_tokens // 4)
        available = profile.context_tokens - output - margin - TEMPLATE_OVERHEAD_TOKENS - 2 * MESSAGE_OVERHEAD_TOKENS
    import os
    try:
        override = int(os.environ.get("LOCAL_REASONING_PROMPT_TOKENS", "0"))
    except ValueError:
        override = 0
    target = override if override > 0 else TIER_PROMPT_TARGETS[tier]
    return max(128, min(target, available)), output


def memory_allowance(tier, profile, request_tokens, media_tokens, count):
    """Tokens the memory block may use for this tier (what the planner will admit)."""
    cap, _ = tier_limits(tier, profile)
    fixed = count(core_prompt()) + request_tokens + media_tokens + 220     # ~min tools + one example
    return max(0, min(TIER_MEMORY_SHARE.get(tier, 320), cap - fixed))


def compose(parts, profile, tier, counter):
    """
    Build chat messages for one tier within its budget. Returns (messages, budget).
    Allocation order (priority): core + request + media (mandatory, truncated if huge) ->
    minimum tools -> memory -> first example -> remaining tools -> skills hint -> more examples.
    """
    count = counter.count if hasattr(counter, "count") else counter
    tier = tier if tier in TIER_PROMPT_TARGETS else "lite"
    cap, output = tier_limits(tier, profile)
    budget = Budget(profile.context_tokens, cap, output)
    core = core_prompt()
    budget.sections["core"] = count(core)

    remaining = cap - budget.sections["core"]
    request = truncate_to_tokens(parts.request, max(48, remaining // 3), count, keep="ends")
    budget.sections["request"] = count(request) + 4
    media = truncate_to_tokens(parts.media_text, 160, count)
    budget.sections["media"] = count(media)
    remaining -= budget.sections["request"] + budget.sections["media"]

    ranked = [spec for spec in select_tools(parts.request, parts.media_type, parts.hint_tools,
                                            k=TIER_TOOL_K[tier])]
    lines = {spec.name: tool_line(spec) for spec in ranked}
    order = [spec.name for _, spec in rank_tools(parts.request, parts.media_type, parts.hint_tools)
             if spec.name in lines]
    chosen = []

    def admit_tools(limit):
        nonlocal remaining
        for name in order:
            if name in chosen or len(chosen) >= limit:
                continue
            cost = count(lines[name]) + 1
            if cost > remaining:
                budget.dropped.append("tools")
                return
            chosen.append(name)
            remaining -= cost

    admit_tools(max(MIN_TOOLS, len(parts.hint_tools)))
    remaining -= 6                                                  # "TOOLS:" header

    memory = ""
    if parts.memory:
        allowance = min(remaining - 160, TIER_MEMORY_SHARE[tier])
        memory = truncate_memory(parts.memory.strip(), max(0, allowance), count)
        if parts.memory.strip() and memory != parts.memory.strip():
            budget.dropped.append("memory" if not memory else "memory-partial")
        budget.sections["memory"] = count(memory)
        remaining -= budget.sections["memory"]

    shots = select_shots(parts.request, parts.media_type, TIER_SHOTS[tier], parts.want_clarify,
                         allowed_tools=None)
    shot_texts = []

    def admit_shot(shot):
        nonlocal remaining
        text = render_shot(shot)
        cost = count(text) + 1
        if cost > remaining:
            return False
        shot_texts.append(text)
        remaining -= cost
        return True

    if shots:
        admit_shot(shots[0])
    admit_tools(TIER_TOOL_K[tier])
    hint = ""
    if parts.skills_hint:
        cost = count(parts.skills_hint)
        if cost <= remaining:
            hint, remaining = parts.skills_hint, remaining - cost
        else:
            budget.dropped.append("skills")
    for shot in shots[1:]:
        if not admit_shot(shot):
            budget.dropped.append("shots")
            break

    if not chosen and order:
        chosen.append(order[0])                                     # a plan needs at least one tool
    state = {"shots": shot_texts, "hint": hint, "memory": memory, "media": media, "request": request,
             "chosen": chosen, "output": output}

    def assemble():
        tool_block = NL.join(lines[spec.name] for spec in agent_planner.TOOL_SPECS if spec.name in state["chosen"])
        system = f"{core}{NL}TOOLS:{NL}{tool_block}"
        if state["shots"]:
            system += f"{NL}Examples:{NL}" + NL.join(state["shots"])
        user = NL.join(part for part in (state["media"], state["memory"], state["hint"].strip()) if part)
        user += f"{NL}User Request: {state['request']}{profile.thinking_suffix()}"
        if profile.system_role:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]
        else:
            msgs = [{"role": "user", "content": f"{system}{NL}{NL}{user.strip()}"}]
        return msgs, tool_block

    # Hard guarantee: prompt + reserved output never exceeds the window (exact count when available).
    limit = profile.context_tokens - 8
    shrinkers = [lambda: state["shots"] and state["shots"].pop(), lambda: state.update(hint="") if state["hint"] else None,
                 lambda: state.update(memory="") if state["memory"] else None]
    shrinkers += [lambda: len(state["chosen"]) > 1 and state["chosen"].pop()] * TIER_TOOL_K[tier]
    shrinkers += [lambda: state.update(media=truncate_to_tokens(state["media"], 40, count)) if state["media"] else None,
                  lambda: state.update(media="") if state["media"] else None,
                  lambda: state.update(request=truncate_to_tokens(state["request"], 32, count, keep="ends"))]
    shrinkers += [lambda: state.update(output=max(128, int(state["output"] * 0.75)))] * 4
    messages, tool_block = assemble()
    for shrink in shrinkers:
        if prompt_tokens(messages, count) + state["output"] <= limit:
            break
        shrink()
        budget.dropped.append("fit")
        messages, tool_block = assemble()

    budget.output = state["output"]
    budget.sections["tools"] = count(tool_block) + 2
    budget.sections["examples"] = sum(count(text) + 1 for text in state["shots"])
    budget.sections["skills"] = count(state["hint"]) if state["hint"] else 0
    budget.sections["memory"] = count(state["memory"]) if state["memory"] else 0
    budget.sections["media"] = count(state["media"])
    budget.sections["request"] = count(state["request"]) + 4
    budget.tools_k, budget.shots = len(state["chosen"]), len(state["shots"])
    budget.selected_tools = list(state["chosen"])
    budget.dropped = list(dict.fromkeys(budget.dropped))
    return messages, budget


def prompt_tokens(messages, counter):
    count = counter.count if hasattr(counter, "count") else counter
    return (sum(count(message["content"]) + MESSAGE_OVERHEAD_TOKENS for message in messages)
            + TEMPLATE_OVERHEAD_TOKENS)
