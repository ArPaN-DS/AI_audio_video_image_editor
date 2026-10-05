"""
Evaluation harness for the small-model planning runtime (slm_runtime / slm_context / slm_profiles).

    # Offline: scripted mock server that behaves like a flaky ~2B model
    ./venv/Scripts/python.exe slm_eval.py offline [--ctx 4096] [--limit N]

    # Live: golden prompts through real local models (loopback only)
    ./venv/Scripts/python.exe slm_eval.py live --url http://127.0.0.1:11434/v1 --models qwen3:1.7b,qwen3:4b
          [--limit N] [--routing always|auto] [--legacy]

Both modes run the golden set of agent_orchestration_eval.py through the real
`agent_processor.query_agent_orchestrator` and score plans with its `check_case`.
Reported per configuration: plan accuracy, first-attempt JSON validity, model
calls/request, repairs, mean prompt tokens, p50/p95 latency, overflow/timeout counts.

The "legacy" configuration disables the runtime (full ~2.5k-token prompt, no
structured output, no repair, always calls the model) to give a before/after.
"""

import argparse
import contextlib
import hashlib
import http.server
import json
import os
import re
import statistics
import sys
import threading
import time
from unittest.mock import patch

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)
os.environ.setdefault("COPILOT_MEMORY", "off")

import agent_orchestration_eval as aoe  # noqa: E402
import agent_planner  # noqa: E402
import agent_processor  # noqa: E402
import reasoning_models  # noqa: E402
import slm_profiles  # noqa: E402
import slm_runtime  # noqa: E402

ENV_KEYS = ("LOCAL_REASONING_URL", "LOCAL_REASONING_MODEL", "LOCAL_REASONING_MODEL_BALANCED",
            "LOCAL_REASONING_MODEL_MAX", "LOCAL_REASONING_ROUTING", "LOCAL_REASONING_TIER",
            "LOCAL_REASONING_CONTEXT_TOKENS", "LOCAL_REASONING_URL_BALANCED", "LOCAL_REASONING_URL_MAX")


# ═══════════════════════════════════════════════════════════════════════════
#  MOCK SMALL-MODEL SERVER
# ═══════════════════════════════════════════════════════════════════════════

FAILURE_MODES = ("ok", "ok", "ok", "ok", "ok", "truncated", "prose", "wrong_tool", "bad_args", "semantic")


def _h(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


class MockModel:
    """
    Answers like a flaky small model. The 'intended' answer is the rule-based plan for the
    request (which matches the golden set); a deterministic failure mode is then applied.
    Constrained decoding (response_format json_schema) removes structural failures but not
    semantic ones, matching published findings (arXiv:2609.23742).
    """

    def __init__(self, n_ctx=4096, chars_per_token=3.3, timeout_every=0, repair_success=0.7, sleep=0.0):
        self.n_ctx = n_ctx
        self.chars_per_token = chars_per_token
        self.timeout_every = timeout_every
        self.repair_success = repair_success
        self.sleep = sleep
        self.oracle = {}
        self.requests = 0

    def expect(self, prompt, media_type, plan):
        self.oracle[prompt.strip()] = (media_type, plan)

    def prompt_tokens(self, messages):
        return int(sum(len(m.get("content", "")) for m in messages) / self.chars_per_token) + 8 * len(messages)

    def respond(self, body):
        """Returns (status, response dict, delay seconds)."""
        self.requests += 1
        messages = body.get("messages") or []
        tokens = self.prompt_tokens(messages)
        if tokens + int(body.get("max_tokens") or 0) > self.n_ctx:
            return 400, {"error": {"message": f"the request exceeds the available context size ({self.n_ctx})"}}, 0
        user_turns = [m["content"] for m in messages if m.get("role") == "user"]
        first_user = next((m for m in user_turns if "User Request:" in m), user_turns[0] if user_turns else "")
        request = first_user.rsplit("User Request:", 1)[-1].replace("/no_think", "").strip()
        repairing = bool(user_turns) and user_turns[-1].startswith("Your answer had problems")
        media_type, plan = self.oracle.get(request, (None, {"tools": [], "reply": "Okay.", "thought": "?"}))
        schema = ((body.get("response_format") or {}).get("json_schema") or {}).get("schema")
        allowed = _allowed_names(schema)
        mode = FAILURE_MODES[_h(request, "mode") % len(FAILURE_MODES)]
        if repairing:
            mode = "ok" if (_h(request, "repair") % 100) < self.repair_success * 100 else mode
        delay = self.sleep
        if self.timeout_every and _h(request, "timeout") % self.timeout_every == 0 and not repairing:
            delay = 999
        content = self._render(plan, mode, schema is not None, allowed)
        usage = {"prompt_tokens": tokens, "completion_tokens": int(len(content) / self.chars_per_token)}
        return 200, {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": usage}, delay

    def _render(self, plan, mode, constrained, allowed):
        tools = [{"name": s["name"], "args": dict(s["args"])} for s in plan.get("tools") or []
                 if s.get("role", "requested") != "prep"]
        out = {"thought": "plan", "tools": tools, "clarification_needed": bool(plan.get("clarification_needed")),
               "clarification_options": plan.get("clarification_options") or [], "reply": plan.get("reply") or "OK",
               "suggested_actions": []}
        if allowed is not None:                       # constrained: a tool outside the shown set is impossible
            for step in out["tools"]:
                if step["name"] not in allowed:
                    step["name"] = sorted(allowed)[0] if allowed else step["name"]
        if constrained and mode in ("truncated", "prose"):
            mode = "ok"
        if mode == "wrong_tool" and tools:
            if constrained and allowed:
                others = sorted(set(allowed) - {tools[0]["name"]})
                tools[0]["name"] = others[0] if others else tools[0]["name"]
            else:
                tools[0]["name"] = tools[0]["name"].replace("_", "-") + "_tool"
        elif mode == "bad_args" and tools:
            args = tools[0]["args"]
            if constrained:
                spec = agent_planner.TOOL_REGISTRY.get(tools[0]["name"])
                for key in list(args):
                    if spec and key in spec.params and not spec.params[key].required:
                        args.pop(key)               # optional args silently omitted (still schema-valid)
            else:
                tools[0]["args"] = {f"{key}_value": value for key, value in args.items()}
        elif mode == "semantic" and tools:
            for key, value in tools[-1]["args"].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    tools[-1]["args"][key] = value * 2 if value else 1
                    break
            else:
                tools.pop()
        text = json.dumps(out)
        if mode == "truncated":
            return text[:max(20, len(text) // 2)]
        if mode == "prose":
            return f"Sure! Here's what I'll do:\n```json\n{text}\n```\nLet me know!" if _h(text) % 2 else \
                "I'll take care of that edit for you right away."
        return text


def _allowed_names(schema):
    if not isinstance(schema, dict):
        return None
    items = ((schema.get("properties") or {}).get("tools") or {}).get("items") or {}
    names = set()
    for variant in items.get("anyOf") or [items]:
        name = (variant.get("properties") or {}).get("name") or {}
        if "const" in name:
            names.add(name["const"])
        names.update(value for value in name.get("enum") or [] if isinstance(value, str))
    return names or None


class MockServer:
    """Loopback HTTP server speaking the OpenAI chat-completions subset (+ /v1/models)."""

    def __init__(self, model, model_id="mock-2b"):
        self.model = model
        self.model_id = model_id
        mock = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"data": [{"id": mock.model_id, "owned_by": "mock"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                status, payload, delay = mock.model.respond(body)
                if delay:
                    time.sleep(min(delay, 3.0))
                    if delay >= 999:
                        return              # client has timed out; drop the response
                self._send(status, payload)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


# ═══════════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def configured(env):
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update({key: value for key, value in env.items() if value is not None})
    reasoning_models.discovery.invalidate()
    slm_profiles.registry = slm_profiles.ProfileRegistry()
    model_manager = sys.modules.get("model_manager")
    if model_manager is not None:
        model_manager.global_quality_governor.reset()
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reasoning_models.discovery.invalidate()


def _disabled_runtime(*args, **kwargs):
    raise RuntimeError("runtime disabled for the legacy baseline")


def run_golden(cases, legacy=False, label=""):
    """Run golden cases through the orchestrator; returns metrics dict."""
    slm_runtime.recent_stats(clear=True)
    results, latencies, first_valid, model_calls = [], [], [], []
    original_request = agent_processor._request_reasoning_content
    call_log = []

    def counting_request(url, payload):
        started = time.monotonic()
        try:
            content = original_request(url, payload)
        except Exception as err:
            call_log.append(("error", type(err).__name__, time.monotonic() - started, payload))
            raise
        call_log.append(("ok", content, time.monotonic() - started, payload))
        return content

    def plan_via_orchestrator(prompt, media_type=None, history=None, **context):
        media_context = dict(context)
        if media_type:
            media_context["type"] = media_type
        return agent_processor.query_agent_orchestrator(prompt, media_context or None, history)

    patches = [patch.object(agent_processor, "_request_reasoning_content", side_effect=counting_request),
               patch.object(aoe, "plan_for", side_effect=plan_via_orchestrator),
               patch("agent_memory.recall_similar_jobs", return_value=[])]
    if legacy:
        patches.append(patch.object(slm_runtime, "PlanningSession", side_effect=_disabled_runtime))
    with contextlib.ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        for prompt, media_type, expectation in cases:
            before = len(call_log)
            started = time.monotonic()
            problems = aoe.check_case(prompt, media_type, expectation)
            latencies.append(time.monotonic() - started)
            calls = call_log[before:]
            model_calls.append(len(calls))
            if calls:
                status, content, _, _ = calls[0]
                first_valid.append(status == "ok" and agent_planner.extract_json_object(content)[0] is not None)
            results.append((prompt, media_type, problems))
    prompt_tokens = []
    for status, _, _, payload in call_log:
        messages = payload.get("messages") or []
        prompt_tokens.append(int(sum(len(m.get("content", "")) for m in messages) / 3.3))
    stats = slm_runtime.recent_stats(clear=True)
    served = [s["prompt_tokens"] for s in stats if s["event"] == "call" and s.get("prompt_tokens")]
    model_latency = [s["latency"] for s in stats if s["event"] == "call"]
    errors = [item[1] for item in call_log if item[0] == "error"]
    passed = sum(1 for _, _, problems in results if not problems)
    metrics = {
        "label": label, "cases": len(results), "accuracy": round(passed / max(1, len(results)), 3),
        "json_valid_first": round(sum(first_valid) / len(first_valid), 3) if first_valid else None,
        "model_calls_per_request": round(sum(model_calls) / max(1, len(model_calls)), 2),
        "requests_with_model_call": sum(1 for count in model_calls if count),
        "repairs": sum(1 for s in stats if s["event"] == "repair"),
        "prompt_tokens_mean_est": int(statistics.mean(prompt_tokens)) if prompt_tokens else 0,
        "prompt_tokens_mean_served": int(statistics.mean(served)) if served else None,
        "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p95": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3) if latencies else None,
        "model_latency_p50": round(statistics.median(model_latency), 3) if model_latency else None,
        "errors": {name: errors.count(name) for name in set(errors)},
        "failures": [(p, m, pr[:2]) for p, m, pr in results if pr][:12],
    }
    return metrics


def offline(ctx=4096, limit=None, timeout_every=0):
    cases = aoe.GOLDEN[:limit] if limit else aoe.GOLDEN
    model = MockModel(n_ctx=ctx, timeout_every=timeout_every)
    with patch("agent_memory.recall_similar_jobs", return_value=[]):
        for prompt, media_type, expectation in cases:
            context = dict(expectation.get("context", {}))
            if media_type:
                context["type"] = media_type
            plan = agent_processor._fallback_intent_parser(prompt, context or None, expectation.get("history"))
            model.expect(prompt, media_type, plan)
    reports = []
    with MockServer(model) as server, patch.dict(reasoning_models.TIER_TIMEOUTS_SEC, {"lite": 1.5}):
        env = {"LOCAL_REASONING_URL": server.url, "LOCAL_REASONING_MODEL": "mock-2b",
               "LOCAL_REASONING_CONTEXT_TOKENS": str(ctx)}
        for label, routing, legacy in (("legacy (before)", "always", True), ("runtime, routing=always", "always", False),
                                       ("runtime, routing=auto", "auto", False)):
            with configured(dict(env, LOCAL_REASONING_ROUTING=routing)):
                reports.append(run_golden(cases, legacy=legacy, label=f"mock ctx={ctx}: {label}"))
    return reports


def live(url, models, limit=None, routing="always", legacy=False):
    if not reasoning_models.normalize_base_url(url):
        raise SystemExit("Only loopback endpoints are allowed.")
    cases = aoe.GOLDEN[:limit] if limit else aoe.GOLDEN
    reports = []
    for model_id in models:
        env = {"LOCAL_REASONING_URL": url, "LOCAL_REASONING_MODEL": model_id, "LOCAL_REASONING_ROUTING": routing}
        configs = [("runtime", False)] + ([("legacy", True)] if legacy else [])
        for name, is_legacy in configs:
            with configured(env):
                if not reasoning_models.reasoning_ladder():
                    reports.append({"label": f"{model_id}: not served", "cases": 0})
                    continue
                profile = slm_profiles.registry.profile(model_id, reasoning_models.normalize_base_url(url), "lite")
                report = run_golden(cases, legacy=is_legacy, label=f"{model_id} [{name}, routing={routing}]")
                report["profile"] = profile.public_summary()
                reports.append(report)
    return reports


def find_local_servers():
    """Loopback OpenAI-compatible servers on the usual ports (LOCAL_REASONING_URL first)."""
    candidates = [os.environ.get("LOCAL_REASONING_URL")] + [f"http://127.0.0.1:{port}/v1"
                                                             for port in (8000, 8080, 11434, 1234)]
    found = []
    for base in candidates:
        base = reasoning_models.normalize_base_url(base or "")
        if not base or base in [b for b, _ in found]:
            continue
        try:
            ids = reasoning_models.ServedModelDiscovery().served(base)
        except Exception:
            ids = frozenset()
        if ids:
            found.append((base, sorted(ids)))
    return found


def _print(reports):
    for report in reports:
        failures = report.pop("failures", [])
        print(json.dumps(report))
        for prompt, media, problems in failures[:6]:
            print(f"    FAIL [{media}] {prompt[:70]!r}: {'; '.join(problems)[:160]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode")
    off = sub.add_parser("offline")
    off.add_argument("--ctx", type=int, nargs="*", default=[4096, 2048])
    off.add_argument("--limit", type=int)
    off.add_argument("--timeouts", type=int, default=25, help="1 in N requests times out (0 = never)")
    on = sub.add_parser("live")
    on.add_argument("--url")
    on.add_argument("--models")
    on.add_argument("--limit", type=int)
    on.add_argument("--routing", default="always", choices=slm_runtime.ROUTING_MODES)
    on.add_argument("--legacy", action="store_true")
    sub.add_parser("discover")
    args = parser.parse_args(argv)
    import logging
    logging.disable(logging.WARNING)
    if args.mode == "live":
        servers = find_local_servers()
        url = args.url or (servers[0][0] if servers else None)
        if not url:
            print("No local OpenAI-compatible server found on loopback (8000/8080/11434/1234). See docs/SETUP.md.")
            return 1
        models = [m for m in (args.models or "").split(",") if m] or dict(servers).get(url, [])[:1]
        _print(live(url, models, args.limit, args.routing, args.legacy))
    elif args.mode == "discover":
        for base, ids in find_local_servers():
            print(base, ids)
    else:
        for ctx in getattr(args, "ctx", [4096, 2048]):
            _print(offline(ctx, getattr(args, "limit", None), getattr(args, "timeouts", 25)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
