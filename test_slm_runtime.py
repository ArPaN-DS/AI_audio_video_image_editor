"""
Small-model runtime tests: profiles, token budgets, prompt compression, tool retrieval,
structured output, repair/escalation/decomposition, and the offline mock-server pipeline.

Run:  ./venv/Scripts/python.exe test_slm_runtime.py
"""

import json
import os
import random
import sys
import unittest
from unittest.mock import patch

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)
os.environ.setdefault("COPILOT_MEMORY", "off")

import agent_planner  # noqa: E402
import reasoning_models  # noqa: E402
import slm_context  # noqa: E402
import slm_profiles  # noqa: E402
import slm_runtime  # noqa: E402
from model_manager import CapabilityVariant, TIER_BALANCED, TIER_LITE, TIER_MAX  # noqa: E402

FORBIDDEN = ['gemma', 'qwen', 'llama', 'whisper', 'vllm', 'ollama', 'ffmpeg', 'torch', 'onnx', 'copilot', 'omni']
ENV = ("LOCAL_REASONING_CONTEXT_TOKENS", "LOCAL_REASONING_STRUCTURED_OUTPUT", "LOCAL_REASONING_SERVER",
       "LOCAL_REASONING_PROBE", "LOCAL_REASONING_ROUTING", "LOCAL_REASONING_PROMPT_TOKENS", "LOCAL_REASONING_SCHEMA")


def variant(tier, model="m", url="http://127.0.0.1:9/v1"):
    return CapabilityVariant(f"{model}-{tier}", tier=TIER_LITE, public_label=tier,
                             options={"url": f"{url}/chat/completions", "timeout": 5, "tier": tier})


def golden():
    import agent_orchestration_eval as aoe
    return aoe.GOLDEN


class EnvTestCase(unittest.TestCase):
    def setUp(self):
        self.saved = {key: os.environ.pop(key, None) for key in ENV}
        self.registry = slm_profiles.registry
        slm_profiles.registry = slm_profiles.ProfileRegistry(metadata_lookup=lambda *a: None, probes={})

    def tearDown(self):
        for key, value in self.saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value
        slm_profiles.registry = self.registry


# ═══════════════════════════════════════════════════════════════════════════
#  1. PROFILES & TOKEN COUNTING
# ═══════════════════════════════════════════════════════════════════════════

class ProfileTests(EnvTestCase):

    def test_family_traits(self):
        self.assertFalse(slm_profiles.ModelProfile("gemma-3n-E2B-it").system_role)
        self.assertTrue(slm_profiles.ModelProfile("google/gemma-4-E2B-it").system_role)
        self.assertEqual(slm_profiles.ModelProfile("gemma-3n-E2B-it").params_b, 2.0)
        self.assertEqual(slm_profiles.ModelProfile("qwen3:1.7b").size_class, "lite")
        self.assertEqual(slm_profiles.ModelProfile("qwen3:8b").size_class, "balanced")
        self.assertEqual(slm_profiles.ModelProfile("something-12b").size_class, "max")
        unknown = slm_profiles.ModelProfile("local-media-copilot")
        self.assertEqual((unknown.family, unknown.context_tokens, unknown.params_b), ("generic", 4096, None))

    def test_thinking_switch_per_server(self):
        self.assertEqual(slm_profiles.ModelProfile("qwen3:4b", server_kind="ollama").thinking_fields(),
                         {"reasoning_effort": "none"})
        self.assertEqual(slm_profiles.ModelProfile("qwen3-4b", server_kind="llamacpp").thinking_fields(),
                         {"chat_template_kwargs": {"enable_thinking": False}})
        lmstudio = slm_profiles.ModelProfile("qwen3-4b", server_kind="lmstudio")
        self.assertEqual((lmstudio.thinking_fields(), lmstudio.thinking_suffix()), ({}, " /no_think"))
        self.assertEqual(slm_profiles.ModelProfile("gemma-4-e2b", server_kind="ollama").thinking_fields(), {})
        self.assertEqual(slm_profiles.ModelProfile("deepseek-r1-distill-qwen-1.5b").output_multiplier(), 3)

    def test_ollama_probe_uses_loaded_or_configured_context(self):
        def fake(url, timeout=1.5, data=None):
            if url.endswith("/api/show"):
                return {"model_info": {"qwen3.context_length": 40960, "general.parameter_count": 2.03e9},
                        "parameters": "temperature 0.6\nnum_ctx 8192"}
            if url.endswith("/api/ps"):
                return {"models": []}
            raise AssertionError(url)
        with patch.object(slm_profiles, "_get_json", side_effect=fake):
            registry = slm_profiles.ProfileRegistry(metadata_lookup=lambda *a: {"owned_by": "library"})
            profile = registry.profile("qwen3:fast", "http://127.0.0.1:11434/v1")
        self.assertEqual((profile.server_kind, profile.context_tokens, profile.train_context), ("ollama", 8192, 40960))
        self.assertEqual(profile.size_class, "lite")

    def test_llamacpp_probe_reads_props_and_enables_exact_tokenizer(self):
        def fake(url, timeout=1.5, data=None):
            if url.endswith("/props"):
                return {"default_generation_settings": {"n_ctx": 2048}, "chat_template": "{{ user }}"}
            raise AssertionError(url)
        meta = {"owned_by": "llamacpp", "meta": {"n_ctx_train": 32768, "n_params": 2.0e9}}
        with patch.object(slm_profiles, "_get_json", side_effect=fake):
            profile = slm_profiles.ProfileRegistry(metadata_lookup=lambda *a: meta).profile(
                "gemma-3n-E2B", "http://127.0.0.1:8080/v1")
        self.assertEqual((profile.server_kind, profile.context_tokens, profile.train_context), ("llamacpp", 2048, 32768))
        self.assertFalse(profile.system_role)
        self.assertEqual(profile.tokenize_url, "http://127.0.0.1:8080/tokenize")

    def test_env_overrides_and_probe_failure_is_safe(self):
        os.environ["LOCAL_REASONING_CONTEXT_TOKENS"] = "3000"
        os.environ["LOCAL_REASONING_STRUCTURED_OUTPUT"] = "json_object"
        with patch.object(slm_profiles, "_get_json", side_effect=OSError("down")):
            profile = slm_profiles.ProfileRegistry(metadata_lookup=lambda *a: {"owned_by": "library"}).profile(
                "x", "http://127.0.0.1:11434/v1")
        self.assertEqual((profile.context_tokens, profile.context_source, profile.structured), (3000, "env", "json_object"))

    def test_structured_downgrade_and_calibration(self):
        registry = slm_profiles.ProfileRegistry(metadata_lookup=lambda *a: None, probes={})
        profile = registry.profile("m", "http://127.0.0.1:1/v1")
        self.assertEqual([registry.downgrade_structured(profile) for _ in range(3)], ["json_object", "off", "off"])
        self.assertEqual(registry.profile("m", "http://127.0.0.1:1/v1").structured, "off")
        registry.calibrate(profile, 4000, 1000 + 40, overhead_tokens=40)
        self.assertAlmostEqual(profile.chars_per_token, 3.68, places=2)        # 4.0 * 0.92 safety
        registry.calibrate(profile, 100000, 10)                                 # absurd values are clamped
        self.assertLessEqual(profile.chars_per_token, 5.0)

    def test_token_counter_exact_then_fallback(self):
        profile = slm_profiles.ModelProfile("m", tokenize_url="http://127.0.0.1:9/tokenize", tokenize_style="llamacpp")
        counter = slm_profiles.TokenCounter(profile)
        with patch.object(slm_profiles, "_get_json", return_value={"tokens": [1, 2, 3]}):
            self.assertEqual(counter.count("hello world"), 3)
        with patch.object(slm_profiles, "_get_json", side_effect=OSError):
            self.assertEqual(counter.count("x" * 30), 10)        # 30 bytes / 3.0 chars per token (generic)


# ═══════════════════════════════════════════════════════════════════════════
#  2. BUDGETS & COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════

class BudgetTests(EnvTestCase):

    def test_never_overflows_any_window(self):
        rng = random.Random(3)
        for _ in range(60):
            ctx = rng.choice([768, 1024, 2048, 4096, 8192, 32768])
            tier = rng.choice(["lite", "balanced", "max"])
            profile = slm_profiles.ModelProfile("m", context_tokens=ctx)
            counter = slm_profiles.TokenCounter(profile)
            parts = slm_context.PromptParts("trim 1 to 2 and " * rng.randint(1, 400), rng.choice(["audio", "video", None]),
                                            "Measured condition: " + "noisy " * rng.randint(0, 300),
                                            "[Memory]\nPreferences:\n" + "- likes mp3\n" * rng.randint(0, 400)
                                            + "Recent turns:\nUSER: hi\n", "")
            messages, budget = slm_context.compose(parts, profile, tier, counter)
            total = slm_context.prompt_tokens(messages, counter)
            with self.subTest(ctx=ctx, tier=tier):
                self.assertLessEqual(total + budget.output, ctx)
                self.assertGreaterEqual(budget.tools_k, 1)

    def test_lite_prompt_is_small_and_far_below_the_legacy_prompt(self):
        profile = slm_profiles.ModelProfile("m", context_tokens=4096)
        counter = slm_profiles.TokenCounter(profile)
        sizes = []
        for prompt, media_type, _ in golden():
            messages, _ = slm_context.compose(slm_context.PromptParts(prompt, media_type, "Current Loaded Media: {}"),
                                              profile, "lite", counter)
            sizes.append(slm_context.prompt_tokens(messages, counter))
        self.assertLessEqual(max(sizes), 1536)
        import agent_processor
        self.assertLess(max(sizes), counter.count(agent_processor.SYSTEM_PROMPT) * 0.6)

    def test_static_prefix_identity_and_privacy(self):
        profile = slm_profiles.ModelProfile("m")
        counter = slm_profiles.TokenCounter(profile)
        a, _ = slm_context.compose(slm_context.PromptParts("make an instrumental", "audio"), profile, "lite", counter)
        b, _ = slm_context.compose(slm_context.PromptParts("upscale 2x", "image"), profile, "lite", counter)
        core = slm_context.core_prompt()
        self.assertTrue(a[0]["content"].startswith(core) and b[0]["content"].startswith(core))
        self.assertIn("Never name or describe the underlying models", core)
        blob = (core + json.dumps(slm_context.FEW_SHOT_POOL)).lower()
        for term in FORBIDDEN:
            self.assertNotIn(term, blob.replace("ai assistant", ""))

    def test_no_system_role_models_get_one_user_turn(self):
        profile = slm_profiles.ModelProfile("gemma-3n-E2B-it")
        messages, _ = slm_context.compose(slm_context.PromptParts("mute it", "video"), profile, "lite",
                                          slm_profiles.TokenCounter(profile))
        self.assertEqual([m["role"] for m in messages], ["user"])
        self.assertTrue(messages[0]["content"].startswith(slm_context.core_prompt()))

    def test_memory_truncation_keeps_preferences_and_newest_turns(self):
        block = ("[Memory]\nPreferences:\n- mp3 320k\nRelated earlier:\n" + "".join(f"- old {i}\n" for i in range(40))
                 + "Recent turns:\nUSER: first\nASSISTANT: second\nUSER: newest")
        count = slm_profiles.TokenCounter(slm_profiles.ModelProfile("m")).count
        out = slm_context.truncate_memory(block, 30, count)
        self.assertLessEqual(count(out), 30)
        self.assertIn("mp3 320k", out)
        self.assertIn("newest", out)
        self.assertNotIn("old 3", out)


class RetrievalTests(unittest.TestCase):

    def test_tool_retrieval_recall_on_golden_set(self):
        hits = total = 0
        misses = []
        for prompt, media_type, expectation in golden():
            expected = set(expectation["tools"])
            if not expected:
                continue
            chosen = {spec.name for spec in slm_context.select_tools(prompt, media_type, (), k=8)}
            total += 1
            if expected <= chosen:
                hits += 1
            else:
                misses.append((prompt, sorted(expected - chosen)))
        recall = hits / total
        print(f"\n[tool retrieval] recall@8 without rule hints = {recall:.3f} ({hits}/{total}); misses={misses[:5]}")
        self.assertGreaterEqual(recall, 0.9)

    def test_rule_hints_are_always_retrieved(self):
        chosen = {s.name for s in slm_context.select_tools("do the thing", "video", ("restore_faces", "compress_video"), 3)}
        self.assertIn("compress_video", chosen)

    def test_few_shots_prefer_same_media_and_clarification(self):
        shots = slm_context.select_shots("speed it up please", "video", 2, want_clarify=True)
        self.assertTrue(shots[0]["plan"]["clarification_needed"])

    def test_strict_schema_pins_names_and_required_args(self):
        specs = [agent_planner.TOOL_REGISTRY[n] for n in ("trim_audio", "adjust_audio_speed")]
        schema = slm_context.plan_schema(specs, skill_ids=("podcast-polish",))
        variants = schema["properties"]["tools"]["items"]["anyOf"]
        self.assertEqual([v["properties"].get("name", {}).get("const") for v in variants],
                         ["trim_audio", "adjust_audio_speed", None])
        self.assertEqual(variants[1]["properties"]["args"]["required"], ["speed"])
        self.assertEqual(list(schema["properties"])[0], "thought")
        json.dumps(schema)


# ═══════════════════════════════════════════════════════════════════════════
#  3. RUNTIME: FIELDS, DIAGNOSIS, ROUTING, REPAIR, ESCALATION
# ═══════════════════════════════════════════════════════════════════════════

class RuntimeTests(EnvTestCase):

    def test_request_fields_per_mechanism(self):
        schema = {"type": "object"}
        profile = slm_profiles.ModelProfile("qwen3:1.7b", server_kind="ollama")
        fields = slm_runtime.request_fields(profile, schema)
        self.assertEqual(fields["response_format"]["type"], "json_schema")
        self.assertEqual((fields["temperature"], fields["reasoning_effort"]), (0.0, "none"))
        profile.structured = "json_object"
        self.assertEqual(slm_runtime.request_fields(profile, schema)["response_format"], {"type": "json_object"})
        profile.structured = "off"
        self.assertNotIn("response_format", slm_runtime.request_fields(profile, schema))
        chat = slm_runtime.request_fields(slm_profiles.ModelProfile("gemma-4-e2b", server_kind="llamacpp"), schema, "chat")
        self.assertEqual((chat["temperature"], chat["cache_prompt"]), (0.7, True))

    def test_diagnose(self):
        d = slm_runtime.diagnose
        self.assertEqual(d(json.dumps({"tools": [{"name": "trim_audio", "args": {"start_sec": 1, "end_sec": 2}}]})), [])
        self.assertIn("cut off", d('{"thought": "x", "tools": [{"name": "trim')[0])
        self.assertIn("one JSON object", d("Sure, trimming now!")[0])
        self.assertEqual(d("LUFS is a loudness unit.", mode="chat"), [])
        self.assertIn("Unknown tool", d(json.dumps({"tools": [{"name": "hyperdrive", "args": {}}]}), allowed={"trim_audio"})[0])
        self.assertIn("valid args: speed",
                      d(json.dumps({"tools": [{"name": "adjust_audio_speed", "args": {"speed": 2, "pitchiness": 1}}]}))[0])
        self.assertIn("needs speed", d(json.dumps({"tools": [{"name": "adjust_audio_speed", "args": {}}]}))[0])
        self.assertEqual(d(json.dumps({"tools": [{"name": "trim", "args": {"start": 1, "end": 2}}]})), [])

    def _session(self, prompt, local, ladder, routing="auto"):
        os.environ["LOCAL_REASONING_ROUTING"] = routing
        return slm_runtime.PlanningSession(prompt, {"type": "audio"}, None, lambda: local, ladder=ladder)

    def test_routing_modes(self):
        confident = {"tools": [{"name": "normalize_audio", "args": {}}], "clarification_needed": False, "validated": True}
        unsure = {"tools": [], "clarification_needed": True}
        lite = [variant(TIER_LITE)]
        self.assertTrue(self._session("normalize", confident, lite).skip_llm)
        self.assertFalse(self._session("normalize", confident, []).skip_llm)          # nothing discovered: legacy
        self.assertTrue(self._session("normalize", confident, [], "cascade").skip_llm)
        self.assertFalse(self._session("normalize", confident, lite, "always").skip_llm)
        self.assertTrue(self._session("trim it", unsure, lite, "local").skip_llm)
        self.assertFalse(self._session("trim it", unsure, lite).skip_llm)
        self.assertEqual(self._session("what is lufs", {"tools": [], "reply": "x"}, lite).mode, "chat")

    def test_repair_then_success(self):
        unsure = {"tools": [], "clarification_needed": True}
        session = self._session("speed it up 1.5x nicely", unsure, [])
        session.payload("Current Loaded Media: {}", "", "", {"model": "x"})
        answers = iter(["I'll speed it up!", json.dumps({"tools": [{"name": "adjust_audio_speed", "args": {"speed": 1.5}}],
                                                         "reply": "ok"})])
        sent = []
        content = session.complete(lambda body: sent.append(body) or next(answers))
        self.assertIn("adjust_audio_speed", content)
        self.assertEqual(len(sent), 2)
        self.assertTrue(sent[1]["messages"][-1]["content"].startswith("Your answer had problems"))

    def test_escalates_simple_requests_from_smallest_tier(self):
        unsure = {"tools": [], "clarification_needed": True}
        ladder = [variant(TIER_MAX), variant(TIER_BALANCED), variant(TIER_LITE)]
        session = self._session("speed it up 2x", unsure, ladder)
        self.assertEqual(session.tier_groups(), [[TIER_LITE], [TIER_MAX, TIER_BALANCED]])
        long_session = self._session("a, b, c, d and e", unsure, ladder)
        self.assertEqual(long_session.tier_groups(), [[TIER_MAX, TIER_BALANCED, TIER_LITE]])
        session.payload("", "", "", {})
        bad = json.dumps({"tools": [{"name": "warp", "args": {}}]})
        good = json.dumps({"tools": [{"name": "adjust_audio_speed", "args": {"speed": 2}}]})
        tiers_seen = []

        def fake(body):
            tiers_seen.append(tuple(body.get("_tiers") or ()))
            return good if body.get("_tiers") == [TIER_MAX, TIER_BALANCED] else bad
        self.assertEqual(session.complete(fake), good)
        self.assertEqual(tiers_seen, [(TIER_LITE,), (TIER_LITE,), (TIER_MAX, TIER_BALANCED)])

    def test_long_requests_are_decomposed_for_a_small_only_ladder(self):
        unsure = {"tools": [], "clarification_needed": True}
        session = self._session("trim 1 to 9, denoise, normalize, fade out, speed up 1.5x, export as flac and louder",
                                unsure, [variant(TIER_LITE)])
        self.assertTrue(session.decompose)
        session.payload("", "", "", {})
        replies = iter([json.dumps({"tools": [{"name": "trim_audio", "args": {"start_sec": 1, "end_sec": 9}},
                                              {"name": "reduce_noise", "args": {}}, {"name": "normalize_audio", "args": {}}]}),
                        json.dumps({"tools": [{"name": "apply_audio_fade", "args": {"fade_in_sec": 0, "fade_out_sec": 2}},
                                              {"name": "adjust_audio_speed", "args": {"speed": 1.5}},
                                              {"name": "convert_audio_format", "args": {"target_format": "flac"}}]}),
                        json.dumps({"tools": [{"name": "adjust_volume", "args": {"gain_db": 6}}]})])
        merged = json.loads(session.complete(lambda body: next(replies)))
        self.assertEqual(len(merged["tools"]), 7)

    def test_guard_rejects_invented_values_and_chat_edits(self):
        unsure = {"tools": [], "clarification_needed": True, "clarification_options": ["Speed up 1.5x"], "reply": "Speed?"}
        invented = json.dumps({"tools": [{"name": "adjust_audio_speed", "args": {"speed": 1.5}}], "reply": "ok"})
        guarded = json.loads(self._session("make it faster", unsure, []).guard(invented))
        self.assertEqual((guarded["tools"], guarded["clarification_needed"]), ([], True))
        stated = self._session("make it 25% faster", unsure, [])
        self.assertEqual(stated.guard(invented.replace("1.5", "1.25")), invented.replace("1.5", "1.25"))
        fmt = json.dumps({"tools": [{"name": "convert_audio_format", "args": {"target_format": "mp3"}}]})
        self.assertTrue(json.loads(self._session("export as m4a", unsure, []).guard(fmt))["clarification_needed"])
        chat = self._session("hello", {"tools": [], "reply": "Hi!"}, [])
        out = json.loads(chat.guard(json.dumps({"tools": [], "clarification_needed": True, "reply": ""})))
        self.assertEqual((out["clarification_needed"], out["reply"]), (False, "Hi!"))

    def test_attempt_payload_renders_per_tier_and_strips_private_keys(self):
        payload = {"model": "x", "messages": [], "_render": lambda v: {"messages": [{"role": "user", "content": v.variant_id}],
                                                                        "_secret": 1}, "_tiers": ["lite"]}
        body = reasoning_models.attempt_payload(variant(TIER_LITE, "abc"), payload)
        self.assertEqual(body, {"messages": [{"role": "user", "content": "abc-lite"}], "model": "abc-lite"})


# ═══════════════════════════════════════════════════════════════════════════
#  4. END TO END AGAINST THE MOCK SMALL-MODEL SERVER
# ═══════════════════════════════════════════════════════════════════════════

class MockServerPipelineTests(unittest.TestCase):

    def setUp(self):
        import logging
        logging.disable(logging.WARNING)

    def tearDown(self):
        import logging
        logging.disable(logging.NOTSET)

    def test_overflow_and_format_rejection_recover_in_tier(self):
        import slm_eval
        model = slm_eval.MockModel(n_ctx=1200)
        model.expect("normalize then fade out over 2 seconds", "audio",
                     {"tools": [{"name": "normalize_audio", "args": {}}], "reply": "ok"})
        with slm_eval.MockServer(model) as server:
            env = {"LOCAL_REASONING_URL": server.url, "LOCAL_REASONING_MODEL": "mock-2b",
                   "LOCAL_REASONING_ROUTING": "always", "LOCAL_REASONING_CONTEXT_TOKENS": "4096"}
            with slm_eval.configured(env):
                import agent_processor
                plan = agent_processor.query_agent_orchestrator("normalize then fade out over 2 seconds", {"type": "audio"})
                events = [s["event"] for s in slm_runtime.recent_stats(clear=True)]
        self.assertIn("recover", events)                 # 4096 assumed, server said 1200 -> shrink and retry
        self.assertIn("call", events)
        self.assertTrue(plan["tools"])

    def test_runtime_beats_legacy_on_mock_small_model(self):
        import slm_eval
        legacy, always, auto = slm_eval.offline(ctx=2048, limit=30, timeout_every=0)
        print(f"\n[mock 2B, ctx=2048] accuracy legacy={legacy['accuracy']} always={always['accuracy']} "
              f"auto={auto['accuracy']}; json-valid legacy={legacy['json_valid_first']} always={always['json_valid_first']}; "
              f"tokens {legacy['prompt_tokens_mean_est']}->{always['prompt_tokens_mean_est']}; errors {legacy['errors']}")
        # The legacy prompt does not fit a 2048 window: every call fails and only the rule fallback answers.
        self.assertGreater(sum(legacy["errors"].values()), 0)
        self.assertEqual(always["errors"], {})
        self.assertGreater(always["json_valid_first"], 0.95)
        self.assertGreaterEqual(auto["accuracy"], legacy["accuracy"])
        self.assertLess(always["prompt_tokens_mean_est"], legacy["prompt_tokens_mean_est"] * 0.6)
        self.assertLess(auto["model_calls_per_request"], always["model_calls_per_request"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
