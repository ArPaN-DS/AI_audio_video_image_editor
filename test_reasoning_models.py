"""
Tiered reasoning model registry tests (small / mid / large local models).

Run:  python test_reasoning_models.py
"""

import json
import os
import socket
import sys
import unittest
import urllib.error
from unittest.mock import patch

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

import model_manager
import reasoning_models
from model_manager import AdaptiveQualityGovernor, ResourceSnapshot, TIER_BALANCED, TIER_LITE, TIER_MAX

BASE = "http://127.0.0.1:8000/v1"
ENV_KEYS = ("LOCAL_REASONING_URL", "VLLM_API_BASE", "LOCAL_REASONING_MODEL", "VLLM_MODEL_NAME",
            "LOCAL_REASONING_MODEL_BALANCED", "LOCAL_REASONING_MODEL_MAX", "LOCAL_REASONING_URL_BALANCED",
            "LOCAL_REASONING_URL_MAX", "LOCAL_REASONING_TIER")


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def governor_for(tier):
    hardware = {TIER_LITE: (1.0, 2, 0.0), TIER_BALANCED: (4.0, 4, 0.0), TIER_MAX: (12.0, 16, 0.0)}[tier]
    return AdaptiveQualityGovernor(probe=lambda: ResourceSnapshot(hardware[0], 16, hardware[1], hardware[2], 0),
                                   clock=FakeClock(), env={})


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


class ReasoningRegistryTests(unittest.TestCase):

    def setUp(self):
        self.saved_env = {key: os.environ.pop(key, None) for key in ENV_KEYS}
        os.environ.update({
            "LOCAL_REASONING_URL": BASE,
            "LOCAL_REASONING_MODEL": "small-2b",
            "LOCAL_REASONING_MODEL_BALANCED": "mid-4b",
            "LOCAL_REASONING_MODEL_MAX": "large-12b",
        })
        self.governor = governor_for(TIER_MAX)
        self.patches = [
            patch.object(reasoning_models, "global_quality_governor", self.governor),
            patch.object(model_manager.global_model_manager, "_flush_system_memory"),
        ]
        for active in self.patches:
            active.start()
        reasoning_models.discovery.invalidate()

    def tearDown(self):
        for active in self.patches:
            active.stop()
        for key, value in self.saved_env.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value
        reasoning_models.discovery.invalidate()

    def ladder_ids(self, served):
        ladder = reasoning_models.reasoning_ladder(served_lookup=lambda url: served)
        return [variant.variant_id for variant in ladder]

    # ── Configuration ────────────────────────────────────────────────────

    def test_today_only_small_model_configured(self):
        os.environ.pop("LOCAL_REASONING_MODEL_BALANCED")
        os.environ.pop("LOCAL_REASONING_MODEL_MAX")
        self.assertEqual(self.ladder_ids(None), ["small-2b"])

    def test_legacy_variable_names_still_work(self):
        for key in ("LOCAL_REASONING_URL", "LOCAL_REASONING_MODEL"):
            os.environ.pop(key)
        os.environ["VLLM_API_BASE"] = "http://localhost:9000/v1"
        os.environ["VLLM_MODEL_NAME"] = "legacy-model"
        tiers = reasoning_models.configured_tiers()
        self.assertEqual((tiers[0].model_id, tiers[0].base_url), ("legacy-model", "http://localhost:9000/v1"))

    def test_non_local_endpoints_are_ignored(self):
        os.environ["LOCAL_REASONING_URL_MAX"] = "https://abc.ngrok.app/v1"
        ids = [tier.model_id for tier in reasoning_models.configured_tiers()]
        self.assertNotIn("large-12b", ids)
        self.assertIsNone(reasoning_models.normalize_base_url("http://10.0.0.5:8000/v1"))
        self.assertIsNone(reasoning_models.normalize_base_url("file:///etc/passwd"))

    # ── Selection ────────────────────────────────────────────────────────

    def test_strong_machine_prefers_largest_served_model(self):
        self.assertEqual(self.ladder_ids({"small-2b", "mid-4b", "large-12b"}),
                         ["large-12b", "mid-4b", "small-2b"])

    def test_only_served_models_are_attempted(self):
        self.assertEqual(self.ladder_ids({"small-2b", "mid-4b"}), ["mid-4b", "small-2b"])

    def test_discovery_unsupported_assumes_configured(self):
        self.assertEqual(self.ladder_ids(None), ["large-12b", "mid-4b", "small-2b"])

    def test_server_offline_yields_empty_ladder(self):
        self.assertEqual(self.ladder_ids(frozenset()), [])

    def test_hardware_tier_caps_model_size(self):
        with patch.object(reasoning_models, "global_quality_governor", governor_for(TIER_BALANCED)):
            self.assertEqual(self.ladder_ids({"small-2b", "mid-4b", "large-12b"}), ["mid-4b", "small-2b"])
        with patch.object(reasoning_models, "global_quality_governor", governor_for(TIER_LITE)):
            self.assertEqual(self.ladder_ids({"small-2b", "mid-4b", "large-12b"}), ["small-2b"])

    def test_pinned_reasoning_tier_overrides_hardware(self):
        os.environ["LOCAL_REASONING_TIER"] = "lite"
        self.assertEqual(self.ladder_ids({"small-2b", "large-12b"}), ["small-2b"])
        os.environ["LOCAL_REASONING_TIER"] = "max"
        with patch.object(reasoning_models, "global_quality_governor", governor_for(TIER_LITE)):
            self.assertEqual(self.ladder_ids({"small-2b", "large-12b"}), ["large-12b", "small-2b"])

    def test_only_a_big_model_served_on_weak_machine_still_works(self):
        with patch.object(reasoning_models, "global_quality_governor", governor_for(TIER_LITE)):
            self.assertEqual(self.ladder_ids({"large-12b"}), ["large-12b"])

    # ── Execution & resilience ───────────────────────────────────────────

    def test_falls_back_and_pauses_slow_large_model(self):
        calls = []

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/models"):
                return Response({"data": [{"id": "small-2b"}, {"id": "mid-4b"}, {"id": "large-12b"}]})
            model = json.loads(req.data.decode("utf-8"))["model"]
            calls.append((model, timeout))
            if model == "large-12b":
                raise socket.timeout("timed out")
            return Response({"choices": [{"message": {"content": f"plan from {model}"}}]})

        with patch.object(reasoning_models.urllib.request, "urlopen", side_effect=fake_urlopen):
            content, label = reasoning_models.request_completion({"messages": []})
            self.assertEqual(content, "plan from mid-4b")
            self.assertEqual(label, "Enhanced Reasoning")
            self.assertTrue(self.governor.is_paused(reasoning_models.CAPABILITY, "large-12b"))
            self.assertEqual(calls[0], ("large-12b", reasoning_models.TIER_TIMEOUTS_SEC[TIER_MAX]))

            calls.clear()
            reasoning_models.request_completion({"messages": []})
            self.assertEqual([model for model, _ in calls], ["mid-4b"])

    def test_wrapped_url_timeout_is_treated_as_slow_tier(self):
        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/models"):
                raise urllib.error.HTTPError(req.full_url, 404, "no", None, None)
            model = json.loads(req.data.decode("utf-8"))["model"]
            if model != "small-2b":
                raise urllib.error.URLError(socket.timeout("timed out"))
            return Response({"choices": [{"message": {"content": "ok"}}]})

        with patch.object(reasoning_models.urllib.request, "urlopen", side_effect=fake_urlopen):
            content, _ = reasoning_models.request_completion({"messages": []})
        self.assertEqual(content, "ok")
        self.assertTrue(self.governor.is_paused(reasoning_models.CAPABILITY, "large-12b"))

    def test_offline_server_fails_fast_without_posting(self):
        posts = []

        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/models"):
                raise urllib.error.URLError(ConnectionRefusedError("refused"))
            posts.append(req)
            raise AssertionError("must not post when offline")

        with patch.object(reasoning_models.urllib.request, "urlopen", side_effect=fake_urlopen):
            with self.assertRaises(ConnectionError):
                reasoning_models.request_completion({"messages": []})
        self.assertEqual(posts, [])

    def test_malformed_response_falls_through(self):
        def fake_urlopen(req, timeout=None):
            if req.full_url.endswith("/models"):
                return Response({"data": [{"id": "small-2b"}, {"id": "large-12b"}]})
            model = json.loads(req.data.decode("utf-8"))["model"]
            if model == "large-12b":
                return Response({"unexpected": True})
            return Response({"choices": [{"message": {"content": "fine"}}]})

        with patch.object(reasoning_models.urllib.request, "urlopen", side_effect=fake_urlopen):
            content, _ = reasoning_models.request_completion({"messages": []})
        self.assertEqual(content, "fine")

    def test_public_status_has_no_model_ids(self):
        with patch.object(reasoning_models.discovery, "served", return_value={"small-2b", "large-12b"}):
            status = reasoning_models.public_status()
        text = json.dumps(status).lower()
        for private in ("small-2b", "large-12b", "mid-4b", "127.0.0.1"):
            self.assertNotIn(private, text)
        self.assertTrue(status["reasoning_available"])
        self.assertEqual(status["reasoning_mode"], "Extended Reasoning")


class DiscoveryCacheTests(unittest.TestCase):

    def test_results_are_cached_and_offline_rechecked_sooner(self):
        clock = FakeClock()
        calls = []

        def fetch(url):
            calls.append(url)
            raise urllib.error.URLError("down")

        discovery = reasoning_models.ServedModelDiscovery(fetch=fetch, clock=clock)
        self.assertEqual(discovery.served(BASE), frozenset())
        self.assertEqual(discovery.served(BASE), frozenset())
        self.assertEqual(len(calls), 1)
        clock.now += reasoning_models.DISCOVERY_OFFLINE_TTL_SEC + 1
        discovery.served(BASE)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
