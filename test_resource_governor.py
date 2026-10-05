"""
Adaptive Quality Governor & Model Lifecycle regression suite.

Covers hardware-tier classification, hysteresis, pinned tiers, capability
ladders, out-of-memory demotion with cooldown, concurrency, memory headroom
guards, mid-session instance release, and the processors/routes wired to it.

Run:  python test_resource_governor.py
"""

import gc
import os
import sys
import tempfile
import subprocess
import threading
import unittest
import weakref
from unittest.mock import patch

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

import model_manager
from model_manager import (
    AdaptiveQualityGovernor,
    CapabilityVariant,
    ModelLifecycleManager,
    ResourceExhaustedError,
    ResourceSnapshot,
    TIER_BALANCED,
    TIER_LITE,
    TIER_MAX,
    is_resource_error,
)

PRIVATE_TERMS = ('gemma', 'whisper', 'vllm', 'isnet', 'u2net', 'birefnet', 'rembg',
                 'edsr', 'fsrcnn', 'onnx', 'torch', 'opencv', 'ffmpeg')


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeHardware:
    def __init__(self, ram_free=2.0, cores=4, gpu_free=0.0):
        self.ram_free = ram_free
        self.cores = cores
        self.gpu_free = gpu_free
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return ResourceSnapshot(self.ram_free, 16.0, self.cores, self.gpu_free, 8.0 if self.gpu_free else 0.0)


def make_governor(ram_free=2.0, cores=4, gpu_free=0.0, env=None):
    hardware = FakeHardware(ram_free, cores, gpu_free)
    clock = FakeClock()
    governor = AdaptiveQualityGovernor(probe=hardware, clock=clock, env=env if env is not None else {})
    return governor, hardware, clock


def ladder(heavy_ram=6.0, heavy_tier=TIER_MAX, heavy_available=True):
    return [
        CapabilityVariant("heavy", tier=heavy_tier, min_ram_gb=heavy_ram, min_cpu_cores=4,
                          is_available=lambda: heavy_available),
        CapabilityVariant("baseline"),
    ]


class TierClassificationTests(unittest.TestCase):

    def test_low_resources_stay_on_lite(self):
        governor, _, _ = make_governor(ram_free=1.5, cores=2)
        self.assertEqual(governor.current_tier(), TIER_LITE)

    def test_balanced_and_max_cpu_paths(self):
        self.assertEqual(make_governor(ram_free=4.0, cores=4)[0].current_tier(), TIER_BALANCED)
        self.assertEqual(make_governor(ram_free=12.0, cores=16)[0].current_tier(), TIER_MAX)

    def test_many_cores_without_ram_is_not_max(self):
        self.assertEqual(make_governor(ram_free=2.5, cores=32)[0].current_tier(), TIER_LITE)

    def test_gpu_memory_promotes_tier(self):
        governor, _, _ = make_governor(ram_free=2.5, cores=4, gpu_free=7.5)
        self.assertEqual(governor.current_tier(), TIER_MAX)

    def test_probe_failure_degrades_safely(self):
        def broken_probe():
            raise OSError("probe exploded")
        governor = AdaptiveQualityGovernor(probe=broken_probe, clock=FakeClock(), env={})
        self.assertEqual(governor.current_tier(), TIER_LITE)
        self.assertEqual(governor.plan("cap", ladder())[0].variant_id, "baseline")

    def test_invalid_probe_result_degrades_safely(self):
        governor = AdaptiveQualityGovernor(probe=lambda: {"ram": 64}, clock=FakeClock(), env={})
        self.assertEqual(governor.current_tier(), TIER_LITE)

    def test_snapshot_is_cached_within_ttl(self):
        governor, hardware, clock = make_governor(ram_free=4.0)
        governor.current_tier()
        governor.current_tier()
        self.assertEqual(hardware.calls, 1)
        clock.advance(AdaptiveQualityGovernor.SNAPSHOT_TTL_SEC + 0.1)
        governor.current_tier()
        self.assertEqual(hardware.calls, 2)

    def test_hysteresis_prevents_flapping(self):
        governor, hardware, clock = make_governor(ram_free=4.0, cores=16)
        self.assertEqual(governor.current_tier(), TIER_BALANCED)

        # Barely over the max threshold: not enough margin to upgrade.
        hardware.ram_free = 8.2
        clock.advance(10)
        self.assertEqual(governor.current_tier(), TIER_BALANCED)

        # Clearly over: upgrade.
        hardware.ram_free = 9.5
        clock.advance(10)
        self.assertEqual(governor.current_tier(), TIER_MAX)

        # Dip just under the floor: hold the tier.
        hardware.ram_free = 7.6
        clock.advance(10)
        self.assertEqual(governor.current_tier(), TIER_MAX)

        # Real pressure: downgrade.
        hardware.ram_free = 3.5
        clock.advance(10)
        self.assertEqual(governor.current_tier(), TIER_BALANCED)

    def test_pinned_tier_override_and_aliases(self):
        governor, _, _ = make_governor(ram_free=32.0, cores=16, env={"MEDIA_QUALITY_TIER": "low"})
        self.assertEqual(governor.current_tier(), TIER_LITE)
        governor, _, _ = make_governor(ram_free=1.0, cores=2, env={"MEDIA_QUALITY_TIER": "MAX"})
        self.assertEqual(governor.current_tier(), TIER_MAX)
        governor, _, _ = make_governor(ram_free=1.0, cores=2, env={"MEDIA_QUALITY_TIER": "turbo"})
        self.assertEqual(governor.current_tier(), TIER_LITE)
        self.assertIsNone(governor.tier_override())


class LadderSelectionTests(unittest.TestCase):

    def test_rich_machine_selects_heavy_variant(self):
        governor, _, _ = make_governor(ram_free=12.0, cores=16)
        self.assertEqual([v.variant_id for v in governor.plan("cap", ladder())], ["heavy", "baseline"])

    def test_baseline_is_always_last_and_only_when_constrained(self):
        governor, _, _ = make_governor(ram_free=1.0, cores=2)
        self.assertEqual([v.variant_id for v in governor.plan("cap", ladder())], ["baseline"])

    def test_tier_ceiling_excludes_heavier_variants(self):
        governor, _, _ = make_governor(ram_free=12.0, cores=16, env={"MEDIA_QUALITY_TIER": "balanced"})
        self.assertEqual(governor.select("cap", ladder()).variant_id, "baseline")

    def test_variant_specific_ram_requirement(self):
        # Max tier via GPU, but the CPU-bound heavy variant needs real RAM.
        governor, _, _ = make_governor(ram_free=2.4, cores=16, gpu_free=7.5)
        self.assertEqual(governor.current_tier(), TIER_MAX)
        self.assertEqual(governor.select("cap", ladder(heavy_ram=6.0)).variant_id, "baseline")

    def test_gpu_variant_uses_gpu_memory(self):
        governor, _, _ = make_governor(ram_free=2.4, cores=16, gpu_free=7.5)
        gpu_ladder = [CapabilityVariant("gpu-heavy", tier=TIER_MAX, min_ram_gb=1.0, min_gpu_gb=4.0),
                      CapabilityVariant("baseline")]
        self.assertEqual(governor.select("cap", gpu_ladder).variant_id, "gpu-heavy")

    def test_unavailable_or_crashing_availability_check_is_skipped(self):
        governor, _, _ = make_governor(ram_free=12.0, cores=16)
        self.assertEqual(governor.select("cap", ladder(heavy_available=False)).variant_id, "baseline")

        def explode():
            raise RuntimeError("disk gone")
        crashing = [CapabilityVariant("heavy", tier=TIER_MAX, is_available=explode), CapabilityVariant("baseline")]
        self.assertEqual(governor.select("cap", crashing).variant_id, "baseline")

    def test_empty_ladder_rejected(self):
        governor, _, _ = make_governor()
        with self.assertRaises(ValueError):
            governor.plan("cap", [])


class FallbackAndDemotionTests(unittest.TestCase):

    def setUp(self):
        self.flush = patch.object(model_manager.global_model_manager, "_flush_system_memory")
        self.flush.start()

    def tearDown(self):
        self.flush.stop()

    def test_success_uses_heavy_variant(self):
        governor, _, _ = make_governor(ram_free=12.0, cores=16)
        result, variant = governor.run("cap", ladder(), lambda v: f"ran-{v.variant_id}")
        self.assertEqual((result, variant.variant_id), ("ran-heavy", "heavy"))

    def test_out_of_memory_falls_back_and_pauses_heavy_variant(self):
        governor, _, clock = make_governor(ram_free=12.0, cores=16)
        calls = []

        def operation(variant):
            calls.append(variant.variant_id)
            if variant.variant_id == "heavy":
                raise MemoryError("bad_alloc")
            return "ok"

        result, variant = governor.run("cap", ladder(), operation)
        self.assertEqual((result, variant.variant_id), ("ok", "baseline"))
        self.assertEqual(calls, ["heavy", "baseline"])
        self.assertTrue(governor.is_paused("cap", "heavy"))
        self.assertIn("cap", governor.public_status()["paused_capabilities"])

        # During cooldown the heavy variant is not retried.
        calls.clear()
        governor.run("cap", ladder(), operation)
        self.assertEqual(calls, ["baseline"])

        # After cooldown it is eligible again.
        clock.advance(AdaptiveQualityGovernor.BASE_COOLDOWN_SEC + 1)
        self.assertFalse(governor.is_paused("cap", "heavy"))
        self.assertEqual(governor.select("cap", ladder()).variant_id, "heavy")

    def test_cooldown_grows_exponentially_and_is_capped(self):
        governor, _, clock = make_governor(ram_free=12.0, cores=16)
        for strike in range(1, 7):
            governor.report_failure("cap", "heavy", MemoryError())
            expected = min(AdaptiveQualityGovernor.BASE_COOLDOWN_SEC * 2 ** (strike - 1),
                           AdaptiveQualityGovernor.MAX_COOLDOWN_SEC)
            self.assertAlmostEqual(governor._banned_until[("cap", "heavy")] - clock.now, expected)

    def test_success_clears_strikes(self):
        governor, _, _ = make_governor(ram_free=12.0, cores=16)
        governor.report_failure("cap", "heavy", ValueError("glitch"))
        governor.report_success("cap", "heavy")
        governor.report_failure("cap", "heavy", ValueError("glitch"))
        self.assertFalse(governor.is_paused("cap", "heavy"))

    def test_non_resource_error_needs_two_strikes(self):
        governor, _, _ = make_governor(ram_free=12.0, cores=16)
        governor.report_failure("cap", "heavy", ValueError("corrupt frame"))
        self.assertFalse(governor.is_paused("cap", "heavy"))
        governor.report_failure("cap", "heavy", ValueError("corrupt frame"))
        self.assertTrue(governor.is_paused("cap", "heavy"))

    def test_baseline_failure_propagates_and_is_never_paused(self):
        governor, _, _ = make_governor(ram_free=1.0, cores=2)

        def operation(variant):
            raise MemoryError("out of memory")

        with self.assertRaises(MemoryError):
            governor.run("cap", ladder(), operation)
        self.assertFalse(governor.is_paused("cap", "baseline"))

    def test_concurrent_runs_are_consistent(self):
        governor, _, _ = make_governor(ram_free=12.0, cores=16)
        errors, results = [], []
        lock = threading.Lock()

        def operation(variant):
            if variant.variant_id == "heavy":
                raise MemoryError("out of memory")
            return variant.variant_id

        def worker():
            try:
                result, _ = governor.run("cap", ladder(), operation)
                with lock:
                    results.append(result)
            except Exception as error:  # pragma: no cover - failure path
                with lock:
                    errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(24)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertEqual(errors, [])
        self.assertEqual(results, ["baseline"] * 24)
        self.assertTrue(governor.is_paused("cap", "heavy"))

    def test_resource_error_detection(self):
        self.assertTrue(is_resource_error(MemoryError()))
        self.assertTrue(is_resource_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")))
        self.assertTrue(is_resource_error(RuntimeError("std::bad_alloc")))
        self.assertTrue(is_resource_error(ResourceExhaustedError("x")))
        self.assertTrue(is_resource_error("Failed to allocate memory for requested buffer"))
        self.assertFalse(is_resource_error(ValueError("invalid literal for int()")))
        self.assertFalse(is_resource_error(FileNotFoundError("missing.png")))


class LifecycleManagerTests(unittest.TestCase):

    def test_headroom_guard_refuses_load_and_reclaims_first(self):
        manager = ModelLifecycleManager()
        loads = []
        with patch.object(model_manager, "_probe_ram_gb", return_value=(0.5, 16.0)), \
             patch.object(manager, "_flush_system_memory") as flush:
            with self.assertRaises(ResourceExhaustedError):
                with manager.session("big", lambda: (loads.append(1) or object(), {}), min_free_ram_gb=4.0):
                    pass
            flush.assert_called()
        self.assertEqual(loads, [])
        self.assertIsNone(manager._active_model_id)
        self.assertEqual(manager._session_depth, 0)

    def test_headroom_guard_allows_load_when_memory_is_free(self):
        manager = ModelLifecycleManager()
        with patch.object(model_manager, "_probe_ram_gb", return_value=(12.0, 16.0)):
            with manager.session("big", lambda: ("instance", {}), min_free_ram_gb=4.0) as (instance, _):
                self.assertEqual(instance, "instance")
        self.assertIsNone(manager._active_model_id)

    def test_release_active_instance_frees_memory_mid_session(self):
        manager = ModelLifecycleManager()

        class HeavyModel:
            pass

        with patch.object(manager, "_flush_system_memory"):
            with manager.session("stt", lambda: (HeavyModel(), {"tier": "large"})) as (model, _meta):
                probe = weakref.ref(model)
                model = None
                manager.release_active_instance()
                gc.collect()
                self.assertIsNone(probe(), "failed model is still referenced after release")
                manager.adopt_active_instance("small-model", {"tier": "small"})
                self.assertEqual(manager._active_model_instance, "small-model")
        self.assertIsNone(manager._active_model_instance)
        self.assertIsNone(manager._active_model_id)

    def test_adopt_without_active_capability_is_rejected(self):
        with self.assertRaises(RuntimeError):
            ModelLifecycleManager().adopt_active_instance(object())

    def test_single_active_model_is_preserved(self):
        manager = ModelLifecycleManager()
        unloaded = []
        with patch.object(manager, "_flush_system_memory"):
            with manager.session("a", lambda: ("A", {}), lambda inst: unloaded.append(inst)):
                with self.assertRaises(RuntimeError):
                    with manager.session("b", lambda: ("B", {})):
                        pass
        self.assertEqual(unloaded, ["A"])


class ProcessorIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from PIL import Image
        cls.tmp = tempfile.mkdtemp(prefix="governor_test_")
        cls.src = os.path.join(cls.tmp, "src.png")
        Image.new("RGB", (48, 48), (120, 60, 30)).save(cls.src)

    def setUp(self):
        self.governor, _, _ = make_governor(ram_free=12.0, cores=16)
        self.swap = patch.object(model_manager, "global_quality_governor", self.governor)
        self.swap.start()
        self.flush = patch.object(model_manager.global_model_manager, "_flush_system_memory")
        self.flush.start()

    def tearDown(self):
        self.flush.stop()
        self.swap.stop()

    def test_upscale_auto_falls_back_on_memory_failure(self):
        import image_processor
        attempts = []

        def fake_superres(cv2, img, model_key, scale, out_path, min_free_ram_gb=0.0):
            attempts.append(model_key)
            if model_key == "best":
                raise MemoryError("bad_alloc")
            return {"engine": "fsrcnn", "scale": scale, "size": (96, 96)}

        out = os.path.join(self.tmp, "up.png")
        with patch.object(image_processor, "_run_superres", side_effect=fake_superres), \
             patch.object(image_processor, "_model_path", return_value=("x", "present")):
            info = image_processor.upscale(self.src, out, scale=2, model_key="auto")
        self.assertEqual(attempts, ["best", "fast"])
        self.assertEqual(info["engine"], "fsrcnn")
        self.assertTrue(info["adaptive"])
        self.assertTrue(self.governor.is_paused("image.superres", "best"))

    def test_upscale_auto_never_crashes(self):
        import image_processor
        out = os.path.join(self.tmp, "up2.png")
        with patch.object(image_processor, "_run_superres", side_effect=MemoryError("out of memory")):
            info = image_processor.upscale(self.src, out, scale=2, model_key="auto")
        self.assertEqual(info["engine"], "lanczos")
        self.assertTrue(os.path.exists(out))

    def test_explicit_upscale_choice_is_respected(self):
        import image_processor
        seen = []

        def fake_superres(cv2, img, model_key, scale, out_path, min_free_ram_gb=0.0):
            seen.append(model_key)
            return {"engine": model_key, "scale": scale, "size": (96, 96)}

        with patch.object(image_processor, "_run_superres", side_effect=fake_superres), \
             patch.object(image_processor, "_model_path", return_value=("x", "present")):
            image_processor.upscale(self.src, os.path.join(self.tmp, "e.png"), scale=2, model_key="fast")
        self.assertEqual(seen, ["fast"])

    def test_cutout_auto_upgrades_and_falls_back(self):
        import image_processor
        attempts = []

        def fake_cutout(model_name, src, out, alpha_matting, min_free_ram_gb=0.0):
            attempts.append(model_name)
            if model_name == image_processor.CUTOUT_MAX_MODEL:
                raise RuntimeError("Failed to allocate memory for requested buffer")
            return {"engine": "baseline", "status": "success"}

        with patch.object(image_processor, "_cutout_with", side_effect=fake_cutout), \
             patch.object(image_processor, "_cutout_model_installed", return_value=True):
            info = image_processor.remove_bg(self.src, os.path.join(self.tmp, "c.png"), model_name="ultra-hd")
        self.assertEqual(attempts, [image_processor.CUTOUT_MAX_MODEL, image_processor.CUTOUT_BASELINE_MODEL])
        self.assertTrue(info["adaptive"])

    def test_cutout_never_upgrades_without_installed_weights(self):
        import image_processor
        attempts = []

        def fake_cutout(model_name, src, out, alpha_matting, min_free_ram_gb=0.0):
            attempts.append(model_name)
            return {"engine": "baseline", "status": "success"}

        with patch.object(image_processor, "_cutout_with", side_effect=fake_cutout), \
             patch.object(image_processor, "_cutout_model_installed", return_value=False):
            image_processor.remove_bg(self.src, os.path.join(self.tmp, "c2.png"), model_name="auto")
        self.assertEqual(attempts, [image_processor.CUTOUT_BASELINE_MODEL])

    def test_explicit_cutout_profiles_bypass_governor(self):
        import image_processor
        attempts = []

        def fake_cutout(model_name, src, out, alpha_matting, min_free_ram_gb=0.0):
            attempts.append(model_name)
            return {"engine": "x", "status": "success"}

        with patch.object(image_processor, "_cutout_with", side_effect=fake_cutout):
            image_processor.remove_bg(self.src, os.path.join(self.tmp, "c3.png"), model_name="fast")
            image_processor.remove_bg(self.src, os.path.join(self.tmp, "c4.png"), model_name="../../etc")
        self.assertEqual(attempts, ["u2netp", image_processor.CUTOUT_BASELINE_MODEL])

    def test_speech_cascade_respects_pin_and_pauses(self):
        import ai_processor
        governor, _, _ = make_governor(ram_free=12.0, cores=16, env={"MEDIA_QUALITY_TIER": "lite"})
        with patch.object(model_manager, "global_quality_governor", governor):
            names = [tier["name"] for tier in ai_processor._eligible_cascade()]
            self.assertEqual(names[0], "small")
            governor.report_failure("speech.transcribe", "small", MemoryError())
            names = [tier["name"] for tier in ai_processor._eligible_cascade()]
            self.assertNotIn("small", names)
            self.assertTrue(names)

    def test_speech_cascade_keeps_smallest_model_when_all_paused(self):
        import ai_processor
        for tier in ai_processor._MODEL_CASCADE:
            self.governor.report_failure("speech.transcribe", tier["name"], MemoryError())
        names = [tier["name"] for tier in ai_processor._eligible_cascade()]
        self.assertEqual(names, [ai_processor._MODEL_CASCADE[-1]["name"]])


class PublicSurfaceTests(unittest.TestCase):

    def test_status_endpoints_are_public_safe(self):
        from app import app
        client = app.test_client()
        for route in ("/api/system/resources", "/health"):
            with self.subTest(route=route):
                resp = client.get(route)
                self.assertEqual(resp.status_code, 200)
                body = resp.get_data(as_text=True).lower()
                for term in PRIVATE_TERMS:
                    self.assertNotIn(term, body)
        status = client.get("/api/system/resources").get_json()
        self.assertIn(status["quality_tier"], (TIER_LITE, TIER_BALANCED, TIER_MAX))
        self.assertIn("hardware", status)
        self.assertIn("adaptive_quality", client.get("/health").get_json())

    def test_enhance_header_exposes_only_public_tier(self):
        import io
        from PIL import Image
        from app import app
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (10, 200, 90)).save(buf, format="PNG")
        buf.seek(0)
        resp = app.test_client().post(
            "/image/enhance", data={"file": (buf, "x.png"), "scale": "2", "model": "auto"},
            content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(resp.headers.get("X-Enhance-Engine"), ("best", "fast", "standard"))

    def test_import_app_does_not_load_heavy_libraries(self):
        """Ensure heavy libraries (torch, onnxruntime, cv2, ctranslate2, scipy) are not loaded on startup."""
        cmd = [
            sys.executable,
            "-c",
            "import app, sys; "
            "heavy = [m for m in ['torch', 'onnxruntime', 'cv2', 'ctranslate2', 'scipy'] if m in sys.modules]; "
            "assert not heavy, f'Heavy modules imported at startup: {heavy}'"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE_DIR)
        self.assertEqual(res.returncode, 0, f"Startup heavy imports detected: {res.stderr or res.stdout}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
