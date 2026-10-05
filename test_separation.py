"""
Source separation tests (vocals / karaoke / 4 stems / voice isolation / lyrics).

Every test builds synthetic mixes in a temporary directory:
  * "vocal": a harmonic tone with vibrato and syllable gating, panned centre
  * "drums": decorrelated (wide) noise bursts plus short centre hi-hat clicks
  * "bass":  low sustained tones, panned centre

Quality is measured as scale-invariant SDR of each estimated stem against its
true source, compared with using the untouched mix as the estimate. That
difference is the target-to-interference improvement the separation achieved.

Run:  ./venv/Scripts/python.exe -m unittest test_separation -v
"""
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np
import soundfile as sf

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

import separation_processor as sp  # noqa: E402
import ai_processor  # noqa: E402
from model_manager import (AdaptiveQualityGovernor, ResourceSnapshot,  # noqa: E402
                           global_model_manager)

SR = 44100
RNG = np.random.default_rng(2024)

# Engine / library names that must never reach users.
PRIVATE_TERMS = ("demucs", "htdemucs", "torch", "pytorch", "librosa", "scipy", "numpy",
                 "soundfile", "ffmpeg", "whisper", "huggingface", "safetensors", "cuda")


# ── Synthetic sources ────────────────────────────────────────────────────────

def make_sources(seconds=8.0, sr=SR, seed=7):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr

    # Vocal: 220 Hz fundamental, 5 Hz vibrato, six harmonics, sung syllables.
    f0 = 220.0 * (1.0 + 0.03 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    vocal = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 7))
    syllable = 0.5 - 0.5 * np.cos(2 * np.pi * np.clip((t % 0.5) / 0.4, 0, 1))
    syllable[(t % 0.5) > 0.4] = 0.0
    vocal = vocal * syllable

    # Drums: wide decorrelated noise bursts every 250 ms + centre clicks.
    env = np.exp(-((t % 0.25) / 0.03))
    drums_l = rng.standard_normal(n) * env * 0.5
    drums_r = rng.standard_normal(n) * env * 0.5
    click_env = np.exp(-((t % 0.125) / 0.004))
    click = np.diff(rng.standard_normal(n + 1)) * click_env * 0.15
    drums = np.stack([drums_l + click, drums_r + click])

    # Bass: sustained low notes (55 / 41.2 / 49 / 36.7 Hz) with a 2nd harmonic.
    notes = np.array([55.0, 41.2, 49.0, 36.7])
    f_bass = notes[(t // 1.0).astype(int) % len(notes)]
    bphase = 2 * np.pi * np.cumsum(f_bass) / sr
    bass = 0.5 * np.sin(bphase) + 0.2 * np.sin(2 * bphase)

    vocal2 = np.stack([vocal, vocal])
    bass2 = np.stack([bass, bass])
    return {"vocals": vocal2, "drums": drums, "bass": bass2}


def mix_of(sources, mono=False):
    mix = sum(sources.values())
    if mono:
        mix = mix.mean(axis=0, keepdims=True)
    return mix


def si_sdr(estimate, reference):
    est = np.asarray(estimate, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    n = min(est.size, ref.size)
    est, ref = est[:n], ref[:n]
    scale = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = scale * ref
    noise = est - target
    return 10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12))


def read(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.T, sr


def write(path, y, sr=SR):
    sf.write(path, np.asarray(y, dtype=np.float32).T, sr, subtype="FLOAT")
    return path


def generous_governor(gpu_free=0.0):
    snap = ResourceSnapshot(ram_free_gb=12.0, ram_total_gb=16.0, cpu_cores=16,
                            gpu_free_gb=gpu_free, gpu_total_gb=8.0 if gpu_free else 0.0)
    return AdaptiveQualityGovernor(probe=lambda: snap, env={})


class SeparationTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="separation_")
        cls.sources = make_sources()
        cls.mix = mix_of(cls.sources)
        peak = np.max(np.abs(cls.mix))
        cls.gain = 0.8 / peak
        cls.sources = {k: v * cls.gain for k, v in cls.sources.items()}
        cls.mix = cls.mix * cls.gain
        cls.mix_path = write(os.path.join(cls.tmp, "song.wav"), cls.mix)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def out_dir(self, name):
        path = os.path.join(self.tmp, name)
        shutil.rmtree(path, ignore_errors=True)
        return path

    def improvement(self, stem_path, reference):
        est, _ = read(stem_path)
        mix = self.mix if est.shape[0] == self.mix.shape[0] else self.mix.mean(0, keepdims=True)
        ref = reference if reference.shape[0] == est.shape[0] else reference.mean(0, keepdims=True)
        return si_sdr(est, ref) - si_sdr(mix, ref)


# ── Baseline (always available) quality ──────────────────────────────────────

class BaselineQualityTests(SeparationTestCase):

    def test_vocals_mode_improves_vocal_and_instrumental(self):
        report = sp.separate(self.mix_path, self.out_dir("vocals"), mode="vocals", quality="fast")
        self.assertEqual(set(report["stems"]), {"vocals", "instrumental"})
        self.assertEqual(report["quality"]["id"], "quick")
        self.assertFalse(report["quality"]["neural"])
        gain_v = self.improvement(report["stems"]["vocals"], self.sources["vocals"])
        accompaniment = self.sources["drums"] + self.sources["bass"]
        gain_i = self.improvement(report["stems"]["instrumental"], accompaniment)
        self.assertGreater(gain_v, 8.0, f"vocal improvement only {gain_v:.1f} dB")
        self.assertGreater(gain_i, 6.0, f"instrumental improvement only {gain_i:.1f} dB")

    def test_four_stem_mode_improves_every_target(self):
        report = sp.separate(self.mix_path, self.out_dir("four"), mode="4stem", quality="fast")
        self.assertEqual(set(report["stems"]), {"vocals", "drums", "bass", "other"})
        for name, floor in (("vocals", 8.0), ("drums", 3.0), ("bass", 8.0)):
            gain = self.improvement(report["stems"][name], self.sources[name])
            self.assertGreater(gain, floor, f"{name} improvement only {gain:.1f} dB")

    def test_stems_sum_back_to_the_mix(self):
        report = sp.separate(self.mix_path, self.out_dir("recon"), mode="4stem", quality="fast")
        total = sum(read(p)[0] for p in report["stems"].values())
        residual = total - self.mix
        err_db = 10 * np.log10(np.sum(residual ** 2) / np.sum(self.mix ** 2))
        self.assertLess(err_db, -40.0)
        self.assertLess(report["checks"]["reconstruction_error_db"], -40.0)
        self.assertTrue(report["checks"]["passed"])

    def test_report_has_loudness_and_timing(self):
        report = sp.separate(self.mix_path, self.out_dir("report"), mode="vocals", quality="fast")
        self.assertAlmostEqual(report["duration"], 8.0, places=2)
        self.assertEqual(report["sample_rate"], SR)
        self.assertEqual(report["channels"], 2)
        for name in ("vocals", "instrumental"):
            loud = report["loudness"][name]
            self.assertGreater(loud["integrated_lufs"], -40.0)
            self.assertLessEqual(loud["sample_peak_dbfs"], 0.0)
        self.assertGreater(report["elapsed_sec"], 0.0)
        self.assertIn("realtime_factor", report)
        self.assertTrue(report["quality_note"])

    def test_mono_input_still_separates(self):
        mono_path = write(os.path.join(self.tmp, "mono.wav"), self.mix.mean(0, keepdims=True))
        report = sp.separate(mono_path, self.out_dir("mono"), mode="vocals", quality="fast")
        vocals, _ = read(report["stems"]["vocals"])
        self.assertEqual(vocals.shape[0], 1)
        ref = self.sources["vocals"].mean(0, keepdims=True)
        gain = si_sdr(vocals, ref) - si_sdr(self.mix.mean(0, keepdims=True), ref)
        self.assertGreater(gain, 4.0, f"mono vocal improvement only {gain:.1f} dB")
        self.assertTrue(any("mono" in w.lower() for w in report["warnings"]))

    def test_voice_isolation_from_noise_and_music_bed(self):
        n = self.mix.shape[1]
        voice = self.sources["vocals"]
        bed = self.sources["bass"] + 0.6 * self.sources["drums"]
        hiss = 0.05 * RNG.standard_normal((2, n))
        noisy = voice + bed + hiss
        path = write(os.path.join(self.tmp, "noisy_voice.wav"), noisy / np.max(np.abs(noisy)) * 0.8)
        out = os.path.join(self.out_dir("voice"), "voice.wav")
        report = sp.isolate_voice(path, out, quality="fast")
        self.assertEqual(set(report["stems"]), {"voice"})
        self.assertEqual(report["stems"]["voice"], out)
        est, _ = read(out)
        scaled = noisy / np.max(np.abs(noisy)) * 0.8
        gain = si_sdr(est, voice) - si_sdr(scaled, voice)
        self.assertGreater(gain, 6.0, f"voice improvement only {gain:.1f} dB")

    def test_karaoke_mode_returns_instrumental_only(self):
        report = sp.separate(self.mix_path, self.out_dir("karaoke"), mode="karaoke", quality="fast")
        self.assertEqual(list(report["stems"]), ["instrumental"])
        self.assertLess(report["checks"]["reconstruction_error_db"], -40.0)


# ── Robustness ───────────────────────────────────────────────────────────────

class RobustnessTests(SeparationTestCase):

    def test_long_input_is_chunked_with_progress_and_seamless_joins(self):
        calls = []
        report = sp.separate(self.mix_path, self.out_dir("chunks"), mode="vocals", quality="fast",
                             progress=lambda frac, stage: calls.append(frac), chunk_seconds=2.0)
        self.assertGreaterEqual(report["chunks"], 4)
        self.assertEqual(calls, sorted(calls))
        self.assertAlmostEqual(calls[-1], 1.0)
        self.assertGreater(len(calls), 4)
        vocals, _ = read(report["stems"]["vocals"])
        inst, _ = read(report["stems"]["instrumental"])
        self.assertEqual(vocals.shape[1], self.mix.shape[1])
        residual = vocals + inst - self.mix
        self.assertLess(np.max(np.abs(residual)), 1e-3)     # no clicks at chunk joins
        gain = self.improvement(report["stems"]["vocals"], self.sources["vocals"])
        self.assertGreater(gain, 6.0)

    def test_clipping_protection_keeps_stems_consistent(self):
        # A float master peaking at +6 dBFS: stems would clip when written as PCM.
        hot = self.mix * 2.5
        path = write(os.path.join(self.tmp, "hot.wav"), hot)
        report = sp.separate(path, self.out_dir("hot"), mode="vocals", quality="fast")
        vocals, _ = read(report["stems"]["vocals"])
        self.assertLessEqual(np.max(np.abs(vocals)), 1.0)
        self.assertLess(report["checks"]["gain_db"], 0.0)
        self.assertFalse(report["checks"]["clipped"])
        inst, _ = read(report["stems"]["instrumental"])
        restored = (vocals + inst) / (10 ** (report["checks"]["gain_db"] / 20))
        err = 10 * np.log10(np.sum((restored - hot) ** 2) / np.sum(hot ** 2))
        self.assertLess(err, -40.0)

    def test_other_sample_rates_are_kept(self):
        y = self.mix[:, ::2][:, : 22050 * 3]
        path = write(os.path.join(self.tmp, "sr22.wav"), y, sr=22050)
        report = sp.separate(path, self.out_dir("sr22"), mode="vocals", quality="fast")
        out, sr = read(report["stems"]["vocals"])
        self.assertEqual(sr, 22050)
        self.assertEqual(out.shape[1], y.shape[1])

    def test_mp3_output(self):
        report = sp.separate(self.mix_path, self.out_dir("mp3"), mode="vocals", quality="fast", fmt="mp3")
        for path in report["stems"].values():
            self.assertTrue(path.endswith(".mp3"))
            self.assertGreater(os.path.getsize(path), 1000)

    def test_honest_errors(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            sp.separate(self.mix_path, self.out_dir("bad"), mode="six")
        with self.assertRaisesRegex(ValueError, "format"):
            sp.separate(self.mix_path, self.out_dir("bad"), fmt="aiff")
        with self.assertRaisesRegex(ValueError, "found"):
            sp.separate(os.path.join(self.tmp, "missing.wav"), self.out_dir("bad"))
        silent = write(os.path.join(self.tmp, "silent.wav"), np.zeros((2, SR)))
        with self.assertRaisesRegex(ValueError, "silent"):
            sp.separate(silent, self.out_dir("bad"), quality="fast")
        tiny = write(os.path.join(self.tmp, "tiny.wav"), self.mix[:, :100])
        with self.assertRaisesRegex(ValueError, "short"):
            sp.separate(tiny, self.out_dir("bad"), quality="fast")
        junk = os.path.join(self.tmp, "junk.wav")
        with open(junk, "wb") as handle:
            handle.write(b"not audio at all" * 10)
        with self.assertRaises(ValueError):
            sp.separate(junk, self.out_dir("bad"), quality="fast")

    def test_errors_and_report_are_brand_private(self):
        report = sp.separate(self.mix_path, self.out_dir("private"), mode="vocals", quality="fast")
        public = json.dumps({k: v for k, v in report.items() if k != "stems"}).lower()
        for term in PRIVATE_TERMS:
            self.assertNotIn(term, public)
        try:
            sp.separate(self.mix_path, self.out_dir("private"), mode="nope")
        except ValueError as err:
            for term in PRIVATE_TERMS:
                self.assertNotIn(term, str(err).lower())


# ── Neural tier: governor fallback and zero-idle memory ─────────────────────

class _FakeModel:
    sources = ["drums", "bass", "other", "vocals"]
    samplerate = 44100
    audio_channels = 2


class NeuralTierTests(SeparationTestCase):

    def setUp(self):
        global_model_manager.unload_active_model()

    def _run_with(self, infer, gpu_free=0.0, quality="auto"):
        unloads = []
        governor = generous_governor(gpu_free)
        with mock.patch.object(sp, "global_quality_governor", governor), \
                mock.patch.object(sp, "_neural_ready", return_value=True), \
                mock.patch.object(sp, "_load_neural_model", return_value=_FakeModel()), \
                mock.patch.object(sp, "_unload_neural_model", side_effect=unloads.append), \
                mock.patch.object(sp, "_neural_infer", side_effect=infer), \
                mock.patch("model_manager._probe_ram_gb", return_value=(12.0, 16.0)),                 mock.patch.object(global_model_manager, "_flush_system_memory"):
            report = sp.separate(self.mix_path, self.out_dir("neural"), mode="vocals", quality=quality)
        return report, unloads, governor

    def test_neural_tier_used_when_installed_and_model_released_after(self):
        def infer(model, x, device):
            # Perfect "oracle" separation at 44.1 kHz stereo.
            n = x.shape[-1]
            vocal = self.sources["vocals"][:, :n]
            rest = x - vocal
            return {"vocals": vocal, "drums": rest * 0.5, "bass": rest * 0.25, "other": rest * 0.25}
        report, unloads, _ = self._run_with(infer)
        self.assertTrue(report["quality"]["neural"])
        self.assertEqual(report["quality"]["label"], "Studio stems")
        self.assertIsNone(global_model_manager._active_model_id)       # zero idle memory
        self.assertGreaterEqual(len(unloads), 1)
        gain = self.improvement(report["stems"]["vocals"], self.sources["vocals"])
        self.assertGreater(gain, 30.0)

    def test_memory_error_falls_back_to_baseline_and_pauses_variant(self):
        report, unloads, governor = self._run_with(mock.Mock(side_effect=MemoryError()))
        self.assertEqual(report["quality"]["id"], "quick")
        self.assertTrue(report["quality"]["fallback"])
        self.assertTrue(governor.is_paused(sp.CAPABILITY, "studio"))
        self.assertIsNone(global_model_manager._active_model_id)
        self.assertTrue(any("lighter" in w.lower() or "quick" in w.lower() for w in report["warnings"]))

    def test_gpu_out_of_memory_retries_on_processor_then_succeeds(self):
        devices = []

        def infer(model, x, device):
            devices.append(str(device))
            if str(device).startswith("cuda"):
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            vocal = self.sources["vocals"][:, : x.shape[-1]]
            rest = x - vocal
            return {"vocals": vocal, "drums": rest, "bass": rest * 0, "other": rest * 0}

        with mock.patch.object(sp, "_best_device", return_value="cuda"):
            report, _, _ = self._run_with(infer, gpu_free=6.0)
        self.assertTrue(report["quality"]["neural"])
        self.assertIn("cpu", devices)
        self.assertIsNone(global_model_manager._active_model_id)

    def test_fast_quality_never_loads_a_model(self):
        loader = mock.Mock()
        with mock.patch.object(sp, "_neural_ready", return_value=True), \
                mock.patch.object(sp, "_load_neural_model", loader):
            report = sp.separate(self.mix_path, self.out_dir("fast"), mode="vocals", quality="fast")
        loader.assert_not_called()
        self.assertEqual(report["quality"]["id"], "quick")

    def test_neural_is_never_used_or_downloaded_when_not_installed(self):
        empty = tempfile.mkdtemp(dir=self.tmp)
        with mock.patch.dict(os.environ, {"SEPARATION_MODELS_DIR": empty}):
            self.assertFalse(sp._neural_ready("htdemucs"))
            status = sp.capability_status()
        self.assertFalse(status["studio_stems_installed"])
        self.assertIn("quick", [t["id"] for t in status["tiers"]])


# ── Lyrics ──────────────────────────────────────────────────────────────────

class LyricsTests(SeparationTestCase):

    def fake_transcribe(self, path):
        self.transcribed_path = path
        return {"available": True, "language": "en", "full_text": "hello there world again",
                "segments": [{"start": 1.0, "end": 2.5, "text": "hello there"},
                             {"start": 3.0, "end": 4.25, "text": "world again"}],
                "model": "secret-model", "device": "secret-device"}

    def test_lyrics_isolates_vocals_then_transcribes(self):
        result = sp.lyrics(self.mix_path, quality="fast", transcribe=self.fake_transcribe)
        self.assertNotEqual(os.path.abspath(self.transcribed_path), os.path.abspath(self.mix_path))
        self.assertEqual(result["text"], "hello there world again")
        self.assertEqual(len(result["lines"]), 2)
        self.assertIn("00:00:01,000 --> 00:00:02,500", result["srt"])
        self.assertTrue(result["vtt"].startswith("WEBVTT"))
        self.assertIn("00:00:03.000 --> 00:00:04.250", result["vtt"])
        self.assertTrue(result["vocals_isolated"])
        self.assertNotIn("model", result)
        self.assertNotIn("device", result)
        self.assertFalse(os.path.exists(self.transcribed_path))      # temp stem cleaned up

    def test_lyrics_unavailable_speech_engine_has_public_error(self):
        def missing(path):
            return {"available": False, "error": "faster-whisper is not installed. pip install faster-whisper"}
        with self.assertRaises(RuntimeError) as ctx:
            sp.lyrics(self.mix_path, quality="fast", transcribe=missing)
        for term in PRIVATE_TERMS:
            self.assertNotIn(term, str(ctx.exception).lower())

    def test_subtitle_formatting(self):
        lines = [{"start": 3661.5, "end": 3662.0, "text": "late line"}]
        self.assertIn("01:01:01,500 --> 01:01:02,000", sp.to_srt(lines))
        self.assertIn("01:01:01.500 --> 01:01:02.000", sp.to_vtt(lines))


# ── Legacy entry point delegates to the new module ──────────────────────────

class LegacyDelegationTests(SeparationTestCase):

    def test_separate_stems_keeps_legacy_file_names(self):
        out = self.out_dir("legacy")
        with mock.patch.object(sp, "_neural_ready", return_value=False):
            res = ai_processor.separate_stems(self.mix_path, out, stems_mode="2")
        self.assertEqual(res["status"], "success")
        self.assertTrue(os.path.isfile(os.path.join(out, "vocals.wav")))
        self.assertTrue(os.path.isfile(os.path.join(out, "no_vocals.wav")))
        self.assertEqual(res["no_vocals"], f"/processed/{os.path.basename(out)}/no_vocals.wav")
        self.assertNotIn("engine", res)

    def test_separate_stems_four(self):
        out = self.out_dir("legacy4")
        with mock.patch.object(sp, "_neural_ready", return_value=False):
            res = ai_processor.separate_stems(self.mix_path, out, stems_mode="4")
        for stem in ("vocals", "drums", "bass", "other"):
            self.assertTrue(os.path.isfile(os.path.join(out, f"{stem}.wav")), stem)


# ── Routes ──────────────────────────────────────────────────────────────────

class RouteTests(SeparationTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from app import app
        cls.app = app
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.original = (app.config["UPLOAD_FOLDER"], app.config["PROCESSED_FOLDER"])
        cls.upload_dir = os.path.join(cls.tmp, "uploads")
        cls.processed_dir = os.path.join(cls.tmp, "processed")
        os.makedirs(cls.upload_dir, exist_ok=True)
        os.makedirs(cls.processed_dir, exist_ok=True)
        app.config["UPLOAD_FOLDER"] = cls.upload_dir
        app.config["PROCESSED_FOLDER"] = cls.processed_dir
        with open(cls.mix_path, "rb") as handle:
            cls.song = handle.read()

    @classmethod
    def tearDownClass(cls):
        cls.app.config["UPLOAD_FOLDER"], cls.app.config["PROCESSED_FOLDER"] = cls.original
        super().tearDownClass()

    def post(self, url, **fields):
        data = {"file": (io.BytesIO(self.song), "song.wav")}
        data.update(fields)
        return self.client.post(url, data=data, content_type="multipart/form-data")

    def fetch(self, url):
        response = self.client.get(url)
        response.close()
        return response.status_code

    def assert_private(self, response):
        blob = (response.get_data(as_text=True) if response.is_json else "").lower()
        headers = " ".join(f"{k}: {v}" for k, v in response.headers.items()).lower()
        for term in PRIVATE_TERMS:
            self.assertNotIn(term, blob, term)
            self.assertNotIn(term, headers, term)

    def test_separate_returns_zip_of_stems(self):
        res = self.post("/ai/separate", mode="vocals", format="wav", quality="fast")
        self.assertEqual(res.status_code, 200, res.get_data()[:300])
        self.assertEqual(res.mimetype, "application/zip")
        names = zipfile.ZipFile(io.BytesIO(res.data)).namelist()
        self.assertEqual(sorted(names), ["song_instrumental.wav", "song_vocals.wav"])
        self.assertEqual(res.headers.get("X-Separation-Quality"), "Quick separation")
        self.assert_private(res)
        res.close()

    def test_separate_karaoke_returns_single_file(self):
        res = self.post("/ai/separate", mode="karaoke", format="mp3", quality="fast")
        self.assertEqual(res.status_code, 200)
        self.assertIn("audio", res.mimetype)
        self.assertIn("instrumental", res.headers.get("Content-Disposition", ""))
        self.assert_private(res)
        res.close()

    def test_separate_json_delivery_for_the_editor(self):
        res = self.post("/ai/separate", mode="4stem", format="wav", quality="fast", delivery="json")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual({s["name"] for s in body["stems"]}, {"vocals", "drums", "bass", "other"})
        for stem in body["stems"]:
            self.assertTrue(stem["url"].startswith("/processed/"))
            self.assertEqual(self.fetch(stem["url"]), 200)
            self.assertIn("integrated_lufs", stem["loudness"])
        self.assertTrue(body["zip_url"].startswith("/processed/"))
        self.assertEqual(body["quality"]["label"], "Quick separation")
        self.assertIn("quality_note", body)
        self.assert_private(res)

    def test_separate_rejects_bad_mode_with_friendly_error(self):
        res = self.post("/ai/separate", mode="eleven", quality="fast")
        self.assertEqual(res.status_code, 400)
        self.assertIn("mode", res.get_json()["error"].lower())
        self.assert_private(res)

    def test_separate_internal_failure_is_generic(self):
        with mock.patch.object(sp, "separate", side_effect=RuntimeError("htdemucs torch CUDA crash")):
            res = self.post("/ai/separate", mode="vocals", quality="fast")
        self.assertEqual(res.status_code, 500)
        self.assert_private(res)

    def test_lyrics_route(self):
        fake = {"available": True, "language": "en", "full_text": "la la",
                "segments": [{"start": 0.5, "end": 1.5, "text": "la la"}], "model": "large-v3"}
        with mock.patch.object(ai_processor, "transcribe_audio", return_value=fake):
            res = self.post("/ai/lyrics", quality="fast")
        self.assertEqual(res.status_code, 200, res.get_data()[:300])
        body = res.get_json()
        self.assertEqual(body["text"], "la la")
        self.assertEqual(body["lines"][0]["text"], "la la")
        srt = self.client.get(body["srt_url"])
        self.assertEqual(srt.status_code, 200)
        self.assertIn(b"00:00:00,500 --> 00:00:01,500", srt.data)
        srt.close()
        self.assertEqual(self.fetch(body["vtt_url"]), 200)
        self.assert_private(res)

    def test_legacy_separate_stems_route_still_works(self):
        with mock.patch.object(sp, "_neural_ready", return_value=False):
            res = self.post("/ai/separate-stems")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertTrue(body["vocals"].startswith("/processed/"))
        self.assertTrue(body["no_vocals"].startswith("/processed/"))
        self.assertEqual(self.fetch(body["no_vocals"]), 200)
        self.assert_private(res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
