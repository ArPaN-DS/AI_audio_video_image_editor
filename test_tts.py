"""
Voiceover (text-to-speech) regression suite.

Covers: the always-available system-voice baseline (real WAV, non-silent,
plausible duration, speed effect), sentence-aware chunking and text
normalisation, input validation, the lexicon pronunciation front end,
adaptive-quality fallback when the natural tier runs out of memory, zero idle
memory after synthesis, thread safety, the Copilot tool hook, and the HTTP
routes including brand privacy (no engine/library/vendor names anywhere in
public payloads or headers).

Run:  python test_tts.py
"""

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

import tts_processor
from model_manager import global_model_manager, global_quality_governor, ResourceSnapshot

# Anything that would reveal a model, library, runtime or OS vendor.
FORBIDDEN = re.compile(
    r"kokoro|misaki|piper|espeak|phonemi|onnx|sapi|system\.speech|microsoft|zira|david|hazel|"
    r"powershell|pyttsx|hexgrad|huggingface|\bsay\b|af_|am_|bf_|bm_|heart|bella|fenrir|"
    r"michael|emma|george|torch|ffmpeg|librosa",
    re.IGNORECASE,
)

SAMPLE = "Hello there. This is a short test of the voiceover feature."


def _duration(path):
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def _rms(path):
    y, _ = sf.read(path, dtype="float32", always_2d=True)
    return float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0


class FakeNeuralEngine:
    """Stands in for the neural runtime: a tone whose length tracks the text."""

    sample_rate = 24000
    instances = []

    def __init__(self, fail_with=None):
        self.fail_with = fail_with
        self.unloaded = False
        self.calls = []
        FakeNeuralEngine.instances.append(self)

    def synthesize_chunk(self, text, voice_key, speed):
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append((text, voice_key, speed))
        seconds = max(0.2, 0.06 * len(text)) / float(speed)
        t = np.arange(int(seconds * self.sample_rate)) / self.sample_rate
        return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    def close(self):
        self.unloaded = True


class NeuralTierHarness(unittest.TestCase):
    """Base: pretend the natural tier is installed and the machine is roomy."""

    fail_with = None

    def setUp(self):
        FakeNeuralEngine.instances = []
        global_quality_governor.reset()
        roomy = lambda: ResourceSnapshot(16.0, 16.0, 16, 0.0, 0.0)
        patches = [
            mock.patch.object(global_quality_governor, "_probe", roomy),
            mock.patch.dict(os.environ, {"MEDIA_QUALITY_TIER": "max"}),
            mock.patch.object(tts_processor, "_neural_variant_installed", lambda variant_id: True),
            mock.patch.object(tts_processor, "_installed_natural_voices",
                              lambda: list(tts_processor._NATURAL_VOICES)),
            mock.patch.object(tts_processor, "_load_neural_engine",
                              lambda variant_id: FakeNeuralEngine(self.fail_with)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(global_quality_governor.reset)
        self.tmp = tempfile.mkdtemp(prefix="tts_neural_")
        self.addCleanup(shutil.rmtree, self.tmp, True)


class TextFrontEndTests(unittest.TestCase):

    def test_normalizes_numbers_currency_and_abbreviations(self):
        out = tts_processor.normalize_text("Dr. Smith paid $5.50 for 3 items, 25% off, on the 3rd.")
        self.assertIn("Doctor Smith", out)
        self.assertIn("five dollars and fifty cents", out)
        self.assertIn("three items", out)
        self.assertIn("twenty-five percent", out)
        self.assertIn("third", out)

    def test_normalizes_years_and_large_numbers(self):
        out = tts_processor.normalize_text("In 2024 we had 1,250,000 views.")
        self.assertIn("twenty twenty-four", out)
        self.assertIn("one million two hundred fifty thousand", out)

    def test_chunks_respect_sentences_and_limit(self):
        sentences = [f"This is sentence number {i} in a long script." for i in range(40)]
        text = " ".join(sentences)
        chunks = tts_processor.split_into_chunks(text, max_chars=200)
        self.assertGreater(len(chunks), 5)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 200)
            self.assertTrue(chunk.text.endswith("."), chunk.text)
        rejoined = " ".join(c.text for c in chunks).split()
        self.assertEqual(rejoined, text.split())

    def test_chunks_split_run_on_sentence_without_losing_words(self):
        text = ("word " * 300).strip()
        chunks = tts_processor.split_into_chunks(text, max_chars=120)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.text) <= 120 for c in chunks))
        self.assertEqual(" ".join(c.text for c in chunks).split(), text.split())

    def test_paragraph_breaks_get_longer_pauses(self):
        chunks = tts_processor.split_into_chunks("First idea. Second idea.\n\nNew paragraph here.",
                                                 max_chars=15)
        pauses = [c.pause_after for c in chunks]
        self.assertEqual(pauses[-1], 0.0)
        self.assertGreater(pauses[1], pauses[0])

    def test_lexicon_pronunciation_with_suffix_rules(self):
        lexicon = {"hello": "həlˈO", "world": "wˈɜɹld", "walk": "wˈɔk", "cat": "kˈæt"}
        g2p = tts_processor.LexiconPronouncer([lexicon])
        self.assertEqual(g2p.phonemize("Hello, world."), "həlˈO, wˈɜɹld.")
        self.assertEqual(g2p.phonemize("walks"), "wˈɔks")
        self.assertEqual(g2p.phonemize("walked"), "wˈɔkt")
        self.assertEqual(g2p.phonemize("cats"), "kˈæts")
        # Unknown words are spelled with letter names rather than dropped.
        self.assertTrue(g2p.phonemize("qz"))


class ValidationTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tts_val_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.out = os.path.join(self.tmp, "out.wav")

    def _assert_rejected(self, **kwargs):
        args = {"text": SAMPLE, "out_path": self.out}
        args.update(kwargs)
        with self.assertRaises(ValueError) as ctx:
            tts_processor.synthesize(**args)
        message = str(ctx.exception)
        self.assertTrue(message and not FORBIDDEN.search(message), message)
        self.assertFalse(os.path.exists(self.out))
        return message

    def test_rejects_empty_text(self):
        self._assert_rejected(text="   \n ")

    def test_rejects_text_over_cap(self):
        message = self._assert_rejected(text="a" * (tts_processor.MAX_TEXT_CHARS + 1))
        self.assertIn(f"{tts_processor.MAX_TEXT_CHARS:,}", message)

    def test_rejects_bad_speed(self):
        for speed in (0, 0.1, 5, "fast", float("nan")):
            self._assert_rejected(speed=speed)

    def test_rejects_bad_pitch(self):
        self._assert_rejected(pitch=40)

    def test_rejects_bad_format(self):
        self._assert_rejected(fmt="exe")

    def test_rejects_unknown_voice(self):
        self._assert_rejected(voice="no-such-voice")

    def test_rejects_text_without_speakable_content(self):
        self._assert_rejected(text="... --- !!!")


@unittest.skipUnless(tts_processor.baseline_available(), "no system voice on this machine")
class BaselineTierTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="tts_base_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_lists_public_system_voices(self):
        voices = tts_processor.list_voices()
        standard = [v for v in voices if v["quality"] == "Standard voice"]
        self.assertTrue(standard)
        for voice in voices:
            self.assertEqual(set(voice) - {"id", "label", "language", "gender", "quality", "default"}, set())
            self.assertFalse(FORBIDDEN.search(json.dumps(voice)), voice)
        self.assertEqual(sum(1 for v in voices if v.get("default")), 1)

    def test_synthesizes_real_non_silent_wav(self):
        out = os.path.join(self.tmp, "hello.wav")
        report = tts_processor.synthesize(SAMPLE, out, quality="standard")
        self.assertTrue(os.path.exists(out))
        with open(out, "rb") as handle:
            self.assertEqual(handle.read(4), b"RIFF")
        duration = _duration(out)
        self.assertGreater(duration, 1.5)
        self.assertLess(duration, 15.0)
        self.assertGreater(_rms(out), 0.01)
        self.assertAlmostEqual(report["duration"], duration, delta=0.1)
        self.assertEqual(report["tier"], "Standard voice")
        self.assertFalse(FORBIDDEN.search(json.dumps(report)), report)

    def test_speed_changes_duration(self):
        slow = os.path.join(self.tmp, "slow.wav")
        fast = os.path.join(self.tmp, "fast.wav")
        tts_processor.synthesize(SAMPLE, slow, speed=0.7, quality="standard")
        tts_processor.synthesize(SAMPLE, fast, speed=1.6, quality="standard")
        self.assertGreater(_duration(slow), _duration(fast) * 1.4)

    def test_long_text_is_chunked_and_concatenated(self):
        text = " ".join(f"Point {i} is covered here." for i in range(1, 13))
        out = os.path.join(self.tmp, "long.wav")
        report = tts_processor.synthesize(text, out, quality="standard")
        self.assertGreater(report["chunks"], 1)
        self.assertGreater(_duration(out), 8.0)

    def test_mp3_output(self):
        out = os.path.join(self.tmp, "hello.mp3")
        tts_processor.synthesize("Quick check.", out, fmt="mp3", quality="standard")
        with open(out, "rb") as handle:
            head = handle.read(3)
        self.assertTrue(head == b"ID3" or head[0] == 0xFF, head)

    def test_pitch_shift_keeps_duration(self):
        base = os.path.join(self.tmp, "base.wav")
        high = os.path.join(self.tmp, "high.wav")
        tts_processor.synthesize("Pitch check one two three.", base, quality="standard")
        tts_processor.synthesize("Pitch check one two three.", high, pitch=3, quality="standard")
        self.assertAlmostEqual(_duration(base), _duration(high), delta=0.25)


class NeuralTierTests(NeuralTierHarness):

    def test_uses_natural_tier_and_unloads_after(self):
        out = os.path.join(self.tmp, "n.wav")
        report = tts_processor.synthesize(SAMPLE, out)
        self.assertIn(report["tier"], ("Studio voice", "Natural voice"))
        self.assertTrue(FakeNeuralEngine.instances)
        self.assertTrue(all(e.unloaded for e in FakeNeuralEngine.instances))
        self.assertIsNone(global_model_manager._active_model_id)
        self.assertGreater(_rms(out), 0.01)

    def test_chunks_are_joined_with_natural_pauses(self):
        text = " ".join(f"Sentence {i} is here." for i in range(30))
        out = os.path.join(self.tmp, "long.wav")
        report = tts_processor.synthesize(text, out)
        engine = FakeNeuralEngine.instances[-1]
        self.assertEqual(report["chunks"], len(engine.calls))
        tone = sum(max(0.2, 0.06 * len(c[0])) for c in engine.calls)
        self.assertGreater(_duration(out), tone + 0.1 * (len(engine.calls) - 1))
        self.assertLess(_duration(out), tone + 1.0 * len(engine.calls))

    def test_speed_is_passed_to_engine(self):
        tts_processor.synthesize(SAMPLE, os.path.join(self.tmp, "s.wav"), speed=1.3)
        self.assertTrue(all(abs(c[2] - 1.3) < 1e-6 for c in FakeNeuralEngine.instances[-1].calls))

    def test_standard_quality_skips_neural(self):
        if not tts_processor.baseline_available():
            self.skipTest("no system voice")
        report = tts_processor.synthesize(SAMPLE, os.path.join(self.tmp, "b.wav"), quality="standard")
        self.assertEqual(report["tier"], "Standard voice")
        self.assertFalse(FakeNeuralEngine.instances)

    def test_concurrent_requests_are_serialised_safely(self):
        errors, reports = [], []

        def work(i):
            try:
                reports.append(tts_processor.synthesize(
                    f"Thread {i} speaking now.", os.path.join(self.tmp, f"t{i}.wav")))
            except Exception as error:  # pragma: no cover - surfaced below
                errors.append(error)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(len(reports), 4)
        self.assertIsNone(global_model_manager._active_model_id)

    def test_voiceover_tool_for_copilot(self):
        result = tts_processor.voiceover_tool({"text": SAMPLE, "speed": 1.0, "format": "wav"}, self.tmp)
        self.assertTrue(os.path.isfile(result["output_file"]))
        self.assertEqual(os.path.dirname(os.path.abspath(result["output_file"])), os.path.abspath(self.tmp))
        self.assertTrue(result["message"])
        self.assertFalse(FORBIDDEN.search(json.dumps(result)), result)
        with self.assertRaises(ValueError):
            tts_processor.voiceover_tool({"text": ""}, self.tmp)


@unittest.skipUnless(tts_processor.baseline_available(), "no system voice on this machine")
class NeuralFallbackTests(NeuralTierHarness):

    fail_with = MemoryError("out of memory")

    def test_memory_error_falls_back_to_baseline_and_pauses_tier(self):
        out = os.path.join(self.tmp, "fb.wav")
        report = tts_processor.synthesize(SAMPLE, out)
        self.assertEqual(report["tier"], "Standard voice")
        self.assertGreater(_rms(out), 0.01)
        self.assertTrue(any(global_quality_governor.is_paused(tts_processor.CAPABILITY, v)
                            for v in tts_processor.NEURAL_VARIANTS))
        self.assertIsNone(global_model_manager._active_model_id)
        self.assertTrue(all(e.unloaded for e in FakeNeuralEngine.instances))
        self.assertFalse(FORBIDDEN.search(json.dumps(report)), report)


@unittest.skipUnless(tts_processor.neural_installed(), "natural voice pack not installed")
class InstalledNeuralTierTests(unittest.TestCase):
    """Runs the real local neural tier when the opt-in pack is present."""

    def test_real_natural_voice(self):
        tmp = tempfile.mkdtemp(prefix="tts_real_")
        self.addCleanup(shutil.rmtree, tmp, True)
        out = os.path.join(tmp, "real.wav")
        report = tts_processor.synthesize(SAMPLE, out)
        self.assertIn(report["tier"], ("Studio voice", "Natural voice"))
        self.assertGreater(_duration(out), 1.5)
        self.assertLess(_duration(out), 12.0)
        self.assertGreater(_rms(out), 0.01)
        self.assertIsNone(global_model_manager._active_model_id)


class RouteTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import app as app_module
        cls.app = app_module.app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()
        cls.tmp = tempfile.mkdtemp(prefix="tts_routes_")
        cls.original = cls.app.config["PROCESSED_FOLDER"]
        cls.app.config["PROCESSED_FOLDER"] = cls.tmp

    @classmethod
    def tearDownClass(cls):
        cls.app.config["PROCESSED_FOLDER"] = cls.original
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _assert_private(self, response):
        for name, value in response.headers.items():
            self.assertNotIn("engine", name.lower())
            self.assertFalse(FORBIDDEN.search(f"{name}: {value}"), (name, value))
        if response.is_json:
            self.assertFalse(FORBIDDEN.search(response.get_data(as_text=True)))

    def test_voices_route_is_public_and_private(self):
        response = self.client.get("/ai/tts/voices")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("voices", payload)
        self.assertIn("max_chars", payload)
        self.assertIn("default_voice", payload)
        self._assert_private(response)

    def test_invalid_requests_return_json_errors(self):
        for body in ({"text": ""}, {"text": "hi", "speed": 9}, {"text": "hi", "format": "exe"},
                     {"text": "x" * (tts_processor.MAX_TEXT_CHARS + 5)}, {"text": "hi", "voice": "nope"}):
            response = self.client.post("/ai/tts", json=body)
            self.assertEqual(response.status_code, 400, body)
            payload = response.get_json()
            self.assertEqual(payload["status"], "error")
            self.assertTrue(payload["error"])
            self._assert_private(response)

    @unittest.skipUnless(tts_processor.baseline_available(), "no system voice on this machine")
    def test_generates_audio_file(self):
        response = self.client.post("/ai/tts", json={"text": SAMPLE, "speed": 1.0, "format": "wav",
                                                     "quality": "standard"})
        self.assertEqual(response.status_code, 200, response.data[:300])
        self.assertTrue(response.mimetype.startswith("audio/"))
        self.assertEqual(response.data[:4], b"RIFF")
        self.assertEqual(response.headers.get("X-Voice-Quality"), "Standard voice")
        self.assertGreater(float(response.headers.get("X-Audio-Duration")), 1.0)
        self._assert_private(response)

    @unittest.skipUnless(tts_processor.baseline_available(), "no system voice on this machine")
    def test_form_post_with_mp3(self):
        response = self.client.post("/ai/tts", data={"text": "Form check.", "format": "mp3",
                                                     "quality": "standard"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/mpeg")
        self._assert_private(response)


if __name__ == "__main__":
    unittest.main(verbosity=2)
