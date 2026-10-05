"""
DSP correctness tests for the media processing core.

Every test builds synthetic signals in a temporary directory, so no fixtures
are needed. Loudness references come from two independent sources:
  * the published ITU-R BS.1770-4 48 kHz K-weighting coefficients (re-implemented
    here, independently of audio_processor), and
  * the bundled ffmpeg ``ebur128`` scanner.

Run:  ./venv/Scripts/python.exe -m unittest test_media_dsp_correctness -v
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_processor  # noqa: E402
import ai_processor  # noqa: E402
import video_processor  # noqa: E402

SR = 48000
RNG = np.random.default_rng(1770)

# ── Independent reference helpers ────────────────────────────────────────────

# ITU-R BS.1770-4 Table 1 / Table 2 coefficients at 48 kHz.
_KW_B1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_KW_A1 = [1.0, -1.69065929318241, 0.73248077421585]
_KW_B2 = [1.0, -2.0, 1.0]
_KW_A2 = [1.0, -1.99004745483398, 0.99007225036621]


def ref_window_loudness(y):
    """Ungated K-weighted loudness of a 48 kHz mono/stereo window (LUFS)."""
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if y.shape[0] > y.shape[1]:
        y = y.T
    z = 0.0
    for ch in y:
        k = sps.lfilter(_KW_B2, _KW_A2, sps.lfilter(_KW_B1, _KW_A1, ch))
        z += np.mean(k ** 2)
    return -0.691 + 10 * np.log10(z + 1e-20)


def ffmpeg_ebur128(path):
    """Return {'I': LUFS, 'LRA': LU, 'TP': dBTP} measured by ffmpeg's scanner."""
    proc = subprocess.run(
        [video_processor.FFMPEG, "-hide_banner", "-nostats", "-i", path,
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    summary = proc.stderr.split("Summary:")[-1]
    out = {}
    for key, pat in (("I", r"I:\s+(-?[\d.]+|-inf)\s+LUFS"),
                     ("LRA", r"LRA:\s+(-?[\d.]+)\s+LU"),
                     ("TP", r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS")):
        m = re.search(pat, summary)
        out[key] = float(m.group(1)) if m else None
    return out


def ffprobe_dims(path):
    proc = subprocess.run(
        [video_processor.FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        capture_output=True, text=True,
    )
    st = json.loads(proc.stdout)["streams"][0]
    return int(st["width"]), int(st["height"])


def sine(freq, seconds, amp, sr=SR, phase=0.0):
    t = np.arange(int(round(seconds * sr))) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def speech_like(seconds, sr=SR, seed=0):
    """Band-limited noise with a 4 Hz syllabic envelope (speech stand-in)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    sos = sps.butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    x = sps.sosfilt(sos, rng.standard_normal(n))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * np.arange(n) / sr)
    return x * env


def scale_to(y, target_lufs):
    return y * 10 ** ((target_lufs - ref_window_loudness(y)) / 20)


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dsp_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def wav(self, name, y, sr=SR, subtype="FLOAT"):
        path = os.path.join(self.tmp, name)
        y = np.asarray(y)
        sf.write(path, y.T if y.ndim == 2 and y.shape[0] < y.shape[1] else y, sr, subtype=subtype)
        return path

    def garbage(self, name="broken.wav"):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as fh:
            fh.write(b"this is not audio at all" * 40)
        return path


# ═════════════════════════════════════════════════════════════════════════════
# 1. Integrated loudness / true peak (ITU-R BS.1770-4, EBU R128)
# ═════════════════════════════════════════════════════════════════════════════

class LoudnessMeterTests(_TempDirCase):

    def test_1khz_sine_at_minus20dbfs_mono_reads_minus23_lufs(self):
        path = self.wav("sine_mono.wav", sine(1000, 10, 0.1))
        m = audio_processor.calculate_lufs(path)
        self.assertAlmostEqual(m["lufs"], -23.0, delta=0.1)

    def test_identical_stereo_channels_sum_to_plus_3_lu(self):
        s = sine(1000, 10, 0.1)
        path = self.wav("sine_stereo.wav", np.vstack([s, s]))
        m = audio_processor.calculate_lufs(path)
        self.assertAlmostEqual(m["lufs"], -20.0, delta=0.1)
        self.assertEqual(m["channels"], 2)

    def test_k_weighting_matches_reference_meter_across_spectrum(self):
        # RMS-based estimates are off by several LU at low/high frequencies.
        for freq in (60, 100, 4000, 10000):
            path = self.wav(f"sine_{freq}.wav", sine(freq, 8, 0.25))
            ours = audio_processor.calculate_lufs(path)["lufs"]
            ref = ffmpeg_ebur128(path)["I"]
            self.assertAlmostEqual(ours, ref, delta=0.5, msg=f"{freq} Hz")

    def test_gating_ignores_silence(self):
        # 10 s of programme then 10 s of digital silence: gated loudness is
        # the programme loudness, not 3 dB lower.
        prog = speech_like(10, seed=1)
        prog = scale_to(prog, -18.0)
        y = np.concatenate([prog, np.zeros(10 * SR)])
        path = self.wav("gated.wav", y)
        ours = audio_processor.calculate_lufs(path)["lufs"]
        ref = ffmpeg_ebur128(path)["I"]
        self.assertAlmostEqual(ours, ref, delta=0.5)

    def test_relative_gate_with_dynamic_programme(self):
        loud = scale_to(speech_like(6, seed=2), -14.0)
        quiet = scale_to(speech_like(6, seed=3), -32.0)
        y = np.concatenate([loud, quiet, loud * 0.5])
        path = self.wav("dynamic.wav", np.vstack([y, 0.7 * y]))
        m = audio_processor.calculate_lufs(path)
        ref = ffmpeg_ebur128(path)
        self.assertAlmostEqual(m["lufs"], ref["I"], delta=0.5)
        self.assertIn("loudness_range_lu", m)
        self.assertAlmostEqual(m["loudness_range_lu"], ref["LRA"], delta=1.5)

    def test_true_peak_detects_inter_sample_overs(self):
        # fs/4 sine at 45 deg phase: every sample sits at 0.707*A, so sample
        # peak under-reads the true peak by ~3 dB.
        y = sine(SR / 4, 5, 0.9, phase=np.pi / 4)
        path = self.wav("isp.wav", y)
        m = audio_processor.calculate_lufs(path)
        self.assertAlmostEqual(m["true_peak_dbtp"], 20 * np.log10(0.9), delta=0.6)
        self.assertLess(m["peak_db"], m["true_peak_dbtp"] - 2.0)
        self.assertAlmostEqual(m["true_peak_dbtp"], ffmpeg_ebur128(path)["TP"], delta=0.6)

    def test_silence_and_very_short_clips_do_not_crash(self):
        m = audio_processor.calculate_lufs(self.wav("silence.wav", np.zeros(SR * 2)))
        self.assertEqual(m["lufs"], -70.0)
        m = audio_processor.calculate_lufs(self.wav("short.wav", sine(1000, 0.2, 0.1)))
        self.assertAlmostEqual(m["lufs"], -23.0, delta=0.3)
        json.dumps(m)  # must stay JSON-serialisable (no inf/nan)

    def test_unreadable_file_raises_value_error(self):
        with self.assertRaises(ValueError):
            audio_processor.calculate_lufs(self.garbage())


# ═════════════════════════════════════════════════════════════════════════════
# 2. Loudness normalisation
# ═════════════════════════════════════════════════════════════════════════════

class NormalizeTests(_TempDirCase):

    def _check(self, out, target, ceiling=-1.0):
        ours = audio_processor.calculate_lufs(out)
        ref = ffmpeg_ebur128(out)
        self.assertAlmostEqual(ours["lufs"], target, delta=0.5)
        self.assertAlmostEqual(ref["I"], target, delta=0.5)
        self.assertLessEqual(ours["true_peak_dbtp"], ceiling + 0.05)
        self.assertLessEqual(ref["TP"], ceiling + 0.15)

    def test_default_normalizes_quiet_speech_to_minus14_lufs(self):
        src = self.wav("quiet.wav", scale_to(speech_like(12, seed=4), -31.0))
        out = os.path.join(self.tmp, "norm.wav")
        result = audio_processor.normalize_audio(src, out)
        self.assertEqual(result, out)  # callers rely on the path return value
        self._check(out, -14.0)

    def test_peaky_material_is_true_peak_limited_and_still_on_target(self):
        y = scale_to(speech_like(12, seed=5), -26.0)
        y[SR * 3: SR * 3 + 40] += 0.6  # transient spikes
        y[SR * 7: SR * 7 + 40] -= 0.6
        src = self.wav("peaky.wav", np.vstack([y, y]))
        out = os.path.join(self.tmp, "peaky_norm.wav")
        audio_processor.normalize_audio(src, out)
        self._check(out, -14.0)

    def test_platform_presets(self):
        self.assertEqual(audio_processor.LOUDNESS_PRESETS["youtube"]["integrated_lufs"], -14.0)
        self.assertEqual(audio_processor.LOUDNESS_PRESETS["spotify"]["integrated_lufs"], -14.0)
        self.assertEqual(audio_processor.LOUDNESS_PRESETS["apple_podcasts"]["integrated_lufs"], -16.0)
        self.assertEqual(audio_processor.LOUDNESS_PRESETS["broadcast"]["integrated_lufs"], -23.0)
        src = self.wav("loud.wav", scale_to(speech_like(10, seed=6), -9.0))
        for preset, target in (("apple_podcasts", -16.0), ("broadcast", -23.0)):
            out = os.path.join(self.tmp, f"{preset}.wav")
            audio_processor.normalize_audio(src, out, preset=preset)
            self._check(out, target)

    def test_explicit_peak_target_keeps_legacy_peak_semantics(self):
        # Callers that ask for a peak level ("normalize peak to -3 dBFS") get it.
        src = self.wav("speech.wav", scale_to(speech_like(4, seed=7), -30.0))
        out = os.path.join(self.tmp, "peak.wav")
        audio_processor.normalize_audio(src, out, target_dbfs=-3.0)
        z, _ = sf.read(out)
        self.assertAlmostEqual(20 * np.log10(np.max(np.abs(z))), -3.0, delta=0.05)

    def test_unreadable_input_raises_value_error(self):
        with self.assertRaises(ValueError):
            audio_processor.normalize_audio(self.garbage(), os.path.join(self.tmp, "x.wav"))


# ═════════════════════════════════════════════════════════════════════════════
# 3. Auto-duck gain staging
# ═════════════════════════════════════════════════════════════════════════════

class AutoDuckTests(_TempDirCase):

    def test_voice_preserved_and_music_bed_sits_under_voice(self):
        dur = 10
        speech = np.zeros(dur * SR)
        for start in (1, 6):  # speech at 1–3 s and 6–8 s
            speech[start * SR:(start + 2) * SR] = speech_like(2, seed=start)
        speech = speech * 10 ** ((-20.0 - ref_window_loudness(speech[SR:3 * SR])) / 20)
        t = np.arange(dur * SR) / SR
        music = sum(np.sin(2 * np.pi * f * t) for f in (110, 220, 277.2, 329.6, 440))
        music = scale_to(music, -18.0)

        sp = self.wav("speech.wav", speech)
        mu = self.wav("music.wav", music)
        out = os.path.join(self.tmp, "ducked.wav")
        ai_processor.auto_duck_music(sp, mu, out)

        y, sr = sf.read(out, always_2d=True)
        self.assertEqual(sr, SR)
        y = y.mean(axis=1)[: dur * SR]
        self.assertLess(np.max(np.abs(y)), 1.0, "output clipped")
        residual = y - speech[: len(y)]  # what the music contributes

        for a, b in ((1.4, 2.6), (6.4, 7.6)):
            win = slice(int(a * SR), int(b * SR))
            voice = ref_window_loudness(speech[win])
            # 1) the voice is not attenuated by the mix (old amix: -6 dB)
            self.assertAlmostEqual(ref_window_loudness(y[win]), voice, delta=0.5)
            # 2) the bed sits 15–21 LU under the voice while speaking
            gap = voice - ref_window_loudness(residual[win])
            self.assertGreaterEqual(gap, 15.0)
            self.assertLessEqual(gap, 21.0)

        # 3) music returns to its own level between phrases
        win = slice(int(4.2 * SR), int(5.6 * SR))
        self.assertAlmostEqual(ref_window_loudness(residual[win]),
                               ref_window_loudness(music[win]), delta=1.0)

    def test_unreadable_input_raises_value_error(self):
        good = self.wav("music.wav", sine(220, 2, 0.2))
        with self.assertRaises(ValueError) as ctx:
            ai_processor.auto_duck_music(self.garbage(), good, os.path.join(self.tmp, "o.wav"))
        self.assertNotIn("ffmpeg", str(ctx.exception).lower())


# ═════════════════════════════════════════════════════════════════════════════
# 4. Silence-gap trimming
# ═════════════════════════════════════════════════════════════════════════════

class TrimSilenceTests(_TempDirCase):
    SR = 44100

    def _programme(self):
        sr = self.SR
        total = 6.5
        t = np.arange(int(total * sr)) / sr
        room = 0.008 * np.sin(2 * np.pi * 100 * t)  # -42 dBFS hum ("room tone")
        y = room.copy()
        fade = int(0.01 * sr)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
        bursts = (0.5, 2.8, 5.1)
        for start in bursts:
            n = int(0.8 * sr)
            seg = 0.3 * np.sin(2 * np.pi * 220 * t[:n]) + 0.3 * np.sin(2 * np.pi * 330 * t[:n])
            seg[:fade] *= ramp
            seg[-fade:] *= ramp[::-1]
            i = int(start * sr)
            y[i:i + n] += seg
        return y, bursts

    @staticmethod
    def _bursts(y, sr):
        hop = int(0.005 * sr)
        frames = y[: len(y) // hop * hop].reshape(-1, hop)
        loud = np.sqrt(np.mean(frames ** 2, axis=1)) > 0.03
        edges = np.flatnonzero(np.diff(loud.astype(int)))
        return [(e + 1) * hop / sr for e in edges]

    def test_cuts_are_click_free_and_keep_breathing_room(self):
        y, bursts = self._programme()
        src = self.wav("speech.wav", y, sr=self.SR, subtype="PCM_24")
        out = os.path.join(self.tmp, "trimmed.wav")
        res = ai_processor.trim_silence_gaps(src, out, min_silence_len=1.0)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["silences_removed"], 2)

        z, sr = sf.read(out)
        self.assertEqual(sr, self.SR)
        self.assertLess(len(z) / sr, 4.6)
        self.assertGreater(len(z) / sr, 3.0)

        # Click metric: largest 2nd difference anywhere in the output must not
        # exceed what already exists in the (click-free) source material.
        src_d2 = np.max(np.abs(np.diff(y, 2)))
        out_d2 = np.max(np.abs(np.diff(z, 2)))
        self.assertLess(out_d2, 1.5 * src_d2, f"click at a cut: {out_d2:.5f} vs {src_d2:.5f}")

        # ~150 ms of natural pause remains between phrases.
        edges = self._bursts(z, sr)
        self.assertEqual(len(edges), 6, edges)
        for k in (1, 3):
            pause = edges[k + 1] - edges[k]
            self.assertGreaterEqual(pause, 0.10, edges)
            self.assertLessEqual(pause, 0.40, edges)

    def test_audio_without_long_pauses_is_returned_intact(self):
        y = sine(220, 2.0, 0.3, sr=self.SR)
        src = self.wav("tone.wav", y, sr=self.SR, subtype="PCM_24")
        out = os.path.join(self.tmp, "tone_out.wav")
        res = ai_processor.trim_silence_gaps(src, out, min_silence_len=1.0)
        self.assertEqual(res["silences_removed"], 0)
        z, _ = sf.read(out)
        self.assertEqual(len(z), len(y))

    def test_unreadable_input_fails_honestly(self):
        out = os.path.join(self.tmp, "never.wav")
        with self.assertRaises(ValueError):
            ai_processor.trim_silence_gaps(self.garbage(), out)
        self.assertFalse(os.path.exists(out), "a copy of the input must not be reported as a result")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Filler words
# ═════════════════════════════════════════════════════════════════════════════

def _fake_transcript(words):
    t = 0.0
    out = []
    for w in words:
        out.append({"word": w, "start": round(t, 3), "end": round(t + 0.3, 3), "confidence": 0.9})
        t += 0.4
    return {"available": True, "segments": [{"start": 0.0, "end": t, "text": " ".join(words), "words": out}]}


class FillerWordTests(unittest.TestCase):
    WORDS = ["So,", "Um,", "I", "like", "this,", "you", "know,", "uhh", "it", "works", "Hmm."]

    def test_default_list_is_conservative(self):
        with mock.patch.object(ai_processor, "transcribe_audio", return_value=_fake_transcript(self.WORDS)):
            res = ai_processor.detect_filler_words("ignored.wav")
        found = sorted(f["word"].lower().strip(".,") for f in res["fillers"])
        self.assertEqual(found, ["hmm", "uhh", "um"])
        self.assertIn("note", res)  # documents the transcription limitation

    def test_discourse_markers_are_opt_in_and_match_phrases(self):
        with mock.patch.object(ai_processor, "transcribe_audio", return_value=_fake_transcript(self.WORDS)):
            res = ai_processor.detect_filler_words("ignored.wav", include_discourse_markers=True)
        found = [f["word"].lower().strip(".,") for f in res["fillers"]]
        self.assertIn("like", found)
        self.assertIn("so", found)
        phrase = [f for f in res["fillers"] if f["word"].lower().startswith("you know")]
        self.assertEqual(len(phrase), 1)
        self.assertAlmostEqual(phrase[0]["end"] - phrase[0]["start"], 0.7, delta=0.01)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Video enhancement keeps aspect ratio
# ═════════════════════════════════════════════════════════════════════════════

class EnhanceVideoAspectTests(_TempDirCase):

    def _clip(self, w, h):
        path = os.path.join(self.tmp, f"src_{w}x{h}.mp4")
        subprocess.run([video_processor.FFMPEG, "-y", "-f", "lavfi", "-i",
                        f"testsrc=duration=0.4:size={w}x{h}:rate=10",
                        "-pix_fmt", "yuv420p", path], capture_output=True, check=True)
        return path

    def test_aspect_ratio_preserved_for_landscape_portrait_and_4_3(self):
        for (w, h), expected in (((320, 180), (1920, 1080)),
                                 ((180, 320), (1080, 1920)),
                                 ((320, 240), (1440, 1080))):
            out = os.path.join(self.tmp, f"enh_{w}x{h}.mp4")
            video_processor.enhance_video_quality(self._clip(w, h), out, mode="1080p")
            self.assertEqual(ffprobe_dims(out), expected, f"{w}x{h}")

    def test_target_size_helper_even_dimensions_and_optional_pad(self):
        size = video_processor._enhance_target_size
        self.assertEqual(size(1080, 1920, "4k"), (2160, 3840))
        ow, oh = size(333, 250, "1080p")
        self.assertEqual((ow % 2, oh % 2), (0, 0))
        self.assertAlmostEqual(ow / oh, 333 / 250, delta=0.01)
        self.assertEqual(size(640, 480, "original"), (640, 480))
        # Padding only when explicitly requested: canvas is exact 16:9.
        out = os.path.join(self.tmp, "padded.mp4")
        video_processor.enhance_video_quality(self._clip(180, 320), out, mode="1080p", pad=True)
        self.assertEqual(ffprobe_dims(out), (1920, 1080))

    def test_rotated_phone_footage_stays_portrait(self):
        rotated = os.path.join(self.tmp, "rotated.mp4")
        subprocess.run([video_processor.FFMPEG, "-y", "-display_rotation", "90",
                        "-i", self._clip(320, 180), "-c", "copy", rotated],
                       capture_output=True, check=True)
        out = os.path.join(self.tmp, "rot_out.mp4")
        video_processor.enhance_video_quality(rotated, out, mode="720p")
        self.assertEqual(ffprobe_dims(out), (720, 1280))

    def test_audio_only_input_fails_honestly(self):
        src = self.wav("tone.wav", sine(440, 1, 0.2))
        with self.assertRaises(ValueError):
            video_processor.enhance_video_quality(src, os.path.join(self.tmp, "x.mp4"))


# ═════════════════════════════════════════════════════════════════════════════
# 7. Honest failures elsewhere
# ═════════════════════════════════════════════════════════════════════════════

class HonestFailureTests(_TempDirCase):

    def test_chapters_do_not_fake_success_when_transcription_fails(self):
        with mock.patch.object(ai_processor, "transcribe_audio",
                               return_value={"available": False, "error": "x"}):
            with self.assertRaises(ValueError):
                ai_processor.generate_youtube_chapters("ignored.wav")

    def test_voice_enhance_fallback_is_labelled_and_brand_neutral(self):
        src = self.wav("voice.wav", scale_to(speech_like(2, seed=9), -20) + 0.003 * RNG.standard_normal(2 * SR))
        out = os.path.join(self.tmp, "enh.wav")
        res = ai_processor.enhance_speech_studio(src, out)
        self.assertTrue(os.path.exists(out))
        blob = json.dumps(res).lower()
        for name in ("deepfilternet", "noisereduce", "demucs"):
            self.assertNotIn(name, blob)

    def test_multitrack_mix_with_missing_source_raises(self):
        good = self.wav("a.wav", sine(220, 1, 0.2))
        tracks = [{"file_path": good, "duration": 1.0},
                  {"file_path": os.path.join(self.tmp, "missing.wav"), "duration": 1.0}]
        with self.assertRaises(ValueError):
            audio_processor.mix_audio_tracks(tracks, os.path.join(self.tmp, "mix.wav"))


def _peak_private_bytes():
    """Peak private commit of this process (Windows PeakPagefileUsage; psutil elsewhere)."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return int(getattr(mi, "peak_pagefile", 0) or mi.rss)
    except ImportError:
        pass
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return int(pmc.PeakPagefileUsage)


@unittest.skipUnless(sys.platform == "win32", "peak-memory probe uses Windows APIs")
class NormalizeMemoryBoundTests(unittest.TestCase):
    def test_ten_minute_stereo_normalize_stays_bounded(self):
        tmp = tempfile.mkdtemp()
        try:
            src, out = os.path.join(tmp, "long.wav"), os.path.join(tmp, "norm.wav")
            rng = np.random.default_rng(7)
            with sf.SoundFile(src, "w", samplerate=SR, channels=2, subtype="FLOAT") as f:
                for i in range(20):  # 20 x 30 s
                    t = (np.arange(30 * SR) + i * 30 * SR) / SR
                    tone = 0.1 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t))
                    blk = np.stack([tone, 0.8 * tone], axis=1) + 0.01 * rng.standard_normal((t.size, 2))
                    f.write(blk.astype(np.float32))
            before = _peak_private_bytes()
            audio_processor.normalize_audio(src, out, target_lufs=-16.0, true_peak_ceiling=-1.0)
            growth = _peak_private_bytes() - before
            self.assertLess(growth, 500 * 1024 * 1024, f"peak private growth {growth / 2**20:.0f} MB")
            res = audio_processor.calculate_lufs(out)
            self.assertAlmostEqual(res["integrated_lufs"], -16.0, delta=0.5)
            self.assertLessEqual(res["true_peak_dbtp"], -0.9)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
