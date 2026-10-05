"""
Audio Processor — Advanced Multitrack Audio Mixing & Signal Processing Engine.

Complements Pydub and Librosa with:
  - Loudness metering per ITU-R BS.1770-4 / EBU R128 (integrated loudness
    with K-weighting and two-stage gating, 4x oversampled true peak,
    loudness range per EBU Tech 3342)
  - Two-pass loudness normalisation with a true-peak limiter
  - 10-Band Parametric Equalizer filter calculation
  - Multi-track audio mixdown with volume, pan, and offset alignment
  - Batch conversion and formatting

Memory model (bounded for hour-long recordings):
  Metering, limiting, normalisation and EQ STREAM the signal in fixed blocks
  (256 Ki frames) with explicit filter state carried across blocks and exact
  look-ahead/look-behind context, so a 1-hour stereo file needs a few tens of
  MB instead of several GB of float64 temporaries. Block boundaries follow the
  same grid the whole-signal implementation used, so results are identical to
  the previous in-memory implementation (bit-exact metering; limiter gain
  within ~1e-9, i.e. identical after 16-bit quantisation).
"""

import runtime_tuning  # noqa: F401  (thread-pool defaults before numeric libraries load)

import logging
import os
import subprocess
import tempfile

import numpy as np

from runtime_tuning import lazy_module

# Heavy libraries load on first use, not at server start.
sf = lazy_module("soundfile")
librosa = lazy_module("librosa")
_sps = lazy_module("scipy.signal")


def __getattr__(name):
    # `audio_processor.AudioSegment` stays available without importing the
    # media-segment library at server start.
    if name == "AudioSegment":
        from pydub import AudioSegment
        return AudioSegment
    raise AttributeError(name)


_log = logging.getLogger("audio_dsp")

# ═══════════════════════════════════════════════════════════════════════════
#  Loudness measurement — ITU-R BS.1770-4 / EBU R128 / EBU Tech 3342
# ═══════════════════════════════════════════════════════════════════════════

LOUDNESS_FLOOR_LUFS = -70.0      # absolute gate; reported for silence
PEAK_FLOOR_DB = -120.0           # keeps results JSON-safe (no -inf)

# Platform delivery targets (integrated loudness, true-peak ceiling).
LOUDNESS_PRESETS = {
    "youtube": {"label": "YouTube", "integrated_lufs": -14.0, "true_peak_dbtp": -1.0},
    "spotify": {"label": "Spotify", "integrated_lufs": -14.0, "true_peak_dbtp": -1.0},
    "apple_podcasts": {"label": "Apple Podcasts", "integrated_lufs": -16.0, "true_peak_dbtp": -1.0},
    "podcast": {"label": "Podcast", "integrated_lufs": -16.0, "true_peak_dbtp": -1.0},
    "broadcast": {"label": "Broadcast (EBU R128)", "integrated_lufs": -23.0, "true_peak_dbtp": -1.0},
}
DEFAULT_LOUDNESS_PRESET = "youtube"

_UNREADABLE_AUDIO = ("This file could not be read as audio. "
                     "Try exporting it as WAV or MP3 and upload it again.")

BLOCK_FRAMES = 1 << 18           # streaming block == true-peak oversampling grid
_TRUE_PEAK_PAD = 32
# In-memory collection of a normalised result below this many samples avoids a
# final re-render pass; above it the result is streamed straight to disk.
COLLECT_MAX_SAMPLES = 24_000_000  # ~190 MB of float64
# Container formats that decode sample-exactly in sequential blocks.
_STREAMABLE_FORMATS = {"WAV", "WAVEX", "FLAC", "AIFF", "AIFC", "W64", "RF64", "CAF", "AU", "SD2", "IRCAM"}


# ── Sources: decoded audio, read in consecutive blocks ─────────────────────

class _ArraySource:
    """An in-memory [channels, samples] signal."""

    def __init__(self, y, sr):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        self.y = y
        self.sr = int(sr)
        self.channels, self.frames = y.shape

    def reader(self):
        return _ArrayReader(self.y)

    def close(self):
        self.y = None


class _ArrayReader:
    def __init__(self, y):
        self.y = y

    def read(self, a, b):
        return self.y[:, a:b]

    def close(self):
        pass


class _FileSource:
    """A PCM container decoded block-by-block (O(block) memory)."""

    def __init__(self, path, info, cleanup=None):
        self.path = path
        self.sr = int(info.samplerate)
        self.channels = int(info.channels)
        self.frames = int(info.frames)
        self._cleanup = cleanup

    def reader(self):
        return _SequentialReader(self.path, self.channels, self.frames)

    def close(self):
        if self._cleanup:
            try:
                os.remove(self._cleanup)
            except OSError:
                pass
            self._cleanup = None


class _SequentialReader:
    """
    Forward-only window over a decoder: ``read(a, b)`` with non-decreasing
    ``a`` and ``b`` returns float32 [channels, b - a]; samples before the
    latest ``a`` are released.
    """

    def __init__(self, path, channels, frames):
        self._file = sf.SoundFile(path)
        self._channels = channels
        self._frames = frames
        self._buf = np.zeros((channels, 0), dtype=np.float32)
        self._start = 0

    def read(self, a, b):
        b = min(b, self._frames)
        have = self._start + self._buf.shape[1]
        if b > have:
            block = self._file.read(b - have, dtype="float32", always_2d=True).T
            if block.shape[1] < b - have:  # truncated stream: pad with silence
                block = np.pad(block, ((0, 0), (0, b - have - block.shape[1])))
            if not np.all(np.isfinite(block)):
                block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
            self._buf = np.concatenate([self._buf, block], axis=1) if self._buf.shape[1] else block
        if a > self._start:
            self._buf = self._buf[:, a - self._start:]
            self._start = a
        return self._buf[:, a - self._start:b - self._start]

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


def _iter_blocks(source, block=BLOCK_FRAMES):
    reader = source.reader()
    try:
        for start in range(0, source.frames, block):
            yield start, reader.read(start, min(source.frames, start + block))
    finally:
        reader.close()


def open_audio_source(path):
    """Open ``path`` for streaming; raises ValueError for unreadable audio."""
    if not path or not os.path.isfile(path):
        raise ValueError("The audio file could not be found.")
    try:
        info = sf.info(path)
        if info.frames > 0 and info.channels > 0 and str(info.format).upper() in _STREAMABLE_FORMATS:
            return _FileSource(path, info)
    except Exception:
        info = None
    y, sr = load_audio_array(path)
    return _ArraySource(y, sr)


def _decode_to_temp_wav(path):
    """Decode compressed/containerised audio (mp3, aac, video) to a float WAV."""
    import video_processor
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        proc = subprocess.run(
            [video_processor.FFMPEG, "-y", "-v", "error", "-i", path, "-vn",
             "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "wav", tmp],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=1800,
        )
        if proc.returncode != 0 or os.path.getsize(tmp) <= 44:
            os.remove(tmp)
            return None
        return tmp
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None


def _fill_from_soundfile(path):
    """Read a whole file into a preallocated [channels, samples] float32 array
    (one copy of the signal, not the read buffer plus its transpose)."""
    with sf.SoundFile(path) as handle:
        channels, frames = handle.channels, handle.frames
        sr = handle.samplerate
        if handle.format.upper() not in _STREAMABLE_FORMATS or frames <= 0:
            data = handle.read(dtype="float32", always_2d=True)
            return np.ascontiguousarray(data.T), sr
        out = np.empty((channels, frames), dtype=np.float32)
        pos = 0
        while pos < frames:
            block = handle.read(min(BLOCK_FRAMES, frames - pos), dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            out[:, pos:pos + block.shape[0]] = block.T
            pos += block.shape[0]
        return out[:, :pos], sr


def load_audio_array(path):
    """Decode any supported file to (float32 array [channels, samples], sr).

    Raises ValueError with a user-facing message when the file has no
    decodable audio, instead of letting a low-level decoder error escape.
    """
    if not path or not os.path.isfile(path):
        raise ValueError("The audio file could not be found.")
    data = sr = None
    try:
        data, sr = _fill_from_soundfile(path)
    except Exception:
        data = None
    if data is None:
        data, sr = _decode_with_ffmpeg(path)
    if data is None or data.size == 0 or not sr:
        raise ValueError(_UNREADABLE_AUDIO)
    y = data
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, int(sr)


def _decode_with_ffmpeg(path):
    """Fallback decoder for compressed/containerised audio (mp3, aac, video)."""
    tmp = _decode_to_temp_wav(path)
    if tmp is None:
        return None, None
    try:
        return _fill_from_soundfile(tmp)
    except Exception:
        return None, None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _open_stream_source(path):
    """Like ``open_audio_source`` but decodes non-PCM containers to a temporary
    float WAV on disk instead of into memory."""
    if not path or not os.path.isfile(path):
        raise ValueError("The audio file could not be found.")
    try:
        info = sf.info(path)
        if info.frames > 0 and info.channels > 0:
            if str(info.format).upper() in _STREAMABLE_FORMATS:
                return _FileSource(path, info)
            y, sr = load_audio_array(path)  # natively decodable (mp3/ogg): same decoder as before
            return _ArraySource(y, sr)
    except ValueError:
        raise
    except Exception:
        pass
    tmp = _decode_to_temp_wav(path)
    if tmp is None:
        raise ValueError(_UNREADABLE_AUDIO)
    try:
        info = sf.info(tmp)
        if info.frames <= 0:
            raise ValueError(_UNREADABLE_AUDIO)
    except ValueError:
        os.remove(tmp)
        raise
    except Exception:
        os.remove(tmp)
        raise ValueError(_UNREADABLE_AUDIO)
    return _FileSource(tmp, info, cleanup=tmp)


# ── Writers ────────────────────────────────────────────────────────────────

class _AudioWriter:
    """Streams [channels, n] float blocks to ``path``; format follows the extension.

    WAV/FLAC are written as 16-bit PCM; lossy formats are encoded by the media
    engine from a float stream piped over stdin (no temporary WAV on disk).
    """

    _CODECS = {
        "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
        "ogg": ["-c:a", "libvorbis", "-q:a", "6"],
        "opus": ["-c:a", "libopus", "-b:a", "192k"],
        "aac": ["-c:a", "aac", "-b:a", "256k"],
        "m4a": ["-c:a", "aac", "-b:a", "256k"],
    }

    def __init__(self, path, sr, channels):
        self.path = path
        self._file = self._proc = self._tmp = None
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext in ("", "wav", "wave"):
            self._file = sf.SoundFile(path, "w", samplerate=sr, channels=channels, subtype="PCM_16", format="WAV")
        elif ext == "flac":
            self._file = sf.SoundFile(path, "w", samplerate=sr, channels=channels, subtype="PCM_16", format="FLAC")
        elif channels <= 2:
            import video_processor
            self._proc = subprocess.Popen(
                [video_processor.FFMPEG, "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", str(channels),
                 "-i", "pipe:0", *self._CODECS.get(ext, []), path],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:  # multichannel: keep the WAV channel mask the encoder reads
            fd, self._tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            self._ext = ext
            self._file = sf.SoundFile(self._tmp, "w", samplerate=sr, channels=channels, subtype="FLOAT",
                                      format="WAV")

    def write(self, block):
        frames = np.asarray(block, dtype=np.float32)
        if frames.ndim == 1:
            frames = frames[np.newaxis, :]
        if self._proc is not None:
            try:
                self._proc.stdin.write(np.ascontiguousarray(frames.T).tobytes())
            except (BrokenPipeError, OSError):
                raise ValueError("The processed audio could not be saved in that format.")
        else:
            self._file.write(frames.T)

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.wait(timeout=1800)
            proc, self._proc = self._proc, None
            if proc.returncode != 0 or not os.path.exists(self.path):
                raise ValueError("The processed audio could not be saved in that format.")
            return
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._tmp:
            import video_processor
            try:
                proc = subprocess.run(
                    [video_processor.FFMPEG, "-y", "-v", "error", "-i", self._tmp,
                     *self._CODECS.get(self._ext, []), self.path],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
                if proc.returncode != 0 or not os.path.exists(self.path):
                    raise ValueError("The processed audio could not be saved in that format.")
            finally:
                try:
                    os.remove(self._tmp)
                except OSError:
                    pass
                self._tmp = None

    def abort(self):
        try:
            if self._proc is not None:
                self._proc.kill()
                self._proc.wait(timeout=30)
            if self._file is not None:
                self._file.close()
        except Exception:
            pass
        for leftover in (self._tmp,):
            if leftover:
                try:
                    os.remove(leftover)
                except OSError:
                    pass
        self._proc = self._file = self._tmp = None


def write_audio_array(path, y, sr):
    """Write [channels, samples] float audio; format follows the extension.

    WAV/FLAC are written as 16-bit PCM for broad compatibility; lossy formats
    are encoded from a float intermediate so no extra clipping is introduced.
    """
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    writer = _AudioWriter(path, sr, y.shape[0])
    try:
        for start in range(0, y.shape[1], BLOCK_FRAMES):
            writer.write(y[:, start:start + BLOCK_FRAMES])
    except Exception:
        writer.abort()
        raise
    writer.close()
    return path


# ── DSP building blocks ────────────────────────────────────────────────────

def _k_weighting_sos(sr):
    """BS.1770-4 K-weighting (shelving pre-filter + RLB high-pass) for any rate.

    Coefficients are derived from the analog prototypes so they match the
    tabulated 48 kHz values exactly and stay correct at 44.1 kHz, 96 kHz, etc.
    """
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b_shelf = [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a_shelf = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]

    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1.0 + k / q + k * k
    b_hp = [1.0, -2.0, 1.0]
    a_hp = [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    return np.array([b_shelf + a_shelf, b_hp + a_hp], dtype=np.float64)


def _channel_weights(n_channels):
    """BS.1770 channel weights. 5.0/5.1 surrounds get +1.5 dB, LFE is excluded."""
    if n_channels == 5:      # L R C Ls Rs
        return np.array([1.0, 1.0, 1.0, 1.41, 1.41])
    if n_channels == 6:      # L R C LFE Ls Rs
        return np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41])
    return np.ones(n_channels)


def _energy_to_lufs(energy):
    return -0.691 + 10.0 * np.log10(np.maximum(energy, 1e-20))


def _gated_mean(energies, relative_gate_lu):
    """Absolute (−70 LUFS) + relative gating; returns (loudness, gated_blocks)."""
    loud = _energy_to_lufs(energies)
    above_abs = energies[loud > LOUDNESS_FLOOR_LUFS]
    if above_abs.size == 0:
        return LOUDNESS_FLOOR_LUFS, above_abs
    threshold = _energy_to_lufs(np.mean(above_abs)) + relative_gate_lu
    gated = above_abs[_energy_to_lufs(above_abs) > threshold]
    if gated.size == 0:
        return LOUDNESS_FLOOR_LUFS, gated
    return max(LOUDNESS_FLOOR_LUFS, float(_energy_to_lufs(np.mean(gated)))), gated


def _oversample_factor(sr):
    if sr < 96000:
        return 4
    if sr < 192000:
        return 2
    return 1


def _to_db(lin, floor=PEAK_FLOOR_DB):
    return float(max(floor, 20.0 * np.log10(lin))) if lin > 0 else float(floor)


_UPSAMPLE_POOL = []


def _upsample_channels(seg, factor):
    """``resample_poly`` per channel; upfirdn releases the GIL so channels run in parallel."""
    if seg.shape[0] < 2:
        return _sps.resample_poly(seg, factor, 1, axis=-1)
    if not _UPSAMPLE_POOL:
        from concurrent.futures import ThreadPoolExecutor
        _UPSAMPLE_POOL.append(ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)))
    rows = list(_UPSAMPLE_POOL[0].map(lambda r: _sps.resample_poly(r, factor, 1), list(seg)))
    return np.stack(rows)


class _TruePeakEnvelope:
    """
    Incremental 4x-oversampled |peak| envelope (per sample, max over channels).

    Consecutive float64 blocks go in; ``(start, stop, env)`` comes out for each
    grid chunk of ``chunk`` samples once its ``pad`` samples of right context
    have arrived — the exact chunking/padding of the whole-signal estimator.
    """

    def __init__(self, sr, channels, chunk=BLOCK_FRAMES, pad=_TRUE_PEAK_PAD):
        self.factor = _oversample_factor(sr)
        self.chunk = chunk
        self.pad = pad if self.factor > 1 else 0
        self.channels = channels
        self._buf = np.zeros((channels, 0))
        self._buf_start = 0
        self._next = 0        # start of the next chunk to emit
        self._seen = 0

    def feed(self, block):
        block = np.asarray(block, dtype=np.float64)
        if block.shape[1]:
            self._buf = np.concatenate([self._buf, block], axis=1) if self._buf.shape[1] else block
            self._seen += block.shape[1]
        return self._emit(final=False)

    def finish(self):
        return self._emit(final=True)

    def _emit(self, final):
        out = []
        while self._next < self._seen:
            start = self._next
            stop = start + self.chunk
            if stop + self.pad > self._seen and not final:
                break
            stop = min(stop, self._seen)
            a, b = max(0, start - self.pad), min(self._seen, stop + self.pad)
            seg = self._buf[:, a - self._buf_start:b - self._buf_start]
            core = seg[:, start - a:stop - a]
            if self.factor == 1:
                env = np.max(np.abs(core), axis=0).astype(np.float64)
            else:
                up = _upsample_channels(seg, self.factor)
                lo = (start - a) * self.factor
                hi = lo + (stop - start) * self.factor
                up = up[:, lo:hi]
                env = np.maximum(up.max(axis=0), -up.min(axis=0)).reshape(stop - start, self.factor).max(axis=1)
                env = np.maximum(env, np.max(np.abs(core), axis=0))
            out.append((start, stop, env))
            self._next = stop
            keep_from = max(0, self._next - self.pad)
            if keep_from > self._buf_start:
                self._buf = self._buf[:, keep_from - self._buf_start:]
                self._buf_start = keep_from
        return out


class _LoudnessMeter:
    """Streaming BS.1770-4 / EBU R128 meter; ``report()`` equals the whole-signal one."""

    def __init__(self, sr, channels, include_true_peak=True):
        self.sr = sr
        self.channels = channels
        self.include_true_peak = include_true_peak
        self.hop = max(1, int(round(sr * 0.1)))
        self._sos = _k_weighting_sos(sr)
        self._weights = _channel_weights(channels)
        self._zi = np.zeros((self._sos.shape[0], channels, 2))
        self._pending = np.zeros(0)
        self._sums = []
        self._total = 0.0
        self.n = 0
        self._sample_peak = 0.0
        self._true_peak = 0.0
        self._env = _TruePeakEnvelope(sr, channels) if include_true_peak else None

    def feed(self, block):
        seg = np.asarray(block, dtype=np.float64)
        if seg.ndim == 1:
            seg = seg[np.newaxis, :]
        if not seg.shape[1]:
            return
        self.n += seg.shape[1]
        filtered, self._zi = _sps.sosfilt(self._sos, seg, axis=-1, zi=self._zi)
        power = np.tensordot(self._weights, filtered * filtered, axes=1)
        self._total += float(power.sum())
        if self._pending.size:
            power = np.concatenate([self._pending, power])
        full = (power.shape[0] // self.hop) * self.hop
        if full:
            self._sums.append(power[:full].reshape(-1, self.hop).sum(axis=1))
        self._pending = power[full:]
        self._sample_peak = max(self._sample_peak, float(np.max(np.abs(block))))
        if self._env is not None:
            for _, _, env in self._env.feed(seg):
                if env.size:
                    self._true_peak = max(self._true_peak, float(env.max()))

    def report(self):
        if self._env is not None:
            for _, _, env in self._env.finish():
                if env.size:
                    self._true_peak = max(self._true_peak, float(env.max()))
            self._env = None
        sub = np.concatenate(self._sums) if self._sums else np.zeros(0)
        hop = self.hop
        block_len = 4 * hop
        mean_power = self._total / self.n if self.n else 0.0
        if sub.size >= 4:
            momentary = np.convolve(sub, np.ones(4), mode="valid") / block_len
        else:
            # Shorter than one 400 ms gating block: measure the whole clip.
            momentary = np.array([mean_power])
        integrated, _ = _gated_mean(momentary, -10.0)

        short_term_max = LOUDNESS_FLOOR_LUFS
        lra = 0.0
        if sub.size >= 30:
            short_term = np.convolve(sub, np.ones(30), mode="valid") / (30 * hop)
            st_lufs = _energy_to_lufs(short_term)
            short_term_max = max(LOUDNESS_FLOOR_LUFS, float(st_lufs.max()))
            # EBU Tech 3342: abs gate −70, relative gate −20 LU, LRA = P95 − P10.
            _, st_gated = _gated_mean(short_term, -20.0)
            if st_gated.size >= 2:
                vals = _energy_to_lufs(st_gated)
                lra = float(np.percentile(vals, 95) - np.percentile(vals, 10))
        momentary_max = max(LOUDNESS_FLOOR_LUFS, float(_energy_to_lufs(momentary).max()))

        report = {
            "integrated_lufs": round(float(integrated), 2),
            "loudness_range_lu": round(lra, 2),
            "momentary_max_lufs": round(momentary_max, 2),
            "short_term_max_lufs": round(short_term_max, 2),
            "sample_peak_dbfs": round(_to_db(self._sample_peak if self.n else 0.0), 2),
        }
        if self.include_true_peak:
            report["true_peak_dbtp"] = round(_to_db(self._true_peak), 2)
        return report


def _measure_source(source, include_true_peak=True):
    meter = _LoudnessMeter(source.sr, source.channels, include_true_peak)
    for _, block in _iter_blocks(source):
        meter.feed(block)
    return meter.report()


def measure_loudness_array(y, sr, include_true_peak=True):
    """Loudness report for a [channels, samples] (or mono 1-D) float array."""
    return _measure_source(_ArraySource(np.atleast_2d(np.asarray(y)), sr), include_true_peak)


def calculate_lufs(audio_path):
    """
    Measure loudness per ITU-R BS.1770-4 / EBU R128.

    Returns integrated loudness (K-weighted, 400 ms blocks with 75 % overlap,
    −70 LUFS absolute and −10 LU relative gates), 4x-oversampled true peak,
    loudness range (EBU Tech 3342), momentary/short-term maxima, and the
    sample peak. Silence reports the −70 LUFS gate floor. ``lufs`` and
    ``peak_db`` keep their historical names for existing callers.
    Streams the file: memory stays constant regardless of duration.
    """
    source = _open_stream_source(audio_path)
    try:
        m = _measure_source(source)
        return {
            "lufs": m["integrated_lufs"],
            "integrated_lufs": m["integrated_lufs"],
            "true_peak_dbtp": m["true_peak_dbtp"],
            "loudness_range_lu": m["loudness_range_lu"],
            "momentary_max_lufs": m["momentary_max_lufs"],
            "short_term_max_lufs": m["short_term_max_lufs"],
            "peak_db": m["sample_peak_dbfs"],
            "sample_rate": source.sr,
            "channels": int(source.channels),
            "duration": round(source.frames / float(source.sr), 3),
            "standard": "ITU-R BS.1770-4 / EBU R128",
        }
    finally:
        source.close()


# ═══════════════════════════════════════════════════════════════════════════
#  True-peak limiter + loudness normalisation
# ═══════════════════════════════════════════════════════════════════════════

class _StreamingLimiter:
    """
    Look-ahead brickwall limiter on the 4x-oversampled peak envelope, streamed.

    Per output sample i (n = signal length, look = look-ahead in samples):
      required[k] = min(1, ceiling / env[k])
      held[k]     = min(required[k-look .. k+look])         (edge-clamped)
      smooth[i]   = mean(held[i-look+1 .. i])                (left edge = held[0])
      gain        = min(smooth, one-pole release of smooth)
    Every value only needs ``2*look`` samples of context on each side, so the
    signal is processed grid chunk by grid chunk with one chunk of delay and
    the release filter state carried across chunks.
    """

    def __init__(self, sr, channels, frames, ceiling_dbtp=-1.0, lookahead_ms=5.0, release_ms=100.0,
                 input_scale=1.0, output_scale=1.0):
        self.sr, self.channels, self.n = sr, channels, frames
        self.ceiling = 10.0 ** ((ceiling_dbtp - 0.1) / 20.0)  # 0.1 dB safety margin
        self.look = max(1, int(round(lookahead_ms * 1e-3 * sr)))
        self.alpha = np.exp(-1.0 / (release_ms * 1e-3 * sr))
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.min_gain = 1.0

    def run(self, source, sink):
        """Stream ``source`` (scaled by input_scale) through the limiter into
        ``sink(block)``; returns the maximum gain reduction in dB."""
        n, look = self.n, self.look
        if n == 0:
            return 0.0
        env_stream = _TruePeakEnvelope(self.sr, self.channels)
        pending_audio = {}       # chunk start -> scaled input chunk
        env_chunks = []          # [(start, stop, env)] still needed
        b_coef, a_coef = [1.0 - self.alpha], [1.0, -self.alpha]
        state = {"zi": None, "held0": None, "next_out": 0}

        def env_window(w0, w1):
            parts = [env[max(w0, s) - s:min(w1, e) - s] for s, e, env in env_chunks if e > w0 and s < w1]
            return np.concatenate(parts) if len(parts) > 1 else parts[0]

        def flush(final):
            while state["next_out"] < n:
                s = state["next_out"]
                e = min(n, s + BLOCK_FRAMES)
                need_until = min(n, e + look)
                have_until = env_chunks[-1][1] if env_chunks else 0
                if have_until < need_until and not final:
                    return
                w0, w1 = max(0, s - 2 * look), min(n, e + look)
                required = np.minimum(1.0, self.ceiling / np.maximum(env_window(w0, w1), 1e-12))
                held = _minimum_filter1d(required, size=2 * look + 1, mode="nearest")
                if state["held0"] is None:
                    state["held0"] = held[0]          # w0 == 0 on the first chunk
                h_from = s - look + 1                  # positions [h_from, e) of held
                if h_from < w0:                        # only at the signal start
                    h = np.concatenate([np.full(w0 - h_from, state["held0"]), held[:e - w0]])
                else:
                    h = held[h_from - w0:e - w0]
                cs = np.concatenate([[0.0], np.cumsum(h)])
                smooth = (cs[look:look + (e - s)] - cs[:e - s]) / look
                if state["zi"] is None:
                    state["zi"] = _sps.lfilter_zi(b_coef, a_coef) * smooth[0]
                released, state["zi"] = _sps.lfilter(b_coef, a_coef, smooth, zi=state["zi"])
                gain = np.minimum(smooth, released)
                gain[gain >= 1.0 - 1e-12] = 1.0
                self.min_gain = min(self.min_gain, float(gain.min()))
                block = pending_audio.pop(s) * gain[np.newaxis, :]
                if self.output_scale != 1.0:
                    block = block * self.output_scale
                sink(block)
                state["next_out"] = e
                keep_from = max(0, e - 2 * look)
                while env_chunks and env_chunks[0][1] <= keep_from:
                    env_chunks.pop(0)

        for start, block in _iter_blocks(source):
            scaled = np.asarray(block, dtype=np.float64)
            if self.input_scale != 1.0:
                scaled = scaled * self.input_scale
            pending_audio[start] = scaled
            env_chunks.extend(env_stream.feed(scaled))
            flush(final=False)
        env_chunks.extend(env_stream.finish())
        flush(final=True)
        if self.min_gain >= 1.0:
            return 0.0
        return float(-20.0 * np.log10(max(self.min_gain, 1e-6)))


def _minimum_filter1d(values, size, mode):
    from scipy.ndimage import minimum_filter1d
    return minimum_filter1d(values, size=size, mode=mode)


class _Collector:
    def __init__(self, channels, frames):
        self.out = np.empty((channels, frames), dtype=np.float64)
        self.pos = 0

    def __call__(self, block):
        self.out[:, self.pos:self.pos + block.shape[1]] = block
        self.pos += block.shape[1]


def true_peak_limit(y, sr, ceiling_dbtp=-1.0, lookahead_ms=5.0, release_ms=100.0):
    """Look-ahead brickwall limiter on the 4x-oversampled peak envelope.

    Gain never exceeds what each sample needs (so the ceiling holds); the gain
    curve ramps down over the look-ahead window and recovers with a smooth
    one-pole release, so there are no step changes (no clicks).
    Returns (limited_audio, max_gain_reduction_db).
    """
    y = np.atleast_2d(np.asarray(y))
    source = _ArraySource(y, sr)
    if source.frames == 0:
        return np.asarray(y, dtype=np.float64), 0.0
    collect = _Collector(source.channels, source.frames)
    limiter = _StreamingLimiter(sr, source.channels, source.frames, ceiling_dbtp, lookahead_ms, release_ms)
    reduction = limiter.run(source, collect)
    return collect.out, reduction


def _resolve_loudness_targets(preset, target_lufs, true_peak_ceiling):
    key = (preset or DEFAULT_LOUDNESS_PRESET)
    key = str(key).strip().lower().replace(" ", "_").replace("-", "_")
    if key not in LOUDNESS_PRESETS:
        raise ValueError("Unknown loudness preset. Choose one of: "
                         + ", ".join(sorted(LOUDNESS_PRESETS)) + ".")
    spec = LOUDNESS_PRESETS[key]
    lufs = float(spec["integrated_lufs"] if target_lufs is None else target_lufs)
    ceiling = float(spec["true_peak_dbtp"] if true_peak_ceiling is None else true_peak_ceiling)
    if not (-50.0 <= lufs <= -5.0):
        raise ValueError("Choose a loudness target between -50 and -5 LUFS.")
    if not (-12.0 <= ceiling <= 0.0):
        raise ValueError("Choose a true-peak ceiling between -12 and 0 dBTP.")
    return key, lufs, ceiling


def _normalize_source(source, target_lufs, true_peak_ceiling, max_limiting_db, final_sink_factory):
    """
    Two-pass loudness normalisation over a streamed source.

    Pass 1 measures integrated loudness; pass 2 applies the static gain and a
    true-peak limiter, metering the limited output on the fly. If limiting
    pulled the loudness under target, gain is re-trimmed and re-limited (at most
    a few iterations, bounded by ``max_limiting_db``). ``final_sink_factory``
    returns (sink, collected_or_None): when it collects in memory the last
    iteration's output is kept, otherwise one final render streams to the sink.
    Returns (report, sink_result).
    """
    before = _measure_source(source)
    report = {
        "input_lufs": before["integrated_lufs"],
        "input_true_peak_dbtp": before["true_peak_dbtp"],
        "target_lufs": target_lufs,
        "true_peak_ceiling_dbtp": true_peak_ceiling,
    }
    if before["integrated_lufs"] <= LOUDNESS_FLOOR_LUFS:
        report.update(status="skipped", within_tolerance=False,
                      output_lufs=before["integrated_lufs"],
                      output_true_peak_dbtp=before["true_peak_dbtp"],
                      gain_db=0.0, limiter_reduction_db=0.0,
                      note="The audio is silent, so there is no loudness to adjust.")
        sink, collected = final_sink_factory()
        for _, block in _iter_blocks(source):
            sink(np.asarray(block, dtype=np.float64))
        return report, collected

    sink, collected = final_sink_factory()
    keep_in_memory = collected is not None

    def render(gain_db, post_scale, deliver):
        meter = _LoudnessMeter(source.sr, source.channels)

        def tee(block):
            meter.feed(block)
            if deliver is not None:
                deliver(block)

        if deliver is not None and keep_in_memory:
            collected.pos = 0
        limiter = _StreamingLimiter(source.sr, source.channels, source.frames, true_peak_ceiling,
                                    input_scale=10.0 ** (gain_db / 20.0), output_scale=post_scale)
        reduction = limiter.run(source, tee)
        return reduction, meter.report()

    base_gain = target_lufs - before["integrated_lufs"]
    gain_db = base_gain
    rendered_gain_db = gain_db   # gain of the output that is actually delivered
    post_scale = 1.0
    reduction, measured = 0.0, before
    for _ in range(5):
        reduction, measured = render(gain_db, 1.0, sink if keep_in_memory else None)
        rendered_gain_db = gain_db
        err = target_lufs - measured["integrated_lufs"]
        if reduction == 0.0 or abs(err) <= 0.1:
            break
        if gain_db + err - base_gain > max_limiting_db:
            break
        gain_db += err

    # Final safety: if inter-sample peaks still poke over, trim statically.
    if measured["true_peak_dbtp"] > true_peak_ceiling:
        trim = true_peak_ceiling - measured["true_peak_dbtp"] - 0.02
        post_scale = 10.0 ** (trim / 20.0)
        if keep_in_memory:
            collected.out *= post_scale
            measured = measure_loudness_array(collected.out, source.sr)
        else:
            _, measured = render(rendered_gain_db, post_scale, None)
        gain_db += trim

    if not keep_in_memory:
        render(rendered_gain_db, post_scale, sink)

    within = (abs(measured["integrated_lufs"] - target_lufs) <= 0.5
              and measured["true_peak_dbtp"] <= true_peak_ceiling + 0.05)
    report.update(
        status="success" if within else "partial",
        within_tolerance=bool(within),
        output_lufs=measured["integrated_lufs"],
        output_true_peak_dbtp=measured["true_peak_dbtp"],
        output_loudness_range_lu=measured["loudness_range_lu"],
        gain_db=round(float(gain_db), 2),
        limiter_reduction_db=round(float(reduction), 2),
    )
    if not within:
        report["note"] = ("The target could not be reached without heavy limiting; "
                          "the result was kept as close to the target as was safe.")
    return report, collected


def normalize_loudness_array(y, sr, target_lufs=-14.0, true_peak_ceiling=-1.0,
                             max_limiting_db=12.0):
    """Two-pass loudness normalisation of an in-memory signal.

    Pass 1 measures integrated loudness; pass 2 applies the static gain and a
    true-peak limiter. If limiting pulled the loudness under target, gain is
    re-trimmed and re-limited (at most a few iterations, bounded by
    ``max_limiting_db`` so dense material is not crushed).
    Returns (audio, report).
    """
    source = _ArraySource(np.atleast_2d(np.asarray(y)), sr)
    collector = _Collector(source.channels, source.frames)
    report, collected = _normalize_source(source, target_lufs, true_peak_ceiling, max_limiting_db,
                                          lambda: (collector, collector))
    return collected.out, report


def normalize_loudness(audio_path, output_path, target_lufs=None, true_peak_ceiling=None,
                       preset=None):
    """File-level loudness normalisation; returns a measurement report dict.

    Streams long recordings (bounded memory); short ones are rendered in memory
    so the final pass is not repeated.
    """
    key, lufs, ceiling = _resolve_loudness_targets(preset, target_lufs, true_peak_ceiling)
    source = _open_stream_source(audio_path)
    writer = None
    try:
        small = source.frames * source.channels <= COLLECT_MAX_SAMPLES

        def sink_factory():
            nonlocal writer
            if small:
                collector = _Collector(source.channels, source.frames)
                return collector, collector
            writer = _AudioWriter(output_path, source.sr, source.channels)
            return writer.write, None

        report, collected = _normalize_source(source, lufs, ceiling, 12.0, sink_factory)
        if collected is not None:
            write_audio_array(output_path, collected.out, source.sr)
        elif writer is not None:
            writer.close()
            writer = None
    except Exception:
        if writer is not None:
            writer.abort()
        raise
    finally:
        source.close()
    if report.get("status") == "partial":
        _log.warning("Loudness normalisation stopped short of target: %s", report)
    report["preset"] = key
    report["output_path"] = output_path
    return report


# ═══════════════════════════════════════════════════════════════════════════
#  Parametric EQ (streamed STFT with exact frame context)
# ═══════════════════════════════════════════════════════════════════════════

EQ_BANDS_HZ = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
_EQ_FFT = 4096
_EQ_HOP = 1024
_EQ_BLOCK = 256 * _EQ_HOP


def _eq_gain_mask(sr, eq_bands):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_EQ_FFT)
    gain_mask = np.ones(_EQ_FFT // 2 + 1, dtype=np.float32)
    for band_freq in EQ_BANDS_HZ:
        band_str = str(int(band_freq))
        gain_db = float(eq_bands.get(band_str, 0.0))
        if gain_db != 0.0:
            # Gaussian shaped gain curve around band frequency
            linear_gain = 10.0 ** (gain_db / 20.0)
            sigma = band_freq * 0.4
            gaussian = np.exp(-0.5 * ((freqs - band_freq) / (sigma + 1e-5)) ** 2)
            gain_mask += (linear_gain - 1.0) * gaussian
    return gain_mask


def _eq_segment(seg, gain_mask):
    out = []
    for ch in range(seg.shape[0]):
        stft = librosa.stft(seg[ch], n_fft=_EQ_FFT, hop_length=_EQ_HOP)
        out.append(librosa.istft(stft * gain_mask[:, np.newaxis], hop_length=_EQ_HOP, length=seg.shape[1]))
    return np.vstack(out)


def apply_parametric_eq(audio_path, output_path, eq_bands=None):
    """
    Apply 10-Band Parametric EQ adjustments to an audio file.
    eq_bands: dict of band_name -> gain_db (e.g. {"32": 2.0, "64": -1.0, ...})

    Streams PCM files block by block. Each block is transformed with one FFT
    length of real signal context on both sides and blocks start on the hop
    grid, so every STFT frame that reaches the output is the same frame the
    whole-file transform would compute.
    """
    if eq_bands is None:
        eq_bands = {}
    try:
        info = sf.info(audio_path)
        streamable = info.frames > 0 and str(info.format).upper() in _STREAMABLE_FORMATS
    except Exception:
        streamable = False
    if not streamable:
        return _apply_parametric_eq_in_memory(audio_path, output_path, eq_bands)

    sr, channels, n = int(info.samplerate), int(info.channels), int(info.frames)
    gain_mask = _eq_gain_mask(sr, eq_bands)
    reader = _SequentialReader(audio_path, channels, n)
    try:
        with sf.SoundFile(output_path, "w", samplerate=sr, channels=channels) as out:
            for s in range(0, n, _EQ_BLOCK):
                e = min(n, s + _EQ_BLOCK)
                a, b = max(0, s - _EQ_FFT), min(n, e + _EQ_FFT)
                seg = reader.read(a, b)
                y_eq = _eq_segment(seg, gain_mask)[:, s - a:e - a]
                out.write(y_eq[0] if channels == 1 else y_eq.T)
    finally:
        reader.close()
    return True


def _apply_parametric_eq_in_memory(audio_path, output_path, eq_bands):
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    is_mono = (y.ndim == 1)
    if is_mono:
        y = y.reshape(1, -1)
    y_out = _eq_segment(y, _eq_gain_mask(sr, eq_bands))
    sf.write(output_path, y_out[0] if is_mono else y_out.T, sr)
    return True


def mix_audio_tracks(tracks_spec, output_path, master_volume=1.0, format="wav"):
    """
    Mix multiple audio track clips together.
    tracks_spec: list of dicts:
      [
        {
          "file_path": "uploads/audio1.mp3",
          "start_time": 0.0,  # seconds in master timeline
          "clip_offset": 0.0, # trim start inside clip
          "duration": 5.0,
          "volume": 0.8,
          "pan": 0.0, # -1.0 to 1.0
          "mute": False
        },
        ...
      ]
    """
    from pydub import AudioSegment

    if not tracks_spec:
        raise ValueError("No tracks provided to mix")

    # Find total master timeline duration
    max_duration = 0.0
    for t in tracks_spec:
        end_time = float(t.get("start_time", 0.0)) + float(t.get("duration", 0.0))
        if end_time > max_duration:
            max_duration = end_time

    if max_duration <= 0.0:
        max_duration = 1.0

    master_ms = int(max_duration * 1000) + 1000
    master_mix = AudioSegment.silent(duration=master_ms)

    for t in tracks_spec:
        if t.get("mute", False):
            continue

        file_path = t.get("file_path")
        if not file_path or not os.path.exists(file_path):
            raise ValueError("One of the tracks in this mix could not be found. "
                             "Re-add the clip and try again.")

        clip = AudioSegment.from_file(file_path)

        # Trim clip
        clip_offset_ms = int(t.get("clip_offset", 0.0) * 1000)
        duration_ms = int(t.get("duration", clip.duration_seconds) * 1000)

        clip_seg = clip[clip_offset_ms : clip_offset_ms + duration_ms]
        del clip  # release the full decode before the next track loads

        # Volume gain (convert scalar volume multiplier to dB)
        vol = float(t.get("volume", 1.0))
        if vol <= 0.001:
            continue
        gain_db = 20 * np.log10(vol)
        clip_seg = clip_seg.apply_gain(gain_db)

        # Pan (-1.0 to 1.0)
        pan = float(t.get("pan", 0.0))
        if pan != 0.0:
            clip_seg = clip_seg.pan(max(-1.0, min(1.0, pan)))

        # Overlay onto master timeline
        start_ms = int(t.get("start_time", 0.0) * 1000)
        master_mix = master_mix.overlay(clip_seg, position=start_ms)

    # Apply master volume
    if master_volume != 1.0 and master_volume > 0.0:
        master_gain = 20 * np.log10(master_volume)
        master_mix = master_mix.apply_gain(master_gain)

    # Trim to exact length
    master_mix = master_mix[: int(max_duration * 1000)]

    # Export
    bitrate = "320k" if format == "mp3" else None
    master_mix.export(output_path, format=format, bitrate=bitrate)
    return output_path


def normalize_audio(audio_path, output_path, target_dbfs=None, target_lufs=None,
                    preset=None, true_peak_ceiling=None, mode="auto"):
    """
    Loudness-normalise audio to a delivery target (default: -14 LUFS
    integrated with a -1 dBTP true-peak ceiling — the YouTube/Spotify preset).

    Two passes: measure BS.1770 integrated loudness, then apply gain plus a
    true-peak limiter and verify the result. ``preset`` selects a platform
    target from ``LOUDNESS_PRESETS``; ``target_lufs``/``true_peak_ceiling``
    override it.

    Backwards compatibility: a caller that passes only ``target_dbfs`` asked
    for the historical peak normalisation (peak level = ``target_dbfs``) and
    gets exactly that; ``mode="peak"``/``"loudness"`` force either behaviour.

    Returns ``output_path`` (callers depend on this); the full measurement
    report is available from ``normalize_loudness``.
    """
    loudness_requested = any(v is not None for v in (target_lufs, preset, true_peak_ceiling))
    if mode == "peak" or (mode == "auto" and target_dbfs is not None and not loudness_requested):
        peak_db = -1.0 if target_dbfs is None else float(target_dbfs)
        if not (-60.0 <= peak_db <= 0.0):
            raise ValueError("Choose a peak level between -60 and 0 dBFS.")
        source = _open_stream_source(audio_path)
        writer = None
        try:
            peak = 0.0
            for _, block in _iter_blocks(source):
                if block.size:
                    peak = max(peak, float(np.max(np.abs(block))))
            factor = (10.0 ** (peak_db / 20.0) / peak) if peak > 0 else None
            writer = _AudioWriter(output_path, source.sr, source.channels)
            for _, block in _iter_blocks(source):
                writer.write(block * factor if factor is not None else block)
            writer.close()
            writer = None
        except Exception:
            if writer is not None:
                writer.abort()
            raise
        finally:
            source.close()
        return output_path

    normalize_loudness(audio_path, output_path, target_lufs=target_lufs,
                       true_peak_ceiling=true_peak_ceiling, preset=preset)
    return output_path


def apply_fades(audio_path, output_path, fade_in_sec=2.0, fade_out_sec=2.0):
    """
    Apply fade in and fade out to an audio file.
    """
    from pydub import AudioSegment

    seg = AudioSegment.from_file(audio_path)
    fade_in_ms = int(max(0.0, float(fade_in_sec)) * 1000)
    fade_out_ms = int(max(0.0, float(fade_out_sec)) * 1000)
    dur_ms = len(seg)
    if fade_in_ms > dur_ms:
        fade_in_ms = dur_ms
    if fade_out_ms > dur_ms:
        fade_out_ms = dur_ms

    if fade_in_ms > 0:
        seg = seg.fade_in(fade_in_ms)
    if fade_out_ms > 0:
        seg = seg.fade_out(fade_out_ms)

    ext = os.path.splitext(output_path)[1].lstrip('.').lower()
    fmt = ext if ext in ("wav", "mp3", "ogg", "flac", "aac") else "wav"
    seg.export(output_path, format=fmt)
    return output_path
