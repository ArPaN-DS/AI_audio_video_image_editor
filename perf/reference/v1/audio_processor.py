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
"""

import logging
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import librosa
from pydub import AudioSegment
from scipy import signal as _sps

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


def load_audio_array(path):
    """Decode any supported file to (float32 array [channels, samples], sr).

    Raises ValueError with a user-facing message when the file has no
    decodable audio, instead of letting a low-level decoder error escape.
    """
    if not path or not os.path.isfile(path):
        raise ValueError("The audio file could not be found.")
    data = sr = None
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        data = None
    if data is None:
        data, sr = _decode_with_ffmpeg(path)
    if data is None or data.size == 0 or not sr:
        raise ValueError(_UNREADABLE_AUDIO)
    y = np.ascontiguousarray(data.T)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, int(sr)


def _decode_with_ffmpeg(path):
    """Fallback decoder for compressed/containerised audio (mp3, aac, video)."""
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
            return None, None
        data, sr = sf.read(tmp, dtype="float32", always_2d=True)
        return data, sr
    except Exception:
        return None, None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def write_audio_array(path, y, sr):
    """Write [channels, samples] float audio; format follows the extension.

    WAV/FLAC are written as 16-bit PCM for broad compatibility; lossy formats
    are encoded from a float intermediate so no extra clipping is introduced.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    frames = y.T
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    if ext in ("", "wav", "wave"):
        sf.write(path, frames, sr, subtype="PCM_16", format="WAV")
        return path
    if ext == "flac":
        sf.write(path, frames, sr, subtype="PCM_16", format="FLAC")
        return path
    import video_processor
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp, frames, sr, subtype="FLOAT", format="WAV")
        codec = {
            "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
            "ogg": ["-c:a", "libvorbis", "-q:a", "6"],
            "opus": ["-c:a", "libopus", "-b:a", "192k"],
            "aac": ["-c:a", "aac", "-b:a", "256k"],
            "m4a": ["-c:a", "aac", "-b:a", "256k"],
        }.get(ext, [])
        proc = subprocess.run(
            [video_processor.FFMPEG, "-y", "-v", "error", "-i", tmp, *codec, path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=1800,
        )
        if proc.returncode != 0 or not os.path.exists(path):
            raise ValueError("The processed audio could not be saved in that format.")
        return path
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


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


def _kweighted_subblocks(y, sr):
    """K-weighted, channel-weighted power summed over consecutive 100 ms hops.

    Streams the file in chunks (filter state carried across chunks) so long
    recordings never need a second full-length float64 copy in memory.
    Returns (subblock_sums, hop_samples, whole_signal_mean_power).
    """
    n_ch, n = y.shape
    hop = max(1, int(round(sr * 0.1)))
    sos = _k_weighting_sos(sr)
    weights = _channel_weights(n_ch)
    zi = np.zeros((sos.shape[0], n_ch, 2))
    chunk = hop * max(1, (1 << 20) // hop)
    sums = []
    total = 0.0
    for start in range(0, n, chunk):
        seg = np.asarray(y[:, start:start + chunk], dtype=np.float64)
        filtered, zi = _sps.sosfilt(sos, seg, axis=-1, zi=zi)
        power = np.tensordot(weights, filtered * filtered, axes=1)
        total += float(power.sum())
        full = (power.shape[0] // hop) * hop
        if full:
            sums.append(power[:full].reshape(-1, hop).sum(axis=1))
    sub = np.concatenate(sums) if sums else np.zeros(0)
    return sub, hop, (total / n if n else 0.0)


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


def _true_peak_chunks(y, sr, chunk=1 << 18, pad=32):
    """Yield (start, stop, per-sample oversampled |peak| over channels)."""
    n_ch, n = y.shape
    factor = _oversample_factor(sr)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        if factor == 1:
            yield start, stop, np.max(np.abs(y[:, start:stop]), axis=0).astype(np.float64)
            continue
        a, b = max(0, start - pad), min(n, stop + pad)
        up = _sps.resample_poly(np.asarray(y[:, a:b], dtype=np.float64), factor, 1, axis=-1)
        lo = (start - a) * factor
        hi = lo + (stop - start) * factor
        env = np.abs(up[:, lo:hi]).reshape(n_ch, stop - start, factor).max(axis=(0, 2))
        # The original samples are always part of the true-peak estimate.
        env = np.maximum(env, np.max(np.abs(y[:, start:stop]), axis=0))
        yield start, stop, env


def _true_peak_linear(y, sr):
    peak = 0.0
    for _, _, env in _true_peak_chunks(y, sr):
        if env.size:
            peak = max(peak, float(env.max()))
    return peak


def _true_peak_envelope(y, sr):
    env = np.zeros(y.shape[1], dtype=np.float64)
    for start, stop, part in _true_peak_chunks(y, sr):
        env[start:stop] = part
    return env


def _to_db(lin, floor=PEAK_FLOOR_DB):
    return float(max(floor, 20.0 * np.log10(lin))) if lin > 0 else float(floor)


def measure_loudness_array(y, sr, include_true_peak=True):
    """Loudness report for a [channels, samples] (or mono 1-D) float array."""
    y = np.atleast_2d(np.asarray(y))
    n_ch, n = y.shape
    sub, hop, mean_power = _kweighted_subblocks(y, sr)
    block_len = 4 * hop

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

    sample_peak = float(np.max(np.abs(y))) if n else 0.0
    report = {
        "integrated_lufs": round(float(integrated), 2),
        "loudness_range_lu": round(lra, 2),
        "momentary_max_lufs": round(momentary_max, 2),
        "short_term_max_lufs": round(short_term_max, 2),
        "sample_peak_dbfs": round(_to_db(sample_peak), 2),
    }
    if include_true_peak:
        report["true_peak_dbtp"] = round(_to_db(_true_peak_linear(y, sr)), 2)
    return report


def calculate_lufs(audio_path):
    """
    Measure loudness per ITU-R BS.1770-4 / EBU R128.

    Returns integrated loudness (K-weighted, 400 ms blocks with 75 % overlap,
    −70 LUFS absolute and −10 LU relative gates), 4x-oversampled true peak,
    loudness range (EBU Tech 3342), momentary/short-term maxima, and the
    sample peak. Silence reports the −70 LUFS gate floor. ``lufs`` and
    ``peak_db`` keep their historical names for existing callers.
    """
    y, sr = load_audio_array(audio_path)
    m = measure_loudness_array(y, sr)
    return {
        "lufs": m["integrated_lufs"],
        "integrated_lufs": m["integrated_lufs"],
        "true_peak_dbtp": m["true_peak_dbtp"],
        "loudness_range_lu": m["loudness_range_lu"],
        "momentary_max_lufs": m["momentary_max_lufs"],
        "short_term_max_lufs": m["short_term_max_lufs"],
        "peak_db": m["sample_peak_dbfs"],
        "sample_rate": sr,
        "channels": int(y.shape[0]),
        "duration": round(y.shape[1] / float(sr), 3),
        "standard": "ITU-R BS.1770-4 / EBU R128",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  True-peak limiter + loudness normalisation
# ═══════════════════════════════════════════════════════════════════════════

def true_peak_limit(y, sr, ceiling_dbtp=-1.0, lookahead_ms=5.0, release_ms=100.0):
    """Look-ahead brickwall limiter on the 4x-oversampled peak envelope.

    Gain never exceeds what each sample needs (so the ceiling holds); the gain
    curve ramps down over the look-ahead window and recovers with a smooth
    one-pole release, so there are no step changes (no clicks).
    Returns (limited_audio, max_gain_reduction_db).
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    n = y.shape[1]
    if n == 0:
        return y, 0.0
    ceiling = 10.0 ** ((ceiling_dbtp - 0.1) / 20.0)  # 0.1 dB safety margin
    env = _true_peak_envelope(y, sr)
    required = np.minimum(1.0, ceiling / np.maximum(env, 1e-12))
    if required.min() >= 1.0:
        return y, 0.0

    from scipy.ndimage import minimum_filter1d
    look = max(1, int(round(lookahead_ms * 1e-3 * sr)))
    # min over [k-look, k+look]: look-ahead for the attack, short hold after.
    held = minimum_filter1d(required, size=2 * look + 1, mode="nearest")
    # Backward box average of length `look`: every averaged value is <= the
    # requirement at the current sample, so the ceiling is still guaranteed.
    # smooth[i] = mean(held[i-look+1 .. i]) (edge-padded with held[0]).
    csum = np.concatenate([[0.0], np.concatenate([np.full(look, held[0]), held]).cumsum()])
    smooth = (csum[look + 1:look + 1 + n] - csum[1:1 + n]) / look
    alpha = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    zi = _sps.lfilter_zi([1.0 - alpha], [1.0, -alpha]) * smooth[0]
    released, _ = _sps.lfilter([1.0 - alpha], [1.0, -alpha], smooth, zi=zi)
    gain = np.minimum(smooth, released)
    reduction_db = float(-20.0 * np.log10(max(gain.min(), 1e-6)))
    return y * gain[np.newaxis, :], reduction_db


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


def normalize_loudness_array(y, sr, target_lufs=-14.0, true_peak_ceiling=-1.0,
                             max_limiting_db=12.0):
    """Two-pass loudness normalisation of an in-memory signal.

    Pass 1 measures integrated loudness; pass 2 applies the static gain and a
    true-peak limiter. If limiting pulled the loudness under target, gain is
    re-trimmed and re-limited (at most a few iterations, bounded by
    ``max_limiting_db`` so dense material is not crushed).
    Returns (audio, report).
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    before = measure_loudness_array(y, sr)
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
        return y, report

    base_gain = target_lufs - before["integrated_lufs"]
    gain_db = base_gain
    out, reduction = y, 0.0
    measured = before
    for _ in range(5):
        out, reduction = true_peak_limit(y * 10.0 ** (gain_db / 20.0), sr, true_peak_ceiling)
        measured = measure_loudness_array(out, sr)
        err = target_lufs - measured["integrated_lufs"]
        if reduction == 0.0 or abs(err) <= 0.1:
            break
        if gain_db + err - base_gain > max_limiting_db:
            break
        gain_db += err

    # Final safety: if inter-sample peaks still poke over, trim statically.
    if measured["true_peak_dbtp"] > true_peak_ceiling:
        trim = true_peak_ceiling - measured["true_peak_dbtp"] - 0.02
        out = out * 10.0 ** (trim / 20.0)
        gain_db += trim
        measured = measure_loudness_array(out, sr)

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
    return out, report


def normalize_loudness(audio_path, output_path, target_lufs=None, true_peak_ceiling=None,
                       preset=None):
    """File-level loudness normalisation; returns a measurement report dict."""
    key, lufs, ceiling = _resolve_loudness_targets(preset, target_lufs, true_peak_ceiling)
    y, sr = load_audio_array(audio_path)
    out, report = normalize_loudness_array(y, sr, lufs, ceiling)
    write_audio_array(output_path, out, sr)
    if report.get("status") == "partial":
        _log.warning("Loudness normalisation stopped short of target: %s", report)
    report["preset"] = key
    report["output_path"] = output_path
    return report

def apply_parametric_eq(audio_path, output_path, eq_bands=None):
    """
    Apply 10-Band Parametric EQ adjustments to an audio file.
    eq_bands: dict of band_name -> gain_db (e.g. {"32": 2.0, "64": -1.0, ...})
    """
    if eq_bands is None:
        eq_bands = {}
        
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    is_mono = (y.ndim == 1)
    if is_mono:
        y = y.reshape(1, -1)
        
    # Standard 10 octave bands (Hz)
    bands = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    
    # Apply FFT-based EQ curve filtering
    fft_len = 4096
    hop = 1024
    
    processed_channels = []
    for ch in range(y.shape[0]):
        stft = librosa.stft(y[ch], n_fft=fft_len, hop_length=hop)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=fft_len)
        
        # Build frequency gain mask
        gain_mask = np.ones(stft.shape[0], dtype=np.float32)
        
        for band_freq in bands:
            band_str = str(int(band_freq))
            gain_db = float(eq_bands.get(band_str, 0.0))
            if gain_db != 0.0:
                # Gaussian shaped gain curve around band frequency
                linear_gain = 10.0 ** (gain_db / 20.0)
                sigma = band_freq * 0.4
                gaussian = np.exp(-0.5 * ((freqs - band_freq) / (sigma + 1e-5)) ** 2)
                gain_mask += (linear_gain - 1.0) * gaussian
                
        # Apply mask & ISTFT
        stft_eq = stft * gain_mask[:, np.newaxis]
        y_eq = librosa.istft(stft_eq, hop_length=hop, length=y.shape[1])
        processed_channels.append(y_eq)
        
    y_out = np.vstack(processed_channels)
    if is_mono:
        y_out = y_out[0]
    else:
        y_out = y_out.T
        
    sf.write(output_path, y_out, sr)
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
        y, sr = load_audio_array(audio_path)
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y * (10.0 ** (peak_db / 20.0) / peak)
        write_audio_array(output_path, y, sr)
        return output_path

    normalize_loudness(audio_path, output_path, target_lufs=target_lufs,
                       true_peak_ceiling=true_peak_ceiling, preset=preset)
    return output_path


def apply_fades(audio_path, output_path, fade_in_sec=2.0, fade_out_sec=2.0):
    """
    Apply fade in and fade out to an audio file.
    """
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

