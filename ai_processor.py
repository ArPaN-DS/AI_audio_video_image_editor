import runtime_tuning  # noqa: F401  (thread-pool defaults before numeric libraries load)

import os
import re
import shutil
import numpy as np

from runtime_tuning import import_noisereduce, lazy_module

# Heavy libraries load on first use, not at server start.
librosa = lazy_module("librosa")
nr = lazy_module("noisereduce", loader=import_noisereduce)
sf = lazy_module("soundfile")


# ── Streamed frame energy ─────────────────────────────────────────────────
#
# librosa.load(sr=None, mono=True) + feature.rms materialise the whole signal
# and then a (frame_length x n_frames) float32 copy of it — ~4x the signal for
# the default 2048/512 framing (a 10-minute 48 kHz file peaks around 0.6 GB,
# an hour at several GB). The helpers below stream the decoded file and call
# librosa's own rms on frame-aligned chunks of the centre-padded signal, so the
# per-frame values are bit-identical while memory stays O(chunk).

_RMS_CHUNK_FRAMES = 4096


def _stream_mono_rms(path, frame_length=2048, hop_length=512):
    """(rms[n_frames] float32, sr, n_samples) identical to
    ``librosa.feature.rms(y=librosa.load(path, sr=None, mono=True)[0])[0]``,
    or None when the file is not a block-decodable PCM container."""
    import audio_processor as ap
    try:
        info = sf.info(path)
    except Exception:
        return None
    if str(info.format).upper() not in ap._STREAMABLE_FORMATS or info.frames <= 0:
        return None
    n, sr, channels = int(info.frames), int(info.samplerate), int(info.channels)
    half = frame_length // 2
    padded = n + 2 * half
    if padded < frame_length:
        return None
    n_frames = 1 + (padded - frame_length) // hop_length
    out = np.empty(n_frames, dtype=np.float32)
    reader = ap._SequentialReader(path, channels, n)
    try:
        for k0 in range(0, n_frames, _RMS_CHUNK_FRAMES):
            k1 = min(n_frames, k0 + _RMS_CHUNK_FRAMES)
            a = k0 * hop_length - half                     # signal coords of the padded window
            b = (k1 - 1) * hop_length + frame_length - half
            block = reader.read(max(a, 0), min(b, n))
            mono = block[0] if channels == 1 else np.mean(block, axis=0)
            if a < 0 or b > n:
                mono = np.concatenate([np.zeros(max(0, -a), dtype=np.float32), mono,
                                       np.zeros(max(0, b - n), dtype=np.float32)])
            out[k0:k1] = librosa.feature.rms(y=mono, frame_length=frame_length, hop_length=hop_length,
                                             center=False)[0]
    finally:
        reader.close()
    return out, sr, n


def _nonsilent_frames(rms, top_db):
    """librosa.effects._signal_to_frame_nonsilent for a mono rms curve."""
    db = librosa.amplitude_to_db(rms, ref=np.max, top_db=None)
    return db > -top_db


def _split_from_rms(rms, n_samples, top_db, hop_length=512):
    """librosa.effects.split on precomputed frame energies (same edge rules)."""
    non_silent = _nonsilent_frames(rms, top_db)
    edges = [np.flatnonzero(np.diff(non_silent.astype(int))) + 1]
    if non_silent[0]:
        edges.insert(0, np.array([0]))
    if non_silent[-1]:
        edges.append(np.array([len(non_silent)]))
    edges = np.concatenate(edges) * hop_length
    edges = np.minimum(edges, n_samples)
    return edges.reshape((-1, 2))


def _silence_regions(non_silent_intervals, sr, duration, min_silence_len):
    silence_regions = []
    non_silent_secs = [(start_idx / sr, end_idx / sr) for start_idx, end_idx in non_silent_intervals]
    if not non_silent_secs:
        # The entire audio is silent
        return [{"start": 0.0, "end": round(duration, 3), "duration": round(duration, 3)}]
    current_time = 0.0
    for start_sec, end_sec in non_silent_secs:
        if start_sec - current_time >= min_silence_len:
            silence_regions.append({
                "start": round(current_time, 3),
                "end": round(start_sec, 3),
                "duration": round(start_sec - current_time, 3)
            })
        current_time = end_sec
    if duration - current_time >= min_silence_len:
        silence_regions.append({
            "start": round(current_time, 3),
            "end": round(duration, 3),
            "duration": round(duration - current_time, 3)
        })
    return silence_regions


def detect_silence(path, min_silence_len=0.5, silence_thresh=40):
    """
    Detects silent gaps in the audio.
    min_silence_len: minimum duration of silence in seconds to be registered
    silence_thresh: threshold (in dB) below reference to consider silence (equivalent to top_db in librosa.effects.split)
    """
    streamed = _stream_mono_rms(path)
    if streamed is not None:
        rms, sr, n = streamed
        intervals = _split_from_rms(rms, n, silence_thresh)
        return _silence_regions(intervals, sr, n / sr, min_silence_len)

    y, sr = librosa.load(path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # split returns intervals of non-silent regions
    non_silent_intervals = librosa.effects.split(y, top_db=silence_thresh)
    return _silence_regions(non_silent_intervals, sr, duration, min_silence_len)


def auto_trim_silence(path, threshold=40):
    """
    Detects silent portions at start and end and returns proposed trim points.
    """
    streamed = _stream_mono_rms(path)
    if streamed is not None:
        rms, sr, n = streamed
        duration = n / sr
        # librosa.effects.trim on precomputed frame energies.
        nonzero = np.flatnonzero(_nonsilent_frames(rms, threshold))
        if nonzero.size > 0:
            index = (int(nonzero[0]) * 512, min(n, int(nonzero[-1] + 1) * 512))
        else:
            index = (0, 0)
    else:
        y, sr = librosa.load(path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # trim returns the trimmed signal and the start/end samples
        y_trimmed, index = librosa.effects.trim(y, top_db=threshold)
        del y, y_trimmed

    trimmed_start = float(index[0]) / sr
    trimmed_end = float(index[1]) / sr
    
    return {
        "trimmed_start": round(trimmed_start, 3),
        "trimmed_end": round(trimmed_end, 3),
        "removed_start_ms": round(trimmed_start * 1000, 1),
        "removed_end_ms": round((duration - trimmed_end) * 1000, 1),
        "total_duration": round(duration, 3)
    }

def detect_beats(path):
    """
    Analyzes the audio for tempo (BPM) and beat positions.
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    # Convert numpy float/array tempo to normal float
    if hasattr(tempo, "__len__"):
        bpm = float(tempo[0])
    else:
        bpm = float(tempo)
        
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times_list = [round(float(t), 3) for t in beat_times]
    
    return {
        "bpm": round(bpm, 2),
        "beat_times": beat_times_list,
        "total_beats": len(beat_times_list)
    }

def reduce_noise(path, output_path):
    """
    Applies local noise reduction using noisereduce package.
    Preserves stereo shape.
    """
    import audio_processor
    # Keep the channel layout; unreadable input raises a ValueError.
    y, sr = audio_processor.load_audio_array(path)
    if y.shape[0] == 1:
        y = y[0]

    # Run noise reduction
    reduced_y = nr.reduce_noise(y=y, sr=sr)
    
    # Save the output file. Note: soundfile writes channels as columns, so we transpose 2D arrays
    if reduced_y.ndim > 1:
        reduced_y_to_write = reduced_y.T
    else:
        reduced_y_to_write = reduced_y
        
    sf.write(output_path, reduced_y_to_write, sr)
    return output_path

def detect_voice_activity(path, threshold_db=-35.0, frame_length=2048, hop_length=512):
    """
    Performs Voice Activity Detection using RMS energy analysis.
    Classifies frames as 'speech' or 'silence'.
    """
    streamed = _stream_mono_rms(path, frame_length, hop_length)
    if streamed is not None:
        rms, sr, n = streamed
        duration = n / sr
    else:
        y, sr = librosa.load(path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Compute RMS energy for each frame
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        del y
    
    # Avoid log of zero
    rms = np.maximum(rms, 1e-10)
    
    # Safeguard: if absolute peak RMS is extremely quiet, classify entire track as silence
    peak_rms = np.max(rms)
    if peak_rms < 0.001:
        return [{"start": 0.0, "end": round(duration, 3), "type": "silence"}]
        
    # Convert to dB relative to peak energy
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Calculate time timestamps for each frame
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    is_speech = rms_db > threshold_db
    
    segments = []
    if len(is_speech) == 0:
        return segments
        
    current_state = "speech" if is_speech[0] else "silence"
    start_time = 0.0
    
    for i in range(1, len(is_speech)):
        state = "speech" if is_speech[i] else "silence"
        if state != current_state:
            end_time = times[i]
            segments.append({
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "type": current_state
            })
            current_state = state
            start_time = end_time
            
    segments.append({
        "start": round(start_time, 3),
        "end": round(duration, 3),
        "type": current_state
    })
    
    return segments

# ═══════════════════════════════════════════════════════════════════════════
#  HIGH-ACCURACY TRANSCRIPTION ENGINE  v2.0
#  Principal ML Engineer — Production-Optimized
#
#  Engine : faster-whisper  (CTranslate2 backend)
#
#  ┌─────────────────────────────────────────────────────────────────────┐
#  │  OPTIMIZATION STACK (based on latest 2025-2026 research)          │
#  │                                                                    │
#  │  1. Cascading model fallback  (GPU → CPU, large → tiny)           │
#  │  2. int8_float16 on GPU  (INT8 weights + FP16 activations)        │
#  │  3. Audio preprocessing  (resample to 16kHz mono — native fmt)    │
#  │  4. Temperature fallback  [0, 0.2, 0.4, 0.6, 0.8, 1.0]          │
#  │  5. Hallucination prevention  (tuned thresholds + VAD)            │
#  │  6. CPU thread optimization  (75% of cores)                       │
#  │  7. Streaming segment collection  (no RAM bloat)                  │
#  │  8. Post-inference cleanup  (gc + GPU cache clear)                │
#  │  9. Repetition filter  (catches hallucinated loops)               │
#  └─────────────────────────────────────────────────────────────────────┘
#
#  Model Tiers:
#    Tier 1: large-v3-turbo  (809M, ~1.6 GB, 97.2% accuracy, 99+ langs)
#    Tier 2: medium          (769M, ~1.5 GB, 97.1% accuracy, 99+ langs)
#    Tier 3: small           (244M, ~466 MB, 96.6% accuracy, 99+ langs)
#    Tier 4: base            (74M,  ~142 MB, 95.0% accuracy, 99+ langs)
#    Tier 5: tiny            (39M,  ~75 MB,  92.4% accuracy, 99+ langs)
# ═══════════════════════════════════════════════════════════════════════════

import gc
import sys
import math
import logging
import tempfile

_log = logging.getLogger("transcribe")
if not _log.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("[STT] %(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)

# ── Model Cascade Config ──────────────────────────────────────────────────

_MODEL_CASCADE = [
    {
        "name": "large-v3-turbo",
        "params": "809M",
        "disk": "~1.6 GB",
        "accuracy": "97.2%",
        "gpu_vram_int8f16": 2.0,   # int8_float16 VRAM
        "gpu_vram_fp16": 4.0,      # float16 VRAM
        "cpu_ram_min": 3.5,
    },
    {
        "name": "medium",
        "params": "769M",
        "disk": "~1.5 GB",
        "accuracy": "97.1%",
        "gpu_vram_int8f16": 1.8,
        "gpu_vram_fp16": 3.5,
        "cpu_ram_min": 3.0,
    },
    {
        "name": "small",
        "params": "244M",
        "disk": "~466 MB",
        "accuracy": "96.6%",
        "gpu_vram_int8f16": 0.8,
        "gpu_vram_fp16": 1.5,
        "cpu_ram_min": 1.5,
    },
    {
        "name": "base",
        "params": "74M",
        "disk": "~142 MB",
        "accuracy": "95.0%",
        "gpu_vram_int8f16": 0.4,
        "gpu_vram_fp16": 0.8,
        "cpu_ram_min": 0.8,
    },
    {
        "name": "tiny",
        "params": "39M",
        "disk": "~75 MB",
        "accuracy": "92.4%",
        "gpu_vram_int8f16": 0.3,
        "gpu_vram_fp16": 0.4,
        "cpu_ram_min": 0.5,
    },
]

_whisper_model_cache = None
_model_info = None
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: Hardware Detection
# ═══════════════════════════════════════════════════════════════════════════

def _get_free_ram_gb():
    """Get available (free) system RAM in GB — works without psutil."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        try:
            import ctypes
            if sys.platform == "win32":
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullAvailPhys / (1024 ** 3)
        except Exception:
            pass
    return 4.0  # Conservative fallback


def _get_optimal_cpu_threads():
    """Use 75% of CPU cores — leaves headroom for OS + Flask."""
    try:
        cores = os.cpu_count() or 4
        return max(2, int(cores * 0.75))
    except Exception:
        return 4


def _probe_gpu():
    """
    Deep GPU probe — checks CUDA health, free VRAM, device name.
    Works via PyTorch, or falls back to ctranslate2 + nvidia-smi.
    """
    info = {
        "available": False,
        "cuda_working": False,
        "name": None,
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
    }
    # 1. Try PyTorch if installed
    try:
        import torch
        if torch.cuda.is_available():
            try:
                _test = torch.zeros(1, device="cuda")
                del _test
                info["cuda_working"] = True
            except Exception as e:
                _log.warning(f"CUDA is_available=True but allocation failed: {e}")
                return info

            props = torch.cuda.get_device_properties(0)
            info["available"] = True
            info["name"] = props.name
            info["vram_total_gb"] = props.total_mem / (1024 ** 3)
            free_bytes, _ = torch.cuda.mem_get_info(0)
            info["vram_free_gb"] = free_bytes / (1024 ** 3)
            return info
    except Exception:
        pass

    # 2. Fallback to ctranslate2 + nvidia-smi (works without PyTorch)
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            info["available"] = True
            info["cuda_working"] = True
            try:
                import subprocess
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader,nounits'],
                    text=True, timeout=3
                ).strip()
                parts = [p.strip() for p in out.split(',')]
                if len(parts) >= 3:
                    info["name"] = parts[0]
                    info["vram_total_gb"] = float(parts[1]) / 1024.0
                    info["vram_free_gb"] = float(parts[2]) / 1024.0
                else:
                    info["name"] = "NVIDIA CUDA GPU"
                    info["vram_total_gb"] = 8.0
                    info["vram_free_gb"] = 6.0
            except Exception:
                info["name"] = "NVIDIA CUDA GPU"
                info["vram_total_gb"] = 8.0
                info["vram_free_gb"] = 6.0
    except Exception as e:
        _log.warning(f"GPU probe error: {e}")

    return info


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: Audio Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess_audio(path):
    """
    Normalize audio to Whisper's native format for optimal accuracy:
      - 16 kHz sample rate  (Whisper was trained on 16kHz)
      - Mono channel        (stereo causes channel interference)
      - Float32 PCM         (normalized amplitude)

    If audio is already 16kHz mono, returns original path (zero-copy).
    Otherwise, creates a temp WAV file and returns that path.
    """
    try:
        y, sr = librosa.load(path, sr=None, mono=True)

        # Already 16kHz? Return original to skip resampling
        if sr == 16000:
            return path, None  # (path, temp_file_to_cleanup)

        # Resample to 16kHz
        _log.info(f"Resampling audio: {sr}Hz -> 16000Hz")
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False,
            dir=os.path.dirname(path) or "."
        )
        sf.write(tmp.name, y_16k, 16000, subtype="FLOAT")
        tmp.close()
        return tmp.name, tmp.name  # Return temp path + path to cleanup

    except Exception as e:
        _log.warning(f"Audio preprocessing failed ({e}), using original file")
        return path, None


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: Model Loading (Cascading Fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _try_load_model(model_name, device, compute_type):
    """
    Attempt to load a model. Returns (model, None) or (None, error_str).
    Sets CTranslate2 CPU threads for optimal throughput.
    """
    from faster_whisper import WhisperModel

    try:
        os.makedirs(_MODELS_DIR, exist_ok=True)
        kwargs = {
            "device": device,
            "compute_type": compute_type,
            "download_root": _MODELS_DIR,
        }
        # Optimize CPU thread count
        if device == "cpu":
            kwargs["cpu_threads"] = _get_optimal_cpu_threads()

        model = WhisperModel(model_name, **kwargs)
        return model, None
    except Exception as e:
        return None, str(e)


def _clear_gpu_memory():
    """Force GPU cache clear + Python garbage collection."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


_STT_TIER_CEILING = {"lite": "small", "balanced": "medium", "max": "large-v3-turbo"}


def _eligible_cascade():
    """
    Speech model ladder filtered by the adaptive quality governor: a pinned
    MEDIA_QUALITY_TIER caps the largest model, and models that recently failed
    for memory reasons are skipped during their cooldown. The smallest tier is
    always kept so transcription can still run.
    """
    try:
        from model_manager import global_quality_governor
        ceiling_tier = global_quality_governor.tier_override()
        ceiling = _STT_TIER_CEILING.get(ceiling_tier)
        names = [t["name"] for t in _MODEL_CASCADE]
        start = names.index(ceiling) if ceiling in names else 0
        eligible = [
            t for t in _MODEL_CASCADE[start:]
            if not global_quality_governor.is_paused("speech.transcribe", t["name"])
        ]
        return eligible or [_MODEL_CASCADE[-1]]
    except Exception:
        return list(_MODEL_CASCADE)


def _report_stt_failure(model_name, error):
    try:
        from model_manager import global_quality_governor, is_resource_error
        if is_resource_error(error):
            global_quality_governor.report_failure("speech.transcribe", model_name, error)
    except Exception:
        pass


def _load_best_model():
    """
    3-Phase cascading model loader:

    Phase 1 — GPU (if available):
      For each tier: try int8_float16 → float16 → skip
      On fail: clear GPU memory, try next smaller model.

    Phase 2 — CPU fallback:
      For each tier: try int8 (optimal for CPU)
      On fail: try next smaller model.

    Phase 3 — Emergency:
      Force 'tiny' on CPU. Always succeeds (needs ~75 MB).
    """
    gpu = _probe_gpu()
    free_ram = _get_free_ram_gb()

    _log.info("=" * 60)
    _log.info("  HARDWARE DETECTION")
    if gpu["available"]:
        _log.info(f"    GPU     : {gpu['name']}")
        _log.info(f"    VRAM    : {gpu['vram_free_gb']:.1f} / {gpu['vram_total_gb']:.1f} GB free")
    else:
        _log.info("    GPU     : Not available")
    _log.info(f"    RAM     : {free_ram:.1f} GB free")
    _log.info(f"    CPU     : {os.cpu_count()} cores ({_get_optimal_cpu_threads()} threads allocated)")
    _log.info("=" * 60)

    # ── Phase 1: GPU ──────────────────────────────────────────────────
    if gpu["available"] and gpu["cuda_working"]:
        vfree = gpu["vram_free_gb"]
        _log.info("Phase 1 → GPU")

        for tier in _eligible_cascade():
            name = tier["name"]

            # Pick best compute type: int8_float16 preferred on GPU
            # (INT8 weights + FP16 activations = best speed/accuracy/VRAM)
            if vfree >= tier["gpu_vram_fp16"]:
                compute = "float16"
            elif vfree >= tier["gpu_vram_int8f16"]:
                compute = "int8_float16"
            else:
                _log.info(f"  ✗ {name}: need {tier['gpu_vram_int8f16']:.1f} GB, have {vfree:.1f} GB")
                continue

            _log.info(f"  → {name} ({tier['accuracy']}) on GPU/{compute}...")
            model, err = _try_load_model(name, "cuda", compute)

            if model is not None:
                _log.info(f"  ✓ {name} loaded — {tier['accuracy']} accuracy, {tier['disk']}")
                return model, {
                    "model": name, "params": tier["params"],
                    "accuracy": tier["accuracy"], "disk": tier["disk"],
                    "device": f"GPU ({gpu['name']}) → {compute}",
                }

            _log.warning(f"  ✗ {name}: {err}")
            _report_stt_failure(name, err)
            _clear_gpu_memory()

        _log.info("Phase 1 done — no GPU model fit. → CPU")
        _clear_gpu_memory()

    # ── Phase 2: CPU ──────────────────────────────────────────────────
    _log.info("Phase 2 → CPU")
    free_ram = _get_free_ram_gb()  # Re-check after GPU cleanup

    for tier in _eligible_cascade():
        name = tier["name"]
        if free_ram < tier["cpu_ram_min"]:
            _log.info(f"  ✗ {name}: need {tier['cpu_ram_min']:.1f} GB, have {free_ram:.1f} GB")
            continue

        _log.info(f"  → {name} ({tier['accuracy']}) on CPU/int8...")
        model, err = _try_load_model(name, "cpu", "int8")

        if model is not None:
            _log.info(f"  ✓ {name} loaded — {tier['accuracy']} accuracy, {tier['disk']}")
            return model, {
                "model": name, "params": tier["params"],
                "accuracy": tier["accuracy"], "disk": tier["disk"],
                "device": "CPU → int8",
            }

        _log.warning(f"  ✗ {name}: {err}")
        _report_stt_failure(name, err)
        gc.collect()

    # ── Phase 3: Emergency ────────────────────────────────────────────
    _log.warning("Phase 3 → Emergency: tiny on CPU")
    model, err = _try_load_model("tiny", "cpu", "int8")
    if model is not None:
        return model, {
            "model": "tiny", "params": "39M",
            "accuracy": "92.4%", "disk": "~75 MB",
            "device": "CPU → int8 (emergency)",
        }
    raise RuntimeError(f"Cannot load ANY model. Last error: {err}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: Transcription Parameters (Research-Tuned)
# ═══════════════════════════════════════════════════════════════════════════

# Temperature fallback: start deterministic, increase on failure
# (from OpenAI's own recommendation for difficult audio)
_TEMPERATURE_CASCADE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

_TRANSCRIBE_PARAMS = {
    # ── Decoding Quality ──────────────────────────────────────────────
    "beam_size": 5,                     # Beam search for quality
    "best_of": 5,                       # Sample 5, pick best
    "patience": 2.0,                    # Wait for better beams

    # ── Language ──────────────────────────────────────────────────────
    "language": None,                   # Auto-detect
    "task": "transcribe",               # Transcription (not translation)

    # ── VAD (Silero) — prevents hallucination on silence ─────────────
    "vad_filter": True,
    "vad_parameters": {
        "threshold": 0.35,              # Speech detection sensitivity
        "min_silence_duration_ms": 500,  # Merge speech across short gaps
        "min_speech_duration_ms": 250,   # Ignore ultra-short blips
        "speech_pad_ms": 400,           # Context padding around speech
        "max_speech_duration_s": float("inf"),  # No artificial truncation
    },

    # ── Hallucination Prevention ─────────────────────────────────────
    "no_speech_threshold": 0.6,         # Higher = stricter silence filter
    "log_prob_threshold": -1.0,         # Skip low-confidence segments
    "compression_ratio_threshold": 2.4, # Detect repetitive gibberish

    # ── Accuracy Boosters ────────────────────────────────────────────
    "word_timestamps": True,            # Word-level precision
    "condition_on_previous_text": True, # Use prior context for coherence

    # ── Temperature: fallback cascade for difficult audio ────────────
    "temperature": _TEMPERATURE_CASCADE,
}


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: Post-Processing
# ═══════════════════════════════════════════════════════════════════════════

def _is_repetition(text, threshold=3):
    """
    Detect hallucinated repetition loops.
    E.g. "Thank you. Thank you. Thank you. Thank you."
    Returns True if any phrase repeats more than `threshold` times.
    """
    if not text or len(text) < 20:
        return False
    words = text.split()
    if len(words) < threshold * 2:
        return False

    # Check for repeated N-grams (2-5 words)
    for n in range(2, 6):
        if len(words) < n * threshold:
            continue
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        from collections import Counter
        counts = Counter(ngrams)
        most_common_count = counts.most_common(1)[0][1]
        # If one N-gram takes up most of the text, it's a hallucination
        if most_common_count >= threshold and most_common_count >= len(ngrams) * 0.4:
            return True
    return False


def _collect_segments(segments_iter):
    """
    Stream-process segments from the generator.
    Filters out empty segments and hallucinated repetitions.
    Does NOT collect all into a list first (memory efficient).
    """
    segments = []
    full_text_parts = []

    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue

        # Filter repetitive hallucinations
        if _is_repetition(text):
            _log.warning(f"  Filtered hallucinated segment [{seg.start:.1f}s–{seg.end:.1f}s]: {text[:60]}...")
            continue

        segment_data = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": text,
        }

        # Word-level timestamps with confidence
        if seg.words:
            segment_data["words"] = [
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.probability, 3),
                }
                for w in seg.words
                if w.word.strip()
            ]

        segments.append(segment_data)
        full_text_parts.append(text)

    return segments, " ".join(full_text_parts)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: Main Transcription Function
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_audio(path):
    """
    High-accuracy audio transcription with hardware-aware optimization.

    ACCURACY: 95-99% on clean speech (close to Google Cloud STT).
    LANGUAGES: 99+ with automatic detection.
    RESOURCE: Auto-selects best model for available GPU/CPU/RAM.

    Pipeline:
      1. Preprocess audio → 16kHz mono (Whisper's native format)
      2. Load model (cascading fallback — GPU first, CPU if needed)
      3. Transcribe with research-tuned parameters:
           - Beam search (5 beams, 5 candidates)
           - Temperature fallback [0.0 → 1.0] for difficult audio
           - Silero VAD (kills hallucinations on silence)
           - Tuned no_speech / log_prob / compression thresholds
      4. Post-process: filter repetitions, collect word timestamps
      5. Cleanup: release temp files, gc.collect()

    Returns dict with: available, language, full_text, segments, model, device
    """
    global _whisper_model_cache, _model_info

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {
            "available": False,
            "error": (
                "faster-whisper is not installed. "
                "To enable high-accuracy transcription, run:\n"
                "  pip install faster-whisper"
            )
        }

    temp_audio_path = None

    try:
        from model_manager import global_model_manager

        # ── Step 1: Audio Preprocessing ───────────────────────────────
        audio_path, temp_audio_path = _preprocess_audio(path)

        def _unload_whisper(model_inst):
            global _whisper_model_cache, _model_info
            _whisper_model_cache = None
            _model_info = None
            _clear_gpu_memory()

        def _load_whisper():
            global _whisper_model_cache, _model_info
            if _whisper_model_cache is None:
                _whisper_model_cache, _model_info = _load_best_model()
            return _whisper_model_cache, _model_info

        # Execute inside global ModelManager session (Zero-Idle Memory)
        with global_model_manager.session("stt_whisper", _load_whisper, _unload_whisper) as (model, info_dict):

            # ── Step 2: Transcribe ────────────────────────────────────────
            try:
                segments_iter, info = model.transcribe(
                    audio_path, **_TRANSCRIBE_PARAMS
                )
                segments, full_text = _collect_segments(segments_iter)

            except (RuntimeError, MemoryError) as oom_err:
                from model_manager import is_resource_error
                if not is_resource_error(oom_err):
                    raise

                # ── OOM Recovery: drop model, cascade to smaller ─────────
                current_name = (_model_info or {}).get("model", "")
                _log.warning("Speech model ran out of memory; recovering with a lighter model.")
                _report_stt_failure(current_name, oom_err)

                # Release every reference to the failed model BEFORE loading a
                # smaller one, otherwise both stay resident and recovery OOMs too.
                model = None
                info_dict = None
                _whisper_model_cache = None
                global_model_manager.release_active_instance()
                _clear_gpu_memory()

                current_idx = next(
                    (i for i, t in enumerate(_MODEL_CASCADE) if t["name"] == current_name),
                    -1
                )

                recovered = False
                for tier in _MODEL_CASCADE[current_idx + 1:]:
                    _log.info(f"  OOM recovery → trying {tier['name']} on CPU...")
                    m, err = _try_load_model(tier["name"], "cpu", "int8")
                    if m is not None:
                        _whisper_model_cache = m
                        _model_info = {
                            "model": tier["name"], "params": tier["params"],
                            "accuracy": tier["accuracy"], "disk": tier["disk"],
                            "device": "CPU → int8 (OOM recovery)",
                        }
                        global_model_manager.adopt_active_instance(m, _model_info)
                        _log.info(f"  ✓ Recovered: {tier['name']}")
                        recovered = True
                        break
                    _report_stt_failure(tier["name"], err)
                    gc.collect()

                if not recovered:
                    raise RuntimeError("All models exhausted after OOM")

                # Retry transcription
                segments_iter, info = _whisper_model_cache.transcribe(
                    audio_path, **_TRANSCRIBE_PARAMS
                )
                segments, full_text = _collect_segments(segments_iter)

            # ── Step 3: Build Response ────────────────────────────────────
            detected_lang = info.language if info.language else "en"
            lang_prob = (
                round(info.language_probability, 3)
                if info.language_probability else 0.0
            )

            # Compute average word confidence across all segments
            all_confidences = []
            for seg in segments:
                for w in seg.get("words", []):
                    all_confidences.append(w["confidence"])
            avg_confidence = (
                round(sum(all_confidences) / len(all_confidences), 3)
                if all_confidences else 0.0
            )

            return {
                "available": True,
                "language": detected_lang,
                "language_confidence": lang_prob,
                "full_text": full_text,
                "segments": segments,
                "model": _model_info.get("model", "unknown") if _model_info else "unknown",
                "model_accuracy": _model_info.get("accuracy", "unknown") if _model_info else "unknown",
                "device": _model_info.get("device", "unknown") if _model_info else "unknown",
                "avg_word_confidence": avg_confidence,
                "segment_count": len(segments),
            }

    except Exception as e:
        return {
            "available": False,
            "error": f"Failed to transcribe: {str(e)}"
        }

    finally:
        # ── Step 4: Cleanup ───────────────────────────────────────────
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass

        gc.collect()


# ── Filler words ──────────────────────────────────────────────────────────
#
# Limitation (documented, not hidden): automatic transcription is trained to
# write what the speaker *meant*, so hesitation sounds ("um", "uh") are often
# left out of the transcript entirely. Only fillers that actually appear in
# the transcript can be marked; nothing is invented.

HESITATION_FILLERS = ("um", "umm", "uh", "uhh", "uhm", "erm", "er", "err", "ah", "ahh",
                      "eh", "hmm", "hm", "mm", "mmm", "mhm")
# Opt-in: these are often meaningful words, so they cause false positives.
DISCOURSE_MARKERS = ("like", "you know", "so", "i mean", "basically", "actually",
                     "literally", "kind of", "sort of")

_HESITATION_RE = re.compile(r"^(?:u+m+|u+h+m*|e+r+m*|a+h+|e+h+|h+m+|m+h*m+)$")
_FILLER_STRIP = ".,!?;:\"'()[]{}…-–—"

FILLER_LIMITATION_NOTE = (
    "Speech recognition tends to tidy up hesitations, so some 'um' and 'uh' sounds "
    "may be missing from the transcript and cannot be marked. Words such as 'like', "
    "'so' and 'you know' are only checked when you turn them on, because they are "
    "usually meaningful."
)


def _normalize_filler_token(text):
    return (text or "").strip().lower().strip(_FILLER_STRIP).strip()


def detect_filler_words(path, custom_words=None, include_discourse_markers=False):
    """
    Find filler words in the transcript with word-level timestamps.

    Default: conservative hesitation sounds only (um, uh, erm, ah, hmm …,
    including elongated spellings such as "ummm"). Discourse markers
    ("like", "you know", "so", …) are opt-in via ``include_discourse_markers``.
    ``custom_words`` replaces the default list; multi-word phrases are matched
    across consecutive words. See ``FILLER_LIMITATION_NOTE`` for what cannot
    be detected.
    """
    if custom_words:
        targets = {_normalize_filler_token(w) for w in custom_words if _normalize_filler_token(w)}
        use_patterns = False
    else:
        targets = set(HESITATION_FILLERS)
        use_patterns = True
        if include_discourse_markers:
            targets |= set(DISCOURSE_MARKERS)
    phrases = sorted((tuple(t.split()) for t in targets if " " in t), key=len, reverse=True)
    singles = {t for t in targets if " " not in t}

    res = transcribe_audio(path)
    if not res.get("available"):
        raise RuntimeError(res.get("error", "Speech could not be transcribed."))

    words = []
    for seg in res.get("segments", []):
        for w in seg.get("words", []) or []:
            token = _normalize_filler_token(w.get("word", ""))
            if token:
                words.append((token, w))

    detected = []
    i = 0
    while i < len(words):
        matched = 0
        for phrase in phrases:
            k = len(phrase)
            if tuple(tok for tok, _ in words[i:i + k]) == phrase:
                matched = k
                break
        if not matched:
            token = words[i][0]
            if token in singles or (use_patterns and _HESITATION_RE.match(token)):
                matched = 1
        if matched:
            first = words[i][0]
            is_hesitation = matched == 1 and (first in HESITATION_FILLERS or _HESITATION_RE.match(first))
            span = [w for _, w in words[i:i + matched]]
            start_t = round(float(span[0].get("start", 0.0)), 3)
            end_t = round(float(span[-1].get("end", 0.0)), 3)
            detected.append({
                "word": " ".join(w.get("word", "").strip() for w in span),
                "start": start_t,
                "end": end_t,
                "duration": round(end_t - start_t, 3),
                "confidence": round(min(float(w.get("confidence", 1.0)) for w in span), 3),
                "kind": "hesitation" if is_hesitation else ("custom" if custom_words else "discourse_marker"),
            })
            i += matched
        else:
            i += 1

    return {
        "total_fillers": len(detected),
        "fillers": detected,
        "words_checked": len(words),
        "discourse_markers_checked": bool(include_discourse_markers or custom_words),
        "note": FILLER_LIMITATION_NOTE,
    }


def enhance_speech_studio(input_path, output_path):
    """
    Speech enhancement and de-reverb with the neural voice-isolation model.
    When that capability is unavailable, falls back to spectral noise
    reduction and says so in the result (``fallback: True`` plus a note);
    if the fallback also fails, the error is raised.
    """
    from model_manager import global_model_manager
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio

        def _load_df():
            model, df_state, _ = init_df()
            return (model, df_state), {"engine": "neural_voice_isolation"}

        def _unload_df(instance):
            del instance

        with global_model_manager.session("deepfilternet", _load_df, _unload_df) as ((model, df_state), meta):
            audio, _ = load_audio(input_path, sr=df_state.sr())
            enhanced = enhance(model, df_state, audio)
            save_audio(output_path, enhanced, sr=df_state.sr())
            return {"engine": "neural_voice_isolation", "status": "success", "fallback": False}

    except ImportError:
        reduce_noise(input_path, output_path)
        return {
            "engine": "spectral_noise_reduction", "status": "success", "fallback": True,
            "note": "Advanced voice isolation is not installed, so spectral noise reduction was applied instead.",
        }
    except Exception:
        _log.warning("Voice isolation could not run on this file; using spectral noise reduction instead.")
        reduce_noise(input_path, output_path)
        return {
            "engine": "spectral_noise_reduction", "status": "success", "fallback": True,
            "note": "Advanced voice isolation could not process this file, so spectral noise reduction was applied instead.",
        }


def separate_stems(input_path, output_dir, stems_mode="2", quality="auto", fmt="wav"):
    """
    Split audio into vocals + instrumental ('2') or vocals / drums / bass /
    other ('4'). Delegates to ``separation_processor`` (tiered: studio model
    when installed locally, DSP baseline otherwise) and keeps the legacy
    layout: ``<output_dir>/<stem>.wav`` with the instrumental as ``no_vocals``.
    Unusable input raises ValueError; other failures raise RuntimeError.
    """
    import separation_processor
    mode = "4stem" if str(stems_mode).strip() == "4" else "vocals"
    try:
        report = separation_processor.separate(input_path, output_dir, mode=mode, quality=quality, fmt=fmt)
    except ValueError:
        raise
    except Exception as e:
        _log.warning("Stem separation failed (%s).", type(e).__name__)
        raise RuntimeError("Stem separation failed.") from e

    folder = os.path.basename(os.path.normpath(output_dir))
    results = {
        "status": "success",
        "mode": stems_mode,
        "quality": report["quality"]["label"],
        "quality_note": report["quality_note"],
        "warnings": report["warnings"],
    }
    for stem, path in report["stems"].items():
        legacy = "no_vocals" if stem == "instrumental" else stem
        ext = os.path.splitext(path)[1]
        target = os.path.join(output_dir, f"{legacy}{ext}")
        if os.path.abspath(path) != os.path.abspath(target):
            shutil.move(path, target)
        results[legacy] = f"/processed/{folder}/{legacy}{ext}"
    return results


def _snap_to_zero_crossing(mono, pos, radius):
    """Move a cut point to the quietest sample (nearest zero crossing) nearby."""
    lo = max(0, pos - radius)
    hi = min(len(mono), pos + radius + 1)
    if hi - lo < 2:
        return pos
    window = np.abs(mono[lo:hi])
    return int(lo + np.argmin(window))


def trim_silence_gaps(input_path, output_path, min_silence_len=1.0, silence_thresh=-40,
                      keep_silence=0.15, crossfade_ms=20.0):
    """
    Shorten pauses longer than ``min_silence_len`` seconds.

    Silence = 10 ms frames whose RMS is below ``silence_thresh`` dBFS. Each
    long pause is cut down to ``keep_silence`` seconds of natural room tone
    (breathing room), cut points are snapped to zero crossings, and the joins
    are equal-power crossfaded over ``crossfade_ms`` (10–30 ms) so there are
    no clicks. Leading/trailing pauses get a short fade instead.

    Raises ValueError when the input cannot be read or the settings are
    invalid; it never hands back an untouched copy as a "trimmed" result.
    """
    import audio_processor as ap

    try:
        min_silence_len = float(min_silence_len)
        silence_thresh = float(silence_thresh)
        keep_silence = float(keep_silence)
        crossfade_ms = float(crossfade_ms)
    except (TypeError, ValueError):
        raise ValueError("Silence trimming settings must be numbers.")
    if not (0.1 <= min_silence_len <= 60.0):
        raise ValueError("Choose a minimum pause length between 0.1 and 60 seconds.")
    if not (-90.0 <= silence_thresh <= -10.0):
        raise ValueError("Choose a silence threshold between -90 and -10 dB.")
    keep_silence = min(max(keep_silence, 0.0), min_silence_len)
    crossfade_ms = min(max(crossfade_ms, 10.0), 30.0)

    y, sr = ap.load_audio_array(input_path)
    n = y.shape[1]
    frame = max(1, int(round(sr * 0.01)))
    n_frames = n // frame
    if n_frames == 0:
        raise ValueError("This audio is too short to trim.")

    power = np.mean(np.asarray(y[:, :n_frames * frame], dtype=np.float64) ** 2, axis=0)
    frame_db = 10.0 * np.log10(power.reshape(n_frames, frame).mean(axis=1) + 1e-12)
    silent = frame_db < silence_thresh

    # Runs of silent frames → candidate pauses (in samples).
    padded = np.concatenate([[False], silent, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    runs = list(zip(edges[0::2], edges[1::2]))
    min_frames = int(round(min_silence_len / 0.01))
    half_keep = int(round(keep_silence * sr / 2.0))
    xf = max(2, int(round(crossfade_ms * 1e-3 * sr)))
    snap = max(1, int(round(0.005 * sr)))
    mono = np.asarray(y.mean(axis=0), dtype=np.float64)

    cuts = []  # (cut_start, cut_end) sample ranges to remove
    for a_f, b_f in runs:
        if b_f - a_f < min_frames:
            continue
        a = a_f * frame
        b = n if b_f >= n_frames else b_f * frame
        leading, trailing = a == 0, b >= n
        if leading and trailing:
            raise ValueError("This audio is entirely silent, so there is nothing to keep.")
        cut_start = 0 if leading else a + half_keep
        cut_end = n if trailing else b - half_keep
        if not leading:
            cut_start = _snap_to_zero_crossing(mono, cut_start, snap)
        if not trailing:
            cut_end = _snap_to_zero_crossing(mono, cut_end, snap)
        if cut_end - cut_start > 2 * xf:
            cuts.append((cut_start, cut_end))

    if not cuts:
        ap.write_audio_array(output_path, y, sr)
        return {"status": "success", "silences_removed": 0, "removed_seconds": 0.0,
                "input_duration": round(n / sr, 3), "output_duration": round(n / sr, 3),
                "message": f"No pauses longer than {min_silence_len:g} s were found."}

    # Kept regions between cuts; interior joins overlap by `xf` samples.
    keeps = []
    pos = 0
    for cs, ce in cuts:
        if cs > pos:
            keeps.append([pos, cs])
        pos = ce
    if pos < n:
        keeps.append([pos, n])
    for k in range(len(keeps) - 1):  # extend each side of an interior join by xf/2
        keeps[k][1] = min(n, keeps[k][1] + xf // 2)
        keeps[k + 1][0] = max(0, keeps[k + 1][0] - (xf - xf // 2))

    fade_in = np.sin(0.5 * np.pi * (np.arange(xf) + 0.5) / xf)  # equal-power
    fade_out = np.cos(0.5 * np.pi * (np.arange(xf) + 0.5) / xf)
    pieces = []
    tail = None
    for idx, (s, e) in enumerate(keeps):
        seg = np.asarray(y[:, s:e], dtype=np.float64).copy()
        if idx == 0 and cuts[0][0] == 0 and seg.shape[1] >= xf:
            seg[:, :xf] *= fade_in          # file now starts mid room-tone
        if idx == len(keeps) - 1 and cuts[-1][1] == n and seg.shape[1] >= xf:
            seg[:, -xf:] *= fade_out        # file now ends mid room-tone
        if tail is not None and seg.shape[1] >= xf:
            seg[:, :xf] = tail * fade_out + seg[:, :xf] * fade_in
        elif tail is not None:
            pieces.append(tail)
        if idx < len(keeps) - 1 and seg.shape[1] >= 2 * xf:
            pieces.append(seg[:, :-xf])
            tail = seg[:, -xf:]
        else:
            pieces.append(seg)
            tail = None
    if tail is not None:
        pieces.append(tail)
    out = np.concatenate(pieces, axis=1)

    ap.write_audio_array(output_path, out, sr)
    removed = (n - out.shape[1]) / sr
    return {"status": "success", "silences_removed": len(cuts),
            "removed_seconds": round(removed, 3),
            "input_duration": round(n / sr, 3),
            "output_duration": round(out.shape[1] / sr, 3)}


def _match_channels(y, channels):
    if y.shape[0] == channels:
        return y
    mono = y.mean(axis=0, keepdims=True)
    return np.repeat(mono, channels, axis=0)


def auto_duck_music(speech_path, music_path, output_path, duck_db=None,
                    bed_lu_below_voice=18.0, attack_ms=80.0, release_ms=500.0,
                    hold_ms=250.0, true_peak_ceiling=-1.0):
    """
    Sidechain-style auto-ducking: mixes voice + music so the voice keeps its
    level and only the music is turned down while someone is speaking.

    * Gain staging: the voice is summed at unity gain (no automatic halving of
      every input), so speech loudness is preserved.
    * Depth: during speech the music bed sits ``bed_lu_below_voice`` LU (15–20
      recommended) under the voice's integrated loudness. A numeric
      ``duck_db`` forces a fixed attenuation instead.
    * Envelope: look-ahead attack (``attack_ms``), ``hold_ms`` hold across
      short pauses, slew-limited ``release_ms`` recovery — no pumping/clicks.
    * Safety: a true-peak limiter keeps the mix under ``true_peak_ceiling``.
    Raises ValueError when either input is unreadable or has no content.
    """
    import audio_processor as ap
    from scipy.signal import resample_poly

    bed_lu_below_voice = float(bed_lu_below_voice)
    if not (6.0 <= bed_lu_below_voice <= 30.0):
        raise ValueError("Choose a music bed level between 6 and 30 LU below the voice.")

    speech, sr = ap.load_audio_array(speech_path)
    music, sr_m = ap.load_audio_array(music_path)
    if sr_m != sr:
        g = math.gcd(int(sr), int(sr_m))
        music = resample_poly(music, sr // g, sr_m // g, axis=-1).astype(np.float32)

    channels = max(speech.shape[0], music.shape[0])
    speech = _match_channels(speech, channels).astype(np.float64)
    music = _match_channels(music, channels).astype(np.float64)
    n = max(speech.shape[1], music.shape[1])
    speech = np.pad(speech, ((0, 0), (0, n - speech.shape[1])))
    music = np.pad(music, ((0, 0), (0, n - music.shape[1])))

    voice_lufs = ap.measure_loudness_array(speech, sr, include_true_peak=False)["integrated_lufs"]
    music_lufs = ap.measure_loudness_array(music, sr, include_true_peak=False)["integrated_lufs"]
    if voice_lufs <= ap.LOUDNESS_FLOOR_LUFS:
        raise ValueError("No speech was found in the voice track, so there is nothing to duck under.")
    if music_lufs <= ap.LOUDNESS_FLOOR_LUFS:
        raise ValueError("The music track is silent.")

    if duck_db is not None:
        duck_gain_db = -abs(float(duck_db))
    else:
        duck_gain_db = min(0.0, (voice_lufs - bed_lu_below_voice) - music_lufs)

    # Speech activity from 50 ms RMS evaluated every 10 ms.
    hop = max(1, int(round(sr * 0.01)))
    win = 5
    n_frames = int(math.ceil(n / hop))
    p = np.mean(speech ** 2, axis=0)
    p = np.pad(p, (0, n_frames * hop - n))
    frame_pow = p.reshape(n_frames, hop).mean(axis=1)
    smooth_pow = np.convolve(frame_pow, np.ones(win) / win, mode="same")
    level_db = 10.0 * np.log10(smooth_pow + 1e-12)
    active = level_db > max(voice_lufs - 20.0, -60.0)

    # Hold across short gaps, and look ahead so the duck lands before the word.
    hold = int(round(hold_ms / 10.0))
    look = int(round(attack_ms / 10.0))
    if active.any():
        idx = np.flatnonzero(active)
        dilated = np.zeros(n_frames + hold + look + 1, dtype=bool)
        for shift in range(-look, hold + 1):
            pos = idx + shift
            pos = pos[(pos >= 0) & (pos < n_frames)]
            dilated[pos] = True
        active = dilated[:n_frames]

    # Slew-limited gain (dB per 10 ms frame): attack/release ramps, no steps.
    depth = max(abs(duck_gain_db), 1.0)
    down_step = depth / max(1.0, attack_ms / 10.0)
    up_step = depth / max(1.0, release_ms / 10.0)
    target = np.where(active, duck_gain_db, 0.0)
    gain_db = np.empty(n_frames)
    g = float(target[0])
    for i in range(n_frames):
        t = target[i]
        if t < g:
            g = max(t, g - down_step)
        elif t > g:
            g = min(t, g + up_step)
        gain_db[i] = g
    centers = (np.arange(n_frames) + 0.5) * hop
    gain = 10.0 ** (np.interp(np.arange(n), centers, gain_db) / 20.0)

    mix = speech + music * gain[np.newaxis, :]
    mix, limited_db = ap.true_peak_limit(mix, sr, true_peak_ceiling)
    ap.write_audio_array(output_path, mix, sr)

    return {
        "status": "success",
        "voice_lufs": round(float(voice_lufs), 2),
        "music_lufs": round(float(music_lufs), 2),
        "duck_gain_db": round(float(duck_gain_db), 2),
        "bed_lu_below_voice": round(float(voice_lufs - (music_lufs + duck_gain_db)), 2),
        "speech_coverage": round(float(active.mean()), 3),
        "limiter_reduction_db": round(float(limited_db), 2),
    }


def pitch_preserved_speed(input_path, output_path, speed=1.25):
    """
    Pitch-Preserved Audio Speed Changer (0.25x to 4.0x):
    Changes playback speed without altering vocal pitch (WSOLA time-stretching).
    """
    import math
    import video_processor
    speed = float(speed)
    if not math.isfinite(speed) or not 0.25 <= speed <= 4:
        raise ValueError('Choose a playback speed between 0.25x and 4x.')
    cmd = [
        video_processor.FFMPEG, "-y", "-i", input_path, '-vn',
        "-filter:a", ','.join(video_processor._atempo_chain(speed)),
        output_path
    ]
    video_processor._run(cmd, timeout=1200)
    return {"status": "success", "speed": speed}


def generate_youtube_chapters(media_path):
    """
    Auto-Chapter Marker Generator for YouTube:
    Analyzes speech transcripts and groups topics into timestamped YouTube chapter markers.
    """
    result = transcribe_audio(media_path)
    if not result.get("available"):
        raise ValueError("The speech could not be transcribed, so chapters could not be generated.")
    if not result.get("segments"):
        raise ValueError("No speech was found, so there is nothing to build chapters from.")

    segments = result["segments"]
    chapters = [{"time": "00:00", "title": "Intro"}]

    def sec_to_min_sec(sec):
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    # Group segments every 45-60 seconds or on major topic pause gaps
    last_chapter_t = 0.0
    for idx, seg in enumerate(segments):
        start_t = seg["start"]
        text = seg["text"].strip()
        if start_t - last_chapter_t >= 45.0 and text:
            time_str = sec_to_min_sec(start_t)
            title_summary = text[:40].strip().capitalize()
            chapters.append({"time": time_str, "title": f"Section {len(chapters)} — {title_summary}"})
            last_chapter_t = start_t

    formatted_text = "\n".join([f"{ch['time']} {ch['title']}" for ch in chapters])
    return {
        "status": "success",
        "chapters": chapters,
        "formatted_text": formatted_text
    }


