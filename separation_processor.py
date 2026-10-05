"""
Source separation — vocals / instrumental (karaoke), 4 stems, voice isolation
and song-to-lyrics. 100 % local.

Tiers (best first; the baseline always works and needs no download)
-------------------------------------------------------------------
``studio_fine``  Fine-tuned hybrid-transformer separator bag (4 models, one per
                 source). Chosen automatically only on a CUDA GPU; on CPU only
                 when the caller asks for ``quality="best"``.
``studio``       Hybrid-transformer separator (single model, ~84 MB). GPU when
                 available, CPU otherwise.
``quick``        DSP baseline: stereo centre extraction (inter-channel
                 coherence, a per-bin Wiener estimate of the centre-panned
                 component), harmonic/percussive decomposition and smooth band
                 weighting. Every stem set is a partition of the mix, so the
                 stems always add back to the input exactly.

Internal notes (never shown to users): the neural tiers are the MIT-licensed
"Demucs v4" htdemucs / htdemucs_ft checkpoints by Meta AI. They are only used
when the ``demucs`` package *and* the weights are installed locally in
``models/stems`` (``python download_models.py --stems``); nothing is ever
downloaded implicitly.

Long inputs are processed in overlapping chunks with complementary
(sin²/cos²) crossfades, so memory stays bounded and joins are seamless. Stems
are written as float intermediates, then a single common gain (only when a
stem would clip) keeps all stems consistent, so ``sum(stems) ≈ mix`` still
holds after clipping protection. Every run is verified: reconstruction error
of the full internal stem set, clipping, near-silent stems, and per-stem
BS.1770 loudness.

User-facing text uses capability language only ("Vocal separation",
"Studio stems", "Voice isolation").
"""

import importlib.util
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf

from model_manager import (CapabilityVariant, TIER_BALANCED, TIER_LITE, TIER_MAX,
                           global_model_manager, global_quality_governor, is_resource_error)

_log = logging.getLogger("separation")

CAPABILITY = "audio.separation"
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

MODES = ("vocals", "karaoke", "4stem", "voice")
FORMATS = ("wav", "mp3")
QUALITIES = ("auto", "fast", "best")

MODE_LABELS = {
    "vocals": "Vocal separation",
    "karaoke": "Instrumental (karaoke)",
    "4stem": "4-stem split",
    "voice": "Voice isolation",
}
STEM_LABELS = {
    "vocals": "Vocals",
    "instrumental": "Instrumental",
    "drums": "Drums",
    "bass": "Bass",
    "other": "Other instruments",
    "voice": "Voice",
    "background": "Background",
}

# Every internal set is a full partition of the mix (used for verification).
_INTERNAL_STEMS = {
    "vocals": ("vocals", "instrumental"),
    "karaoke": ("vocals", "instrumental"),
    "4stem": ("vocals", "drums", "bass", "other"),
    "voice": ("voice", "background"),
}
_OUTPUT_STEMS = {
    "vocals": ("vocals", "instrumental"),
    "karaoke": ("instrumental",),
    "4stem": ("vocals", "drums", "bass", "other"),
    "voice": ("voice",),
}

TIERS = {
    # model: internal checkpoint name; ram/gpu: peak estimates in GB for one
    # 60 s chunk (model + activations + chunk buffers), fed to the governor
    # and to the lifecycle manager's ``min_free_ram_gb`` headroom guard.
    # Measured (CPU, 16 cores): the single model adds ~1.0 GB to the process
    # working set on a 60 s stereo chunk; estimates keep ~50 % margin. The
    # 4-model bag keeps all four sets of weights resident (+~0.3 GB).
    "studio_fine": {"label": "Studio stems (fine)", "model": "htdemucs_ft", "ram_gb": 2.0, "gpu_gb": 3.0,
                    "neural": True},
    "studio": {"label": "Studio stems", "model": "htdemucs", "ram_gb": 1.5, "gpu_gb": 2.0, "neural": True},
    "quick": {"label": "Quick separation", "model": None, "ram_gb": 0.5, "gpu_gb": 0.0, "neural": False},
}

QUALITY_NOTES = {
    "studio": ("Separated with the studio separation model. Expect clean stems; a faint trace of "
               "other instruments can remain on very dense mixes."),
    "quick_stereo": ("Quick separation uses the stereo image and the tonal or percussive character of "
                     "the mix. It works best when the vocal is panned to the centre; reverb, backing "
                     "vocals and centre-panned instruments may bleed between stems. Install Studio stems "
                     "for cleaner results."),
    "quick_mono": ("Quick separation on a mono recording can only use the tonal or percussive character "
                   "and frequency range of each part, so expect noticeable bleed between stems. Install "
                   "Studio stems for cleaner results."),
}

MIN_SECONDS = 0.5
DSP_CHUNK_SECONDS = 30.0
NEURAL_CHUNK_SECONDS = 60.0
CHUNK_OVERLAP_SECONDS = 1.0
SILENCE_PEAK = 1e-5                 # -100 dBFS: the input is treated as silent
NEAR_SILENT_REL_DB = -45.0          # stem energy relative to the mix
RECONSTRUCTION_LIMIT_DB = -25.0     # roadmap quality gate (residual vs mix)
WAV_CLIP_LEVEL = 0.9999
WAV_CEILING = 10 ** (-0.1 / 20)     # -0.1 dBFS
MP3_CEILING = 10 ** (-1.0 / 20)     # -1 dBFS leaves room for encoder overshoot
LOUDNESS_MAX_SECONDS = 30 * 60      # longer stems report peak only (bounded RAM)
TRUE_PEAK_MAX_SECONDS = 10 * 60

_UNREADABLE = "This file could not be read as audio. Try exporting it as WAV or MP3 and upload it again."


class SeparationUnavailableError(RuntimeError):
    """A dependency of the requested capability is not available; message is public."""


# ═══════════════════════════════════════════════════════════════════════════
#  Validation and capability discovery
# ═══════════════════════════════════════════════════════════════════════════

def validate_request(mode="vocals", fmt="wav", quality="auto"):
    """Normalise and validate user options; raises ValueError with a public message."""
    mode = str(mode or "vocals").strip().lower()
    fmt = str(fmt or "wav").strip().lower().lstrip(".")
    quality = str(quality or "auto").strip().lower()
    aliases = {"2": "vocals", "2stem": "vocals", "two": "vocals", "4": "4stem", "four": "4stem",
               "stems": "4stem", "instrumental": "karaoke", "isolate": "voice", "voice_only": "voice"}
    mode = aliases.get(mode, mode)
    if mode not in MODES:
        raise ValueError("Choose a separation mode: vocals and instrumental, karaoke, 4 stems or voice only.")
    if fmt not in FORMATS:
        raise ValueError("Choose an output format: WAV or MP3.")
    if quality not in QUALITIES:
        raise ValueError("Choose a quality setting: auto, fast or best.")
    return mode, fmt, quality


def _models_dir():
    return os.environ.get("SEPARATION_MODELS_DIR") or os.path.join(WORKSPACE_DIR, "models", "stems")


def _bag_signatures(name, root=None):
    """Checkpoint signatures listed in the local bag definition, or None."""
    path = os.path.join(root or _models_dir(), f"{name}.yaml")
    if not os.path.isfile(path):
        return None
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as handle:
            bag = yaml.safe_load(handle) or {}
        sigs = [str(s) for s in bag.get("models") or []]
        return sigs or None
    except Exception:
        return None


def _checkpoint_path(root, sig):
    """Prefer safetensors (no pickle); accept checksummed legacy checkpoints."""
    st = os.path.join(root, f"{sig}.safetensors")
    if os.path.isfile(st) and os.path.getsize(st) > 0:
        return st
    try:
        for entry in os.listdir(root):
            if entry.startswith(f"{sig}-") and entry.endswith(".th"):
                return os.path.join(root, entry)
    except OSError:
        pass
    return None


def _neural_ready(name):
    """True only when the separator package AND every weight file are local."""
    if importlib.util.find_spec("demucs") is None:
        return False
    root = _models_dir()
    sigs = _bag_signatures(name, root)
    return bool(sigs) and all(_checkpoint_path(root, s) for s in sigs)


def _cuda_usable():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _best_device():
    return "cuda" if _cuda_usable() else "cpu"


def capability_status():
    """Public description of which separation tiers are installed (no vendor names)."""
    studio = _neural_ready(TIERS["studio"]["model"])
    fine = _neural_ready(TIERS["studio_fine"]["model"])
    tiers = [
        {"id": "studio_fine", "label": TIERS["studio_fine"]["label"], "installed": fine},
        {"id": "studio", "label": TIERS["studio"]["label"], "installed": studio},
        {"id": "quick", "label": TIERS["quick"]["label"], "installed": True},
    ]
    return {
        "studio_stems_installed": bool(studio or fine),
        "tiers": tiers,
        "modes": [{"id": m, "label": MODE_LABELS[m]} for m in MODES],
        "formats": list(FORMATS),
        "install_hint": None if (studio or fine) else
        "Studio stems are an optional download. Run the model downloader with the --stems option.",
    }


def _ladder(quality):
    quick = CapabilityVariant("quick", tier=TIER_LITE, public_label=TIERS["quick"]["label"])
    if quality == "fast":
        return [quick]
    best = quality == "best"
    fine_name, studio_name = TIERS["studio_fine"]["model"], TIERS["studio"]["model"]
    fine = CapabilityVariant(
        "studio_fine",
        tier=TIER_BALANCED if best else TIER_MAX,
        min_ram_gb=TIERS["studio_fine"]["ram_gb"],
        min_cpu_cores=4,
        # Automatic selection only on a usable GPU: on CPU the 4-model bag is
        # ~4x slower than the single model for ~0.3 dB.
        min_gpu_gb=0.0 if best else TIERS["studio_fine"]["gpu_gb"],
        is_available=(lambda: _neural_ready(fine_name)) if best
        else (lambda: _neural_ready(fine_name) and _cuda_usable()),
        public_label=TIERS["studio_fine"]["label"],
    )
    studio = CapabilityVariant(
        "studio",
        tier=TIER_BALANCED,
        min_ram_gb=TIERS["studio"]["ram_gb"],
        min_cpu_cores=4,
        is_available=lambda: _neural_ready(studio_name),
        public_label=TIERS["studio"]["label"],
    )
    return [fine, studio, quick]


# ═══════════════════════════════════════════════════════════════════════════
#  Streaming source reader
# ═══════════════════════════════════════════════════════════════════════════

class _Source:
    """Block reader over the input; non-PCM containers are decoded once to a temp WAV."""

    def __init__(self, src, work_dir):
        if not src or not os.path.isfile(src):
            raise ValueError("The audio file could not be found.")
        path = src
        try:
            info = sf.info(src)
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError
        except Exception:
            path = self._decode(src, work_dir)
        try:
            self._file = sf.SoundFile(path)
        except Exception:
            raise ValueError(_UNREADABLE)
        self.sr = int(self._file.samplerate)
        self.source_channels = int(self._file.channels)
        self.channels = 1 if self.source_channels == 1 else 2
        self.frames = int(self._file.frames)
        if self.frames <= 0:
            self.close()
            raise ValueError(_UNREADABLE)

    @staticmethod
    def _decode(src, work_dir):
        import video_processor
        out = os.path.join(work_dir, "decoded.wav")
        try:
            proc = subprocess.run(
                [video_processor.FFMPEG, "-y", "-v", "error", "-i", src, "-vn", "-map", "0:a:0",
                 "-c:a", "pcm_f32le", "-rf64", "auto", "-f", "wav", out],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600,
            )
        except Exception:
            raise ValueError(_UNREADABLE)
        if proc.returncode != 0 or not os.path.isfile(out) or os.path.getsize(out) <= 44:
            raise ValueError(_UNREADABLE)
        return out

    def read(self, start, stop):
        self._file.seek(start)
        data = self._file.read(stop - start, dtype="float32", always_2d=True).T
        if data.shape[1] < stop - start:
            data = np.pad(data, ((0, 0), (0, stop - start - data.shape[1])))
        if self.source_channels > 2:
            # Surround → stereo: odd channels to the left, even to the right.
            data = np.stack([data[0::2].mean(axis=0), data[1::2].mean(axis=0)])
        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(data, dtype=np.float32)

    def scan(self, block_seconds=60.0):
        """Return (peak, energy) of the whole input with bounded memory."""
        block = max(1, int(block_seconds * self.sr))
        peak, energy = 0.0, 0.0
        for start in range(0, self.frames, block):
            x = self.read(start, min(self.frames, start + block)).astype(np.float64)
            peak = max(peak, float(np.max(np.abs(x))) if x.size else 0.0)
            energy += float(np.sum(x * x))
        return peak, energy

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Baseline (DSP) separator
# ═══════════════════════════════════════════════════════════════════════════

def _stft_params(sr):
    n_fft = 4096 if sr >= 32000 else (2048 if sr >= 16000 else 1024)
    return n_fft, n_fft // 4


def _soft_band(freqs, lo=None, hi=None, slope_oct=0.25):
    """Smooth band weighting on a log-frequency axis (0.5 at the corner)."""
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1.0)
    g = np.ones_like(f)
    k = 4.0 / slope_oct
    if lo:
        g *= 1.0 / (1.0 + np.exp(-k * np.log2(f / lo)))
    if hi:
        g *= 1.0 / (1.0 + np.exp(k * np.log2(f / hi)))
    return g[:, np.newaxis]


def _dsp_separate(x, sr, mode):
    """Mask-based separation of one chunk; stems partition ``x`` exactly."""
    import librosa
    from scipy.ndimage import median_filter, uniform_filter

    ch, n = x.shape
    n_fft, hop = _stft_params(sr)
    if n < n_fft:
        n_fft = max(256, 1 << int(math.log2(max(n, 256))))
        hop = n_fft // 4
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop)                 # [ch, F, T]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = np.mean(np.abs(X) ** 2, axis=0)
    mag = np.sqrt(power)

    # Harmonic / percussive soft masks (H + P = 1): ~0.5 s horizontal and
    # ~500 Hz vertical median kernels.
    k_time = max(5, int(round(0.5 * sr / hop)) | 1)
    k_freq = max(5, int(round(500.0 / (sr / n_fft))) | 1)
    H, P = librosa.decompose.hpss(mag, kernel_size=(k_time, k_freq), power=2.0, mask=True)

    centre = None
    if ch == 2:
        # Inter-channel coherence: for a centre source C plus decorrelated
        # side content S, E[Re(L·R*)] / E[(|L|²+|R|²)/2] = |C|²/(|C|²+|S|²),
        # i.e. a Wiener gain for the centre-panned component.
        left, right = X[0], X[1]
        cross = uniform_filter(np.real(left * np.conj(right)), size=(3, 7))
        energy = uniform_filter(0.5 * (np.abs(left) ** 2 + np.abs(right) ** 2), size=(3, 7))
        centre = np.clip(cross / (energy + 1e-12), 0.0, 1.0)

    def smooth(mask):
        return np.clip(median_filter(mask, size=(1, 3)), 0.0, 1.0)

    def apply(mask):
        return librosa.istft(X * mask[np.newaxis].astype(np.float32), hop_length=hop, length=n).astype(np.float32)

    if mode == "voice":
        band = _soft_band(freqs, lo=100.0, hi=8000.0)
        spatial = 0.15 + 0.85 * centre ** 1.5 if centre is not None else 1.0
        floor = np.percentile(power, 20, axis=1, keepdims=True)       # stationary noise floor
        snr_gain = power / (power + 2.0 * floor + 1e-12)
        tonal = 0.35 + 0.65 * H                                        # keep some consonants
        voice = apply(smooth(band * spatial * snr_gain * tonal))
        return {"voice": voice, "background": x - voice}

    band = _soft_band(freqs, lo=120.0, hi=12000.0)
    if centre is not None:
        m_vocal = smooth(centre ** 1.5 * band * H)
    else:
        m_vocal = smooth(band * H)
    vocals = apply(m_vocal)
    if mode in ("vocals", "karaoke"):
        return {"vocals": vocals, "instrumental": x - vocals}

    rest = 1.0 - m_vocal
    m_drums = rest * P
    m_bass = rest * H * _soft_band(freqs, hi=250.0)
    drums = apply(m_drums)
    bass = apply(m_bass)
    return {"vocals": vocals, "drums": drums, "bass": bass, "other": x - vocals - drums - bass}


# ═══════════════════════════════════════════════════════════════════════════
#  Neural separator (optional, local weights only)
# ═══════════════════════════════════════════════════════════════════════════

def _load_neural_model(name):
    """Load a separator bag from the local models folder; never touches the network."""
    root = _models_dir()
    sigs = _bag_signatures(name, root)
    if not sigs:
        raise SeparationUnavailableError("Studio stems are not installed.")
    import yaml
    from demucs.apply import BagOfModels

    with open(os.path.join(root, f"{name}.yaml"), "r", encoding="utf-8") as handle:
        bag = yaml.safe_load(handle) or {}
    models = []
    for sig in sigs:
        path = _checkpoint_path(root, sig)
        if path is None:
            raise SeparationUnavailableError("Studio stems are not installed.")
        if path.endswith(".safetensors"):
            from demucs.hf import load_safetensors_model
            models.append(load_safetensors_model(path))
        else:
            from demucs.repo import check_checksum
            from demucs.states import load_model
            from pathlib import Path
            check_checksum(Path(path), os.path.basename(path)[:-3].rsplit("-", 1)[1])
            models.append(load_model(path))
    model = BagOfModels(models, bag.get("weights"), bag.get("segment"))
    model.eval()
    return model


def _unload_neural_model(model):
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _neural_infer(model, x, device):
    """Run the separator on one stereo chunk at the model rate → {source: [2, n]}."""
    import torch
    from demucs.apply import apply_model

    wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    def run(fp16):
        with torch.inference_mode():
            if fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = apply_model(model, wav[None], device=device, shifts=0, split=True,
                                      overlap=0.25, progress=False, num_workers=0)
            else:
                out = apply_model(model, wav[None], device=device, shifts=0, split=True,
                                  overlap=0.25, progress=False, num_workers=0)
        return out[0].float().cpu()

    on_gpu = str(device).startswith("cuda")
    fp16 = on_gpu and os.environ.get("SEPARATION_GPU_FP16", "1") != "0"
    out = run(fp16)
    if fp16 and not torch.isfinite(out).all():
        # Half precision overflowed on this material: redo in full precision.
        out = run(False)
    out = out * std + mean
    return {name: out[i].numpy() for i, name in enumerate(model.sources)}


def _resample(y, sr_from, sr_to):
    if sr_from == sr_to:
        return y
    from scipy.signal import resample_poly
    g = math.gcd(int(sr_from), int(sr_to))
    return resample_poly(y, int(sr_to) // g, int(sr_from) // g, axis=-1).astype(np.float32)


def _fit_length(y, n):
    if y.shape[-1] >= n:
        return y[..., :n]
    return np.pad(y, ((0, 0), (0, n - y.shape[-1])))


def _neural_fn(model, device, mode):
    model_sr = int(getattr(model, "samplerate", 44100))

    def separate_chunk(x, sr):
        ch, n = x.shape
        stereo = x if ch == 2 else np.repeat(x[:1], 2, axis=0)
        est = _neural_infer(model, _resample(stereo, sr, model_sr), device)
        out = {}
        for name, y in est.items():
            y = _fit_length(_resample(np.asarray(y, dtype=np.float32), model_sr, sr), n)
            out[name] = y.mean(axis=0, keepdims=True) if ch == 1 else y
        # Two-stem modes: accompaniment = mix - vocals, so the pair always adds
        # back to the input exactly (the model's own sources leave a ~-25 dB
        # residual, which would otherwise fail the reconstruction check).
        accompaniment = x - out["vocals"]
        if mode in ("vocals", "karaoke"):
            return {"vocals": out["vocals"], "instrumental": accompaniment}
        if mode == "voice":
            return {"voice": out["vocals"], "background": accompaniment}
        missing = [k for k in ("vocals", "drums", "bass", "other") if k not in out]
        if missing:
            raise RuntimeError("Separator returned an unexpected stem layout.")
        return {k: out[k] for k in ("vocals", "drums", "bass", "other")}

    return separate_chunk


# ═══════════════════════════════════════════════════════════════════════════
#  Chunked overlap-add pipeline
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_starts(n, chunk, overlap):
    if n <= chunk:
        return [0]
    step = chunk - overlap
    starts, s = [], 0
    while True:
        starts.append(s)
        if s + chunk >= n:
            return starts
        s += step


def _process(source, separate_chunk, names, work_dir, chunk_seconds, report):
    """Separate the whole source chunk by chunk into float temp files."""
    sr, n, ch = source.sr, source.frames, source.channels
    chunk = max(int(chunk_seconds * sr), 2048)
    overlap = int(min(CHUNK_OVERLAP_SECONDS, chunk_seconds * 0.25) * sr) if n > chunk else 0
    starts = _chunk_starts(n, chunk, overlap)
    big = n * ch * 4 > 3.5e9
    paths, writers, pending = {}, {}, {}
    peaks = {k: 0.0 for k in names}
    energy = {k: 0.0 for k in names}
    residual_energy = 0.0
    if overlap:
        ramp = np.sin(0.5 * np.pi * (np.arange(overlap) + 0.5) / overlap) ** 2
        fade_in, fade_out = ramp.astype(np.float32), (1.0 - ramp).astype(np.float32)
    try:
        for name in names:
            paths[name] = os.path.join(work_dir, f"{name}.wav")
            writers[name] = sf.SoundFile(paths[name], "w", samplerate=sr, channels=ch,
                                         format="RF64" if big else "WAV", subtype="FLOAT")
        for k, start in enumerate(starts):
            stop = min(n, start + chunk)
            last = k == len(starts) - 1
            x = source.read(start, stop)
            stems = separate_chunk(x, sr)
            length = stop - start
            weight = np.ones(length, dtype=np.float32)
            if overlap and k > 0:
                weight[:overlap] = fade_in
            if overlap and not last:
                weight[length - overlap:] = fade_out
            residual = x.astype(np.float64)
            for name in names:
                y = np.asarray(stems[name], dtype=np.float32)
                if y.shape != x.shape:
                    raise RuntimeError("Separator returned a stem of the wrong shape.")
                if not np.all(np.isfinite(y)):
                    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                residual = residual - y
                stems[name] = y
            stems["__residual__"] = residual.astype(np.float32)
            for name, y in stems.items():
                y = y * weight
                if pending.get(name) is not None:
                    y[:, :overlap] += pending[name]
                if not last and overlap:
                    final, pending[name] = y[:, :length - overlap], y[:, length - overlap:].copy()
                else:
                    final, pending[name] = y, None
                if name == "__residual__":
                    residual_energy += float(np.sum(final.astype(np.float64) ** 2))
                    continue
                if final.size:
                    peaks[name] = max(peaks[name], float(np.max(np.abs(final))))
                    energy[name] += float(np.sum(final.astype(np.float64) ** 2))
                    writers[name].write(final.T)
            report(k + 1, len(starts))
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
    return {"paths": paths, "peaks": peaks, "energy": energy,
            "residual_energy": residual_energy, "chunks": len(starts)}


# ═══════════════════════════════════════════════════════════════════════════
#  Output: clipping protection, encoding, verification
# ═══════════════════════════════════════════════════════════════════════════

def _db(ratio, floor=-120.0):
    return round(float(max(floor, 10.0 * math.log10(ratio))) if ratio > 0 else floor, 2)


def _write_output(temp_path, out_path, fmt, gain, sr, frames):
    clipped = 0
    if fmt == "wav":
        block = max(1, sr * 30)
        with sf.SoundFile(temp_path) as src, \
                sf.SoundFile(out_path, "w", samplerate=sr, channels=src.channels,
                             format="RF64" if frames * src.channels * 2 > 3.5e9 else "WAV",
                             subtype="PCM_16") as dst:
            while True:
                data = src.read(block, dtype="float32", always_2d=True)
                if not len(data):
                    break
                data = data * np.float32(gain)
                clipped += int(np.count_nonzero(np.abs(data) > 1.0))
                dst.write(np.clip(data, -1.0, 1.0))
        return clipped
    import video_processor
    volume = [] if gain >= 0.999999 else ["-af", f"volume={gain:.8f}"]
    proc = subprocess.run(
        [video_processor.FFMPEG, "-y", "-v", "error", "-i", temp_path, *volume,
         "-c:a", "libmp3lame", "-b:a", "320k", out_path],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600,
    )
    if proc.returncode != 0 or not os.path.isfile(out_path):
        raise RuntimeError("The separated audio could not be encoded.")
    return clipped


def _stem_loudness(temp_path, gain, sr, duration):
    if duration > LOUDNESS_MAX_SECONDS:
        return None
    import audio_processor
    data, _ = sf.read(temp_path, dtype="float32", always_2d=True)
    m = audio_processor.measure_loudness_array(data.T * np.float32(gain), sr,
                                               include_true_peak=duration <= TRUE_PEAK_MAX_SECONDS)
    keys = ("integrated_lufs", "true_peak_dbtp", "sample_peak_dbfs", "loudness_range_lu")
    return {k: m[k] for k in keys if k in m}


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def separate(src, out_dir, mode="vocals", quality="auto", fmt="wav", progress=None, chunk_seconds=None):
    """
    Separate ``src`` into stems written to ``out_dir`` as ``<stem>.<fmt>``.

    mode:    'vocals' (vocals + instrumental), 'karaoke' (instrumental only),
             '4stem' (vocals / drums / bass / other), 'voice' (voice isolation)
    quality: 'auto' (best tier the hardware can sustain), 'fast' (quick DSP
             only), 'best' (allow the fine tier even on CPU)
    progress: optional ``callable(fraction 0..1, stage)``; monotonic.

    Returns a report: stems → paths, quality tier (public label), duration,
    per-stem loudness, verification checks, warnings and timing.
    Raises ValueError (public message) for unusable input or options.
    """
    mode, fmt, quality = validate_request(mode, fmt, quality)
    started = time.perf_counter()
    last = [0.0]

    def emit(fraction, stage):
        fraction = max(last[0], min(1.0, float(fraction)))
        last[0] = fraction
        if progress is not None:
            try:
                progress(fraction, stage)
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=".separation-", dir=out_dir)
    source = None
    try:
        emit(0.0, "Reading audio")
        source = _Source(src, work)
        duration = source.frames / float(source.sr)
        if duration < MIN_SECONDS:
            raise ValueError("This recording is too short to separate (under half a second).")
        peak, mix_energy = source.scan()
        if peak < SILENCE_PEAK:
            raise ValueError("The recording is silent, so there is nothing to separate.")
        emit(0.03, "Analysing")

        names = _INTERNAL_STEMS[mode]
        ladder = _ladder(quality)
        attempts = []

        def report_chunks(done, total):
            emit(0.05 + 0.85 * done / total, "Separating")

        def operate(variant):
            attempts.append(variant.variant_id)
            attempt_dir = tempfile.mkdtemp(dir=work)
            if variant.variant_id == "quick":
                fn = lambda x, sr: _dsp_separate(x, sr, mode)  # noqa: E731
                return _process(source, fn, names, attempt_dir, chunk_seconds or DSP_CHUNK_SECONDS,
                                report_chunks)
            return _run_neural(variant, source, mode, names, attempt_dir,
                               chunk_seconds or NEURAL_CHUNK_SECONDS, report_chunks, work)

        result, variant = global_quality_governor.run(CAPABILITY, ladder, operate)
        tier = variant.variant_id
        fallback = len(attempts) > 1

        emit(0.92, "Saving stems")
        outputs = _OUTPUT_STEMS[mode]
        stem_peak = max(result["peaks"][s] for s in outputs)
        ceiling, clip_level = (WAV_CEILING, WAV_CLIP_LEVEL) if fmt == "wav" else (MP3_CEILING, MP3_CEILING)
        gain = 1.0 if stem_peak <= clip_level else ceiling / stem_peak
        stems, loudness, clipped, silent = {}, {}, 0, []
        for name in outputs:
            out_path = os.path.join(out_dir, f"{name}.{fmt}")
            clipped += _write_output(result["paths"][name], out_path, fmt, gain, source.sr, source.frames)
            stems[name] = out_path
            loud = _stem_loudness(result["paths"][name], gain, source.sr, duration)
            if loud is None:
                loud = {"sample_peak_dbfs": round(20 * math.log10(max(result["peaks"][name] * gain, 1e-6)), 2)}
            loudness[name] = loud
            if _db(result["energy"][name] / mix_energy) < NEAR_SILENT_REL_DB:
                silent.append(name)

        recon_db = _db(result["residual_energy"] / mix_energy)
        warnings = []
        if source.channels == 1:
            warnings.append("This recording is mono, so separation cannot use the stereo image.")
        if source.source_channels > 2:
            warnings.append("Surround audio was folded down to stereo before separation.")
        if fallback:
            warnings.append("Studio stems could not run within the available memory, so a lighter "
                            "separation (Quick separation) was used instead.")
        for name in silent:
            label = STEM_LABELS[name]
            warnings.append(f"The {label.lower()} stem is nearly silent; this recording may not contain much "
                            f"{label.lower()}.")
        if gain < 1.0:
            warnings.append(f"All stems were lowered by {abs(20 * math.log10(gain)):.1f} dB together to prevent "
                            "clipping; they still add back up to the original balance.")
        passed = recon_db <= RECONSTRUCTION_LIMIT_DB and clipped == 0
        if recon_db > RECONSTRUCTION_LIMIT_DB:
            warnings.append("The stems do not add back up to the original as closely as expected.")

        if TIERS[tier]["neural"]:
            note = QUALITY_NOTES["studio"]
        else:
            note = QUALITY_NOTES["quick_mono" if source.channels == 1 else "quick_stereo"]
        elapsed = time.perf_counter() - started
        emit(1.0, "Done")
        return {
            "status": "success",
            "mode": mode,
            "mode_label": MODE_LABELS[mode],
            "format": fmt,
            "stems": stems,
            "stem_labels": {k: STEM_LABELS[k] for k in stems},
            "quality": {"id": tier, "label": TIERS[tier]["label"], "neural": TIERS[tier]["neural"],
                        "fallback": fallback},
            "quality_note": note,
            "duration": round(duration, 3),
            "sample_rate": source.sr,
            "channels": source.channels,
            "source_channels": source.source_channels,
            "chunks": result["chunks"],
            "loudness": loudness,
            "checks": {
                "reconstruction_error_db": recon_db,
                "clipped": clipped > 0,
                "gain_db": round(20 * math.log10(gain), 2),
                "silent_stems": silent,
                "passed": passed,
            },
            "warnings": warnings,
            "elapsed_sec": round(elapsed, 3),
            "realtime_factor": round(elapsed / duration, 3),
        }
    finally:
        if source is not None:
            source.close()
        shutil.rmtree(work, ignore_errors=True)


def _run_neural(variant, source, mode, names, attempt_dir, chunk_seconds, report_chunks, work):
    """One neural attempt inside a lifecycle session; GPU OOM retries on the CPU."""
    spec = TIERS[variant.variant_id]
    model_name = spec["model"]

    def _load():
        return _load_neural_model(model_name), {"capability": "stem_separation", "tier": variant.variant_id}

    with global_model_manager.session(f"stem_separation_{variant.variant_id}", _load, _unload_neural_model,
                                      min_free_ram_gb=spec["ram_gb"]) as (model, _meta):
        device = _best_device()
        try:
            return _process(source, _neural_fn(model, device, mode), names, attempt_dir, chunk_seconds,
                            report_chunks)
        except Exception as error:
            if not (str(device).startswith("cuda") and is_resource_error(error)):
                raise
            _log.warning("Studio stems ran out of graphics memory; continuing on the processor.")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            retry_dir = tempfile.mkdtemp(dir=work)
            return _process(source, _neural_fn(model, "cpu", mode), names, retry_dir, chunk_seconds,
                            report_chunks)


def isolate_voice(src, out, quality="auto", progress=None):
    """Isolate the voice from noise or a music bed into ``out`` (format from its extension)."""
    ext = os.path.splitext(out)[1].lstrip(".").lower() or "wav"
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix=".voice-", dir=out_dir)
    try:
        report = separate(src, work, mode="voice", quality=quality, fmt=ext, progress=progress)
        shutil.move(report["stems"]["voice"], out)
        report["stems"] = {"voice": out}
        return report
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ── Lyrics ──────────────────────────────────────────────────────────────────

MAX_LINE_CHARS = 42
MAX_LINE_SECONDS = 7.0


def _timestamp(seconds, sep):
    ms = int(round(max(0.0, float(seconds)) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(lines):
    blocks = []
    for i, line in enumerate(lines, 1):
        blocks.append(f"{i}\n{_timestamp(line['start'], ',')} --> {_timestamp(line['end'], ',')}\n"
                      f"{line['text']}\n")
    return "\n".join(blocks)


def to_vtt(lines):
    cues = [f"{_timestamp(l['start'], '.')} --> {_timestamp(l['end'], '.')}\n{l['text']}\n" for l in lines]
    return "WEBVTT\n\n" + "\n".join(cues)


def _lines_from_segments(segments):
    """Readable lyric lines: split long segments at word boundaries when word timing exists."""
    lines = []
    for seg in segments or []:
        text = re.sub(r"\s+", " ", str(seg.get("text", ""))).strip()
        if not text:
            continue
        words = [w for w in seg.get("words") or [] if str(w.get("word", "")).strip()]
        start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        if not words or (len(text) <= MAX_LINE_CHARS and end - start <= MAX_LINE_SECONDS):
            lines.append({"start": round(start, 3), "end": round(max(end, start), 3), "text": text})
            continue
        current = []
        for w in words:
            candidate = " ".join(x["word"].strip() for x in current + [w])
            too_long = len(candidate) > MAX_LINE_CHARS or (
                current and float(w["end"]) - float(current[0]["start"]) > MAX_LINE_SECONDS)
            if current and too_long:
                lines.append({"start": round(float(current[0]["start"]), 3),
                              "end": round(float(current[-1]["end"]), 3),
                              "text": " ".join(x["word"].strip() for x in current)})
                current = []
            current.append(w)
        if current:
            lines.append({"start": round(float(current[0]["start"]), 3),
                          "end": round(float(current[-1]["end"]), 3),
                          "text": " ".join(x["word"].strip() for x in current)})
    return lines


def lyrics(src, quality="auto", transcribe=None, progress=None):
    """
    Song → lyrics: isolate the vocals, then run the existing speech-to-text
    capability on the vocal stem. Returns text, timed lines, SRT and VTT.
    """
    if transcribe is None:
        import ai_processor
        transcribe = ai_processor.transcribe_audio

    def emit(fraction, stage):
        if progress is not None:
            try:
                progress(fraction, stage)
            except Exception:
                pass

    work = tempfile.mkdtemp(prefix="lyrics-")
    try:
        report = separate(src, work, mode="vocals", quality=quality, fmt="wav",
                          progress=lambda f, s: emit(0.6 * f, s))
        emit(0.65, "Transcribing lyrics")
        try:
            result = transcribe(report["stems"]["vocals"])
        except Exception:
            _log.warning("Lyric transcription failed.")
            result = {"available": False}
        if not result or not result.get("available"):
            raise SeparationUnavailableError(
                "Speech to text is not available on this machine, so the lyrics could not be transcribed.")
        lines = _lines_from_segments(result.get("segments"))
        text = re.sub(r"\s+", " ", str(result.get("full_text") or " ".join(l["text"] for l in lines))).strip()
        warnings = list(report["warnings"])
        if not lines:
            warnings.append("No sung words were recognised in this recording.")
        emit(1.0, "Done")
        return {
            "status": "success",
            "text": text,
            "lines": lines,
            "srt": to_srt(lines),
            "vtt": to_vtt(lines),
            "language": result.get("language"),
            "language_confidence": result.get("language_confidence"),
            "vocals_isolated": True,
            "quality": report["quality"],
            "quality_note": report["quality_note"],
            "duration": report["duration"],
            "warnings": warnings,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
