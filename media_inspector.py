"""
Media Inspector — fast, bounded perception of a media file's condition.

The copilot uses these measurements to decide *how* to fulfil a request
(e.g. clean a noisy recording before transcribing it) instead of only
following the literal words.  Everything here is:

  * bounded   — audio is analysed on a capped excerpt, images on a capped
                centre crop, video on one small frame plus its soundtrack;
  * cached    — keyed by (real path, size, mtime) so repeated planning on the
                same upload costs nothing;
  * safe      — any failure returns ``None`` / partial data, never raises into
                the planner.

All public text produced here uses capability language only.
"""

import os
import math
import logging
import subprocess
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image

import video_processor

_log = logging.getLogger("agent_processor")

ANALYSIS_SAMPLE_RATE = 16000
MAX_AUDIO_EXCERPT_SEC = 30.0
MAX_IMAGE_ANALYSIS_EDGE = 512
DECODE_TIMEOUT_SEC = 20
_CACHE_LIMIT = 48

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.aiff'}

# Condition thresholds (dB values are dBFS unless stated otherwise).
CLEAN_BACKGROUND_DB = -55.0      # background below this is effectively silence
CLEAN_GAP_DB = 32.0              # foreground this far above background is clean
NOISE_FLATNESS = 0.30            # broadband (hiss-like) background above this flatness
HUM_MAX_HZ = 180.0               # tonal background below this is mains hum, not music
QUIET_PEAK_DB = -20.0
QUIET_ACTIVE_RMS_DB = -38.0
SILENT_PEAK_DB = -70.0
BLUR_LAPLACIAN_VAR = 60.0
NOISY_IMAGE_SIGMA = 6.0
DARK_FRAME_LUMA = 40.0

_cache = OrderedDict()
_cache_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────

def perceive(path, media_type=None):
    """Return a condition report for ``path`` or ``None`` when it cannot be read."""
    try:
        if not path or not os.path.isfile(path):
            return None
        stat = os.stat(path)
        key = (os.path.realpath(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return None
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    try:
        report = _perceive_uncached(path, media_type)
    except Exception as err:  # perception must never break planning
        _log.warning("Media condition analysis unavailable (%s).", type(err).__name__)
        report = None
    with _cache_lock:
        _cache[key] = report
        while len(_cache) > _CACHE_LIMIT:
            _cache.popitem(last=False)
    return report


def clear_cache():
    with _cache_lock:
        _cache.clear()


def audio_peak_db(path):
    """Peak level of the first excerpt of ``path`` in dBFS (``None`` if unreadable)."""
    samples = _decode_audio_excerpt(path)
    if samples is None:
        return None
    if not samples.size:
        return -120.0
    peak = float(np.max(np.abs(samples)))
    return 20 * math.log10(peak) if peak > 0 else -120.0


def describe(report):
    """Short capability-language summary lines for prompts and reasoning traces."""
    if not report:
        return []
    lines = []
    audio = report.get('audio')
    if audio:
        if audio.get('silent'):
            lines.append('Soundtrack is silent.')
        else:
            background = audio.get('background')
            if background == 'clean':
                lines.append('Clean recording with a quiet background.')
            elif background == 'music':
                lines.append(f"Background music detected about {audio['speech_to_background_db']:.0f} dB under the foreground.")
            elif background in ('noise', 'hum'):
                kind = 'Steady hum' if background == 'hum' else 'Background noise'
                lines.append(f"{kind} detected (signal-to-noise about {audio['snr_db']:.0f} dB).")
            if audio.get('quiet'):
                lines.append(f"Very quiet level (peak {audio['peak_db']:.0f} dBFS).")
            if audio.get('clipping'):
                lines.append('Some clipping detected.')
            lines.append(f"Loudness about {audio['loudness_lufs']:.1f} LUFS, peak {audio['peak_db']:.1f} dBFS.")
            if audio.get('speech_likely'):
                lines.append('Speech-like activity present.')
    image = report.get('image')
    if image:
        lines.append(f"Image {image['width']}x{image['height']} pixels.")
        if image.get('blurry'):
            lines.append('Image looks soft or out of focus.')
        if image.get('noisy'):
            lines.append('Visible grain or compression noise.')
        if image.get('has_alpha'):
            lines.append('Image already has transparency.')
    video = report.get('video')
    if video:
        lines.append(f"Video {video.get('width', 0)}x{video.get('height', 0)}, {video.get('duration', 0):.1f}s, "
                     f"{'with' if video.get('has_audio') else 'without'} a soundtrack.")
        if video.get('dark'):
            lines.append('Footage is very dark.')
    return lines


# ─────────────────────────────────────────────────────────────────────────
#  Internals
# ─────────────────────────────────────────────────────────────────────────

def _perceive_uncached(path, media_type):
    extension = os.path.splitext(path)[1].lower()
    kind = (media_type or '').lower() if isinstance(media_type, str) else ''
    if kind not in ('image', 'audio', 'video'):
        kind = 'image' if extension in IMAGE_EXTENSIONS else 'audio' if extension in AUDIO_EXTENSIONS else 'video'
    if kind == 'image' or extension in IMAGE_EXTENSIONS:
        return {'type': 'image', 'image': analyze_image(path)}
    if kind == 'audio' or extension in AUDIO_EXTENSIONS:
        audio = analyze_audio(path)
        return {'type': 'audio', 'audio': audio} if audio else None
    info = video_processor.probe_media(path)
    report = {'type': 'video' if info.get('has_video') else 'audio', 'video': {
        'width': info.get('width', 0), 'height': info.get('height', 0),
        'duration': info.get('duration', 0.0), 'has_audio': bool(info.get('has_audio')),
        'has_video': bool(info.get('has_video')),
    }}
    if info.get('has_video'):
        luma = _frame_luma(path, min(1.0, max(0.0, info.get('duration', 0) / 2)))
        if luma is not None:
            report['video']['mean_luma'] = round(luma, 1)
            report['video']['dark'] = luma < DARK_FRAME_LUMA
    if info.get('has_audio'):
        report['audio'] = analyze_audio(path)
    return report


def _decode_audio_excerpt(path, max_seconds=MAX_AUDIO_EXCERPT_SEC):
    cmd = [video_processor.FFMPEG, '-v', 'error', '-nostdin', '-t', f'{max_seconds:.3f}', '-i', path,
           '-vn', '-ac', '1', '-ar', str(ANALYSIS_SAMPLE_RATE), '-f', 's16le', '-']
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=DECODE_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    data = np.frombuffer(proc.stdout, dtype='<i2')
    return data.astype(np.float32) / 32768.0


def _studio_loudness(path, max_seconds=MAX_AUDIO_EXCERPT_SEC):
    """Integrated loudness / true peak of a bounded excerpt using the studio's loudness meter."""
    import tempfile
    import audio_processor
    folder = tempfile.mkdtemp(prefix='media-inspect-')
    excerpt = os.path.join(folder, 'excerpt.wav')
    try:
        cmd = [video_processor.FFMPEG, '-v', 'error', '-nostdin', '-y', '-t', f'{max_seconds:.3f}', '-i', path,
               '-vn', '-c:a', 'pcm_s16le', excerpt]
        proc = subprocess.run(cmd, capture_output=True, timeout=DECODE_TIMEOUT_SEC)
        if proc.returncode != 0 or not os.path.isfile(excerpt):
            return None
        metrics = audio_processor.calculate_lufs(excerpt)
        lufs = metrics.get('integrated_lufs', metrics.get('lufs'))
        if lufs is None or not math.isfinite(float(lufs)):
            return None
        result = {'loudness_lufs': round(float(lufs), 2)}
        true_peak = metrics.get('true_peak_dbtp')
        if true_peak is not None and math.isfinite(float(true_peak)):
            result['true_peak_dbtp'] = round(float(true_peak), 2)
        return result
    except Exception:
        return None
    finally:
        try:
            if os.path.exists(excerpt):
                os.remove(excerpt)
            os.rmdir(folder)
        except OSError:
            pass


def _frames(samples, frame, hop):
    if samples.size < frame:
        samples = np.pad(samples, (0, frame - samples.size))
    count = 1 + (samples.size - frame) // hop
    strides = (samples.strides[0] * hop, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=(count, frame), strides=strides)


def analyze_audio(path, samples=None):
    """Measure loudness, background type, signal-to-noise and speech presence."""
    measure_loudness = samples is None
    if samples is None:
        samples = _decode_audio_excerpt(path)
    if samples is None:
        return None
    sr = ANALYSIS_SAMPLE_RATE
    if samples.size == 0:
        return {'silent': True, 'peak_db': -120.0, 'loudness_lufs': -120.0, 'background': 'clean',
                'snr_db': 0.0, 'speech_to_background_db': 0.0, 'noisy': False, 'music': False,
                'quiet': False, 'clipping': False, 'speech_likely': False, 'analyzed_seconds': 0.0}

    peak = float(np.max(np.abs(samples)))
    peak_db = 20 * math.log10(peak) if peak > 0 else -120.0
    total_rms = float(np.sqrt(np.mean(samples ** 2)))
    loudness = (20 * math.log10(total_rms) - 0.6) if total_rms > 0 else -120.0
    clip_ratio = float(np.mean(np.abs(samples) >= 0.995))

    frame, hop = 512, 256  # 32 ms frames, 16 ms hop
    frames = _frames(samples, frame, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    frame_db = 20 * np.log10(np.maximum(rms, 1e-6))
    silent = peak_db < SILENT_PEAK_DB

    background_db = float(np.percentile(frame_db, 10))
    foreground_db = float(np.percentile(frame_db, 95))
    gap = foreground_db - background_db
    active = frame_db > max(-60.0, foreground_db - 30)
    active_rms_db = float(20 * np.log10(max(np.sqrt(np.mean(rms[active] ** 2)) if active.any() else 1e-6, 1e-6)))

    # Spectral character of the quietest frames (what sits "under" the content).
    window = np.hanning(frame).astype(np.float32)
    quiet_idx = np.where(frame_db <= np.percentile(frame_db, 20))[0]
    if quiet_idx.size > 300:
        quiet_idx = quiet_idx[np.linspace(0, quiet_idx.size - 1, 300).astype(int)]
    spectrum = np.abs(np.fft.rfft(frames[quiet_idx] * window, axis=1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    band = (freqs >= 100) & (freqs <= 7000)
    band_power = spectrum[:, band]
    flatness = float(np.median(np.exp(np.mean(np.log(band_power), axis=1)) / np.mean(band_power, axis=1)))
    mean_spectrum = np.mean(spectrum, axis=0)
    audible = freqs >= 40
    dominant_hz = float(freqs[audible][int(np.argmax(mean_spectrum[audible]))])

    # Syllabic modulation of the energy envelope: speech pulses at roughly 2-8 Hz.
    envelope = rms.astype(np.float64)
    depth = float(np.std(envelope) / (np.mean(envelope) + 1e-9))
    env = envelope - envelope.mean()
    mod = np.abs(np.fft.rfft(env)) ** 2
    mod_freqs = np.fft.rfftfreq(env.size, hop / sr)
    in_band = mod[(mod_freqs >= 2) & (mod_freqs <= 8)].sum()
    broad = mod[(mod_freqs >= 0.5) & (mod_freqs <= 20)].sum() + 1e-12
    syllabic_ratio = float(in_band / broad)
    speech_likely = (not silent) and depth > 0.25 and syllabic_ratio > 0.25

    if silent or background_db < CLEAN_BACKGROUND_DB or gap > CLEAN_GAP_DB:
        background = 'clean'
    elif flatness > NOISE_FLATNESS:
        background = 'noise'
    elif dominant_hz < HUM_MAX_HZ:
        background = 'hum'
    else:
        background = 'music'

    studio = _studio_loudness(path) if measure_loudness and not silent else None
    if studio:
        loudness = studio['loudness_lufs']
    return {
        'silent': silent,
        'peak_db': round(peak_db, 2),
        'true_peak_dbtp': (studio or {}).get('true_peak_dbtp', round(peak_db, 2)),
        'loudness_lufs': round(loudness, 2),
        'active_rms_db': round(active_rms_db, 2),
        'background_db': round(background_db, 2),
        'snr_db': round(gap, 2),
        'speech_to_background_db': round(gap, 2),
        'background': background,
        'background_flatness': round(flatness, 3),
        'background_dominant_hz': round(dominant_hz, 1),
        'modulation_depth': round(depth, 3),
        'syllabic_ratio': round(syllabic_ratio, 3),
        'speech_likely': bool(speech_likely),
        'noisy': background in ('noise', 'hum'),
        'music': background == 'music',
        'quiet': (not silent) and (peak_db < QUIET_PEAK_DB or active_rms_db < QUIET_ACTIVE_RMS_DB),
        'clipping': clip_ratio > 0.001,
        'analyzed_seconds': round(samples.size / sr, 2),
    }


def analyze_image(path):
    """Resolution, transparency, focus (Laplacian variance) and grain estimate."""
    with Image.open(path) as image:
        width, height = image.size
        fmt = (image.format or '').upper()
        has_alpha = False
        alpha_coverage = 0.0
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            alpha = image.convert('RGBA').getchannel('A')
            has_alpha = alpha.getextrema()[0] < 255
            if has_alpha:
                thumb = np.asarray(alpha.resize((min(256, width), min(256, height))), dtype=np.uint8)
                alpha_coverage = float(np.mean(thumb < 128))
        jpeg_quality_low = False
        quantization = getattr(image, 'quantization', None)
        if fmt == 'JPEG' and quantization:
            table = quantization.get(0)
            if table:
                jpeg_quality_low = float(np.mean(list(table))) > 12.0
        gray = image.convert('L')
        # Full-resolution centre crop keeps grain and focus measurements honest.
        crop_w, crop_h = min(width, MAX_IMAGE_ANALYSIS_EDGE), min(height, MAX_IMAGE_ANALYSIS_EDGE)
        left, top = (width - crop_w) // 2, (height - crop_h) // 2
        pixels = np.asarray(gray.crop((left, top, left + crop_w, top + crop_h)), dtype=np.float32)
    report = {'width': width, 'height': height, 'format': fmt.lower(), 'has_alpha': has_alpha,
              'alpha_coverage': round(alpha_coverage, 3),
              'jpeg_quality_low': jpeg_quality_low, 'brightness': round(float(pixels.mean()), 1) if pixels.size else 0.0}
    if pixels.shape[0] >= 3 and pixels.shape[1] >= 3:
        center = pixels[1:-1, 1:-1]
        laplacian = (pixels[:-2, 1:-1] + pixels[2:, 1:-1] + pixels[1:-1, :-2] + pixels[1:-1, 2:] - 4 * center)
        lap_var = float(laplacian.var())
        # Immerkaer noise operator with a median (MAD) statistic so sparse edges do not count as grain.
        noise_kernel = (pixels[:-2, :-2] - 2 * pixels[:-2, 1:-1] + pixels[:-2, 2:]
                        - 2 * pixels[1:-1, :-2] + 4 * center - 2 * pixels[1:-1, 2:]
                        + pixels[2:, :-2] - 2 * pixels[2:, 1:-1] + pixels[2:, 2:])
        sigma = float(np.median(np.abs(noise_kernel)) / 0.6745 / 6.0)
        report.update({'sharpness': round(lap_var, 2), 'noise_sigma': round(sigma, 2),
                       'blurry': lap_var < BLUR_LAPLACIAN_VAR,
                       'noisy': sigma > NOISY_IMAGE_SIGMA or jpeg_quality_low})
    else:
        report.update({'sharpness': 0.0, 'noise_sigma': 0.0, 'blurry': False, 'noisy': False})
    report['small'] = min(width, height) < 320
    return report


def _frame_luma(path, at_sec):
    cmd = [video_processor.FFMPEG, '-v', 'error', '-nostdin', '-ss', f'{at_sec:.3f}', '-i', path,
           '-frames:v', '1', '-vf', 'scale=64:-2', '-pix_fmt', 'gray', '-f', 'rawvideo', '-']
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=DECODE_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return float(np.frombuffer(proc.stdout, dtype=np.uint8).mean())
