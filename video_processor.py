"""
Video Processor — FFmpeg-based engine for the Video Editor page.

Everything runs locally through the same FFmpeg binary the audio cutter
already requires. No cloud APIs, no uploads to third parties.

Responsibilities:
  - probe_media()      : read duration / dimensions / streams via ffprobe
  - generate_filmstrip(): thumbnail strip for timeline clips
  - extract_audio()    : video -> MP3/WAV extraction (quick tool)
  - export_project()   : render the full timeline (clips + effects + text
                         overlays + music track) into MP4, or audio-only
                         MP3/WAV of the same timeline
"""

import os
import json
import subprocess

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

TARGET_FPS = 30
AUDIO_RATE = 44100

# Canvas presets (aspect ratios, CapCut-style)
RESOLUTIONS = {
    "1080p": (1920, 1080),   # 16:9
    "720p": (1280, 720),     # 16:9 small
    "vertical": (1080, 1920),  # 9:16 (Shorts/Reels)
    "square": (1080, 1080),  # 1:1
}

FONT_CANDIDATES = [
    r"C:/Windows/Fonts/arialbd.ttf",
    r"C:/Windows/Fonts/arial.ttf",
    r"C:/Windows/Fonts/segoeui.ttf",
    r"C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _run(cmd, timeout=1800):
    """Run an ffmpeg/ffprobe command; raise with stderr tail on failure."""
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-1200:]
        raise RuntimeError(f"FFmpeg failed (code {proc.returncode}):\n{tail}")
    return proc


def probe_media(path):
    """Return {duration, width, height, fps, has_video, has_audio} for a file."""
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type,width,height,avg_frame_rate",
        "-of", "json", path,
    ]
    proc = _run(cmd, timeout=60)
    data = json.loads(proc.stdout)

    info = {
        "duration": float(data.get("format", {}).get("duration", 0) or 0),
        "width": 0, "height": 0, "fps": 0.0,
        "has_video": False, "has_audio": False,
    }
    for stream in data.get("streams", []):
        ctype = stream.get("codec_type")
        if ctype == "video" and not info["has_video"]:
            # Ignore attached cover art (mjpeg streams report as video too,
            # but real video always has avg_frame_rate > 0/0)
            rate = stream.get("avg_frame_rate", "0/0")
            try:
                num, den = rate.split("/")
                fps = float(num) / float(den) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                fps = 0.0
            info["has_video"] = True
            info["width"] = int(stream.get("width") or 0)
            info["height"] = int(stream.get("height") or 0)
            info["fps"] = round(fps, 3)
        elif ctype == "audio":
            info["has_audio"] = True
    return info


def generate_filmstrip(path, out_dir, duration, count=6, height=90):
    """Extract `count` evenly spaced frames as small JPEGs for the timeline."""
    os.makedirs(out_dir, exist_ok=True)
    made = []
    if duration <= 0:
        return made
    for i in range(count):
        # Sample from 2%..98% of the clip so we skip black lead-in frames
        t = duration * (0.02 + 0.96 * (i / max(count - 1, 1)))
        out_path = os.path.join(out_dir, f"thumb_{i}.jpg")
        cmd = [
            FFMPEG, "-y", "-ss", f"{t:.3f}", "-i", path,
            "-frames:v", "1", "-vf", f"scale=-2:{height}",
            "-q:v", "5", out_path,
        ]
        try:
            _run(cmd, timeout=60)
            made.append(out_path)
        except RuntimeError:
            break  # unreadable frame near EOF — keep what we have
    return made


def extract_audio(video_path, output_path, fmt="mp3", bitrate="192k"):
    """Quick tool: pull the audio track out of a video (like extract_audio.py)."""
    cmd = [FFMPEG, "-y", "-i", video_path, "-vn"]
    if fmt == "wav":
        cmd += ["-acodec", "pcm_s16le", "-ar", str(AUDIO_RATE)]
    else:
        cmd += ["-acodec", "libmp3lame", "-b:a", bitrate]
    cmd.append(output_path)
    _run(cmd)
    return output_path


# ─────────────────────────────────────────────
#  QUICK TOOLS — one-file-in, one-file-out
#  (no timeline needed; friendly for beginners)
# ─────────────────────────────────────────────

QUICK_COMPRESS_CRF = {"light": 23, "balanced": 28, "strong": 33}
QUICK_SCALE = {  # long-edge caps
    "original": None, "1080p": 1080, "720p": 720, "480p": 480, "360p": 360,
}


def _scale_filter(cap):
    """Scale so the height is `cap`, keeping aspect and even dimensions."""
    if not cap:
        return None
    return f"scale=-2:'min({cap},ih)'"


def quick_convert(src, out_path, container="mp4", quality="720p"):
    """Re-encode into another container / resolution."""
    cmd = [FFMPEG, "-y", "-i", src]
    vf = _scale_filter(QUICK_SCALE.get(quality))
    if vf:
        cmd += ["-vf", vf]
    if container == "webm":
        cmd += ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32",
                "-c:a", "libopus"]
    else:  # mp4 / mkv share h264+aac
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-movflags", "+faststart"] if container == "mp4" else []
    cmd.append(out_path)
    _run(cmd)
    return out_path


def quick_compress(src, out_path, level="balanced"):
    """Shrink file size using a higher CRF + capped resolution."""
    crf = QUICK_COMPRESS_CRF.get(level, 28)
    cmd = [FFMPEG, "-y", "-i", src,
           "-vf", _scale_filter(720),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", out_path]
    _run(cmd)
    return out_path


def quick_to_gif(src, out_path, start=0.0, duration=5.0, fps=12, width=480):
    """Convert a slice of video to an optimized GIF (two-pass palette)."""
    palette = out_path + ".png"
    filt = f"fps={fps},scale={width}:-1:flags=lanczos"
    _run([FFMPEG, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
          "-i", src, "-vf", f"{filt},palettegen=stats_mode=diff", palette])
    _run([FFMPEG, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
          "-i", src, "-i", palette,
          "-lavfi", f"{filt}[x];[x][1:v]paletteuse=dither=bayer",
          out_path])
    try:
        os.remove(palette)
    except OSError:
        pass
    return out_path


def quick_extract_frame(src, out_path, t=0.0):
    """Grab a single frame at time `t` as a JPEG screenshot."""
    _run([FFMPEG, "-y", "-ss", f"{t:.3f}", "-i", src,
          "-frames:v", "1", "-q:v", "2", out_path])
    return out_path


def quick_mute(src, out_path):
    """Strip the audio track, keep the video as-is (stream copy)."""
    _run([FFMPEG, "-y", "-i", src, "-an", "-c:v", "copy",
          "-movflags", "+faststart", out_path])
    return out_path


# ─────────────────────────────────────────────
#  TIMELINE EXPORT
# ─────────────────────────────────────────────

def _atempo_chain(speed):
    """FFmpeg atempo only accepts 0.5–2.0 per instance; chain for more."""
    parts = []
    s = float(speed)
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    parts.append(f"atempo={s:.6f}")
    return parts


def _find_font():
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            return f
    return None


def _esc_drawtext(text):
    """Escape a literal string for use inside drawtext=text=..."""
    out = text.replace("\\", "\\\\")
    out = out.replace("'", "’")  # avoid quote-nesting hell entirely
    for ch in (":", ",", ";", "[", "]", "%", "="):
        out = out.replace(ch, "\\" + ch)
    return out


def _esc_path(path):
    """Escape a file path for use inside a filter option value."""
    return path.replace("\\", "/").replace(":", "\\:")


def _text_position(pos):
    """Map a 9-grid position key to drawtext x/y expressions."""
    xs = {
        "l": "w*0.05",
        "c": "(w-text_w)/2",
        "r": "w-text_w-w*0.05",
    }
    ys = {
        "t": "h*0.06",
        "m": "(h-text_h)/2",
        "b": "h-text_h-h*0.08",
    }
    pos = pos if isinstance(pos, str) and len(pos) == 2 else "bc"
    return xs.get(pos[1], xs["c"]), ys.get(pos[0], ys["b"])


def _clip_duration(clip):
    trimmed = max(0.0, float(clip["out"]) - float(clip["in"]))
    speed = float(clip.get("speed", 1.0)) or 1.0
    return trimmed / speed


def _build_clip_command(clip, src, out_path, target, is_first, is_last,
                        transition, audio_only=False):
    """Render one timeline clip into a normalized intermediate file."""
    t_in = float(clip["in"])
    t_dur = max(0.05, float(clip["out"]) - t_in)
    speed = min(4.0, max(0.25, float(clip.get("speed", 1.0)) or 1.0))
    out_dur = t_dur / speed

    fade_black = transition == "fadeblack"
    v_fade_in = bool(clip.get("fadeIn")) or (fade_black and not is_first)
    v_fade_out = bool(clip.get("fadeOut")) or (fade_black and not is_last)
    fade_d = min(1.0 if (clip.get("fadeIn") or clip.get("fadeOut")) else 0.5,
                 out_dur / 2)

    has_audio = bool(clip.get("hasAudio", True)) and not clip.get("muted")
    volume = min(2.0, max(0.0, float(clip.get("volume", 1.0))))

    cmd = [FFMPEG, "-y", "-ss", f"{t_in:.3f}", "-t", f"{t_dur:.3f}", "-i", src]
    if not has_audio:
        cmd += ["-f", "lavfi", "-t", f"{out_dur:.3f}",
                "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]

    # ── audio chain ──
    a_src = "0:a" if has_audio else "1:a"
    a_parts = []
    if has_audio:
        a_parts += _atempo_chain(speed)
        if volume != 1.0:
            a_parts.append(f"volume={volume:.3f}")
        if v_fade_in:
            a_parts.append(f"afade=t=in:st=0:d={fade_d:.3f}")
        if v_fade_out:
            a_parts.append(
                f"afade=t=out:st={max(0, out_dur - fade_d):.3f}:d={fade_d:.3f}")
    a_parts.append(f"aresample={AUDIO_RATE}")
    a_chain = ",".join(a_parts)

    if audio_only:
        cmd += ["-filter_complex", f"[{a_src}]{a_chain}[a]",
                "-map", "[a]", "-ac", "2",
                "-c:a", "pcm_s16le", out_path]
        return cmd

    # ── video chain ──
    tw, th = target
    f = clip.get("filters", {}) or {}
    v_parts = [f"fps={TARGET_FPS}"]

    rotate = int(clip.get("rotate", 0)) % 360
    if rotate == 90:
        v_parts.append("transpose=1")
    elif rotate == 180:
        v_parts.append("transpose=1,transpose=1")
    elif rotate == 270:
        v_parts.append("transpose=2")

    v_parts.append(
        f"scale={tw}:{th}:force_original_aspect_ratio=decrease")
    v_parts.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black")
    v_parts.append("setsar=1")

    brightness = min(1.0, max(-1.0, float(f.get("brightness", 0))))
    contrast = min(3.0, max(0.0, float(f.get("contrast", 1))))
    saturation = min(3.0, max(0.0, float(f.get("saturation", 1))))
    if brightness != 0 or contrast != 1 or saturation != 1:
        v_parts.append(
            f"eq=brightness={brightness:.3f}"
            f":contrast={contrast:.3f}:saturation={saturation:.3f}")
    if f.get("grayscale"):
        v_parts.append("hue=s=0")
    if f.get("sepia"):
        v_parts.append(
            "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131:0")

    if speed != 1.0:
        v_parts.append(f"setpts=PTS/{speed:.6f}")
    if v_fade_in:
        v_parts.append(f"fade=t=in:st=0:d={fade_d:.3f}")
    if v_fade_out:
        v_parts.append(
            f"fade=t=out:st={max(0, out_dur - fade_d):.3f}:d={fade_d:.3f}")

    v_chain = ",".join(v_parts)
    cmd += [
        "-filter_complex", f"[0:v]{v_chain}[v];[{a_src}]{a_chain}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-ac", "2",
        "-shortest", out_path,
    ]
    return cmd


def _build_drawtext(texts, target_h):
    """Build a chained drawtext filter string for all text overlays."""
    font = _find_font()
    filters = []
    for t in texts:
        content = str(t.get("text", "")).strip()
        if not content:
            continue
        size = int(round(float(t.get("size", 48)) * target_h / 1080.0))
        color = str(t.get("color", "#FFFFFF")).lstrip("#")
        x, y = _text_position(t.get("position", "bc"))
        start = float(t.get("start", 0))
        end = float(t.get("end", start + 3))
        parts = [
            f"text='{_esc_drawtext(content)}'",
            f"fontsize={max(10, size)}",
            f"fontcolor=0x{color}",
            f"x={x}", f"y={y}",
            f"enable='between(t\\,{start:.3f}\\,{end:.3f})'",
        ]
        if font:
            parts.insert(0, f"fontfile='{_esc_path(font)}'")
        if t.get("bg", True):
            parts += ["box=1", "boxcolor=black@0.45",
                      f"boxborderw={max(6, size // 4)}"]
        filters.append("drawtext=" + ":".join(parts))
    return ",".join(filters)


def export_project(spec, resolve_source, work_dir, output_path):
    """
    Render a timeline spec to `output_path`.

    spec = {
      "format": "mp4" | "mp3" | "wav",
      "resolution": "original" | "1080p" | "720p" | "vertical" | "square",
      "transition": "none" | "fadeblack",
      "clips":  [{source, in, out, speed, volume, muted, hasAudio, rotate,
                  fadeIn, fadeOut, filters:{...}}, ...],
      "texts":  [{text, start, end, size, color, position, bg}, ...],
      "music":  {source, volume, loop} | None
    }
    resolve_source(name) -> absolute path of an uploaded media file (or None).
    """
    clips = spec.get("clips") or []
    if not clips:
        raise ValueError("Timeline is empty — add at least one clip.")

    fmt = spec.get("format", "mp4")
    audio_only = fmt in ("mp3", "wav")
    transition = spec.get("transition", "none")
    texts = spec.get("texts") or []
    music = spec.get("music") or None

    os.makedirs(work_dir, exist_ok=True)

    # ── target canvas ──
    res_key = spec.get("resolution", "original")
    if res_key in RESOLUTIONS:
        target = RESOLUTIONS[res_key]
    else:
        first_src = resolve_source(clips[0]["source"])
        if not first_src:
            raise ValueError("Source media not found on server.")
        info = probe_media(first_src)
        w = info["width"] or 1280
        h = info["height"] or 720
        target = (w - w % 2, h - h % 2)  # h264 needs even dimensions

    # ── pass 1: render each clip to a normalized intermediate ──
    ext = "wav" if audio_only else "mp4"
    intermediates = []
    for i, clip in enumerate(clips):
        src = resolve_source(clip["source"])
        if not src or not os.path.exists(src):
            raise ValueError(f"Source media missing for clip {i + 1}.")
        part = os.path.join(work_dir, f"part_{i:03d}.{ext}")
        cmd = _build_clip_command(
            clip, src, part, target,
            is_first=(i == 0), is_last=(i == len(clips) - 1),
            transition=transition, audio_only=audio_only,
        )
        _run(cmd)
        intermediates.append(part)

    # ── pass 2: concat ──
    merged = os.path.join(work_dir, f"merged.{ext}")
    if len(intermediates) == 1:
        os.replace(intermediates[0], merged)
    else:
        list_path = os.path.join(work_dir, "concat.txt")
        with open(list_path, "w", encoding="utf-8") as fh:
            for p in intermediates:
                fh.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")
        concat_cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_path]
        # WAV headers don't survive stream-copy concat; re-encode PCM instead
        concat_cmd += ["-c:a", "pcm_s16le"] if audio_only else ["-c", "copy"]
        concat_cmd.append(merged)
        _run(concat_cmd)

    # ── pass 3: overlays + music + final encode ──
    music_path = None
    if music and music.get("source"):
        music_path = resolve_source(music["source"])
        if music_path and not os.path.exists(music_path):
            music_path = None

    drawtext = "" if audio_only else _build_drawtext(texts, target[1])

    if not drawtext and not music_path:
        # Nothing global to apply — transcode/copy straight to output
        if audio_only:
            cmd = [FFMPEG, "-y", "-i", merged]
            cmd += (["-c:a", "libmp3lame", "-b:a", "320k"] if fmt == "mp3"
                    else ["-c:a", "pcm_s16le"])
            cmd.append(output_path)
            _run(cmd)
        else:
            _run([FFMPEG, "-y", "-i", merged, "-c", "copy",
                  "-movflags", "+faststart", output_path])
        return output_path

    cmd = [FFMPEG, "-y", "-i", merged]
    if music_path:
        if music.get("loop"):
            cmd += ["-stream_loop", "-1"]
        cmd += ["-i", music_path]

    graphs = []
    if drawtext:
        graphs.append(f"[0:v]{drawtext}[vout]")
    if music_path:
        mvol = min(2.0, max(0.0, float(music.get("volume", 0.6))))
        graphs.append(
            f"[1:a]volume={mvol:.3f},aresample={AUDIO_RATE}[m];"
            f"[0:a][m]amix=inputs=2:duration=first:normalize=0[aout]")

    cmd += ["-filter_complex", ";".join(graphs)]
    cmd += ["-map", "[vout]" if drawtext else "0:v"] if not audio_only else []
    if audio_only:
        cmd += ["-map", "[aout]" if music_path else "0:a"]
        cmd += (["-c:a", "libmp3lame", "-b:a", "320k"] if fmt == "mp3"
                else ["-c:a", "pcm_s16le"])
    else:
        cmd += ["-map", "[aout]" if music_path else "0:a"]
        if drawtext:
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-c:v", "copy"]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest",
                "-movflags", "+faststart"]
    cmd.append(output_path)
    _run(cmd)
    return output_path


def detect_scenes(video_path, threshold=3.0, mode="adaptive"):
    """
    Detects scene cut boundaries in video using scenedetect (PySceneDetect).
    mode: 'adaptive' (AdaptiveDetector - best for dynamic shots & camera motion) or 'content' (ContentDetector).
    Returns list of cut intervals: [{'start': sec, 'end': sec, 'duration': sec}]
    """
    try:
        from scenedetect import detect, AdaptiveDetector, ContentDetector
        if mode == "adaptive":
            detector = AdaptiveDetector(adaptive_threshold=float(threshold))
        else:
            detector = ContentDetector(threshold=float(threshold))

        scene_list = detect(video_path, detector)
        scenes = []
        for i, scene in enumerate(scene_list):
            start_sec = round(scene[0].get_seconds(), 3)
            end_sec = round(scene[1].get_seconds(), 3)
            scenes.append({
                "scene_num": i + 1,
                "start": start_sec,
                "end": end_sec,
                "duration": round(end_sec - start_sec, 3)
            })
        return scenes
    except Exception as e:
        # Fallback to ContentDetector
        try:
            from scenedetect import detect, ContentDetector
            scene_list = detect(video_path, ContentDetector(threshold=27.0))
            return [{"scene_num": i + 1, "start": round(s[0].get_seconds(), 3), "end": round(s[1].get_seconds(), 3), "duration": round(s[1].get_seconds() - s[0].get_seconds(), 3)} for i, s in enumerate(scene_list)]
        except Exception as err:
            raise RuntimeError(f"Scene detection failed: {str(e)}")


def burn_subtitles(video_path, output_path, style="yellow_box"):
    """
    Transcribes audio via Whisper and burns stylized hardcoded subtitles onto the video via FFmpeg.
    style: 'yellow_box' (CapCut modern yellow text with dark box), 'classic_white', 'neon_cyan'
    """
    import tempfile
    from ai_processor import transcribe_audio

    result = transcribe_audio(video_path)
    if not result.get("available") or not result.get("segments"):
        raise RuntimeError("Speech transcription failed or no spoken text found.")

    segments = result["segments"]

    def sec_to_srt_time(sec):
        hrs = int(sec // 3600)
        mins = int((sec % 3600) // 60)
        secs = int(sec % 60)
        millis = int(round((sec - int(sec)) * 1000))
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    srt_lines = []
    for idx, seg in enumerate(segments, start=1):
        start_t = sec_to_srt_time(seg["start"])
        end_t = sec_to_srt_time(seg["end"])
        text = seg["text"].strip()
        if text:
            srt_lines.append(f"{idx}\n{start_t} --> {end_t}\n{text}\n")

    if not srt_lines:
        raise RuntimeError("No spoken subtitle text found in video.")

    srt_content = "\n".join(srt_lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as srt_file:
        srt_file.write(srt_content)
        srt_path = srt_file.name

    escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")

    if style == "yellow_box":
        force_style = "Fontname=Arial,Fontsize=22,PrimaryColour=&H0000FFFF,BackColour=&H80000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=30,Bold=1"
    elif style == "neon_cyan":
        force_style = "Fontname=Arial,Fontsize=22,PrimaryColour=&H00FFFF00,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=30,Bold=1"
    else:
        force_style = "Fontname=Arial,Fontsize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,MarginV=30,Bold=1"

    vf_arg = f"subtitles='{escaped_srt}':force_style='{force_style}'"

    cmd = [
        FFMPEG_BINARY, "-y",
        "-i", video_path,
        "-vf", vf_arg,
        "-c:a", "copy",
        "-preset", "fast",
        output_path
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg subtitle burn failed: {proc.stderr[-500:]}")
        return {"status": "success", "subtitles_burned": len(segments)}
    finally:
        if os.path.exists(srt_path):
            try: os.remove(srt_path)
            except Exception: pass
