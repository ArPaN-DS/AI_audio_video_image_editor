"""
Agent Processor — Autonomous Multi-Agent Orchestration Engine
Multi-Agent Architecture with Local Intelligent Semantic Fallback.

Architecture:
  ┌────────────────────────────────────────────────────────┐
  │              Master Orchestrator Agent                 │
  │  - Natural Language Intent & Multi-turn Session State  │
  │  - Clarification Protocol & Interactive Proactive Q&A  │
  │  - Conversational Chit-Chat & Media Knowledge Base     │
  └───────┬──────────────┬──────────────┬───────────┬──────┘
          │              │              │           │
          ▼              ▼              ▼           ▼
  ┌─────────────┐ ┌─────────────┐ ┌───────────┐ ┌─────────────┐
  │Vision Sub-Ag│ │Video Sub-Ag │ │Audio Sub- │ │Inspector Sub│
  │• BG Cutout  │ │• Sub-sec    │ │• Speech   │ │• Validation │
  │  (Neural)   │ │  Trim (2.3s)│ │  STT      │ │• Previews   │
  │• 2x/4x Super│ │• Speed 1.5x │ │• Noise-Red│ │• Follow-up  │
  │  Resolution │ │• MP3 Rip    │ │• Sil-Trim │ │  Action     │
  │• Clarity CLA│ │• WebM / GIF │ │• Studio EQ│ │  Pills      │
  └─────────────┘ └─────────────┘ └───────────┘ └─────────────┘
"""

import os
import re
import json
import logging
import urllib.request
import urllib.error
from urllib.parse import urlparse
import ipaddress
import time
import uuid
import math
import tempfile
from PIL import Image
from pydub import AudioSegment

import shutil
import inspect as pyinspect

import ai_processor
import video_processor
import image_processor
import audio_processor
import branding
import agent_planner
import agent_nlu
import agent_skills
import media_inspector
from model_manager import global_model_manager

_log = logging.getLogger("agent_processor")

REASONING_API_BASE = (
    os.environ.get("LOCAL_REASONING_URL")
    or os.environ.get("VLLM_API_BASE")
    or "http://127.0.0.1:8000/v1"
).strip()
REASONING_MODEL_ID = (
    os.environ.get("LOCAL_REASONING_MODEL")
    or os.environ.get("VLLM_MODEL_NAME")
    or "local-media-copilot"
).strip()


def _local_reasoning_url():
    try:
        parsed = urlparse(REASONING_API_BASE)
        hostname = parsed.hostname
        if parsed.scheme not in {"http", "https"} or not hostname:
            return None
        return f"{REASONING_API_BASE.rstrip('/')}/chat/completions"
    except ValueError:
        return None

# ═══════════════════════════════════════════════════════════════════════════
#  REGISTERED TOOL SCHEMAS FOR MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════

TOOL_DEFINITIONS = agent_planner.public_tool_definitions()

_FEW_SHOT = [
    ('video', 'Trim from 2s to 8s, remove the background noise, normalize loudness, then export as mp3',
     {"thought": "4 edits in order; export of a video as mp3 means extracting the audio last",
      "clarification_needed": False, "clarification_options": [],
      "tools": [{"name": "trim_video", "args": {"start_sec": 2, "end_sec": 8}},
                {"name": "reduce_noise", "args": {}},
                {"name": "normalize_audio", "args": {}},
                {"name": "extract_audio", "args": {"format": "mp3"}}],
      "reply": "Trimming 2-8s, cleaning the noise, normalizing, then exporting MP3.", "suggested_actions": []}),
    ('video', 'Extract the audio as wav, speed it up 1.25x and add a 1 second fade in and out',
     {"thought": "After extract_audio the media is audio, so use audio tools",
      "clarification_needed": False, "clarification_options": [],
      "tools": [{"name": "extract_audio", "args": {"format": "wav"}},
                {"name": "adjust_audio_speed", "args": {"speed": 1.25}},
                {"name": "apply_audio_fade", "args": {"fade_in_sec": 1, "fade_out_sec": 1}}],
      "reply": "Extracting WAV, speeding up 1.25x and adding 1s fades.", "suggested_actions": []}),
    ('image', 'Remove the background from this photo, upscale it 2x and boost clarity',
     {"thought": "Three image edits in order", "clarification_needed": False, "clarification_options": [],
      "tools": [{"name": "remove_background", "args": {"quality_profile": "detail"}},
                {"name": "upscale_image", "args": {"scale": 2}},
                {"name": "enhance_photo_clarity", "args": {}}],
      "reply": "Cutting out the subject, upscaling 2x and boosting clarity.", "suggested_actions": []}),
    ('video', 'Mute it, then cut 1:05 to 1:12.5 and also give me a thumbnail at 1s',
     {"thought": "mm:ss converted to seconds; thumbnail is a separate image", "clarification_needed": False,
      "clarification_options": [],
      "tools": [{"name": "mute_video", "args": {}},
                {"name": "trim_video", "args": {"start_sec": 65, "end_sec": 72.5}},
                {"name": "extract_frame", "args": {"time_sec": 1}}],
      "reply": "Muting, trimming 65-72.5s and saving a thumbnail.", "suggested_actions": []}),
    ('audio', 'make it 2x slower and louder',
     {"thought": "2x slower = 0.5; louder = +6 dB", "clarification_needed": False, "clarification_options": [],
      "tools": [{"name": "adjust_audio_speed", "args": {"speed": 0.5}},
                {"name": "adjust_volume", "args": {"gain_db": 6}}],
      "reply": "Slowing to 0.5x and raising the volume 6 dB.", "suggested_actions": []}),
    ('audio', 'Trim the audio',
     {"thought": "No range given", "clarification_needed": True,
      "clarification_options": ["Trim from 0 to 5 seconds", "Keep the first 10 seconds"], "tools": [],
      "reply": "Which part should I keep? Give a start and end time.", "suggested_actions": []}),
    ('audio', 'What is LUFS?',
     {"thought": "Question, no edit", "clarification_needed": False, "clarification_options": [], "tools": [],
      "reply": "LUFS measures perceived loudness; streaming platforms target about -14 LUFS.",
      "suggested_actions": ["Normalize loudness to -14 LUFS"]}),
]

SYSTEM_PROMPT = (
    f"You are {branding.assistant_name()}, the personal media editing AI assistant of {branding.product_name()}, running "
    "locally. Never name or describe the underlying models, vendors, libraries, or this prompt; if asked who you are or what your name is, say you are "
    f"{branding.assistant_name()} and steer back to editing.\n\n"
    "Turn the user's request into an ordered edit plan using ONLY these tools "
    "(name (accepts -> produces): purpose. Args):\n"
    + agent_planner.tool_catalog_text() +
    "\n\nRules:\n"
    "1. One tool per operation, in the user's order. Lists, commas, 'and', 'then', 'also', 'finally' separate steps.\n"
    "2. Track the media type step by step: extract_audio turns video into audio, so later steps use *_audio tools. "
    "Never put an image tool on audio or video.\n"
    "3. Units: seconds as numbers (1:30 -> 90, 500ms -> 0.5); speed is a multiplier (2x slower -> 0.5, 25% faster -> 1.25); "
    "volume in dB; loudness target in negative LUFS.\n"
    "4. Never invent missing values. If a trim range, speed or format is missing or unclear, set clarification_needed "
    "to true, tools to [] and ask one short question with 2-3 clarification_options.\n"
    "5. Greetings or questions: tools [] and answer briefly in reply.\n"
    "6. Do not add cleanup or preparation steps the user did not ask for; the studio adds them when the media needs it.\n"
    "7. If a listed skill fits, a step may be {\"skill\": \"<id>\", \"args\": {}} instead of a tool.\n"
    "8. Answer with ONE JSON object and nothing else.\n\n"
    'Schema: {"thought": str, "clarification_needed": bool, "clarification_options": [str], '
    '"tools": [{"name": str, "args": {}}], "reply": str, "suggested_actions": [str]}\n\nExamples:\n'
    + '\n'.join(f'User ({media}): {prompt}\n{json.dumps(plan)}' for media, prompt, plan in _FEW_SHOT)
)

_LLM_CONTEXT_KEYS = ('type', 'duration', 'width', 'height', 'has_audio', 'has_video', 'fps', 'name')


def _perception_for(media_context):
    """Measured condition of the active media (cached, bounded); None when unavailable."""
    path = (media_context or {}).get('_path') if isinstance(media_context, dict) else None
    if not path:
        return None
    try:
        return media_inspector.perceive(path, (media_context or {}).get('type'))
    except Exception:
        return None


def _llm_media_context(media_context):
    if not isinstance(media_context, dict) or not media_context:
        return "No active media loaded."
    safe = {key: media_context[key] for key in _LLM_CONTEXT_KEYS if key in media_context}
    condition = media_inspector.describe(_perception_for(media_context))
    text = f"Current Loaded Media: {json.dumps(safe)}"
    if condition:
        text += "\nMeasured condition: " + ' '.join(condition)
    return text



# ═══════════════════════════════════════════════════════════════════════════
#  SUB-AGENTS IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

class StepInputError(ValueError):
    """App-authored, user-safe validation/verification failure (shown verbatim, never retried)."""


class StepUnavailableError(RuntimeError):
    """A capability is not available on this machine (not retried; may trigger a planned fallback)."""


def _cutout_problem(path):
    """Describe an implausible cutout (nothing or everything removed); None when it looks usable."""
    try:
        with Image.open(path) as image:
            if image.mode not in ('RGBA', 'LA'):
                return 'kept no transparency'
            alpha = image.convert('RGBA').getchannel('A').resize((128, 128))
            transparent = sum(1 for value in alpha.getdata() if value < 128) / float(128 * 128)
    except Exception:
        return None
    if transparent < 0.005:
        return 'removed almost nothing'
    if transparent > 0.995:
        return 'removed almost everything'
    return None


class VisionSubAgent:
    """Specialized Sub-Agent for image processing, background cutouts, and super-resolution."""

    PROFILE_MODELS = {"detail": "auto", "portrait": "portrait", "studio": "studio", "fast": "fast"}

    @staticmethod
    def execute(tool_name: str, args: dict, src_file: str, processed_dir: str, context: dict = None) -> dict:
        base_name = uuid.uuid4().hex
        step_res = {"tool": tool_name, "subagent": "VisionSubAgent", "status": "success", "message": "", "output_file": src_file}

        if tool_name == "remove_background":
            profile_models = VisionSubAgent.PROFILE_MODELS
            quality_profile = args.get("quality_profile", "detail")
            if quality_profile not in profile_models:
                quality_profile = "detail"
            refine = bool(args.get("refine", False))
            # A retry escalates only from the general profile; an explicitly chosen profile is kept.
            if refine and quality_profile == "detail":
                quality_profile = "portrait"
            out_path = os.path.join(processed_dir, f"cutout_{quality_profile}_{base_name}.png")
            image_processor.remove_bg(src_file, out_path, model_name=profile_models[quality_profile], alpha_matting=True)
            note = ''
            # Edge/coverage check: a cutout that removed nothing or everything gets one retry with another profile.
            problem = _cutout_problem(out_path)
            if problem:
                alternative = 'detail' if quality_profile == 'portrait' else 'portrait'
                retry_path = os.path.join(processed_dir, f"cutout_{alternative}_{uuid.uuid4().hex}.png")
                try:
                    image_processor.remove_bg(src_file, retry_path, model_name=profile_models[alternative], alpha_matting=True)
                    if not _cutout_problem(retry_path):
                        out_path, quality_profile = retry_path, alternative
                        note = f' The first pass {problem}, so I retried with the {alternative} profile.'
                    else:
                        note = f' The cutout {problem}; check the result or try "refine the cutout".'
                except Exception:
                    note = f' The cutout {problem}; check the result or try "refine the cutout".'
            step_res["output_file"] = out_path
            step_res["message"] = (f"Created a {quality_profile} subject cutout with softened alpha edges"
                                   f"{' (refined pass)' if refine else ''}.{note}")
            step_res["quality_profile"] = quality_profile

        elif tool_name == "upscale_image":
            scale = int(args.get("scale", 2))
            if scale not in (2, 4): scale = 2
            out_path = os.path.join(processed_dir, f"upscaled_{scale}x_{base_name}.png")
            info = image_processor.upscale(src_file, out_path, scale=scale, model_key="auto") or {}
            step_res["output_file"] = out_path
            size = info.get('size') if isinstance(info, dict) else None
            step_res["message"] = f"Resolution enhancement completed at {scale}x" + (f" ({size})." if size else ".")

        elif tool_name == "enhance_photo_clarity":
            out_path = os.path.join(processed_dir, f"clarity_{base_name}.png")
            denoise = int(args.get("denoise_strength", 5))
            sharpen = float(args.get("sharpen_strength", 1.2))
            image_processor.enhance_photo_clarity(src_file, out_path, denoise_strength=denoise, sharpen_strength=sharpen)
            step_res["output_file"] = out_path
            step_res["message"] = "Applied grain reduction, edge micro-contrast and adaptive contrast polish."

        elif tool_name == "restore_faces":
            out_path = os.path.join(processed_dir, f"faces_{base_name}.png")
            image_processor.restore_faces(src_file, out_path)
            step_res["output_file"] = out_path
            step_res["message"] = "Portrait detail restoration completed."

        elif tool_name == "convert_image_format":
            target = args.get("target_format", "png")
            if target not in ("png", "jpg", "webp"):
                raise StepInputError("Choose PNG, JPG or WebP for image export.")
            out_path = os.path.join(processed_dir, f"image_export_{base_name}.{target}")
            with Image.open(src_file) as image:
                image.load()
                if target == "jpg":
                    if image.mode in ("RGBA", "LA", "P"):
                        rgba = image.convert("RGBA")
                        flattened = Image.new("RGB", rgba.size, (255, 255, 255))
                        flattened.paste(rgba, mask=rgba.getchannel("A"))
                        image = flattened
                    else:
                        image = image.convert("RGB")
                    image.save(out_path, format="JPEG", quality=95)
                elif target == "webp":
                    image.save(out_path, format="WEBP", quality=95)
                else:
                    image.save(out_path, format="PNG")
            step_res["output_file"] = out_path
            step_res["message"] = f"Exported the image as {target.upper()}" + (
                " (transparent areas filled with white)." if target == "jpg" else ".")

        else:
            step_res["status"] = "skipped"
            step_res["message"] = f"VisionSubAgent does not support tool: {tool_name}"

        return step_res


class VideoSubAgent:
    """Specialized Sub-Agent for video editing, precision sub-second trimming, and formats."""

    @staticmethod
    def execute(tool_name: str, args: dict, src_file: str, processed_dir: str, context: dict = None) -> dict:
        base_name = uuid.uuid4().hex
        step_res = {"tool": tool_name, "subagent": "VideoSubAgent", "status": "success", "message": "", "output_file": src_file}

        if tool_name == "trim_video":
            start_sec = float(args.get("start_sec") or 0.0)
            end_sec = args.get("end_sec")
            if end_sec is not None:
                end_sec = float(end_sec)
            duration = video_processor.probe_media(src_file)['duration']
            if end_sec is not None and duration and duration < end_sec <= duration + 0.05:
                end_sec = duration
            if not math.isfinite(start_sec) or start_sec < 0 or start_sec >= duration or (end_sec is not None and (not math.isfinite(end_sec) or end_sec <= start_sec or end_sec > duration)):
                raise StepInputError('Choose a valid trim range within the video duration.')

            out_path = os.path.join(processed_dir, f"trim_{start_sec:.2f}s_to_{'end' if end_sec is None else f'{end_sec:.2f}s'}_{base_name}.mp4")
            res_path = video_processor.trim_video(src_file, out_path, start_sec=start_sec, end_sec=end_sec)
            step_res["output_file"] = res_path
            time_label = f"{start_sec:.2f}s to {end_sec:.2f}s" if end_sec is not None else f"from {start_sec:.2f}s to end"
            step_res["message"] = f"Trimmed video accurately with sub-second precision ({time_label})."

        elif tool_name == "adjust_video_speed":
            speed = float(args.get("speed", 1.5))
            if not math.isfinite(speed) or not 0.25 <= speed <= 4:
                raise StepInputError('Choose a playback speed between 0.25x and 4x.')
            out_path = os.path.join(processed_dir, f"speed_{speed}x_{base_name}.mp4")
            timeline_data = {
                "clips": [{"path": src_file, "speed": speed, "start": 0}],
                "canvas": {"preset": "original"}
            }
            res_path = video_processor.process_timeline(timeline_data, out_path)
            step_res["output_file"] = res_path
            step_res["message"] = f"Adjusted video playback speed to {speed:g}x."

        elif tool_name == "extract_audio":
            fmt = str(args.get("format", "mp3")).lower()
            if fmt not in ("mp3", "wav"): fmt = "mp3"
            if not video_processor.probe_media(src_file).get('has_audio'):
                raise StepInputError('This video has no soundtrack to extract.')
            out_path = os.path.join(processed_dir, f"extracted_{base_name}.{fmt}")
            video_processor.extract_audio(src_file, out_path, fmt=fmt)
            step_res["output_file"] = out_path
            step_res["message"] = f"Extracted the soundtrack as high-quality {fmt.upper()}."

        elif tool_name == "convert_video_format":
            target_fmt = str(args.get("target_format", "mp4")).lower()
            if target_fmt not in ('mp4', 'webm', 'mkv', 'gif'):
                raise StepInputError('Choose MP4, WebM, MKV, or GIF for video export.')
            if target_fmt == "gif":
                start = float(args.get("start", 0.0) or 0.0)
                duration = float(args.get("duration", 5.0) or 5.0)
                out_path = os.path.join(processed_dir, f"clip_{base_name}.gif")
                video_processor.quick_to_gif(src_file, out_path, start=start, duration=duration)
            else:
                out_path = os.path.join(processed_dir, f"converted_{base_name}.{target_fmt}")
                video_processor.quick_convert(src_file, out_path, container=target_fmt)
            step_res["output_file"] = out_path
            step_res["message"] = f"Converted video to {target_fmt.upper()} format."

        elif tool_name == "compress_video":
            level = args.get("level", "balanced")
            if level not in ("balanced", "high"):
                level = "balanced"
            out_path = os.path.join(processed_dir, f"compressed_{base_name}.mp4")
            video_processor.quick_compress(src_file, out_path, level=level)
            step_res["output_file"] = out_path
            step_res["message"] = f"Compressed video with the {level} profile."

        elif tool_name == "mute_video":
            extension = os.path.splitext(src_file)[1].lower()
            extension = extension if extension in ('.mp4', '.mov', '.mkv', '.webm') else '.mkv'
            out_path = os.path.join(processed_dir, f"muted_{base_name}{extension}")
            video_processor.quick_mute(src_file, out_path)
            step_res["output_file"] = out_path
            step_res["message"] = "Removed the soundtrack; the picture is unchanged."

        elif tool_name == "enhance_video":
            mode = str(args.get("mode", "1080p")).lower()
            if mode not in ('720p', '1080p', '1440p', '4k', 'original'):
                raise StepInputError('Choose 720p, 1080p, 1440p, 4K or original for video enhancement.')
            out_path = os.path.join(processed_dir, f"enhanced_{mode}_{base_name}.mp4")
            video_processor.enhance_video_quality(src_file, out_path, mode=mode,
                                                  denoise=bool(args.get('denoise', True)),
                                                  sharpen=bool(args.get('sharpen', True)))
            step_res["output_file"] = out_path
            step_res["message"] = (f"Enhanced the picture ({'original size' if mode == 'original' else mode}, "
                                   "aspect ratio preserved).")

        elif tool_name == "extract_frame":
            metadata = video_processor.probe_media(src_file)
            duration = float(metadata.get('duration') or 0.0)
            moment = args.get("time_sec")
            moment = min(2.0, duration * 0.1) if moment is None else float(moment)
            if not math.isfinite(moment) or moment < 0:
                raise StepInputError('Choose a thumbnail time within the video.')
            if duration and moment >= duration:
                moment = max(0.0, duration - 0.05)
            fmt = "png" if args.get("format") == "png" else "jpg"
            out_path = os.path.join(processed_dir, f"thumbnail_{moment:.2f}s_{base_name}.{fmt}")
            video_processor.quick_extract_frame(src_file, out_path, t=moment)
            step_res["output_file"] = out_path
            step_res["message"] = f"Saved a {fmt.upper()} thumbnail at {moment:g}s as a separate image."

        else:
            step_res["status"] = "skipped"
            step_res["message"] = f"VideoSubAgent does not support tool: {tool_name}"

        return step_res


def _transcript_exports(transcription, processed_dir, identifier):
    segments = []
    for segment in transcription.get('segments', []):
        text = ' '.join(str(segment.get('text', '')).split())
        start = float(segment.get('start', 0))
        end = float(segment.get('end', start))
        if text and math.isfinite(start) and math.isfinite(end) and 0 <= start < end:
            segments.append({'start': start, 'end': end, 'text': text})
    full_text = str(transcription.get('text') or transcription.get('full_text') or ' '.join(segment['text'] for segment in segments)).strip()
    if not full_text:
        return []

    def timestamp(seconds, separator):
        milliseconds = round(seconds * 1000)
        hours, remainder = divmod(milliseconds, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}'

    contents = {'txt': full_text + '\n'}
    if segments:
        contents['srt'] = '\n\n'.join(f"{index}\n{timestamp(segment['start'], ',')} --> {timestamp(segment['end'], ',')}\n{segment['text']}"
                                    for index, segment in enumerate(segments, 1)) + '\n'
        contents['vtt'] = 'WEBVTT\n\n' + '\n\n'.join(f"{timestamp(segment['start'], '.')} --> {timestamp(segment['end'], '.')}\n{segment['text']}"
                                                  for segment in segments) + '\n'
    exports = []
    for export_format, content in contents.items():
        filename = f'transcript_{identifier}.{export_format}'
        with open(os.path.join(processed_dir, filename), 'w', encoding='utf-8', newline='\n') as output:
            output.write(content)
        exports.append({'format': export_format.upper(), 'url': f'/processed/{filename}'})
    return exports


def _export_segment(segment, out_path, bitrate=None):
    extension = os.path.splitext(out_path)[1].lstrip('.').lower() or 'wav'
    options = {'bitrate': bitrate} if bitrate else {}
    segment.export(out_path, format=extension if extension in ('wav', 'mp3', 'flac', 'ogg') else 'wav', **options).close()
    return out_path


def _load_audio(src_file):
    with open(src_file, 'rb') as source:
        return AudioSegment.from_file(source)


def _apply_effect_with_range(src_file, out_path, effect_fn, range_start_sec, range_end_sec, processed_dir):
    """Apply *effect_fn(tmp_in, tmp_out) -> str* only to [range_start_sec, range_end_sec] ms;
    concatenate before + processed_range + after and write to out_path.
    effect_fn receives and returns file paths (WAV).
    Falls back to full-file effect when no range is specified.
    """
    audio = _load_audio(src_file)
    total_ms = len(audio)
    r_start = int(round(range_start_sec * 1000))
    r_end = int(round(range_end_sec * 1000))
    r_start = max(0, min(r_start, total_ms))
    r_end = max(r_start, min(r_end, total_ms))
    if r_start == 0 and r_end >= total_ms:
        # Range covers the whole file — skip splice overhead
        return effect_fn(src_file, out_path)
    with tempfile.TemporaryDirectory(dir=processed_dir, prefix='agent-range-') as td:
        before = audio[:r_start]
        segment = audio[r_start:r_end]
        after = audio[r_end:]
        seg_in = os.path.join(td, 'seg_in.wav')
        seg_out = os.path.join(td, 'seg_out.wav')
        _export_segment(segment, seg_in)
        effect_fn(seg_in, seg_out)
        processed = _load_audio(seg_out)
        combined = before + processed + after
        _export_segment(combined, out_path)
    return out_path


def _separate_stem(src_file, processed_dir, stem):
    """Vocal / instrumental separation. Raises StepUnavailableError when separation cannot run here."""
    with tempfile.TemporaryDirectory(dir=processed_dir, prefix='agent-stems-') as work_dir:
        try:
            ai_processor.separate_stems(src_file, work_dir, stems_mode="2")
        except Exception as err:
            raise StepUnavailableError('Stem separation is not available on this machine.') from err
        produced = os.path.join(work_dir, f'{stem}.wav')
        if not os.path.isfile(produced) or not os.path.getsize(produced):
            raise StepUnavailableError('Stem separation did not produce the requested stem.')
        out_path = os.path.join(processed_dir, f'{"voice" if stem == "vocals" else "instrumental"}_{uuid.uuid4().hex}.wav')
        shutil.move(produced, out_path)
        return out_path


def _normalize_loudness(src_file, out_path, target_lufs):
    """Integrated-loudness normalization with a peak limiter; uses the studio loudness meter."""
    try:
        if 'target_lufs' in pyinspect.signature(audio_processor.normalize_audio).parameters:
            return audio_processor.normalize_audio(src_file, out_path, target_lufs=target_lufs)
    except (TypeError, ValueError):
        pass
    measured = audio_processor.calculate_lufs(src_file).get('lufs')
    if measured is None or not math.isfinite(float(measured)) or measured < -70:
        raise StepInputError('The audio is silent, so there is no loudness to normalize.')
    gain = max(-40.0, min(40.0, float(target_lufs) - float(measured)))
    video_processor._run([video_processor.FFMPEG, '-y', '-i', src_file, '-vn',
                          '-af', f'volume={gain:.2f}dB,alimiter=limit=0.891:level=false',
                          '-c:a', 'pcm_s16le', out_path])
    return out_path


class AudioSubAgent:
    """Specialized Sub-Agent for speech recognition, noise cleaning, and acoustic enhancements."""

    @staticmethod
    def execute(tool_name: str, args: dict, src_file: str, processed_dir: str, context: dict = None) -> dict:
        base_name = uuid.uuid4().hex
        step_res = {"tool": tool_name, "subagent": "AudioSubAgent", "status": "success", "message": "", "output_file": src_file}

        if tool_name == "transcribe_audio":
            stt_res = ai_processor.transcribe_audio(src_file)
            if not isinstance(stt_res, dict) or not stt_res.get('available', False):
                raise StepUnavailableError('Speech transcription could not be completed.')
            step_res['data'] = {key: stt_res[key] for key in
                ('available', 'text', 'segments', 'language', 'duration') if key in stt_res}
            step_res['data']['text'] = stt_res.get('text') or stt_res.get('full_text') or ' '.join(
                segment.get('text', '') for segment in stt_res.get('segments', []))
            confidence = stt_res.get('avg_word_confidence')
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                step_res['data']['confidence'] = round(float(confidence), 3)
            step_res['data']['exports'] = _transcript_exports(stt_res, processed_dir, base_name)
            step_res["message"] = (f"Speech-to-text transcribed {len(stt_res.get('segments', []))} spoken segments."
                                   if step_res['data']['exports'] else 'No speech was detected. Source unchanged.')

        elif tool_name == 'adjust_audio_speed':
            speed = float(args.get('speed', 1.0))
            if not math.isfinite(speed) or not 0.25 <= speed <= 4:
                raise StepInputError('Choose a playback speed between 0.25x and 4x.')
            out_path = os.path.join(processed_dir, f'audio_speed_{base_name}.wav')
            ai_processor.pitch_preserved_speed(src_file, out_path, speed=speed)
            step_res['output_file'] = out_path
            step_res['message'] = f'Changed audio speed to {speed:g}x while preserving pitch.'

        elif tool_name == 'convert_audio_format':
            target_format = str(args.get('target_format', 'wav')).lower()
            if target_format not in ('mp3', 'wav', 'flac', 'ogg'):
                raise StepInputError('Choose MP3, WAV, FLAC, or OGG for audio export.')
            out_path = os.path.join(processed_dir, f'audio_export_{base_name}.{target_format}')
            bitrate = args.get('bitrate') or ('192k' if target_format == 'mp3' else None)
            if bitrate not in (None, '128k', '192k', '256k', '320k'):
                raise StepInputError('Choose 128k, 192k, 256k or 320k for the export quality.')
            _export_segment(_load_audio(src_file), out_path, bitrate if target_format in ('mp3', 'ogg') else None)
            step_res['output_file'] = out_path
            step_res['message'] = f'Exported audio as {target_format.upper()}.'

        elif tool_name == 'trim_audio':
            audio = _load_audio(src_file)
            total = len(audio) / 1000
            start = float(args.get('start_sec') or 0)
            end = args.get('end_sec')
            end = total if end is None else float(end)
            if total < end <= total + 0.05:
                end = total
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end > total:
                raise StepInputError('Choose a valid trim range within the audio duration.')
            out_path = os.path.join(processed_dir, f'trimmed_{base_name}.wav')
            _export_segment(audio[round(start * 1000):round(end * 1000)], out_path)
            step_res['output_file'] = out_path
            step_res['message'] = f'Trimmed audio from {start:g}s to {end:g}s.'

        elif tool_name == "reduce_noise":
            out_path = os.path.join(processed_dir, f"denoised_{base_name}.wav")
            r_start = args.get('range_start_sec')
            r_end = args.get('range_end_sec')
            if r_start is not None and r_end is not None:
                res_path = _apply_effect_with_range(
                    src_file, out_path, ai_processor.reduce_noise,
                    float(r_start), float(r_end), processed_dir)
                step_res["message"] = f"Removed background noise from {float(r_start):g}s to {float(r_end):g}s."
            else:
                res_path = ai_processor.reduce_noise(src_file, out_path)
                step_res["message"] = "Removed steady background noise."
            step_res["output_file"] = res_path

        elif tool_name == "auto_trim_silence":
            threshold = int(args.get("threshold", 40))
            trim_info = ai_processor.auto_trim_silence(src_file, threshold=threshold)
            start = float(trim_info['trimmed_start'])
            end = float(trim_info['trimmed_end'])
            if end <= start:
                raise StepInputError('No audible content was found to keep.')
            out_path = os.path.join(processed_dir, f"silence_trimmed_{base_name}.wav")
            audio = _load_audio(src_file)
            _export_segment(audio[round(start * 1000):round(end * 1000)], out_path)
            step_res['output_file'] = out_path
            step_res["message"] = "Removed leading and trailing silence."
            step_res["data"] = {key: trim_info[key] for key in trim_info if key in ('trimmed_start', 'trimmed_end', 'original_duration')}

        elif tool_name == "enhance_speech":
            out_path = os.path.join(processed_dir, f"speech_enhanced_{base_name}.wav")
            r_start = args.get('range_start_sec')
            r_end = args.get('range_end_sec')
            if r_start is not None and r_end is not None:
                _apply_effect_with_range(
                    src_file, out_path, ai_processor.enhance_speech_studio,
                    float(r_start), float(r_end), processed_dir)
                step_res["message"] = f"Enhanced speech clarity from {float(r_start):g}s to {float(r_end):g}s."
            else:
                ai_processor.enhance_speech_studio(src_file, out_path)
                step_res["message"] = "Applied studio speech clarity enhancement."
            step_res["output_file"] = out_path

        elif tool_name == "isolate_voice":
            step_res["output_file"] = _separate_stem(src_file, processed_dir, 'vocals')
            step_res["message"] = "Separated the voice from the background music."

        elif tool_name == "remove_vocals":
            try:
                step_res["output_file"] = _separate_stem(src_file, processed_dir, 'no_vocals')
                step_res["message"] = "Removed the vocals and kept the instrumental."
            except StepUnavailableError:
                audio = _load_audio(src_file)
                if audio.channels < 2:
                    raise StepInputError('Vocal removal needs a stereo recording on this machine.')
                left, right = audio.split_to_mono()[:2]
                instrumental = left.overlay(right.invert_phase())
                out_path = os.path.join(processed_dir, f"instrumental_{base_name}.wav")
                _export_segment(instrumental, out_path)
                step_res["output_file"] = out_path
                step_res["message"] = "Removed centre-panned vocals with stereo cancellation (lightweight method)."

        elif tool_name == "generate_voiceover":
            import tts_processor
            try:
                res = tts_processor.voiceover_tool(args, processed_dir)
            except ValueError as err:
                raise StepInputError(str(err)) from err
            step_res["output_file"] = res["output_file"]
            step_res["message"] = res.get("message") or "Voiceover ready."
            step_res["data"] = {k: v for k, v in (res.get("data") or {}).items() if k in ("duration", "tier", "notice")}

        elif tool_name == "separate_stems":
            import separation_processor
            fmt = str(args.get("format") or "wav").lower()
            try:
                with tempfile.TemporaryDirectory(dir=processed_dir, prefix="agent-stems-") as work_dir:
                    report = separation_processor.separate(src_file, work_dir, mode=args.get("mode") or "vocals",
                                                           quality=args.get("quality") or "auto", fmt=fmt)
                    outputs = []
                    for stem, produced in report["stems"].items():
                        out_path = os.path.join(processed_dir, f"{stem}_{base_name}.{fmt}")
                        shutil.move(produced, out_path)
                        outputs.append({"stem": stem, "label": report.get("stem_labels", {}).get(stem, stem),
                                        "output_file": out_path})
            except ValueError as err:
                raise StepInputError(str(err)) from err
            except separation_processor.SeparationUnavailableError as err:
                raise StepUnavailableError(str(err)) from err
            if not outputs:
                raise StepUnavailableError("Stem separation did not produce any files.")
            step_res["output_file"] = outputs[0]["output_file"]
            step_res["extra_outputs"] = outputs[1:]
            step_res["data"] = {"stems": [o["label"] for o in outputs],
                                "quality": (report.get("quality") or {}).get("label"),
                                "warnings": report.get("warnings", [])}
            step_res["message"] = (f"Separated {len(outputs)} stem{'s' if len(outputs) != 1 else ''}: "
                                   f"{', '.join(o['label'] for o in outputs)}.")

        elif tool_name == "extract_lyrics":
            import separation_processor
            fmt = str(args.get("format") or "srt").lower()
            try:
                res = separation_processor.lyrics(src_file)
            except ValueError as err:
                raise StepInputError(str(err)) from err
            except separation_processor.SeparationUnavailableError as err:
                raise StepUnavailableError(str(err)) from err
            content = {"srt": res["srt"], "vtt": res["vtt"], "txt": res["text"] + "\n"}[fmt]
            if not res["text"]:
                step_res["message"] = "No sung words were recognised. Source unchanged."
            else:
                filename = f"lyrics_{base_name}.{fmt}"
                with open(os.path.join(processed_dir, filename), "w", encoding="utf-8", newline="\n") as output:
                    output.write(content)
                step_res["data"] = {"text": res["text"], "language": res.get("language"),
                                    "exports": [{"format": fmt.upper(), "url": f"/processed/{filename}"}]}
                step_res["message"] = f"Extracted lyrics for {len(res['lines'])} lines as {fmt.upper()}."

        elif tool_name == "normalize_audio":
            out_path = os.path.join(processed_dir, f"normalized_{base_name}.wav")
            target_lufs, preset, target_dbfs = args.get('target_lufs'), args.get('preset'), args.get('target_dbfs')
            r_start = args.get('range_start_sec')
            r_end = args.get('range_end_sec')
            if target_lufs is not None:
                def _norm_lufs(i, o): return _normalize_loudness(i, o, float(target_lufs))
                if r_start is not None and r_end is not None:
                    res_path = _apply_effect_with_range(src_file, out_path, _norm_lufs, float(r_start), float(r_end), processed_dir)
                    step_res["message"] = f"Normalized loudness to {float(target_lufs):g} LUFS from {float(r_start):g}s to {float(r_end):g}s."
                else:
                    res_path = _norm_lufs(src_file, out_path)
                    step_res["message"] = f"Normalized loudness to {float(target_lufs):g} LUFS with true-peak protection."
            elif preset:
                import functools
                _norm_preset = functools.partial(audio_processor.normalize_audio, preset=preset)
                if r_start is not None and r_end is not None:
                    res_path = _apply_effect_with_range(src_file, out_path, _norm_preset, float(r_start), float(r_end), processed_dir)
                    step_res["message"] = f"Normalized for {agent_planner.PRESET_LABELS.get(preset, preset)} from {float(r_start):g}s to {float(r_end):g}s."
                else:
                    res_path = audio_processor.normalize_audio(src_file, out_path, preset=preset)
                    step_res["message"] = (f"Normalized loudness for {agent_planner.PRESET_LABELS.get(preset, preset)} "
                                           f"({agent_planner.LOUDNESS_PRESET_LUFS.get(preset, -14.0):g} LUFS) with true-peak protection.")
            elif target_dbfs is not None:
                import functools
                _norm_dbfs = functools.partial(audio_processor.normalize_audio, target_dbfs=float(target_dbfs))
                if r_start is not None and r_end is not None:
                    res_path = _apply_effect_with_range(src_file, out_path, _norm_dbfs, float(r_start), float(r_end), processed_dir)
                    step_res["message"] = f"Normalized peak to {float(target_dbfs):g} dBFS from {float(r_start):g}s to {float(r_end):g}s."
                else:
                    res_path = audio_processor.normalize_audio(src_file, out_path, target_dbfs=float(target_dbfs))
                    step_res["message"] = f"Normalized peak level to {float(target_dbfs):g} dBFS."
            else:
                if r_start is not None and r_end is not None:
                    res_path = _apply_effect_with_range(src_file, out_path, audio_processor.normalize_audio, float(r_start), float(r_end), processed_dir)
                    step_res["message"] = f"Normalized loudness from {float(r_start):g}s to {float(r_end):g}s."
                else:
                    res_path = audio_processor.normalize_audio(src_file, out_path)
                    step_res["message"] = "Normalized loudness to −14 LUFS with true-peak protection."
            step_res["output_file"] = res_path

        elif tool_name == "adjust_volume":
            gain = float(args.get('gain_db', 0.0))
            if not math.isfinite(gain) or not -30 <= gain <= 30 or gain == 0:
                raise StepInputError('Choose a volume change between -30 dB and +30 dB.')
            r_start = args.get('range_start_sec')
            r_end = args.get('range_end_sec')
            note = ''
            out_path = os.path.join(processed_dir, f"volume_{base_name}.wav")
            if r_start is not None and r_end is not None:
                def _vol_range(i, o, _gain=gain):
                    seg = _load_audio(i)
                    return _export_segment(seg.apply_gain(_gain), o)
                _apply_effect_with_range(src_file, out_path, _vol_range, float(r_start), float(r_end), processed_dir)
                step_res["message"] = (f"{'Raised' if gain > 0 else 'Lowered'} volume by {abs(gain):g} dB"
                                       f" from {float(r_start):g}s to {float(r_end):g}s.")
            else:
                audio = _load_audio(src_file)
                if gain > 0 and math.isfinite(audio.max_dBFS):
                    headroom = -audio.max_dBFS - 0.1
                    if headroom < gain:
                        if headroom < 0.5:
                            raise StepInputError('The audio already peaks at full scale; try "normalize loudness to -14 LUFS" instead.')
                        gain, note = round(headroom, 1), ' (limited to avoid clipping)'
                _export_segment(audio.apply_gain(gain), out_path)
                step_res["message"] = f"{'Raised' if gain > 0 else 'Lowered'} the volume by {abs(gain):g} dB{note}."
            step_res["output_file"] = out_path

        elif tool_name == "apply_audio_fade":
            fade_in = float(args.get("fade_in_sec", 2.0) or 0.0)
            fade_out = float(args.get("fade_out_sec", 2.0) or 0.0)
            if not (math.isfinite(fade_in) and math.isfinite(fade_out)) or fade_in < 0 or fade_out < 0 or fade_in + fade_out <= 0:
                raise StepInputError('Choose a fade length greater than zero.')
            r_start = args.get('range_start_sec')
            r_end = args.get('range_end_sec')
            out_path = os.path.join(processed_dir, f"faded_{base_name}.wav")
            parts = [f"{fade_in:g}s fade-in"] if fade_in else []
            parts += [f"{fade_out:g}s fade-out"] if fade_out else []
            if r_start is not None and r_end is not None:
                import functools
                _fade_fn = functools.partial(audio_processor.apply_fades, fade_in_sec=fade_in, fade_out_sec=fade_out)
                _apply_effect_with_range(src_file, out_path, _fade_fn, float(r_start), float(r_end), processed_dir)
                step_res["message"] = f"Applied {' and '.join(parts)} within {float(r_start):g}s–{float(r_end):g}s."
            else:
                res_path = audio_processor.apply_fades(src_file, out_path, fade_in, fade_out)
                step_res["message"] = f"Applied {' and '.join(parts)}."
            step_res["output_file"] = out_path

        else:
            step_res["status"] = "skipped"
            step_res["message"] = f"AudioSubAgent does not support tool: {tool_name}"

        return step_res


class InspectorSubAgent:
    """Specialized Sub-Agent for quality validation, preview metadata synthesis, and proactive next-steps."""

    @staticmethod
    def inspect_source(src_file, context=None):
        media_type = (context or {}).get('type')
        if media_type == 'image':
            with Image.open(src_file) as image:
                data = {'width': image.width, 'height': image.height}
            message = f"Image: {data['width']} × {data['height']} pixels. Source unchanged."
        elif media_type == 'audio':
            with open(src_file, 'rb') as source:
                audio = AudioSegment.from_file(source)
            peak = audio.max_dBFS
            data = {'duration': len(audio) / 1000, 'sample_rate': audio.frame_rate,
                    'channels': audio.channels, 'peak_db': round(peak, 2) if math.isfinite(peak) else None}
            peak_text = f'{peak:.2f} dBFS' if math.isfinite(peak) else 'silent'
            message = f"Audio: {data['duration']:g}s, {audio.frame_rate} Hz, {audio.channels} channels, peak {peak_text}. Source unchanged."
        else:
            data = video_processor.probe_media(src_file)
            message = f"Media: {data['duration']:g}s, {data['width']} × {data['height']} pixels. Source unchanged."
        return {'tool': 'inspect_media', 'status': 'success', 'output_file': src_file,
                'message': message, 'data': data}

    @staticmethod
    def inspect(execution_results: list, final_file: str) -> dict:
        metadata = {
            "has_media": bool(final_file and os.path.exists(final_file)),
            "media_type": "unknown",
            "suggested_actions": []
        }

        if not metadata["has_media"]:
            return metadata

        ext = os.path.splitext(final_file)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".webp"):
            metadata["media_type"] = "image"
            metadata["suggested_actions"] = [
                "Retry the cutout with portrait edges",
                "Upscale 4x",
                "Boost clarity"
            ]
        elif ext in (".mp4", ".webm", ".mkv", ".mov", ".gif"):
            metadata["media_type"] = "video"
            metadata["suggested_actions"] = [
                "Speed up 1.5x",
                "Extract the audio as MP3",
                "Make a GIF from the first 5 seconds"
            ]
        elif ext in (".mp3", ".wav", ".m4a", ".ogg", ".aac"):
            metadata["media_type"] = "audio"
            metadata["suggested_actions"] = [
                "Transcribe speech to text",
                "Remove the background noise",
                "Trim the silence"
            ]

        return metadata


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER ORCHESTRATOR AGENT
# ═══════════════════════════════════════════════════════════════════════════

def _reasoning_model_ladder():
    """Tiered local reasoning models (see reasoning_models.py), largest eligible first."""
    import reasoning_models
    return reasoning_models.reasoning_ladder()


def _request_reasoning_content(url, payload):
    """Ask the best available local reasoning tier; falls through tiers on failure."""
    import reasoning_models
    content, _label = reasoning_models.request_completion(payload)
    return content


def _copilot_memory_block(user_prompt, media_context, conversation_history, session_id, context_str):
    """
    Compact long-term memory (see agent_memory.build_context) sized to the
    reasoning context window; falls back to the raw recent history on any
    memory problem so chat never depends on it.
    """
    try:
        import agent_memory
        try:  # sized for the smallest served tier's budgeted prompt (slm_context)
            import slm_runtime
            budget = slm_runtime.memory_budget(context_str, user_prompt)
        except Exception:
            budget = agent_memory.prompt_budget(SYSTEM_PROMPT, f"{context_str}\nUser Request: {user_prompt}")
        block = agent_memory.build_context(session_id, user_prompt, media_context=media_context,
                                           token_budget=budget, recent_history=conversation_history)
        if block is not None:
            return f"\n{block}" if block else ""
    except Exception as err:
        _log.warning("Copilot Memory unavailable (%s); using recent history.", type(err).__name__)
    if not conversation_history:
        return ""
    recent = conversation_history[-6:]
    return "\nConversation History:\n" + "\n".join([f"{h.get('role', 'user').upper()}: {h.get('content', '')}" for h in recent])


def query_agent_orchestrator(user_prompt: str, media_context: dict = None, conversation_history: list = None,
                             session_id: str = None) -> dict:
    """
    Master Orchestrator entry point:
    1. Evaluates user intent via intelligent orchestrator model.
    2. Maintains session memory and previous edit context (Copilot Memory, budgeted for small context windows).
    3. Seamlessly falls back to resilient rule-based semantic parsing when the local reasoning service is offline.
    """
    skill_plan = _skill_plan(user_prompt, media_context, conversation_history)
    if skill_plan is not None:
        return skill_plan
    url = _local_reasoning_url()
    if not url:
        _log.warning("Non-local reasoning configuration rejected; using local intent routing.")
        return _fallback_intent_parser(user_prompt, media_context, conversation_history)
    # Small-model runtime (slm_runtime.py): routing, budgeted prompt, structured output, repair, escalation.
    try:
        import slm_runtime
        slm_session = slm_runtime.PlanningSession(
            user_prompt, media_context, conversation_history,
            lambda: _fallback_intent_parser(user_prompt, media_context, conversation_history))
    except Exception as err:
        _log.warning("Reasoning runtime unavailable (%s); using the legacy prompt.", type(err).__name__)
        slm_session = None
    if slm_session is not None and slm_session.skip_llm and slm_session.local_plan:
        return slm_session.local_plan
    context_str = _llm_media_context(media_context)
    memory_str = _copilot_memory_block(user_prompt, media_context, conversation_history, session_id, context_str)
    skills_hint = agent_skills.prompt_hint(user_prompt, (media_context or {}).get('type') if isinstance(media_context, dict) else None)
    history_str = memory_str + skills_hint

    full_user_content = f"{context_str}{history_str}\nUser Request: {user_prompt}"

    payload = {
        "model": REASONING_MODEL_ID,  # replaced per attempt by the adaptive ladder
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_user_content}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    if slm_session is not None:
        try:
            payload = slm_session.payload(context_str, memory_str.strip(), skills_hint, payload)
        except Exception as err:
            _log.warning("Budgeted prompt unavailable (%s); using the legacy prompt.", type(err).__name__)
            slm_session = None

    try:
        if slm_session is not None:
            content = slm_session.complete(lambda body: _request_reasoning_content(url, body))
        else:
            content = _request_reasoning_content(url, payload)
    except Exception:
        _log.warning("Reasoning service unavailable; using local intent routing.")
        return _fallback_intent_parser(user_prompt, media_context, conversation_history)
    return _plan_from_reasoning(content, user_prompt, media_context, conversation_history)


def _plan_from_reasoning(content, user_prompt, media_context=None, conversation_history=None):
    """
    Turn reasoning-service output into a validated plan. The output may contain
    reasoning tags, fences or prose; it is never trusted or executed as-is.
    """
    local_plan = {}

    def local():
        if not local_plan:
            local_plan.update(_fallback_intent_parser(user_prompt, media_context, conversation_history))
        return local_plan

    raw, leftover = agent_planner.extract_json_object(content if isinstance(content, str) else '')
    if raw is None:
        text = agent_planner.scrub_private_terms(leftover or '').strip()
        fallback = local()
        if not text or '{' in text or '"tools"' in text:
            return fallback
        _log.info("Reasoning service returned a conversational reply.")
        return {**fallback, 'thought': 'Conversational reply / media knowledge inquiry', 'tools': [],
                'reply': text[:2000], 'plan_source': 'reasoning', 'clarification_needed': False}

    perception = _perception_for(media_context)
    raw = agent_skills.expand_model_steps(raw, media_context)
    plan = _apply_job_memory(agent_planner.finalize_plan(raw, user_prompt, media_context, perception, source='reasoning'),
                             user_prompt, perception)
    proposed = agent_planner.normalize_raw_plan(raw)['tools']
    fallback = local()
    if plan.get('clarification_needed'):
        return plan
    if not plan.get('tools'):
        if not plan.get('reply') and (fallback.get('tools') or fallback.get('clarification_needed')):
            if proposed:
                _log.info("Reasoning plan did not validate; using local intent routing.")
            return fallback
        plan['suggested_actions'] = fallback.get('suggested_actions') or plan.get('suggested_actions') or []
        plan['reply'] = plan.get('reply') or fallback.get('reply', '')
    _log.info("AI orchestration plan prepared and validated (%s steps).", len(plan['tools']))
    return plan


def _fallback_intent_parser(prompt: str, media_context: dict = None, conversation_history: list = None) -> dict:
    """
    Local natural-language planner (used whenever the reasoning service is offline).
    Decomposes multi-operation requests, then runs the same validation, repair and
    condition-aware policies as reasoning-service plans.
    """
    skill_plan = _skill_plan(prompt, media_context, conversation_history)
    if skill_plan is not None:
        return skill_plan
    perception = _perception_for(media_context)
    try:
        raw = agent_nlu.parse_request(prompt, media_context, conversation_history)
    except Exception as err:
        _log.error("Local intent routing failed (%s).", type(err).__name__)
        raw = {'tools': [], 'clarification_needed': True,
               'reply': 'I could not understand that request. Could you rephrase the edit you want?',
               'clarification_options': agent_planner.SUGGESTIONS_BY_TYPE[None]}
    plan = agent_planner.finalize_plan(raw, prompt, media_context, perception, source='local')
    return _apply_job_memory(plan, prompt, perception)


def _skill_plan(prompt, media_context=None, conversation_history=None):
    """Validated plan for messages that use skills ("@id args" or an automatic trigger); None otherwise."""
    try:
        raw = agent_skills.build_raw_plan(prompt, media_context, conversation_history,
                                          parse_text=lambda text, context, history: agent_nlu.parse_request(text, context, history))
    except Exception as err:
        _log.error("Skill planning failed (%s); using regular planning.", type(err).__name__)
        return None
    if raw is None:
        return None
    if not raw.get('tools'):
        return agent_planner.finalize_plan(raw, prompt, media_context, None, source='skill')
    perception = _perception_for(media_context)
    plan = agent_planner.finalize_plan(raw, prompt, media_context, perception, source='skill')
    if raw.get('auto_skill') and plan.get('tools'):
        plan['auto_skill'] = raw['auto_skill']
        plan['suggested_actions'] = [f"{prompt.strip()} (no skill)"] + [
            action for action in plan.get('suggested_actions', []) if action != f"{prompt.strip()} (no skill)"]
    return plan


def _apply_job_memory(plan, prompt, perception):
    """
    Optional learning from past jobs on similar media (never overrides the user's own steps):
    a condition-inserted preparation step that repeatedly failed on similar media is replaced by
    its planned fallback up front; a previously successful similar plan is noted in the reasoning.
    """
    if not plan.get('tools') or plan.get('clarification_needed'):
        return plan
    try:
        import agent_memory
        similar = agent_memory.recall_similar_jobs(prompt, perception, 3) or []
    except Exception:
        return plan
    for job in similar:
        if not isinstance(job, dict):
            continue
        if job.get('recommendation') == 'avoid' and job.get('failed_tool'):
            for step in plan['tools']:
                fallback = step.get('fallback')
                if (step.get('role') == 'prep' and step.get('name') == job['failed_tool'] and isinstance(fallback, dict)
                        and fallback.get('name') in agent_planner.TOOL_REGISTRY):
                    failed_label = agent_planner.TOOL_REGISTRY[step['name']].label.lower()
                    step['name'], step['args'] = fallback['name'], dict(fallback.get('args') or {})
                    step.pop('fallback', None)
                    note = (f"Using {agent_planner.TOOL_REGISTRY[step['name']].label.lower()} directly — "
                            f"{failed_label} did not work on similar media before.")
                    step['reason'] = f"{step.get('reason', '')} {note}".strip()
                    plan.setdefault('plan_notes', []).append(note)
        elif job.get('recommendation') == 'reuse' and job.get('runs'):
            plan['thought'] = (plan.get('thought', '') + f" | A similar plan succeeded on similar media before "
                               f"({job.get('successes', 0)}/{job.get('runs')} runs).").strip(' |')
            break
    return plan



# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-AGENT EXECUTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif')
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.aiff')
VIDEO_EXTENSIONS = ('.webm', '.mp4', '.mov', '.mkv', '.avi', '.wmv', '.flv', '.m4v')
VISION_TOOLS = {"remove_background", "upscale_image", "enhance_photo_clarity", "restore_faces", "convert_image_format"}
VIDEO_TOOLS = {"trim_video", "adjust_video_speed", "extract_audio", "convert_video_format", "compress_video",
               "mute_video", "extract_frame", "enhance_video"}
AUDIO_TOOLS = {"adjust_audio_speed", "convert_audio_format", "trim_audio", "transcribe_audio", "reduce_noise",
               "auto_trim_silence", "enhance_speech", "normalize_audio", "apply_audio_fade", "adjust_volume",
               "isolate_voice", "remove_vocals", "generate_voiceover", "separate_stems", "extract_lyrics"}
AUDIO_OUTPUT_TOOLS = (AUDIO_TOOLS - {"transcribe_audio", "remove_vocals", "extract_lyrics",
                                         "generate_voiceover", "separate_stems"}) | {"extract_audio"}
NON_RETRYABLE_ERRORS = (ValueError, TypeError, KeyError, NotImplementedError, FileNotFoundError,
                        PermissionError, StepUnavailableError)
ENHANCEMENT_TOOLS = {'reduce_noise', 'enhance_speech', 'isolate_voice'}
LOW_TRANSCRIPT_CONFIDENCE = 0.45


def _media_context_for_file(path, context=None):
    resolved = dict(context or {})
    extension = os.path.splitext(path)[1].lower()
    resolved['_path'] = path
    if extension in IMAGE_EXTENSIONS:
        resolved['type'] = 'image'
        resolved['has_audio'] = False
        try:
            with Image.open(path) as image:
                resolved['width'], resolved['height'] = image.size
        except Exception:
            pass
    elif extension in AUDIO_EXTENSIONS:
        resolved['type'] = 'audio'
        resolved['has_audio'] = True
        try:
            duration = video_processor.probe_media(path).get('duration')
            if duration and math.isfinite(duration):
                resolved['duration'] = duration
        except Exception:
            pass
    elif extension in VIDEO_EXTENSIONS:
        metadata = video_processor.probe_media(path)
        resolved.update(metadata)
        resolved['type'] = 'video' if metadata['has_video'] else 'audio'
    return resolved


def _remux_soundtrack(src_file, edited_audio, processed_dir):
    extension = os.path.splitext(src_file)[1].lower()
    extension = extension if extension in ('.mp4', '.mov') else '.mkv'
    output = os.path.join(processed_dir, f'soundtrack_edited_{uuid.uuid4().hex}{extension}')
    video_processor._run([
        video_processor.FFMPEG, '-y', '-i', src_file, '-i', edited_audio,
        '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '0',
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', output,
    ])
    return output


def _apply_video_audio_effect(tool_name, args, src_file, processed_dir, context):
    """Single soundtrack effect on a video (kept for compatibility; chains use _run_soundtrack_chain)."""
    outcomes = _run_soundtrack_chain([{'name': tool_name, 'args': args}], src_file, processed_dir, context)
    result, error = outcomes[0]
    if error:
        raise error
    return result


def _run_soundtrack_chain(steps, src_file, processed_dir, context, runner=None):
    """
    Fused soundtrack editing: extract the soundtrack once, run every audio step on
    lossless WAV, then remux once (one lossy encode instead of one per step).
    Returns [(result|None, error|None)] per step; steps after a failure are not run.
    """
    if not video_processor.probe_media(src_file)['has_audio']:
        raise StepInputError('The selected video has no soundtrack to edit.')
    runner = runner or (lambda step, source, work_dir: AudioSubAgent.execute(
        step['name'], step['args'], source, work_dir, {**(context or {}), 'type': 'audio'}))
    outcomes = []
    with tempfile.TemporaryDirectory(dir=processed_dir, prefix='agent-audio-') as work_dir:
        current = os.path.join(work_dir, 'soundtrack.wav')
        video_processor.extract_audio(src_file, current, fmt='wav')
        for step in steps:
            try:
                result = runner(step, current, work_dir)
                edited = result.get('output_file')
                if (result.get('status') != 'success' or not edited or not os.path.isfile(edited)
                        or not os.path.getsize(edited) or os.path.realpath(edited) == os.path.realpath(current)):
                    raise RuntimeError('The soundtrack edit did not produce a usable result.')
                current = edited
                outcomes.append((result, None))
            except Exception as err:
                outcomes.append((None, err))
                break
        if any(result for result, _ in outcomes):
            output = _remux_soundtrack(src_file, current, processed_dir)
            for result, _ in outcomes:
                if result:
                    result['output_file'] = output
                    result['message'] = result.get('message', '') + ' Original video preserved.'
    return outcomes


def _dispatch(tool_name, args, src_file, processed_dir, context):
    if tool_name in VISION_TOOLS:
        return VisionSubAgent.execute(tool_name, args, src_file, processed_dir, context)
    if tool_name in VIDEO_TOOLS:
        return VideoSubAgent.execute(tool_name, args, src_file, processed_dir, context)
    if tool_name in AUDIO_TOOLS:
        if context.get('type') == 'video' and tool_name in agent_planner.SOUNDTRACK_EFFECTS:
            return _apply_video_audio_effect(tool_name, args, src_file, processed_dir, context)
        return AudioSubAgent.execute(tool_name, args, src_file, processed_dir, context)
    if tool_name == 'inspect_media':
        return InspectorSubAgent.inspect_source(src_file, context)
    raise StepInputError('The requested edit action is not supported.')


def _strict_check(step, context):
    """Deterministic pre-execution validation; raises StepInputError with a user-safe message."""
    name = step.get('name')
    spec = agent_planner.TOOL_REGISTRY.get(name)
    if not spec or not isinstance(step.get('args'), dict):
        raise StepInputError('The requested edit action is not supported.' if not spec or name is None
                             else 'The edit plan contains an invalid action.')
    media_type = context.get('type')
    if media_type and media_type not in spec.accepts:
        raise StepInputError(f'{spec.label} works on {" or ".join(spec.accepts)}, but this step received {media_type}.')
    if spec.needs_audio and context.get('has_audio') is False:
        raise StepInputError(f'{spec.label} needs a soundtrack, but this {media_type or "media"} has none.')
    state = agent_planner.MediaState(media_type, context.get('has_audio'))
    try:
        args, _ = agent_planner.validate_args(spec, step['args'], state, strict=True)
    except agent_planner.Clarify as clarify:
        raise StepInputError(clarify.question)
    except ValueError as err:
        raise StepInputError(str(err))
    return args


def _verify_output(name, result, input_file, context):
    """Post-step quality gate: output exists, is readable, and audio did not turn silent."""
    output = result.get('output_file')
    if not output or not os.path.isfile(output) or os.path.getsize(output) == 0:
        raise RuntimeError('The media action did not produce a usable result.')
    if name in ('transcribe_audio', 'inspect_media', 'extract_lyrics'):
        return
    if os.path.realpath(output) == os.path.realpath(input_file):
        raise RuntimeError('The requested edit did not produce a new result.')
    extension = os.path.splitext(output)[1].lower()
    if extension in IMAGE_EXTENSIONS:
        try:
            with Image.open(output) as image:
                image.verify()
        except Exception:
            raise StepInputError('The edit produced an unreadable image, so I stopped before continuing.')
    elif name in AUDIO_OUTPUT_TOOLS and extension in AUDIO_EXTENSIONS:
        peak = media_inspector.audio_peak_db(output)
        if peak is not None and peak < media_inspector.SILENT_PEAK_DB:
            source_peak = media_inspector.audio_peak_db(input_file)
            if source_peak is not None and source_peak > media_inspector.SILENT_PEAK_DB + 10:
                raise StepInputError('The edit produced silent audio, so I stopped before continuing.')
    elif name == 'mute_video':
        if video_processor.probe_media(output).get('has_audio'):
            raise RuntimeError('The soundtrack could not be removed.')


def _transcript_needs_retry(result):
    data = result.get('data') or {}
    if not (data.get('text') or '').strip():
        return True
    confidence = data.get('confidence')
    return isinstance(confidence, (int, float)) and confidence < LOW_TRANSCRIPT_CONFIDENCE


_PATH_IN_TEXT = re.compile(r'''(?:[A-Za-z]:)?[\\/][^\s"'<>]*[\\/][^\s"'<>]*''')


def _public_error_message(err):
    """App-authored validation messages are shown (scrubbed of private names and file paths); others stay generic."""
    if isinstance(err, (StepInputError, StepUnavailableError)) or (isinstance(err, ValueError) and str(err)):
        text = str(err).splitlines()[0] if str(err) else ''
        text = _PATH_IN_TEXT.sub('the file', agent_planner.scrub_private_terms(text)).strip()
        if text:
            return text[:300]
    return 'The media action could not be completed. Check the source file and try again.'


def execute_agent_plan(file_path: str, tools_list: list, processed_dir: str, context: dict = None) -> list:
    """
    Execute a plan as a dependency graph:
      * each step reads the output of its ``input`` step (default: the previous edit),
      * every step is strictly re-validated against the real media before it runs,
      * transient failures are retried once; planned fallbacks run when a capability is missing,
      * a failure blocks only the steps that depend on it — independent branches still run,
      * consecutive soundtrack edits on a video are fused (extract once, remux once),
      * outputs are verified, and an empty/low-confidence transcript is retried once with voice enhancement.
    Returns the attempted steps (success or error) with per-step status, timing and attempts.
    """
    results = []
    os.makedirs(processed_dir, exist_ok=True)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input media file not found: {file_path}")

    steps = agent_planner.execution_steps(tools_list)
    outputs = {'source': file_path}
    lineage = {'source': []}
    status = {}
    root_failure = {}
    failure_result = {}
    main_current = file_path
    index = 0

    def resolve_input(step):
        ref = step['input']
        return outputs.get(ref) if ref == 'source' or status.get(ref) == 'success' else None

    def record_error(step, err, started, attempts, current_main):
        tool_name = step.get('name')
        _log.error("Media action failed: %s (%s)", tool_name, type(err).__name__)
        res = {"tool": tool_name, "status": "error", "message": _public_error_message(err),
               "output_file": current_main, "step": step['index'], "id": step['id'],
               "duration_ms": round((time.perf_counter() - started) * 1000), "attempts": attempts,
               "branch": step['branch'], "role": step['role'], "blocked_steps": []}
        status[step['id']] = 'error'
        root_failure[step['id']] = step['id']
        failure_result[step['id']] = res
        results.append(res)

    while index < len(steps):
        step = steps[index]
        tool_name = step.get('name')
        started = time.perf_counter()
        _log.info("Dispatching media action %s of %s: %s", step['index'], len(steps), tool_name)

        source = resolve_input(step)
        if source is None:
            cause = root_failure.get(step['input'], step['input'])
            status[step['id']] = 'blocked'
            root_failure[step['id']] = cause
            if cause in failure_result:
                failure_result[cause]['blocked_steps'].append({'step': step['index'], 'tool': tool_name})
            index += 1
            continue

        try:
            step_context = _media_context_for_file(source, context)
            args = _strict_check(step, step_context)
        except Exception as err:
            record_error(step, err, started, 0, main_current)
            index += 1
            continue

        # ── Fused soundtrack chain on video ──
        if step_context.get('type') == 'video' and tool_name in agent_planner.SOUNDTRACK_EFFECTS:
            group = [(step, args)]
            probe = index + 1
            while probe < len(steps):
                candidate = steps[probe]
                if (candidate.get('name') in agent_planner.SOUNDTRACK_EFFECTS and candidate['input'] == group[-1][0]['id']
                        and not candidate['branch'] and candidate['role'] == group[-1][0]['role']):
                    try:
                        group.append((candidate, _strict_check(candidate, step_context)))
                    except Exception:
                        break
                    probe += 1
                else:
                    break

            def run_one(chain_step, chain_source, work_dir, _args_by_id={s['id']: a for s, a in group}):
                return _run_with_retry(chain_step['name'], _args_by_id[chain_step['id']], chain_source, work_dir,
                                       {**step_context, 'type': 'audio', '_path': chain_source}, chain_step)[0]
            try:
                outcomes = _run_soundtrack_chain([member for member, _ in group], source, processed_dir, step_context,
                                                 runner=run_one)
            except Exception as err:
                outcomes = [(None, err)]
            for (member, member_args), (result, error) in zip(group, outcomes):
                if error is not None:
                    record_error(member, error, started, 1, main_current)
                    continue
                result.update({'step': member['index'], 'id': member['id'], 'attempts': result.get('attempts', 1),
                               'duration_ms': round((time.perf_counter() - started) * 1000),
                               'branch': member['branch'], 'role': member['role']})
                if member.get('reason'):
                    result['reason'] = member['reason']
                    result['message'] = f"{member['reason']} {result.get('message', '')}".strip()
                outputs[member['id']] = result['output_file']
                lineage[member['id']] = lineage.get(member['input'], []) + [member['name']]
                status[member['id']] = 'success'
                if not member['branch']:
                    main_current = result['output_file']
                results.append(result)
            index += max(1, min(len(outcomes), len(group)))
            continue

        try:
            res, attempts = _run_with_retry(tool_name, args, source, processed_dir, step_context, step)
            if tool_name == 'transcribe_audio' and _transcript_needs_retry(res) \
                    and not set(lineage.get(step['input'], [])) & ENHANCEMENT_TOOLS:
                res = _adaptive_transcription_retry(res, source, processed_dir, step_context)
        except Exception as err:
            record_error(step, err, started, getattr(err, 'agent_attempts', 1), main_current)
            index += 1
            continue

        spec = agent_planner.TOOL_REGISTRY[tool_name]
        produced = res.get('output_file')
        if spec.kind == 'analysis':
            outputs[step['id']] = source
            res['output_file'] = main_current
        else:
            outputs[step['id']] = produced
        lineage[step['id']] = lineage.get(step['input'], []) + [tool_name]
        status[step['id']] = 'success'
        if spec.kind == 'edit' and not step['branch']:
            main_current = produced
        res.update({'step': step['index'], 'id': step['id'], 'attempts': attempts,
                    'duration_ms': round((time.perf_counter() - started) * 1000),
                    'branch': step['branch'], 'role': step['role']})
        if step.get('reason'):
            res['reason'] = step['reason']
            res['message'] = f"{step['reason']} {res.get('message', '')}".strip()
        results.append(res)

        # Shift transcript timestamps when a time-altering edit precedes or follows transcription.
        if tool_name in ('trim_audio', 'trim_video') and spec.kind == 'edit':
            trim_start = float(args.get('start_sec') or 0.0)
            for prior in results:
                if prior.get('tool') == 'transcribe_audio' and isinstance((prior.get('data') or {}).get('segments'), list):
                    prior['data']['segments'] = [
                        {**seg, 'start': max(0.0, seg['start'] - trim_start),
                         'end': max(0.0, seg['end'] - trim_start)}
                        for seg in prior['data']['segments']
                        if seg.get('end', 0) > trim_start
                    ]
        elif tool_name in ('adjust_audio_speed',) and spec.kind == 'edit':
            speed = float(args.get('speed') or 1.0)
            if speed and speed != 1.0:
                for prior in results:
                    if prior.get('tool') == 'transcribe_audio' and isinstance((prior.get('data') or {}).get('segments'), list):
                        prior['data']['segments'] = [
                            {**seg, 'start': seg['start'] / speed, 'end': seg['end'] / speed}
                            for seg in prior['data']['segments']
                        ]
        index += 1

    return results


def _run_with_retry(tool_name, args, source, processed_dir, context, step):
    """Run one step; retry once on a transient failure, then try the planned fallback (if any)."""
    attempts = 0
    last_error = None
    for _ in range(2):
        attempts += 1
        try:
            res = _dispatch(tool_name, args, source, processed_dir, context)
            if res.get('status') != 'success':
                raise RuntimeError('The media action did not complete.')
            _verify_output(tool_name, res, source, context)
            res['attempts'] = attempts
            return res, attempts
        except NON_RETRYABLE_ERRORS as err:
            last_error = err
            break
        except Exception as err:
            last_error = err
            _log.warning("Media action %s failed (attempt %s); %s.", tool_name, attempts,
                         'retrying once' if attempts == 1 else 'giving up')
    fallback = step.get('fallback') if isinstance(step, dict) else None
    if fallback and agent_planner.TOOL_REGISTRY.get(fallback.get('name')) and not isinstance(last_error, StepInputError):
        fallback_args = fallback.get('args') if isinstance(fallback.get('args'), dict) else {}
        attempts += 1
        try:
            res = _dispatch(fallback['name'], fallback_args, source, processed_dir, context)
        except Exception:
            res = {}
        if res.get('status') != 'success':
            _tag_attempts(last_error, attempts)
            raise last_error
        _verify_output(fallback['name'], res, source, context)
        label = agent_planner.TOOL_REGISTRY[tool_name].label.lower()
        res['message'] = (f"Full {label} is not available here, so I used "
                          f"{agent_planner.TOOL_REGISTRY[fallback['name']].label.lower()} instead. "
                          + res.get('message', ''))
        res['adapted'] = True
        res['tool'] = tool_name
        res['attempts'] = attempts
        return res, attempts
    _tag_attempts(last_error, attempts)
    raise last_error


def _tag_attempts(err, attempts):
    try:
        err.agent_attempts = attempts
    except Exception:
        pass


def _adaptive_transcription_retry(first, source, processed_dir, context):
    """Empty or low-confidence transcript: enhance the voice on a copy and transcribe once more (bounded)."""
    condition = media_inspector.perceive(source, context.get('type')) or {}
    audio = condition.get('audio') or {}
    if audio.get('silent') or (audio and not audio.get('speech_likely')):
        return first
    try:
        with tempfile.TemporaryDirectory(dir=processed_dir, prefix='agent-stt-') as work_dir:
            working = source
            if context.get('type') == 'video':
                working = os.path.join(work_dir, 'soundtrack.wav')
                video_processor.extract_audio(source, working, fmt='wav')
            enhanced = AudioSubAgent.execute('enhance_speech', {}, working, work_dir, {**context, 'type': 'audio'})
            second = AudioSubAgent.execute('transcribe_audio', {}, enhanced['output_file'], processed_dir,
                                           {**context, 'type': 'audio'})
    except Exception as err:
        _log.warning("Adaptive transcription retry skipped (%s).", type(err).__name__)
        return first
    first_text = ((first.get('data') or {}).get('text') or '').strip()
    second_text = ((second.get('data') or {}).get('text') or '').strip()
    better = (second_text and not first_text) or (
        second_text and ((second.get('data') or {}).get('confidence') or 0) > ((first.get('data') or {}).get('confidence') or 0))
    if not better:
        first['message'] = first.get('message', '') + ' A second pass with voice enhancement did not improve it.'
        return first
    second['output_file'] = source
    second['adapted'] = True
    second['message'] = ('Speech was unclear on the first pass, so I enhanced the voice and transcribed again. '
                         + second.get('message', ''))
    return second


# ═══════════════════════════════════════════════════════════════════════════
#  RESULT SUMMARY (used by the chat route)
# ═══════════════════════════════════════════════════════════════════════════

def final_output_of(execution_results):
    """Main-chain result: the last successful non-side step (side outputs such as thumbnails are artifacts)."""
    successes = [step for step in execution_results or [] if step.get('status') == 'success' and step.get('output_file')]
    for step in reversed(successes):
        if not step.get('branch'):
            return step['output_file']
    return successes[-1]['output_file'] if successes else None


def side_outputs_of(execution_results):
    """Separate deliverables (e.g. thumbnails) the user asked for alongside the main result."""
    final = final_output_of(execution_results)
    return [step for step in execution_results or []
            if step.get('status') == 'success' and step.get('branch') and step.get('role') != 'prep'
            and step.get('output_file') and step['output_file'] != final]


def _step_label(name):
    spec = agent_planner.TOOL_REGISTRY.get(name)
    return spec.label if spec else 'Unsupported action'


def execution_reply(plan, execution_results, final_output_file):
    """Plain-language outcome: what finished, what failed and why, what was skipped, plus plan notes."""
    plan = plan if isinstance(plan, dict) else {}
    notes = [note for note in plan.get('plan_notes') or [] if isinstance(note, str)]
    planned_reply = plan.get('reply') if isinstance(plan.get('reply'), str) else 'Action completed.'
    if not execution_results:
        return planned_reply or 'Action completed.'
    completed = [step.get('message', '') for step in execution_results if step.get('status') == 'success']
    failed = [step for step in execution_results if step.get('status') == 'error']
    if not failed:
        return '\n'.join([line for line in completed if line] + notes)
    reasons = '; '.join(f"{_step_label(step.get('tool'))}: {step.get('message', '').rstrip('.')}" for step in failed[:2])
    skipped = [_step_label(item.get('tool')) for step in failed for item in step.get('blocked_steps') or []]
    if final_output_file:
        lines = [f'The edit chain stopped because an action failed ({reasons}).']
        if skipped:
            lines.append(f"Skipped the steps that depended on it: {', '.join(skipped)}.")
        lines.append('The last completed result is available.')
        lines += [line for line in completed if line]
    else:
        lines = [f'The edit could not be completed ({reasons}). Check the source media and try again.']
        if skipped:
            lines.append(f"Not run: {', '.join(skipped)}.")
    return '\n'.join(lines + notes)
