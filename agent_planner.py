"""
Agent Planner — tool registry, request decomposition, plan validation/repair and
condition-aware policies for the media assistant.

Pipeline (identical for the local reasoning service and the rule-based parser):

    raw plan  ──►  normalize structure  ──►  validate + repair each step while
    (LLM or        (types, aliases)          tracking media state (type, soundtrack,
     parser)                                  duration, measured condition)
                                         ──►  condition policies insert / skip
                                              preparatory steps with reasons
                                         ──►  explicit ids, inputs, dependencies
                                         ──►  validated plan (never executed raw)

Everything user-facing here uses capability language; internal technology
names never appear in descriptions, replies, notes or reasons.
"""

import copy
import json
import math
import re

# ═══════════════════════════════════════════════════════════════════════════
#  TOOL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

MEDIA_TYPES = ('image', 'audio', 'video')
SOUNDTRACK_EFFECTS = {'reduce_noise', 'normalize_audio', 'apply_audio_fade', 'enhance_speech',
                      'adjust_volume', 'isolate_voice', 'remove_vocals'}


class Param:
    __slots__ = ('type', 'default', 'required', 'minimum', 'maximum', 'enum', 'unit',
                 'description', 'nullable', 'out_of_range', 'internal')

    def __init__(self, type, default=None, required=False, minimum=None, maximum=None, enum=None,
                 unit=None, description='', nullable=False, out_of_range='clamp', internal=False):
        self.type = type
        self.default = default
        self.required = required
        self.minimum = minimum
        self.maximum = maximum
        self.enum = tuple(enum) if enum else None
        self.unit = unit
        self.description = description
        self.nullable = nullable
        self.out_of_range = out_of_range  # 'clamp' | 'clarify' | 'nearest'
        self.internal = internal


class ToolSpec:
    __slots__ = ('name', 'agent', 'label', 'description', 'accepts', 'output', 'kind',
                 'needs_audio', 'params', 'example')

    def __init__(self, name, agent, label, description, accepts, output='same', kind='edit',
                 needs_audio=False, params=None, example=''):
        self.name = name
        self.agent = agent
        self.label = label
        self.description = description
        self.accepts = tuple(accepts)
        self.output = output          # 'same' | media type | callable(args) -> media type
        self.kind = kind              # 'edit' (advances the chain) | 'analysis' | 'branch' (side output)
        self.needs_audio = needs_audio
        self.params = params or {}
        self.example = example

    def output_type(self, args, current):
        if callable(self.output):
            return self.output(args)
        return current if self.output == 'same' else self.output


_TIME = 'seconds'
TOOL_SPECS = [
    ToolSpec('inspect_media', 'InspectorSubAgent', 'Inspect media',
             'Read media dimensions, duration, loudness and audio properties without changing the source.',
             MEDIA_TYPES, kind='analysis', example='Inspect this file'),
    ToolSpec('remove_background', 'VisionSubAgent', 'Remove background',
             'Create a clean transparent subject cutout with optional edge refinement.', ('image',),
             params={'quality_profile': Param('string', 'detail', enum=('detail', 'portrait', 'studio', 'fast'),
                                              description='detail = general subjects, portrait = people'),
                     'refine': Param('boolean', False, description='True when the user asks for a cleaner retry')},
             example='Remove the background'),
    ToolSpec('upscale_image', 'VisionSubAgent', 'Upscale',
             'Super-resolution enlargement (2x or 4x) with learned detail reconstruction.', ('image',),
             params={'scale': Param('integer', 2, enum=(2, 4), out_of_range='nearest', unit='x')},
             example='Upscale 2x'),
    ToolSpec('enhance_photo_clarity', 'VisionSubAgent', 'Boost clarity',
             'Photo clarity polish: grain reduction, edge sharpening and adaptive local contrast.', ('image',),
             params={'denoise_strength': Param('integer', 5, minimum=0, maximum=20),
                     'sharpen_strength': Param('number', 1.2, minimum=0.0, maximum=3.0)},
             example='Boost clarity'),
    ToolSpec('restore_faces', 'VisionSubAgent', 'Restore faces',
             'Portrait face detail restoration and gentle skin smoothing.', ('image',), example='Restore faces'),
    ToolSpec('convert_image_format', 'VisionSubAgent', 'Export image',
             'Export the image as PNG, JPG or WebP.', ('image',),
             params={'target_format': Param('string', None, required=True, enum=('png', 'jpg', 'webp'))},
             example='Export as PNG'),
    ToolSpec('trim_video', 'VideoSubAgent', 'Trim video',
             'Keep only the part of the video between start_sec and end_sec (sub-second precision).', ('video',),
             params={'start_sec': Param('number', 0.0, minimum=0.0, unit=_TIME),
                     'end_sec': Param('number', None, minimum=0.0, unit=_TIME, nullable=True,
                                      description='omit or null = until the end'),
                     'keep_last_sec': Param('number', None, minimum=0.0, unit=_TIME, nullable=True, internal=True),
                     'drop_last_sec': Param('number', None, minimum=0.0, unit=_TIME, nullable=True, internal=True)},
             example='Trim from 2s to 8s'),
    ToolSpec('trim_audio', 'AudioSubAgent', 'Trim audio',
             'Keep only the part of the audio between start_sec and end_sec.', ('audio',),
             params={'start_sec': Param('number', 0.0, minimum=0.0, unit=_TIME),
                     'end_sec': Param('number', None, minimum=0.0, unit=_TIME, nullable=True),
                     'keep_last_sec': Param('number', None, minimum=0.0, unit=_TIME, nullable=True, internal=True),
                     'drop_last_sec': Param('number', None, minimum=0.0, unit=_TIME, nullable=True, internal=True)},
             example='Trim from 2s to 8s'),
    ToolSpec('adjust_video_speed', 'VideoSubAgent', 'Change video speed',
             'Speed up or slow down the video (0.25x to 4x; below 1 is slower).', ('video',),
             params={'speed': Param('number', None, required=True, minimum=0.25, maximum=4.0, unit='x',
                                    out_of_range='clarify')},
             example='Speed up 1.5x'),
    ToolSpec('adjust_audio_speed', 'AudioSubAgent', 'Change audio speed',
             'Change audio playback speed while preserving pitch (0.25x to 4x).', ('audio',),
             params={'speed': Param('number', None, required=True, minimum=0.25, maximum=4.0, unit='x',
                                    out_of_range='clarify')},
             example='Speed up 1.25x'),
    ToolSpec('extract_audio', 'VideoSubAgent', 'Extract audio',
             'Extract the soundtrack of a video as MP3 or WAV audio.', ('video',), output='audio', needs_audio=True,
             params={'format': Param('string', 'mp3', enum=('mp3', 'wav'))}, example='Extract the audio as WAV'),
    ToolSpec('convert_video_format', 'VideoSubAgent', 'Convert video',
             'Convert video to MP4, WebM, MKV or an animated GIF (GIF uses start/duration).', ('video',),
             output=lambda args: 'image' if args.get('target_format') == 'gif' else 'video',
             params={'target_format': Param('string', None, required=True, enum=('mp4', 'webm', 'mkv', 'gif')),
                     'start': Param('number', 0.0, minimum=0.0, unit=_TIME),
                     'duration': Param('number', 5.0, minimum=0.1, maximum=30.0, unit=_TIME)},
             example='Convert to WebM'),
    ToolSpec('compress_video', 'VideoSubAgent', 'Compress video',
             'Reduce video file size with balanced or high compression.', ('video',),
             params={'level': Param('string', 'balanced', enum=('balanced', 'high'))}, example='Compress this video'),
    ToolSpec('mute_video', 'VideoSubAgent', 'Mute video',
             'Remove the soundtrack and keep the picture unchanged.', ('video',), needs_audio=True,
             example='Mute the video'),
    ToolSpec('enhance_video', 'VideoSubAgent', 'Enhance video',
             'Improve picture quality: gentle grain reduction, sharpening and contrast, resized to a target resolution '
             'with the aspect ratio preserved.', ('video',),
             params={'mode': Param('string', '1080p', enum=('720p', '1080p', '1440p', '4k', 'original'),
                                   description='target short-edge resolution; original keeps the size'),
                     'denoise': Param('boolean', True), 'sharpen': Param('boolean', True)},
             example='Enhance the video to 1080p'),
    ToolSpec('extract_frame', 'VideoSubAgent', 'Thumbnail',
             'Save one still frame (thumbnail) at time_sec as a separate image; the video itself is unchanged.',
             ('video',), output='image', kind='branch',
             params={'time_sec': Param('number', None, minimum=0.0, unit=_TIME, nullable=True,
                                       description='omit = a representative early frame'),
                     'format': Param('string', 'jpg', enum=('jpg', 'png'))},
             example='Give me a thumbnail at 1s'),
    ToolSpec('transcribe_audio', 'AudioSubAgent', 'Transcribe',
             'Speech-to-text transcript with timestamps and TXT/SRT/VTT subtitle exports.', ('audio', 'video'),
             kind='analysis', needs_audio=True, example='Transcribe the speech'),
    ToolSpec('reduce_noise', 'AudioSubAgent', 'Reduce noise',
             'Suppress steady background noise, hiss and hum.', ('audio', 'video'), needs_audio=True,
             example='Remove the background noise'),
    ToolSpec('isolate_voice', 'AudioSubAgent', 'Isolate voice',
             'Separate the voice from background music (vocal stem).', ('audio', 'video'), needs_audio=True,
             example='Isolate the vocals'),
    ToolSpec('remove_vocals', 'AudioSubAgent', 'Remove vocals',
             'Create an instrumental by removing the vocals.', ('audio', 'video'), needs_audio=True,
             example='Make an instrumental'),
    ToolSpec('auto_trim_silence', 'AudioSubAgent', 'Trim silence',
             'Remove leading and trailing silence from audio.', ('audio',),
             params={'threshold': Param('integer', 40, minimum=10, maximum=80, unit='dB below peak')},
             example='Trim the silence'),
    ToolSpec('enhance_speech', 'AudioSubAgent', 'Enhance voice',
             'Studio speech clarity: voice-focused cleanup and presence boost.', ('audio', 'video'),
             needs_audio=True, example='Enhance voice clarity'),
    ToolSpec('normalize_audio', 'AudioSubAgent', 'Normalize',
             'Normalize loudness to a platform target (default -14 LUFS, true-peak protected). Use preset for a '
             'platform, target_lufs for an exact loudness, or target_dbfs only for plain peak normalization.',
             ('audio', 'video'), needs_audio=True,
             params={'preset': Param('string', None, enum=('youtube', 'spotify', 'apple_podcasts', 'podcast', 'broadcast'),
                                     nullable=True, description='platform loudness target'),
                     'target_lufs': Param('number', None, minimum=-40.0, maximum=-5.0, unit='LUFS', nullable=True),
                     'target_dbfs': Param('number', None, minimum=-20.0, maximum=0.0, unit='dBFS', nullable=True)},
             example='Normalize loudness to -14 LUFS'),
    ToolSpec('adjust_volume', 'AudioSubAgent', 'Change volume',
             'Make audio louder or quieter by gain_db decibels (positive = louder).', ('audio', 'video'),
             needs_audio=True,
             params={'gain_db': Param('number', None, required=True, minimum=-30.0, maximum=30.0, unit='dB')},
             example='Make it louder'),
    ToolSpec('apply_audio_fade', 'AudioSubAgent', 'Fade',
             'Apply a fade-in and/or fade-out (seconds; 0 = no fade on that end).', ('audio', 'video'),
             needs_audio=True,
             params={'fade_in_sec': Param('number', 2.0, minimum=0.0, maximum=30.0, unit=_TIME),
                     'fade_out_sec': Param('number', 2.0, minimum=0.0, maximum=30.0, unit=_TIME)},
             example='Fade in and out over 1 second'),
    ToolSpec('convert_audio_format', 'AudioSubAgent', 'Export audio',
             'Export audio as MP3, WAV, FLAC or OGG.', ('audio',),
             params={'target_format': Param('string', None, required=True, enum=('mp3', 'wav', 'flac', 'ogg')),
                     'bitrate': Param('string', None, enum=('128k', '192k', '256k', '320k'), nullable=True,
                                      description='MP3/OGG quality; MP3 defaults to 192k')},
             example='Export as MP3'),
    ToolSpec('generate_voiceover', 'AudioSubAgent', 'Voiceover',
             'Turn a script into a spoken voiceover audio file (needs no input media; the source is left unchanged).',
             MEDIA_TYPES, output='audio', kind='branch',
             params={'text': Param('string', None, required=True, description='the words to speak, up to 5000 characters'),
                     'voice': Param('string', None, nullable=True, description='voice id; omit for the default voice'),
                     'speed': Param('number', 1.0, minimum=0.5, maximum=2.0, unit='x'),
                     'format': Param('string', 'wav', enum=('wav', 'mp3'))},
             example='Make a voiceover of this text'),
    ToolSpec('separate_stems', 'AudioSubAgent', 'Separate stems',
             'Split a song or recording into stems: vocals and instrumental, karaoke (instrumental only), '
             '4 stems (vocals, drums, bass, other) or voice only. Delivers each stem as its own file.',
             ('audio', 'video'), output='audio', kind='branch', needs_audio=True,
             params={'mode': Param('string', 'vocals', enum=('vocals', 'karaoke', '4stem', 'voice')),
                     'format': Param('string', 'wav', enum=('wav', 'mp3')),
                     'quality': Param('string', 'auto', enum=('auto', 'fast', 'best'))},
             example='Split this song into stems'),
    ToolSpec('extract_lyrics', 'AudioSubAgent', 'Extract lyrics',
             'Get timed lyrics from a song as SRT, VTT or plain text.', ('audio', 'video'), kind='analysis',
             needs_audio=True, params={'format': Param('string', 'srt', enum=('srt', 'vtt', 'txt'))},
             example='Get the lyrics'),
]

TOOL_REGISTRY = {spec.name: spec for spec in TOOL_SPECS}
AGENT_FOR_TOOL = {spec.name: spec.agent for spec in TOOL_SPECS}

# Names a reasoning model (or older clients) may use for a registered tool.
TOOL_ALIASES = {
    'trim': 'trim', 'cut': 'trim', 'clip': 'trim', 'trim_media': 'trim', 'trim_clip': 'trim',
    'speed': 'speed', 'change_speed': 'speed', 'adjust_speed': 'speed', 'set_speed': 'speed',
    'denoise': 'reduce_noise', 'noise_reduction': 'reduce_noise', 'remove_noise': 'reduce_noise',
    'clean_audio': 'reduce_noise', 'reduce_background_noise': 'reduce_noise',
    'normalize': 'normalize_audio', 'normalize_loudness': 'normalize_audio', 'loudness_normalize': 'normalize_audio',
    'fade': 'apply_audio_fade', 'add_fade': 'apply_audio_fade', 'audio_fade': 'apply_audio_fade', 'fade_audio': 'apply_audio_fade',
    'transcribe': 'transcribe_audio', 'speech_to_text': 'transcribe_audio', 'stt': 'transcribe_audio',
    'generate_subtitles': 'transcribe_audio', 'subtitles': 'transcribe_audio', 'transcribe_video': 'transcribe_audio',
    'remove_bg': 'remove_background', 'background_removal': 'remove_background', 'cutout': 'remove_background',
    'upscale': 'upscale_image', 'super_resolution': 'upscale_image', 'enhance_resolution': 'upscale_image',
    'clarity': 'enhance_photo_clarity', 'sharpen': 'enhance_photo_clarity', 'enhance_image': 'enhance_photo_clarity',
    'enhance_clarity': 'enhance_photo_clarity', 'enhance_photo': 'enhance_photo_clarity',
    'face_restore': 'restore_faces', 'restore_face': 'restore_faces',
    'mute': 'mute_video', 'remove_audio': 'mute_video', 'strip_audio': 'mute_video',
    'thumbnail': 'extract_frame', 'screenshot': 'extract_frame', 'frame': 'extract_frame',
    'extract_thumbnail': 'extract_frame', 'snapshot': 'extract_frame',
    'volume': 'adjust_volume', 'gain': 'adjust_volume', 'change_volume': 'adjust_volume', 'louder': 'adjust_volume',
    'compress': 'compress_video', 'reduce_size': 'compress_video',
    'extract_soundtrack': 'extract_audio', 'rip_audio': 'extract_audio',
    'convert': 'convert', 'export': 'convert', 'convert_format': 'convert', 'export_as': 'convert', 'change_format': 'convert',
    'make_gif': 'convert_video_format', 'gif': 'convert_video_format',
    'trim_silence': 'auto_trim_silence', 'remove_silence': 'auto_trim_silence',
    'voice_enhance': 'enhance_speech', 'enhance_voice': 'enhance_speech', 'speech_enhance': 'enhance_speech',
    'isolate_vocals': 'isolate_voice', 'separate_vocals': 'isolate_voice', 'vocal_isolation': 'isolate_voice',
    'stem_separation': 'isolate_voice', 'instrumental': 'remove_vocals', 'karaoke': 'remove_vocals',
    'inspect': 'inspect_media', 'analyze': 'inspect_media', 'probe': 'inspect_media',
    '_enhance': '_enhance', 'enhance': '_enhance', 'improve': '_enhance',
    '_auto_clean': '_auto_clean', 'auto_clean': '_auto_clean', 'clean_audio_auto': '_auto_clean',
    'enhance_video_quality': 'enhance_video', 'video_enhance': 'enhance_video', 'improve_video': 'enhance_video',
}

AUDIO_FORMATS = ('mp3', 'wav', 'flac', 'ogg')
LOUDNESS_PRESET_LUFS = {'youtube': -14.0, 'spotify': -14.0, 'apple_podcasts': -16.0, 'podcast': -16.0, 'broadcast': -23.0}
DEFAULT_LOUDNESS_LUFS = -14.0
PRESET_LABELS = {'youtube': 'YouTube', 'spotify': 'Spotify', 'apple_podcasts': 'Apple Podcasts', 'podcast': 'podcast',
                 'broadcast': 'broadcast'}


def loudness_target(args):
    """Integrated-loudness target a normalize step aims for (None for plain peak normalization)."""
    if args.get('target_lufs') is not None:
        return float(args['target_lufs'])
    if args.get('preset') in LOUDNESS_PRESET_LUFS:
        return LOUDNESS_PRESET_LUFS[args['preset']]
    if args.get('target_dbfs') is not None:
        return None
    return DEFAULT_LOUDNESS_LUFS
VIDEO_FORMATS = ('mp4', 'webm', 'mkv')
IMAGE_FORMATS = ('png', 'jpg', 'webp')
FORMAT_ALIASES = {'jpeg': 'jpg', 'jpe': 'jpg', 'oga': 'ogg', 'vorbis': 'ogg', 'mpeg4': 'mp4', 'matroska': 'mkv'}


def public_tool_definitions():
    """Privacy-safe, JSON-serialisable tool catalog (served by /api/agent/tools)."""
    definitions = []
    for spec in TOOL_SPECS:
        params = {}
        for name, param in spec.params.items():
            if param.internal:
                continue
            entry = {'type': param.type}
            if param.default is not None:
                entry['default'] = param.default
            if param.enum:
                entry['enum'] = list(param.enum)
            if param.minimum is not None:
                entry['minimum'] = param.minimum
            if param.maximum is not None:
                entry['maximum'] = param.maximum
            if param.unit:
                entry['unit'] = param.unit
            if param.description:
                entry['description'] = param.description
            if param.required:
                entry['required'] = True
            params[name] = entry
        output = spec.output if isinstance(spec.output, str) else 'video, or image for GIF'
        definitions.append({'name': spec.name, 'agent': spec.agent, 'description': spec.description,
                            'accepts': list(spec.accepts), 'produces': output, 'kind': spec.kind,
                            'parameters': params})
    return definitions


def tool_catalog_text():
    """Compact catalog for the reasoning prompt."""
    lines = []
    for spec in TOOL_SPECS:
        produces = spec.output if isinstance(spec.output, str) else 'video (image for gif)'
        produces = 'same type' if produces == 'same' else produces
        args = []
        for name, param in spec.params.items():
            if param.internal:
                continue
            detail = param.type
            if param.enum:
                detail = '|'.join(str(value) for value in param.enum)
            elif param.minimum is not None or param.maximum is not None:
                detail += f" {'' if param.minimum is None else param.minimum}..{'' if param.maximum is None else param.maximum}"
            if param.unit:
                detail += f' {param.unit}'
            if param.required:
                detail += ' REQUIRED'
            elif param.default is not None:
                detail += f' default {param.default}'
            elif param.nullable:
                detail += ' optional'
            args.append(f'{name}: {detail}')
        kind = '' if spec.kind == 'edit' else f' [{spec.kind}: does not change the main result]'
        lines.append(f"- {spec.name} ({'/'.join(spec.accepts)} -> {produces}){kind}: {spec.description}"
                     + (f" Args: {'; '.join(args)}." if args else ' No args.'))
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  PRIVACY
# ═══════════════════════════════════════════════════════════════════════════

_PRIVATE_TERMS = [
    (r'faster[-_ ]?whisper|whisper(?:[-_ ]?(?:large|medium|small|base|tiny)[\w.-]*)?', 'speech recognition'),
    (r'gemma[\w.-]*', 'reasoning model'), (r'v-?llm', 'reasoning service'),
    (r'isnet[\w-]*|u2net[\w-]*|rembg', 'cutout model'), (r'ff(?:mpeg|probe)', 'media engine'),
    (r'open-?cv|cv2', 'vision engine'), (r'librosa|pydub|noisereduce|soundfile', 'audio engine'),
    (r'(?:py)?torch|onnx(?:runtime)?|ctranslate2', 'inference runtime'), (r'edsr|fsrcnn|real-?esrgan', 'detail model'),
    (r'demucs|htdemucs|deepfilternet|deep-?filter', 'separation model'),
]
_PRIVATE_PATTERNS = [(re.compile(rf'\b(?:{pattern})\b', re.IGNORECASE), replacement)
                     for pattern, replacement in _PRIVATE_TERMS]


def scrub_private_terms(text):
    if not isinstance(text, str):
        return text
    for pattern, replacement in _PRIVATE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
#  VALUE PARSING (units, timestamps, multipliers)
# ═══════════════════════════════════════════════════════════════════════════

UNIT_RE = r'(?:milliseconds?|millisecs?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)'
NUMBER_RE = r'(?:\d+(?::\d{1,2}){1,2}(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)'
_NOT_TIME_AFTER = r'(?!\s*(?:x\b|×|times\b|%|percent|db|lufs|lu\b|k\b|p\b|fps|hz|khz|px|bit|kbps|mb|gb|kb|frames?\b))'
NOT_TIME_AFTER = _NOT_TIME_AFTER
_END = r'(?![\w%]|[.:]\d)'
TIME_RE = rf'(?<![\w.:-])({NUMBER_RE})\s*({UNIT_RE})?{_END}{_NOT_TIME_AFTER}'
TIME_PATTERN = re.compile(TIME_RE)
RANGE_PATTERN = re.compile(
    rf'(?:\b(?:from|between)\s+)?(?<![\w.:-])({NUMBER_RE})\s*({UNIT_RE})?{_END}{_NOT_TIME_AFTER}\s*'
    rf'(?:to|-|–|until|till|through|thru|->|→|&)\s*(?:the\s+)?'
    rf'(?<![\w.:])({NUMBER_RE})\s*({UNIT_RE})?{_END}{_NOT_TIME_AFTER}')
MULTIPLIER_PATTERN = re.compile(r'(?<![\w.])(-?\d*\.?\d+)\s*(?:x|×|times)(?![a-z])')


def _unit_factor(unit):
    unit = (unit or 's').lower()
    if unit.startswith('ms') or unit.startswith('milli') or unit.startswith('msec'):
        return 0.001
    if unit in ('m', 'min', 'mins') or unit.startswith('minute'):
        return 60.0
    if unit in ('h', 'hr', 'hrs') or unit.startswith('hour'):
        return 3600.0
    return 1.0


def parse_clock(value):
    """'1:30' -> 90, '00:01:02.5' -> 62.5, '2.5' -> 2.5."""
    value = str(value).strip()
    if ':' in value:
        parts = value.split(':')
        try:
            numbers = [float(part) for part in parts]
        except ValueError:
            return None
        seconds = 0.0
        for number in numbers:
            seconds = seconds * 60 + number
        return seconds
    try:
        return float(value)
    except ValueError:
        return None


def time_to_seconds(value, unit=None):
    seconds = parse_clock(value)
    if seconds is None:
        return None
    if ':' in str(value):
        return round(seconds, 6)
    return round(seconds * _unit_factor(unit), 6)


def coerce_seconds(value):
    """Accept numbers, '2.5', '2.5s', '500ms', '1:30', '1m30s' from a plan."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    compound = re.fullmatch(r'(\d+(?:\.\d+)?)\s*m(?:in(?:utes?)?)?\s*(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?', text)
    if compound:
        return float(compound.group(1)) * 60 + float(compound.group(2))
    match = re.fullmatch(rf'({NUMBER_RE})\s*({UNIT_RE})?', text)
    if not match:
        return None
    return time_to_seconds(match.group(1), match.group(2))


def coerce_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower().replace('×', 'x')
        match = re.fullmatch(r'([+-]?\d*\.?\d+)\s*(x|%|db|dbfs|lufs|lu|times)?', text)
        if match:
            number = float(match.group(1))
            return number / 100.0 if match.group(2) == '%' else number
    return None


def fmt_seconds(value):
    if value is None:
        return 'the end'
    return f'{value:g}s'


# ═══════════════════════════════════════════════════════════════════════════
#  MEDIA STATE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class MediaState:
    """What the media will look like at a given point of the plan."""

    def __init__(self, media_type=None, has_audio=None, duration=None, width=None, height=None, condition=None):
        self.type = media_type if media_type in MEDIA_TYPES else None
        self.has_audio = has_audio
        self.duration = duration
        self.width = width
        self.height = height
        self.condition = condition or {}
        self.applied = []

    @classmethod
    def from_context(cls, context, perception=None):
        context = context if isinstance(context, dict) else {}
        media_type = str(context.get('type') or '').lower() or None
        if media_type and media_type.startswith('video'):
            media_type = 'video'
        elif media_type and media_type.startswith('audio'):
            media_type = 'audio'
        elif media_type and media_type.startswith('image'):
            media_type = 'image'
        has_audio = context.get('has_audio')
        has_audio = bool(has_audio) if isinstance(has_audio, bool) else None
        duration = coerce_number(context.get('duration'))
        duration = duration if duration and math.isfinite(duration) and duration > 0 else None
        width = coerce_number(context.get('width'))
        height = coerce_number(context.get('height'))
        condition = {}
        if perception:
            media_type = media_type or perception.get('type')
            if perception.get('audio'):
                condition['audio'] = dict(perception['audio'])
            if perception.get('image'):
                condition['image'] = dict(perception['image'])
                width = width or perception['image'].get('width')
                height = height or perception['image'].get('height')
            video = perception.get('video')
            if video:
                condition['video'] = dict(video)
                has_audio = video.get('has_audio') if has_audio is None else has_audio
                duration = duration or video.get('duration') or None
                width = width or video.get('width')
                height = height or video.get('height')
        if media_type == 'audio':
            has_audio = True
        elif media_type == 'image':
            has_audio = False
        return cls(media_type, has_audio, duration, width, height, condition)

    def copy(self):
        clone = MediaState(self.type, self.has_audio, self.duration, self.width, self.height,
                           copy.deepcopy(self.condition))
        clone.applied = list(self.applied)
        return clone

    @property
    def audio(self):
        return self.condition.get('audio') or {}

    @property
    def image(self):
        return self.condition.get('image') or {}

    def apply(self, name, args):
        """Advance the state through a step."""
        spec = TOOL_REGISTRY.get(name)
        self.applied.append(name)
        if not spec:
            return
        if name in ('trim_video', 'trim_audio'):
            start = args.get('start_sec') or 0.0
            end = args.get('end_sec')
            if end is None and self.duration:
                end = self.duration
            if end is not None:
                self.duration = max(0.0, end - start)
        elif name in ('adjust_video_speed', 'adjust_audio_speed') and self.duration and args.get('speed'):
            self.duration = self.duration / args['speed']
        elif name == 'extract_audio':
            self.has_audio = True
            self.width = self.height = None
            self.condition.pop('video', None)
            self.condition.pop('image', None)
        elif name == 'mute_video':
            self.has_audio = False
            self.condition.pop('audio', None)
        elif name == 'convert_video_format' and args.get('target_format') == 'gif':
            self.has_audio = False
            self.duration = min(self.duration or args.get('duration', 5.0), args.get('duration', 5.0))
            self.condition.pop('audio', None)
        elif name == 'upscale_image':
            scale = args.get('scale', 2)
            if self.width:
                self.width *= scale
            if self.height:
                self.height *= scale
            if self.image:
                self.condition['image']['small'] = False
        elif name == 'enhance_photo_clarity' and self.image:
            self.condition['image'].update({'noisy': False, 'blurry': False, 'jpeg_quality_low': False})
        elif name == 'remove_background' and self.image:
            self.condition['image']['has_alpha'] = True
        elif name in ('reduce_noise', 'enhance_speech', 'isolate_voice') and self.audio:
            self.condition['audio'].update({'noisy': False, 'music': False, 'background': 'clean'})
        elif name in ('normalize_audio', 'adjust_volume') and self.audio:
            self.condition['audio']['quiet'] = False
            if name == 'normalize_audio':
                target = loudness_target(args)
                if target is not None:
                    self.condition['audio']['loudness_lufs'] = target
                    self.condition['audio']['peak_db'] = min(self.audio.get('peak_db', -1.0), -1.0)
                else:
                    self.condition['audio']['peak_db'] = args['target_dbfs']
        new_type = spec.output_type(args, self.type)
        if spec.kind == 'edit' and new_type:
            self.type = new_type
            if new_type == 'audio':
                self.has_audio = True
            elif new_type == 'image':
                self.has_audio = False


# ═══════════════════════════════════════════════════════════════════════════
#  CONDITION POLICIES (perception -> preparatory steps)
# ═══════════════════════════════════════════════════════════════════════════

PREP_OVERRIDE_PATTERNS = [
    (re.compile(r"\b(?:as[- ]is|raw|untouched|without (?:any )?(?:processing|changes|cleanup|cleaning|prep\w*|enhanc\w*))\b"), {'all'}),
    (re.compile(r"\b(?:don'?t|do not|dont|no need to|never|without|skip|no)\s+(?:any\s+|the\s+)?(?:de-?nois\w*|noise[- ](?:removal|reduction)|nois\w*|clean\w*)"), {'reduce_noise'}),
    (re.compile(r"\b(?:don'?t|do not|dont|no need to|never|without|skip|no)\s+(?:any\s+|the\s+)?(?:enhanc\w*|voice[- ]?enhanc\w*|isolat\w*|separat\w*)"), {'enhance_speech', 'isolate_voice'}),
    (re.compile(r"\b(?:don'?t|do not|dont|no need to|never|without|skip|no)\s+(?:any\s+|the\s+)?(?:normali[sz]\w*|level\w*|loudness)"), {'normalize_audio'}),
    (re.compile(r"\b(?:don'?t|do not|dont|no need to|never|without|skip|no)\s+(?:any\s+|the\s+)?(?:sharpen\w*|clarity|clean\w* (?:the )?(?:photo|image))"), {'enhance_photo_clarity'}),
]


def detect_overrides(prompt):
    text = (prompt or '').lower().replace('’', "'")
    blocked = set()
    for pattern, tools in PREP_OVERRIDE_PATTERNS:
        if pattern.search(text):
            blocked |= tools
    return blocked


def _blocked(tool, overrides):
    return 'all' in overrides or tool in overrides


def _policies_for(name, args, state, overrides, lineage):
    """
    Return dict(prep=[...], post=[...], skip=reason|None, adjust={...}, notes=[...]).
    prep/post entries: {'name', 'args', 'reason', 'fallback'?}.
    Rules fire only from *measured* conditions and never when the user opted out
    or an equivalent step already ran earlier in the chain.
    """
    result = {'prep': [], 'post': [], 'skip': None, 'adjust': {}}
    audio, image = state.audio, state.image
    done = set(lineage)

    if name == 'transcribe_audio' and audio and audio.get('silent'):
        result['skip'] = 'No sound was detected in the recording, so there is nothing to transcribe.'
        return result
    if name == 'transcribe_audio' and audio and not audio.get('silent'):
        if audio.get('music') and not done & {'isolate_voice', 'enhance_speech'} \
                and not _blocked('isolate_voice', overrides):
            result['prep'].append({
                'name': 'isolate_voice', 'args': {},
                'fallback': {'name': 'enhance_speech', 'args': {}},
                'reason': (f"Background music detected ({audio.get('speech_to_background_db', 0):.0f} dB under the "
                           'speech) — isolating the voice before transcription.')})
        elif audio.get('noisy') and not done & {'reduce_noise', 'enhance_speech', 'isolate_voice'} \
                and not _blocked('reduce_noise', overrides):
            kind = 'Steady hum' if audio.get('background') == 'hum' else 'Background noise'
            result['prep'].append({
                'name': 'reduce_noise', 'args': {},
                'reason': (f"{kind} detected (signal-to-noise about {audio.get('snr_db', 0):.0f} dB) — "
                           'cleaning the audio before transcription.')})
        if audio.get('quiet') and not done & {'normalize_audio', 'adjust_volume'} \
                and not _blocked('normalize_audio', overrides):
            result['prep'].append({
                'name': 'normalize_audio', 'args': {},
                'reason': f"Very quiet recording (peak {audio.get('peak_db', 0):.0f} dBFS) — raising the level before transcription."})

    elif name == 'upscale_image' and image:
        if image.get('noisy') and 'enhance_photo_clarity' not in done and not _blocked('enhance_photo_clarity', overrides):
            cause = 'compression artifacts' if image.get('jpeg_quality_low') else 'grain'
            result['prep'].append({
                'name': 'enhance_photo_clarity', 'args': {'denoise_strength': 9, 'sharpen_strength': 0.8},
                'reason': f'Visible {cause} detected — cleaning it first so upscaling does not magnify it.'})
        elif image.get('blurry') and not _blocked('enhance_photo_clarity', overrides):
            result['post'].append({
                'name': 'enhance_photo_clarity', 'args': {'denoise_strength': 3, 'sharpen_strength': 1.6},
                'reason': 'Soft focus detected — sharpening after upscaling to recover edge detail.'})

    elif name == 'enhance_photo_clarity' and image and 'sharpen_strength' not in args.get('_explicit', ()):
        if image.get('blurry'):
            result['adjust'] = {'sharpen_strength': 1.8}
            result['reason'] = 'Soft focus detected — using stronger sharpening.'
        elif image.get('noisy'):
            result['adjust'] = {'denoise_strength': 9}
            result['reason'] = 'Visible grain detected — using stronger grain reduction.'

    elif name == 'remove_background' and image and image.get('has_alpha') and not args.get('refine') \
            and 'remove_background' not in done and (image.get('alpha_coverage') or 0) >= 0.05:
        result['skip'] = ('The image already has a transparent background — skipping the cutout. '
                          'Say "refine the cutout" to redo the edges.')

    elif name == 'normalize_audio' and audio and not audio.get('silent') and 'normalize_audio' not in done \
            and not state.applied_audio_change():
        target = loudness_target(args)
        if target is not None and audio.get('loudness_lufs') is not None:
            peak_ok = audio.get('true_peak_dbtp', audio.get('peak_db', 0.0)) <= -1.0 + 0.05
            if abs(audio['loudness_lufs'] - target) <= 0.5 and peak_ok:
                result['skip'] = (f"Loudness is already on target ({audio['loudness_lufs']:.1f} LUFS vs {target:g}) — "
                                  'skipping normalization.')
        elif target is None and audio.get('peak_db') is not None and abs(audio['peak_db'] - args['target_dbfs']) <= 0.3:
            result['skip'] = f"Peak level is already at {args['target_dbfs']:g} dBFS — skipping normalization."

    elif name == 'adjust_volume' and audio and (args.get('gain_db') or 0) > 0 and audio.get('peak_db') is not None \
            and not state.applied_audio_change():
        headroom = -audio['peak_db'] - 0.1
        if headroom < args['gain_db']:
            if headroom < 0.5:
                result['skip'] = ('The audio already peaks at full scale, so a plain boost would distort it — '
                                  'try "normalize loudness to -14 LUFS" instead.')
            else:
                result['adjust'] = {'gain_db': round(headroom, 1)}
                result['reason'] = f'Boost limited to {headroom:.1f} dB to avoid clipping.'
    return result


def _applied_audio_change(self):
    return any(tool in self.applied for tool in ('trim_audio', 'trim_video', 'normalize_audio', 'adjust_volume',
                                                 'reduce_noise', 'enhance_speech', 'isolate_voice', 'remove_vocals',
                                                 'apply_audio_fade', 'adjust_audio_speed', 'adjust_video_speed'))


MediaState.applied_audio_change = _applied_audio_change


# ═══════════════════════════════════════════════════════════════════════════
#  STEP VALIDATION & REPAIR
# ═══════════════════════════════════════════════════════════════════════════

class Clarify(Exception):
    def __init__(self, question, options=None):
        super().__init__(question)
        self.question = question
        self.options = list(options or [])


class Drop(Exception):
    def __init__(self, note):
        super().__init__(note)
        self.note = note


def _type_label(media_type):
    return {'image': 'an image', 'audio': 'audio', 'video': 'a video'}.get(media_type, 'this media')


def resolve_tool_name(name):
    if not isinstance(name, str):
        return None
    key = re.sub(r'[^a-z0-9_]+', '_', name.strip().lower()).strip('_')
    if key in TOOL_REGISTRY:
        return key
    return TOOL_ALIASES.get(key)


def _normalize_format(value):
    if not isinstance(value, str):
        return None
    value = value.strip().lower().lstrip('.')
    return FORMAT_ALIASES.get(value, value)


def repair_for_media(name, args, state):
    """
    Map a (possibly generic or mismatched) tool onto the right tool(s) for the
    media type at this point of the plan. Returns list[(name, args, note|None)].
    Raises Drop / Clarify.
    """
    media = state.type
    args = dict(args)

    if name == 'trim':
        name = 'trim_audio' if media == 'audio' else 'trim_video'
    elif name == 'speed':
        name = 'adjust_audio_speed' if media == 'audio' else 'adjust_video_speed'
    elif name == '_auto_clean':
        # Noise vs music vs hum aware cleanup, chosen from the measured condition.
        audio = state.audio
        if media == 'image':
            raise Drop('Skipped "audio cleanup": an image has no sound.')
        if audio.get('silent'):
            raise Drop('Skipped audio cleanup: the recording is silent.')
        if audio.get('music'):
            return [('isolate_voice', {}, f"Background music detected ({audio.get('speech_to_background_db', 0):.0f} dB "
                                          'under the voice), so I am isolating the voice instead of filtering noise.')]
        if audio and audio.get('background') == 'clean':
            raise Drop('The recording is already clean (quiet background), so no noise removal was needed.')
        if audio.get('background') == 'hum':
            return [('reduce_noise', {}, 'Steady hum detected, so I am removing it.')]
        return [('reduce_noise', {}, None)]
    elif name == '_enhance':
        if media == 'image':
            return [('enhance_photo_clarity', {}, None)]
        if media == 'audio':
            return [('enhance_speech', {}, None)]
        raise Clarify('What should I enhance in this video?',
                      ['Enhance voice clarity', 'Remove the background noise', 'Normalize loudness'])
    elif name == 'convert':
        target = _normalize_format(args.get('target_format') or args.get('format'))
        if target in AUDIO_FORMATS:
            name = 'convert_audio_format'
        elif target in VIDEO_FORMATS or target == 'gif':
            name = 'convert_video_format'
        elif target in IMAGE_FORMATS:
            name = 'convert_image_format'
        else:
            raise Clarify('Which format should I export to?',
                          {'audio': ['Export as MP3', 'Export as WAV', 'Export as FLAC'],
                           'image': ['Export as PNG', 'Export as JPG', 'Export as WebP']}
                          .get(media, ['Convert to MP4', 'Convert to WebM', 'Make a GIF']))
        args = {'target_format': target}

    if media is None:
        return [(name, args, None)]

    if name == 'trim_video' and media == 'audio':
        name = 'trim_audio'
    elif name == 'trim_audio' and media == 'video':
        name = 'trim_video'
    elif name == 'adjust_video_speed' and media == 'audio':
        name = 'adjust_audio_speed'
    elif name == 'adjust_audio_speed' and media == 'video':
        name = 'adjust_video_speed'
    elif name == 'extract_audio' and media == 'audio':
        fmt = _normalize_format(args.get('format'))
        if fmt:
            return [('convert_audio_format', {'target_format': fmt}, None)]
        raise Drop('Skipped "extract audio": the media is already audio.')
    elif name == 'convert_audio_format' and media == 'video':
        target = _normalize_format(args.get('target_format')) or 'mp3'
        if target in ('mp3', 'wav'):
            return [('extract_audio', {'format': target}, None)]
        return [('extract_audio', {'format': 'wav'}, None),
                ('convert_audio_format', {'target_format': target}, None)]
    elif name == 'convert_video_format' and media == 'audio':
        target = _normalize_format(args.get('target_format'))
        if target in AUDIO_FORMATS:
            return [('convert_audio_format', {'target_format': target}, None)]
        raise Drop(f'Skipped "{target or "video"} export": audio has no picture to convert into video.')
    elif name == 'compress_video' and media == 'audio':
        return [('convert_audio_format', {'target_format': 'mp3'},
                 'For audio, a smaller file means a compressed MP3 export.')]
    elif name == 'convert_image_format' and media == 'video':
        fmt = _normalize_format(args.get('target_format'))
        return [('extract_frame', {'format': 'png' if fmt == 'png' else 'jpg'},
                 'A video cannot become a single image, so I will save a still frame instead.')]
    return [(name, args, None)]


def _coerce_param(name, param, value):
    if value is None:
        return None
    if param.type == 'boolean':
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ('true', 'yes', '1', 'false', 'no', '0'):
            return value.strip().lower() in ('true', 'yes', '1')
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError(name)
    if param.type == 'string':
        if not isinstance(value, str):
            raise ValueError(name)
        text = value.strip().lower()
        if name in ('target_format', 'format'):
            text = _normalize_format(text)
        if name == 'level':
            text = {'strong': 'high', 'max': 'high', 'maximum': 'high', 'heavy': 'high', 'medium': 'balanced',
                    'normal': 'balanced', 'light': 'balanced', 'low': 'balanced'}.get(text, text)
        if name == 'preset':
            text = re.sub(r'[\s-]+', '_', text)
            text = {'apple': 'apple_podcasts', 'apple_podcast': 'apple_podcasts', 'ebu': 'broadcast', 'tv': 'broadcast',
                    'streaming': 'spotify'}.get(text, text)
        if name == 'quality_profile':
            text = {'person': 'portrait', 'human': 'portrait', 'people': 'portrait', 'product': 'studio',
                    'quick': 'fast', 'general': 'detail', 'auto': 'detail', 'default': 'detail'}.get(text, text)
        return text
    number = coerce_seconds(value) if param.unit == _TIME else coerce_number(value)
    if number is None or not math.isfinite(number):
        raise ValueError(name)
    if param.type == 'integer' and param.out_of_range != 'nearest':
        number = int(round(number))
    return number


PARAM_ALIASES = {
    'start': 'start_sec', 'start_time': 'start_sec', 'from': 'start_sec', 'begin': 'start_sec', 'start_seconds': 'start_sec',
    'end': 'end_sec', 'end_time': 'end_sec', 'to': 'end_sec', 'stop': 'end_sec', 'end_seconds': 'end_sec',
    'rate': 'speed', 'factor': 'speed', 'multiplier': 'speed', 'playback_speed': 'speed',
    'fade_in': 'fade_in_sec', 'fade_out': 'fade_out_sec', 'in': 'fade_in_sec', 'out': 'fade_out_sec',
    'fmt': 'format', 'output_format': 'target_format', 'extension': 'target_format',
    'lufs': 'target_lufs', 'target': 'target_lufs', 'db': 'gain_db', 'gain': 'gain_db', 'volume_db': 'gain_db',
    'time': 'time_sec', 'at': 'time_sec', 'timestamp': 'time_sec', 'profile': 'quality_profile',
    'denoise': 'denoise_strength', 'sharpen': 'sharpen_strength', 'model_name': None, 'model': None, 'engine': None,
}


def validate_args(spec, args, state, strict=False):
    """
    Coerce, default, clamp and sanity-check arguments.
    Returns (clean_args, notes). Raises Clarify (planning) or ValueError (strict execution).
    """
    notes = []
    raw = {}
    for key, value in (args or {}).items():
        if not isinstance(key, str):
            continue
        key = key.strip().lower()
        if key not in spec.params and key in PARAM_ALIASES:
            key = PARAM_ALIASES[key]
        if key in spec.params:
            raw[key] = value
    if spec.name == 'convert_video_format' and 'format' in (args or {}) and 'target_format' not in raw:
        raw['target_format'] = args['format']
    if spec.name in ('convert_audio_format', 'convert_image_format') and 'format' in (args or {}) and 'target_format' not in raw:
        raw['target_format'] = args['format']
    if spec.name == 'extract_audio' and 'target_format' in (args or {}) and 'format' not in raw:
        raw['format'] = args['target_format']

    clean = {}
    for key, param in spec.params.items():
        value = raw.get(key)
        try:
            value = _coerce_param(key, param, value)
        except ValueError:
            if strict:
                raise ValueError(f'Invalid value for {key.replace("_", " ")}.')
            if param.required:
                raise Clarify(*_clarification_for(spec, key, state))
            notes.append(f'Ignored an invalid {key.replace("_", " ")} for {spec.label.lower()} and used the default.')
            value = None
        if value is None:
            if param.required:
                if strict:
                    raise ValueError(f'{spec.label} needs a {key.replace("_", " ")}.')
                raise Clarify(*_clarification_for(spec, key, state))
            if param.default is not None or not param.nullable:
                value = param.default
            if value is None and not param.nullable and not param.internal:
                continue
            if value is None:
                if not param.internal:
                    clean[key] = None
                continue
        if param.enum:
            if value not in param.enum:
                if param.out_of_range == 'nearest' and isinstance(value, (int, float)):
                    nearest = min(param.enum, key=lambda option: (abs(option - value), -option))
                    if strict:
                        raise ValueError(f'{spec.label} supports {", ".join(map(str, param.enum))}.')
                    notes.append(f'{spec.label} supports {" or ".join(f"{option}x" for option in param.enum)}; using {nearest}x.')
                    value = nearest
                else:
                    if strict:
                        raise ValueError(f'Unsupported {key.replace("_", " ")} for {spec.label.lower()}.')
                    raise Clarify(*_clarification_for(spec, key, state, bad_value=value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            low, high = param.minimum, param.maximum
            if (low is not None and value < low) or (high is not None and value > high):
                if strict or param.out_of_range == 'clarify':
                    if strict:
                        raise ValueError(f'{spec.label}: {key.replace("_", " ")} must be between '
                                         f'{low if low is not None else "-∞"} and {high if high is not None else "∞"}.')
                    raise Clarify(*_clarification_for(spec, key, state, bad_value=value))
                clamped = min(max(value, low if low is not None else value), high if high is not None else value)
                unit = f' {param.unit}' if param.unit and param.unit != _TIME else ('s' if param.unit == _TIME else '')
                notes.append(f'Adjusted {key.replace("_", " ")} from {value:g}{unit} to {clamped:g}{unit} (allowed range).')
                value = clamped
            if param.type == 'integer':
                value = int(value)
        if param.type == 'integer' and isinstance(value, float) and value.is_integer():
            value = int(value)
        clean[key] = value
    _check_semantics(spec, clean, state, notes, strict)
    return {key: value for key, value in clean.items() if value is not None or key == 'end_sec'}, notes


def _check_semantics(spec, args, state, notes, strict):
    duration = state.duration
    if spec.name in ('trim_video', 'trim_audio'):
        keep_last = args.pop('keep_last_sec', None)
        if keep_last is not None:
            if not duration:
                if strict:
                    raise ValueError('The media length is unknown for a "last seconds" trim.')
                raise Clarify('How long is the clip? I need exact start and end times to keep the last part.',
                              [f'Trim from 0 to {keep_last:g} seconds'])
            args['start_sec'] = max(0.0, round(duration - keep_last, 3))
            args['end_sec'] = None
        drop_last = args.pop('drop_last_sec', None)
        if drop_last is not None:
            if not duration:
                if strict:
                    raise ValueError('The media length is unknown for removing the last seconds.')
                raise Clarify('How long is the clip? I need exact start and end times to remove the ending.',
                              ['Trim from 0 to 5 seconds'])
            if drop_last >= duration:
                raise Clarify(f'The media is only {duration:.2f}s long. Which part should I keep?',
                              [f'Trim from 0 to {max(0.1, duration / 2):g} seconds'])
            args['end_sec'] = round(duration - drop_last, 3)
            if args['end_sec'] <= (args.get('start_sec') or 0.0):
                if strict:
                    raise ValueError('Nothing would be left after removing both ends.')
                raise Clarify(f'The media is only {duration:.2f}s long, so removing both ends leaves nothing. '
                              'Which part should I keep?', [f'Trim from 0 to {max(0.1, duration / 2):g} seconds'])
        start, end = args.get('start_sec') or 0.0, args.get('end_sec')
        if end is not None and end <= start:
            if strict:
                raise ValueError('Choose an end time after the start time.')
            if end < start:
                args['start_sec'], args['end_sec'] = end, start
                start, end = end, start
                notes.append(f'Read the range as {start:g}s to {end:g}s.')
            else:
                raise Clarify('The start and end times are the same. Which range should I keep?',
                              [f'Trim from {start:g} to {start + 5:g} seconds'])
        if start == 0 and end is None and not strict:
            raise Clarify('Which part should I keep? Please give a start and end time.',
                          ['Trim from 0 to 5 seconds', 'Keep the first 10 seconds'])
        if duration and not strict:
            if start >= duration:
                raise Clarify(f'The media is only {duration:.2f}s long, so {start:g}s is past the end. '
                              'Which range should I keep?',
                              [f'Trim from 0 to {min(duration, 5):g} seconds'])
            if end is not None and end > duration + 0.05:
                notes.append(f'The media ends at {duration:.2f}s, so the trim ends there instead of {end:g}s.')
                args['end_sec'] = round(duration, 3)
    elif spec.name == 'apply_audio_fade':
        if not args.get('fade_in_sec') and not args.get('fade_out_sec'):
            if strict:
                raise ValueError('Choose a fade length greater than zero.')
            args['fade_in_sec'] = args['fade_out_sec'] = 2.0
        if duration and not strict:
            total = args['fade_in_sec'] + args['fade_out_sec']
            if total > duration:
                ratio = duration / total
                args['fade_in_sec'] = round(args['fade_in_sec'] * ratio, 3)
                args['fade_out_sec'] = round(args['fade_out_sec'] * ratio, 3)
                notes.append(f'Shortened the fades to fit the {duration:.2f}s clip.')
    elif spec.name == 'extract_frame':
        moment = args.get('time_sec')
        if moment is not None and duration and moment >= duration and not strict:
            args['time_sec'] = round(max(0.0, duration - 0.05), 3)
            notes.append(f'The video is {duration:.2f}s long, so the thumbnail comes from {args["time_sec"]:g}s.')
    elif spec.name == 'convert_video_format' and args.get('target_format') != 'gif':
        args.pop('start', None)          # the GIF window never applies to full-length conversions
        args.pop('duration', None)
    elif spec.name == 'convert_video_format' and args.get('target_format') == 'gif':
        if duration and args.get('start', 0) >= duration and not strict:
            args['start'] = 0.0
            notes.append('The GIF start was past the end of the video, so it starts at 0s.')
    elif spec.name == 'normalize_audio':
        if (args.get('target_lufs') is not None or args.get('preset')) and args.get('target_dbfs') is not None:
            args['target_dbfs'] = None
        if args.get('target_lufs') is not None and args.get('preset'):
            args['preset'] = None


def _clarification_for(spec, key, state, bad_value=None):
    if key == 'speed':
        if bad_value is not None:
            return (f'{bad_value:g}x is outside the supported range. What playback speed should I use? '
                    'Choose a multiplier between 0.25x and 4x.', ['Speed up 1.5x', 'Speed up 2x', 'Slow down to 0.5x'])
        return ('What playback speed should I use? Choose a multiplier between 0.25x and 4x.',
                ['Speed up 1.25x', 'Speed up 1.5x', 'Slow down to 0.5x'])
    if key == 'target_format':
        options = {'convert_audio_format': ['Export as MP3', 'Export as WAV', 'Export as FLAC'],
                   'convert_video_format': ['Convert to MP4', 'Convert to WebM', 'Make a GIF'],
                   'convert_image_format': ['Export as PNG', 'Export as JPG', 'Export as WebP']}[spec.name]
        return 'Which format should I export to?', options
    if key == 'gain_db':
        return 'Should I make it louder or quieter, and by how much?', ['Make it 6 dB louder', 'Make it 6 dB quieter']
def _is_number_grounded(value, prompt_lower):
    if value is None or not prompt_lower:
        return False
    if isinstance(value, (int, float)):
        str_val = f"{value:g}"
        if str_val in prompt_lower:
            return True
        if isinstance(value, float) and value.is_integer() and str(int(value)) in prompt_lower:
            return True
        if value == 2 and any(w in prompt_lower for w in ('double', 'twice', '2x')):
            return True
        if value == 0.5 and any(w in prompt_lower for w in ('half', '0.5x')):
            return True
        if value == 3 and any(w in prompt_lower for w in ('triple', '3x')):
            return True
    return False


def _check_grounding(spec, clean_args, prompt, state):
    prompt_lower = (prompt or '').lower()

    # 1. Number grounding check for required parameters without defaults
    for key, value in clean_args.items():
        param = spec.params.get(key)
        if not param or param.internal:
            continue
        if key == 'speed' and param.required and isinstance(value, (int, float)):
            if not _is_number_grounded(value, prompt_lower):
                return Clarify(*_clarification_for(spec, key, state))

    # 2. Format grounding check (never change requested formats or accept unsupported formats)
    if spec.name in ('convert_audio_format', 'convert_video_format', 'convert_image_format', 'extract_audio'):
        fmt_key = 'target_format' if 'target_format' in clean_args else 'format'
        planned_fmt = str(clean_args.get(fmt_key) or '').lower()
        format_matches = re.findall(
            r'\b(mp3|wav|flac|ogg|mp4|webm|mkv|gif|png|jpe?g|webp|m4a|aac|mov|avi|wma|opus|tiff?|bmp|heic)\b',
            prompt_lower
        )
        if format_matches:
            requested_fmt = format_matches[0].lower()
            if requested_fmt in ('jpg', 'jpeg'):
                requested_fmt = 'jpg'
            unsupported = {'m4a', 'aac', 'mov', 'avi', 'wma', 'opus', 'tif', 'tiff', 'bmp', 'heic'}
            if requested_fmt in unsupported:
                options = {'image': ['Export as PNG', 'Export as JPG', 'Export as WebP'],
                           'video': ['Convert to MP4', 'Convert to WebM', 'Extract the audio as MP3']
                          }.get(state.type, ['Export as MP3', 'Export as WAV', 'Export as FLAC'])
                return Clarify(f'{requested_fmt.upper()} export is not available. Which format should I use?', options)
            if planned_fmt and requested_fmt != planned_fmt and not (requested_fmt in ('jpg', 'jpeg') and planned_fmt in ('jpg', 'jpeg')):
                return Clarify(*_clarification_for(spec, fmt_key, state, bad_value=planned_fmt))

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  PLAN FINALIZATION (shared by the reasoning service and the local parser)
# ═══════════════════════════════════════════════════════════════════════════

def _as_text(value, default=''):
    if isinstance(value, str):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float, bool)):
        return str(value)
    return default


def _as_text_list(value, limit=6):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    items = []
    for item in value:
        if isinstance(item, dict):
            item = item.get('label') or item.get('text') or item.get('prompt')
        if isinstance(item, (str, int, float)) and not isinstance(item, bool):
            text = scrub_private_terms(str(item)).strip()
            if text and text not in items:
                items.append(text[:120])
    return items[:limit]


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', 'yes', '1')
    return False


def normalize_raw_plan(raw):
    """Coerce arbitrary model output into {tools: [{name, args, ...}], reply, ...} without trusting types."""
    if not isinstance(raw, dict):
        return {'tools': [], 'reply': '', 'thought': '', 'clarification_needed': False,
                'clarification_options': [], 'suggested_actions': []}
    tools = raw.get('tools')
    for alias in ('steps', 'plan', 'actions', 'tool_calls'):
        if not isinstance(tools, list) and isinstance(raw.get(alias), list):
            tools = raw[alias]
    steps = []
    for item in tools if isinstance(tools, list) else []:
        if isinstance(item, str):
            steps.append({'name': item, 'args': {}})
            continue
        if not isinstance(item, dict):
            steps.append({'name': None, 'args': {}})
            continue
        function = item.get('function') if isinstance(item.get('function'), dict) else {}
        name = item.get('name') or item.get('tool') or item.get('action') or function.get('name')
        args = item.get('args')
        for alias in ('arguments', 'params', 'parameters', 'input'):
            if not isinstance(args, dict) and alias in item:
                args = item[alias]
        if not isinstance(args, dict) and 'arguments' in function:
            args = function['arguments']
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if not isinstance(args, dict):
            args = {key: value for key, value in item.items()
                    if key not in ('name', 'tool', 'action', 'function', 'id', 'depends_on', 'input', 'reason')}
        step = {'name': name if isinstance(name, str) else None, 'args': args}
        for key in ('reason', 'role', 'fallback', 'input', 'branch'):
            if key in item:
                step[key] = item[key]
        if isinstance(item.get('skill'), dict) and isinstance(item['skill'].get('id'), str):
            step['skill'] = {'id': item['skill']['id'][:48], 'title': _as_text(item['skill'].get('title'))[:80]}
        steps.append(step)
    return {
        'tools': steps,
        'thought': scrub_private_terms(_as_text(raw.get('thought') or raw.get('reasoning'))),
        'reply': scrub_private_terms(_as_text(raw.get('reply') or raw.get('response') or raw.get('message'))),
        'clarification_needed': _as_bool(raw.get('clarification_needed')),
        'clarification_options': _as_text_list(raw.get('clarification_options')),
        'suggested_actions': _as_text_list(raw.get('suggested_actions')),
        'delegated_subagent': _as_text(raw.get('delegated_subagent'), 'Orchestrator'),
        'notes': [scrub_private_terms(note) for note in _as_text_list(raw.get('notes'), limit=10)],
        'keep_order': _as_bool(raw.get('keep_order')),
        'conversational': _as_bool(raw.get('conversational')),
    }


SUGGESTIONS_BY_TYPE = {
    'image': ['Upscale 2x', 'Boost clarity', 'Remove the background'],
    'video': ['Trim from 0 to 5 seconds', 'Extract the audio as MP3', 'Make a GIF from the first 5 seconds'],
    'audio': ['Transcribe speech to text', 'Remove the background noise', 'Normalize loudness to -14 LUFS'],
    None: ['Remove the background', 'Trim from 2.3 to 5.3 seconds', 'Transcribe speech to text'],
}


def _describe_step(name, args):
    spec = TOOL_REGISTRY[name]
    if name in ('trim_video', 'trim_audio'):
        return f"Trim {fmt_seconds(args.get('start_sec') or 0.0)} to {fmt_seconds(args.get('end_sec'))}"
    if name in ('adjust_video_speed', 'adjust_audio_speed'):
        return f"Change speed to {args['speed']:g}x"
    if name == 'upscale_image':
        return f"Upscale {args['scale']}x"
    if name == 'extract_audio':
        return f"Extract the audio as {args['format'].upper()}"
    if name in ('convert_audio_format', 'convert_image_format'):
        return f"Export as {args['target_format'].upper()}"
    if name == 'convert_video_format':
        if args['target_format'] == 'gif':
            return f"Make a {args['duration']:g}s GIF from {args['start']:g}s"
        return f"Convert to {args['target_format'].upper()}"
    if name == 'apply_audio_fade':
        parts = []
        if args.get('fade_in_sec'):
            parts.append(f"{args['fade_in_sec']:g}s fade-in")
        if args.get('fade_out_sec'):
            parts.append(f"{args['fade_out_sec']:g}s fade-out")
        return 'Add ' + ' and '.join(parts)
    if name == 'adjust_volume':
        return f"{'Raise' if args['gain_db'] > 0 else 'Lower'} volume by {abs(args['gain_db']):g} dB"
    if name == 'normalize_audio':
        if args.get('target_lufs') is not None:
            return f"Normalize loudness to {args['target_lufs']:g} LUFS"
        if args.get('preset'):
            return f"Normalize loudness for {PRESET_LABELS.get(args['preset'], args['preset'])} ({loudness_target(args):g} LUFS)"
        if args.get('target_dbfs') is not None:
            return f"Normalize peaks to {args['target_dbfs']:g} dBFS"
        return f'Normalize loudness to {DEFAULT_LOUDNESS_LUFS:g} LUFS'
    if name == 'extract_frame':
        moment = args.get('time_sec')
        return f"Save a thumbnail{'' if moment is None else f' at {moment:g}s'}"
    if name == 'compress_video':
        return f"Compress ({args['level']})"
    if name == 'enhance_video':
        return f"Enhance the picture ({args['mode']})"
    if name == 'remove_background':
        return f"Remove the background ({args['quality_profile']}{', refined' if args.get('refine') else ''})"
    return spec.label


def finalize_plan(raw, prompt='', media_context=None, perception=None, source='local'):
    """
    Validate + repair + apply condition policies. Returns the canonical plan:
      {thought, delegated_subagent, clarification_needed, clarification_options, tools,
       reply, suggested_actions, plan_notes, validated: True, plan_source}
    Every step: {id, name, args, input, depends_on, branch, role, reason?, fallback?}.
    """
    plan = normalize_raw_plan(raw)
    state = MediaState.from_context(media_context, perception)
    initial_type = state.type
    overrides = detect_overrides(prompt)
    notes = list(plan.get('notes') or [])
    clarifications = []
    steps = []
    main_head = 'source'
    main_lineage = []
    counter = [0]

    current_skill = [None]

    def emit(name, args, role='requested', reason=None, fallback=None, input_ref=None, branch=False):
        counter[0] += 1
        step_id = f's{counter[0]}'
        if name == 'isolate_voice' and not fallback:
            fallback = {'name': 'enhance_speech', 'args': {}}
        ref = input_ref or main_head
        step = {'id': step_id, 'name': name, 'args': args, 'input': ref,
                'depends_on': [] if ref == 'source' else [ref], 'branch': bool(branch), 'role': role}
        if reason:
            step['reason'] = reason
        if fallback:
            step['fallback'] = fallback
        if current_skill[0]:
            step['skill'] = current_skill[0]
        steps.append(step)
        return step_id

    pending = list(plan['tools'])
    reordered = False
    if not (plan.get('keep_order') or KEEP_ORDER_RE.search((prompt or '').lower())):
        pending, order_notes = canonicalize_order(pending)
        notes.extend(order_notes)
        reordered = bool(order_notes)
    while pending:
        raw_step = pending.pop(0)
        current_skill[0] = raw_step.get('skill') if isinstance(raw_step.get('skill'), dict) else None
        raw_name = raw_step.get('name')
        name = resolve_tool_name(raw_name)
        args = raw_step.get('args') if isinstance(raw_step.get('args'), dict) else {}
        explicit = tuple(key for key in args if not str(key).startswith('_'))
        from_result = bool(args.pop('_from_result', False)) if isinstance(args, dict) else False
        if not name:
            label = re.sub(r'[^A-Za-z0-9 _-]', '', str(raw_name or 'unnamed'))[:40].replace('_', ' ') or 'unnamed'
            notes.append(f'Skipped "{label}": that action is not available yet.')
            continue
        try:
            resolved = repair_for_media(name, args, state)
        except Drop as drop:
            notes.append(drop.note)
            continue
        except Clarify as clarify:
            clarifications.append(clarify)
            continue

        for index, (tool_name, tool_args, repair_note) in enumerate(resolved):
            spec = TOOL_REGISTRY.get(tool_name)
            if not spec:
                notes.append('Skipped an action that is not available yet.')
                continue
            if repair_note:
                notes.append(repair_note)
            if state.type and state.type not in spec.accepts:
                notes.append(f'Skipped "{spec.label.lower()}": it works on {" or ".join(spec.accepts)}, '
                             f'but the media at that point is {state.type}.')
                continue
            if spec.needs_audio and state.has_audio is False:
                notes.append(f'Skipped "{spec.label.lower()}": the {state.type or "media"} has no soundtrack'
                             f'{" after muting" if "mute_video" in state.applied else ""}.')
                continue
            try:
                clean_args, arg_notes = validate_args(spec, tool_args, state)
            except Clarify as clarify:
                clarifications.append(clarify)
                continue
            if source == 'reasoning' and prompt:
                grounding_clarify = _check_grounding(spec, clean_args, prompt, state)
                if grounding_clarify is not None:
                    clarifications.append(grounding_clarify)
                    continue
            notes.extend(arg_notes)
            policy = _policies_for(tool_name, dict(clean_args, _explicit=explicit), state, overrides, main_lineage)
            if policy.get('note'):
                notes.append(policy['note'])
            if policy['skip']:
                notes.append(policy['skip'])
                continue
            if policy['adjust']:
                clean_args.update(policy['adjust'])
            step_reason = policy.get('reason')

            if spec.kind == 'analysis':
                aux_head, aux_state = main_head, state.copy()
                prep_steps = policy['prep']
                if prep_steps and state.type == 'video':
                    prep_steps = [{'name': 'extract_audio', 'args': {'format': 'wav'},
                                   'reason': 'Working on a copy of the soundtrack so the video stays untouched.'}] + prep_steps
                for prep in prep_steps:
                    prep_spec = TOOL_REGISTRY[prep['name']]
                    prep_args, _ = validate_args(prep_spec, prep['args'], aux_state)
                    aux_head = emit(prep['name'], prep_args, role='prep', reason=prep.get('reason'),
                                    fallback=prep.get('fallback'), input_ref=aux_head, branch=True)
                    aux_state.apply(prep['name'], prep_args)
                emit(tool_name, clean_args, reason=step_reason, input_ref=aux_head, branch=False)
                continue

            if spec.kind == 'branch':
                ref = main_head if (from_result or main_head == 'source' or initial_type != 'video') else 'source'
                if ref == 'source' and initial_type != state.type:
                    ref = main_head
                emit(tool_name, clean_args, reason=step_reason, input_ref=ref, branch=True)
                continue

            for prep in policy['prep']:
                prep_spec = TOOL_REGISTRY[prep['name']]
                prep_args, _ = validate_args(prep_spec, prep['args'], state)
                main_head = emit(prep['name'], prep_args, role='prep', reason=prep.get('reason'),
                                 fallback=prep.get('fallback'))
                state.apply(prep['name'], prep_args)
                main_lineage.append(prep['name'])
            main_head = emit(tool_name, clean_args, reason=step_reason)
            state.apply(tool_name, clean_args)
            main_lineage.append(tool_name)
            for post in policy['post']:
                post_spec = TOOL_REGISTRY[post['name']]
                post_args, _ = validate_args(post_spec, post['args'], state)
                main_head = emit(post['name'], post_args, role='post', reason=post.get('reason'))
                state.apply(post['name'], post_args)
                main_lineage.append(post['name'])

    steps = _dedupe_adjacent(steps)
    result = {
        'thought': '', 'delegated_subagent': 'Orchestrator', 'clarification_needed': False,
        'clarification_options': [], 'tools': [], 'reply': '', 'suggested_actions': [],
        'plan_notes': _unique(notes), 'validated': True, 'plan_source': source,
    }
    condition_lines = []
    if perception:
        try:
            import media_inspector
            condition_lines = media_inspector.describe(perception)
        except Exception:
            condition_lines = []

    if clarifications or plan['clarification_needed']:
        if clarifications:
            question = clarifications[0].question
            options = clarifications[0].options
        else:
            question = plan['reply'] or 'Could you clarify what you would like me to change?'
            options = plan['clarification_options']
        if len(clarifications) > 1:
            question += ' Also: ' + ' '.join(item.question for item in clarifications[1:3])
        result.update({
            'clarification_needed': True, 'clarification_options': options or SUGGESTIONS_BY_TYPE.get(initial_type, [])[:3],
            'reply': question, 'suggested_actions': options or [],
            'thought': 'Missing or ambiguous details; asking before changing the media.',
            'delegated_subagent': _agent_for([step['name'] for step in steps]) if steps else 'Orchestrator',
        })
        return result

    if not steps:
        reply = plan['reply'] if plan['reply'] and source != 'local' else ''
        if notes:
            reply = (reply + '\n' if reply else '') + ' '.join(_unique(notes))
        result.update({'reply': reply or plan['reply'], 'thought': plan['thought'],
                       'suggested_actions': plan['suggested_actions'] or SUGGESTIONS_BY_TYPE.get(initial_type, [])})
        return result

    for step in steps:
        step['args'] = {key: value for key, value in step['args'].items() if not str(key).startswith('_')}
    descriptions = []
    for number, step in enumerate(steps, 1):
        text = f'{number}. {_describe_step(step["name"], step["args"])}'
        if step['role'] == 'prep':
            text += ' (preparation)'
        elif step['branch'] and TOOL_REGISTRY[step['name']].kind == 'branch':
            text += ' (separate file)'
        descriptions.append(text)
    reasons = _unique([step['reason'] for step in steps if step.get('reason')])
    used_skills = []
    for step in steps:
        if step.get('skill') and step['skill'] not in used_skills:
            used_skills.append(step['skill'])
    reply_lines = [f"Using {skill['title'] or skill['id']} (@{skill['id']})." for skill in used_skills]
    reply_lines += ['Plan:'] + descriptions
    reply_lines += reasons
    if result['plan_notes']:
        reply_lines += result['plan_notes']
    thought_parts = []
    if condition_lines:
        thought_parts.append('Media condition: ' + ' '.join(condition_lines))
    thought_parts.append(' -> '.join(_describe_step(step['name'], step['args']) for step in steps))
    if plan['thought'] and source != 'local':
        thought_parts.insert(0, plan['thought'])
    final_state_type = state.type
    result.update({
        'tools': steps,
        'reply': '\n'.join(reply_lines),
        'thought': scrub_private_terms(' | '.join(part for part in thought_parts if part)),
        'delegated_subagent': _agent_for([step['name'] for step in steps]),
        'skills_used': used_skills,
        'suggested_actions': (['Run exactly in my order'] if reordered else [])
        + SUGGESTIONS_BY_TYPE.get(final_state_type, SUGGESTIONS_BY_TYPE[None]),
    })
    return result


KEEP_ORDER_RE = re.compile(r"\b(?:exactly in (?:my|this|the given|that) order|in (?:my|this|that) exact order|"
                           r"keep (?:my|the) order|in the (?:exact )?order i (?:gave|said|wrote|asked)|don'?t reorder)\b")
_CLEANUP = {'reduce_noise', 'isolate_voice', 'enhance_speech'}
_LEVEL = {'normalize_audio', 'adjust_volume'}
_SAME_TYPE_EDITS = {'trim', 'trim_video', 'trim_audio', 'speed', 'adjust_video_speed', 'adjust_audio_speed',
                    'reduce_noise', 'isolate_voice', 'enhance_speech', 'normalize_audio', 'adjust_volume',
                    'apply_audio_fade', 'auto_trim_silence', 'mute_video', 'upscale_image', 'enhance_photo_clarity',
                    'restore_faces', 'remove_vocals'}


def _is_encode(name, args):
    target = _normalize_format((args or {}).get('target_format') or (args or {}).get('format'))
    if name in ('convert_audio_format', 'convert_image_format', 'compress_video'):
        return True
    return name in ('convert', 'convert_video_format') and target not in (None, 'gif')


def canonicalize_order(raw_steps):
    """
    Apply the studio's ordering rules to adjacent steps (stable, bounded):
      * clean-up (noise/voice) before loudness changes, so levels are set on the clean signal;
      * format export / compression after same-type edits, so media is encoded once;
      * upscaling before face restoration, so faces are restored at the final size.
    Returns (steps, notes). Users can opt out with "exactly in my order".
    """
    steps = list(raw_steps)
    notes = []

    def key(step):
        return resolve_tool_name(step.get('name')) if isinstance(step, dict) else None

    for _ in range(len(steps)):
        changed = False
        for index in range(len(steps) - 1):
            first, second = steps[index], steps[index + 1]
            a, b = key(first), key(second)
            a_args = first.get('args') if isinstance(first, dict) and isinstance(first.get('args'), dict) else {}
            reason = None
            if a in _LEVEL and b in _CLEANUP:
                reason = 'Cleaning up the audio before changing loudness, so levels are set on the clean signal.'
            elif _is_encode(a, a_args) and b in _SAME_TYPE_EDITS:
                reason = 'Moved the export to the end so the media is encoded only once.'
            elif _is_encode(a, a_args) and a != 'compress_video' and b == 'compress_video' \
                    and _normalize_format(a_args.get('target_format') or a_args.get('format')) in VIDEO_FORMATS:
                reason = 'Compressing before the format conversion so the final file is in the format you asked for.'
            elif a == 'restore_faces' and b == 'upscale_image':
                reason = 'Upscaling before face restoration, so faces are restored at the final size.'
            if reason:
                steps[index], steps[index + 1] = second, first
                if reason not in notes:
                    notes.append(reason)
                changed = True
        if not changed:
            break
    if notes:
        notes[-1] += ' Say "run exactly in my order" to keep your order.'
    return steps, notes


def _agent_for(names):
    agents = {AGENT_FOR_TOOL.get(name) for name in names if name in AGENT_FOR_TOOL}
    agents.discard('InspectorSubAgent') if len(agents) > 1 else None
    return agents.pop() if len(agents) == 1 else 'Orchestrator'


def _unique(items):
    seen = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen


def _dedupe_adjacent(steps):
    """Merge consecutive identical requested steps (e.g. 'remove background and make it transparent')."""
    result = []
    remap = {}
    for step in steps:
        previous = result[-1] if result else None
        if (previous and previous['name'] == step['name'] == 'apply_audio_fade' and step['input'] == previous['id']
                and previous['role'] == step['role'] == 'requested'):
            merged = dict(previous['args'])
            for key in ('fade_in_sec', 'fade_out_sec'):
                if step['args'].get(key):
                    merged[key] = step['args'][key]
            previous['args'] = merged
            remap[step['id']] = previous['id']
            continue
        kind = TOOL_REGISTRY[step['name']].kind
        same_input = previous and (step['input'] == previous['id'] if kind == 'edit' else step['input'] == previous['input'])
        if (same_input and previous['name'] == step['name'] and previous['args'] == step['args']
                and previous['role'] == step['role'] == 'requested'):
            remap[step['id']] = previous['id']
            continue
        step['input'] = remap.get(step['input'], step['input'])
        step['depends_on'] = [remap.get(ref, ref) for ref in step['depends_on']]
        result.append(step)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  EXECUTION-SIDE NORMALIZATION (for plans that did not come through finalize)
# ═══════════════════════════════════════════════════════════════════════════

def execution_steps(tools_list):
    """Give every step an id, input and dependency list using the default chaining rules."""
    steps = []
    main_head = 'source'
    known_ids = set()
    for index, tool in enumerate(tools_list or []):
        if not isinstance(tool, dict):
            tool = {'name': None, 'args': None}
        name = tool.get('name') if isinstance(tool.get('name'), str) else None
        step_id = tool.get('id') if isinstance(tool.get('id'), str) and tool.get('id') not in known_ids else f'step{index + 1}'
        known_ids.add(step_id)
        spec = TOOL_REGISTRY.get(name)
        kind = spec.kind if spec else 'edit'
        ref = tool.get('input')
        if not (isinstance(ref, str) and (ref == 'source' or ref in known_ids - {step_id})):
            ref = 'source' if kind == 'branch' else main_head
        step = {'id': step_id, 'name': name, 'args': tool.get('args', {}), 'input': ref,
                'branch': bool(tool.get('branch')) or kind == 'branch', 'role': tool.get('role') or 'requested',
                'reason': tool.get('reason') if isinstance(tool.get('reason'), str) else None,
                'fallback': tool.get('fallback') if isinstance(tool.get('fallback'), dict) else None,
                'index': index + 1}
        steps.append(step)
        if kind == 'edit' and not step['branch']:
            main_head = step_id
    return steps


# ═══════════════════════════════════════════════════════════════════════════
#  REASONING-SERVICE OUTPUT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_json_object(content):
    """
    Pull the first JSON object out of model output that may contain reasoning tags,
    markdown fences or prose. Returns (dict|None, leftover_text).
    """
    if not isinstance(content, str):
        return None, ''
    text = re.sub(r'<think>.*?</think>', ' ', content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'^.*?</think>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*$', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    stripped = re.sub(r'```(?:json|JSON)?', ' ', text)
    start = stripped.find('{')
    while start != -1:
        depth, in_string, escape = 0, False, False
        for position in range(start, len(stripped)):
            char = stripped[position]
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = stripped[start:position + 1]
                    try:
                        parsed = json.loads(candidate)
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        return parsed, (stripped[:start] + stripped[position + 1:]).strip()
                    break
        start = stripped.find('{', start + 1)
    return None, stripped.strip()
