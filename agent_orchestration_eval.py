"""
Offline evaluation harness for the Autonomous Media Copilot planner and executor.

    ./venv/Scripts/python.exe agent_orchestration_eval.py

Covers:
  1. Golden set: natural-language prompts -> expected tool sequence / args / clarification
     (multi-operation chains, lists, follow-ups, units, ambiguity, unsupported and
     media-type-incompatible requests, regression cases for known bugs).
  2. Validation/repair layer unit tests (applied identically to reasoning-service plans).
  3. Reasoning-service output handling (fences, reasoning tags, prose, malformed types).
  4. Condition-aware planning on synthetic media (noisy / music / quiet / clean audio,
     blurry / grainy / sharp images) — the plan must change with the measured condition.
  5. Execution on small synthetic media (chaining, side outputs, fused soundtrack edits,
     dependency-aware failure isolation, retries, fallbacks, verification, adaptive STT).

Prints a pass-rate summary for the golden set. No network or reasoning service needed.
"""

import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

import agent_nlu  # noqa: E402
import agent_planner  # noqa: E402
import agent_processor  # noqa: E402
import media_inspector  # noqa: E402
import video_processor  # noqa: E402

FORBIDDEN_TERMS = ['gemma', 'whisper', 'vllm', 'isnet', 'u2net', 'rembg', 'ffmpeg', 'ffprobe', 'opencv', 'librosa',
                   'pydub', 'pytorch', 'torch', 'onnx', 'edsr', 'fsrcnn', 'demucs', 'deepfilternet', 'noisereduce',
                   'ctranslate2']


def plan_for(prompt, media_type=None, history=None, **context):
    media_context = dict(context)
    if media_type:
        media_context['type'] = media_type
    return agent_processor._fallback_intent_parser(prompt, media_context or None, history)


def names(plan):
    return [tool['name'] for tool in plan['tools']]


# ═══════════════════════════════════════════════════════════════════════════
#  1. GOLDEN SET
# ═══════════════════════════════════════════════════════════════════════════
# Each case: (prompt, media_type, expectation). Expectation keys:
#   tools: exact tool-name sequence ([] = nothing executed)
#   args:  {step_index: {arg: expected}} (numbers compared with tolerance)
#   clarify: expected clarification flag (default False)
#   note / reply: substring expected in plan notes / reply
#   history, context: optional conversation history / extra media context
#   inputs: {step_index: 'source' | 'previous'}
H = lambda *messages: [{'role': 'user' if index % 2 == 0 else 'assistant', 'content': text}
                       for index, text in enumerate(messages)]

GOLDEN = [
    # Headline multi-operation requests
    ('Trim from 2s to 8s, remove the background noise, normalize loudness, then export as mp3', 'video',
     {'tools': ['trim_video', 'reduce_noise', 'normalize_audio', 'extract_audio'],
      'args': {0: {'start_sec': 2, 'end_sec': 8}, 3: {'format': 'mp3'}}}),
    ('Trim from 2s to 8s, remove the background noise, normalize loudness, then export as mp3', 'audio',
     {'tools': ['trim_audio', 'reduce_noise', 'normalize_audio', 'convert_audio_format'],
      'args': {3: {'target_format': 'mp3'}}}),
    ('Extract the audio as wav, speed it up 1.25x and add a 1 second fade in and out', 'video',
     {'tools': ['extract_audio', 'adjust_audio_speed', 'apply_audio_fade'],
      'args': {0: {'format': 'wav'}, 1: {'speed': 1.25}, 2: {'fade_in_sec': 1, 'fade_out_sec': 1}}}),
    ('Remove the background from this photo, upscale it 2x and boost clarity', 'image',
     {'tools': ['remove_background', 'upscale_image', 'enhance_photo_clarity'], 'args': {1: {'scale': 2}}}),
    ('Mute the video, then cut 0.5 to 3.2 seconds, and also give me a thumbnail at 1s', 'video',
     {'tools': ['mute_video', 'trim_video', 'extract_frame'],
      'args': {1: {'start_sec': 0.5, 'end_sec': 3.2}, 2: {'time_sec': 1}}, 'inputs': {2: 'source'}}),
    # Lists and connectors
    ('1. trim 0:02 to 0:10\n2. normalize to -14 LUFS\n3. fade out over 3 seconds', 'audio',
     {'tools': ['trim_audio', 'normalize_audio', 'apply_audio_fade'],
      'args': {0: {'start_sec': 2, 'end_sec': 10}, 1: {'target_lufs': -14}, 2: {'fade_in_sec': 0, 'fade_out_sec': 3}}}),
    ('- compress it\n- make a gif from 1 to 3 seconds', 'video',
     {'tools': ['compress_video', 'convert_video_format'], 'args': {1: {'target_format': 'gif', 'start': 1, 'duration': 2}}}),
    ('first trim 1 to 3, after that speed it up 1.5x; finally export as wav', 'audio',
     {'tools': ['trim_audio', 'adjust_audio_speed', 'convert_audio_format'],
      'args': {0: {'start_sec': 1, 'end_sec': 3}, 1: {'speed': 1.5}, 2: {'target_format': 'wav'}}}),
    ('Transcribe it and also extract the audio as mp3', 'video',
     {'tools': ['transcribe_audio', 'extract_audio'], 'args': {1: {'format': 'mp3'}}}),
    ('hey there, can you mute this and give me a 3 second gif?', 'video',
     {'tools': ['mute_video', 'convert_video_format'], 'args': {1: {'target_format': 'gif', 'duration': 3}}}),
    ('trim silence and normalize', 'audio', {'tools': ['auto_trim_silence', 'normalize_audio']}),
    ('Upscale 2x then remove background', 'image', {'tools': ['upscale_image', 'remove_background']}),
    ('make it hd and sharpen it a lot', 'image',
     {'tools': ['upscale_image', 'enhance_photo_clarity'], 'args': {0: {'scale': 2}, 1: {'sharpen_strength': 2.0}}}),
    # Units and timestamps (regressions: mm:ss, "2x slower", "over N seconds")
    ('Trim from 00:02 to 00:10', 'video', {'tools': ['trim_video'], 'args': {0: {'start_sec': 2, 'end_sec': 10}}}),
    ('1:30 to 2:00', 'video', {'tools': ['trim_video'], 'args': {0: {'start_sec': 90, 'end_sec': 120}}}),
    ('trim from 1m30s to 2m', 'video', {'tools': ['trim_video'], 'args': {0: {'start_sec': 90, 'end_sec': 120}}}),
    ('cut 500ms to 1500ms', 'video', {'tools': ['trim_video'], 'args': {0: {'start_sec': 0.5, 'end_sec': 1.5}}}),
    ('Trim between 1 and 4 seconds', 'audio', {'tools': ['trim_audio'], 'args': {0: {'start_sec': 1, 'end_sec': 4}}}),
    ('keep the first 10 seconds', 'video', {'tools': ['trim_video'], 'args': {0: {'start_sec': 0, 'end_sec': 10}}}),
    ('keep the last 5 seconds', 'video',
     {'tools': ['trim_video'], 'args': {0: {'start_sec': 25, 'end_sec': None}}, 'context': {'duration': 30}}),
    ('keep the last 5 seconds', 'video', {'tools': [], 'clarify': True}),
    ('Trim from 8 to 2 seconds', 'video',
     {'tools': ['trim_video'], 'args': {0: {'start_sec': 2, 'end_sec': 8}}, 'note': 'Read the range'}),
    ('trim from 2 to 15', 'video',
     {'tools': ['trim_video'], 'args': {0: {'end_sec': 10}}, 'context': {'duration': 10}, 'note': 'ends at'}),
    ('make it 2x slower', 'video', {'tools': ['adjust_video_speed'], 'args': {0: {'speed': 0.5}}}),
    ('2 times slower', 'audio', {'tools': ['adjust_audio_speed'], 'args': {0: {'speed': 0.5}}}),
    ('speed up by 25%', 'audio', {'tools': ['adjust_audio_speed'], 'args': {0: {'speed': 1.25}}}),
    ('slow motion please', 'video', {'tools': ['adjust_video_speed'], 'args': {0: {'speed': 0.5}}}),
    ('fade out over 3 seconds', 'audio', {'tools': ['apply_audio_fade'], 'args': {0: {'fade_in_sec': 0, 'fade_out_sec': 3}}}),
    ('Fade edges', 'audio', {'tools': ['apply_audio_fade'], 'args': {0: {'fade_in_sec': 2, 'fade_out_sec': 2}}}),
    ('louder by 4 dB', 'audio', {'tools': ['adjust_volume'], 'args': {0: {'gain_db': 4}}}),
    ('make it a bit quieter', 'audio', {'tools': ['adjust_volume'], 'args': {0: {'gain_db': -3}}}),
    ('normalize for youtube', 'audio', {'tools': ['normalize_audio'], 'args': {0: {'preset': 'youtube'}}}),
    ('make it podcast-ready', 'audio', {'tools': ['normalize_audio'], 'args': {0: {'preset': 'podcast'}}}),
    ('normalize for broadcast', 'video', {'tools': ['normalize_audio'], 'args': {0: {'preset': 'broadcast'}}}),
    # Argument binding & user order within one clause (regression)
    ('upscale 4x and speed up 2x', 'image',
     {'tools': ['upscale_image'], 'args': {0: {'scale': 4}}, 'note': 'change video speed'}),
    ('make it 2x faster and trim from 2 to 5', 'video',
     {'tools': ['adjust_video_speed', 'trim_video'], 'args': {0: {'speed': 2}, 1: {'start_sec': 2, 'end_sec': 5}}}),
    ('Upscale 3x', 'image', {'tools': ['upscale_image'], 'args': {0: {'scale': 4}}, 'note': 'using 4x'}),
    # Media-type tracking, repair and incompatibility
    ('convert to flac', 'video', {'tools': ['extract_audio', 'convert_audio_format'],
                                  'args': {0: {'format': 'wav'}, 1: {'target_format': 'flac'}}}),
    ('Extract audio as WAV then trim from 0.2 to 0.8 seconds', 'video', {'tools': ['extract_audio', 'trim_audio']}),
    ('mute it then normalize the audio', 'video', {'tools': ['mute_video'], 'note': 'no soundtrack'}),
    ('save it as png', 'video', {'tools': ['extract_frame'], 'args': {0: {'format': 'png'}}, 'note': 'still frame'}),
    ('upscale this', 'audio', {'tools': [], 'note': 'works on image'}),
    ('restore faces', 'video', {'tools': [], 'note': 'works on image'}),
    ('Extract audio', 'audio', {'tools': [], 'note': 'already audio'}),
    ('remove the background', 'video', {'tools': ['reduce_noise'], 'note': 'background noise'}),
    ('reduce noise in this photo', 'image', {'tools': ['enhance_photo_clarity'], 'args': {0: {'denoise_strength': 10}}}),
    ('export as jpg', 'image', {'tools': ['convert_image_format'], 'args': {0: {'target_format': 'jpg'}}}),
    ('Export video as WebM', 'video', {'tools': ['convert_video_format'], 'args': {0: {'target_format': 'webm'}}}),
    ('isolate the vocals', 'audio', {'tools': ['isolate_voice']}),
    ('make an instrumental', 'audio', {'tools': ['remove_vocals']}),
    ('make it sound better', 'audio', {'tools': ['enhance_speech']}),
    ('enhance', 'image', {'tools': ['enhance_photo_clarity']}),
    ('enhance', 'video', {'tools': [], 'clarify': True}),
    # Canonical ordering (and opting out)
    ('normalize then denoise', 'audio', {'tools': ['reduce_noise', 'normalize_audio'], 'note': 'Cleaning up'}),
    ('normalize then denoise exactly in my order', 'audio', {'tools': ['normalize_audio', 'reduce_noise']}),
    ('restore faces then upscale 2x', 'image', {'tools': ['upscale_image', 'restore_faces']}),
    ('export as mp3 then trim from 1 to 2', 'audio', {'tools': ['trim_audio', 'convert_audio_format']}),
    # Ambiguity -> clarification (nothing executes)
    ('Normalize then do something unknown', 'audio', {'tools': [], 'clarify': True}),
    ('restore the surface texture', 'image', {'tools': [], 'clarify': True}),
    ('Make it faster', 'audio', {'tools': [], 'clarify': True}),
    ('Adjust speed to 5x', 'audio', {'tools': [], 'clarify': True}),
    ('speed 0x', 'audio', {'tools': [], 'clarify': True}),
    ('Trim the video', 'video', {'tools': [], 'clarify': True}),
    ('export as m4a', 'audio', {'tools': [], 'clarify': True}),
    # Unsupported operations explained without breaking the rest
    ('rotate the video and compress it', 'video', {'tools': ['compress_video'], 'note': 'Rotating'}),
    ('cut out 2 to 3 seconds', 'audio', {'tools': [], 'note': 'middle section'}),
    # Conversation, greetings, negation
    ('hello', None, {'tools': [], 'reply': 'image, video, audio'}),
    ('hey trim 2 to 5', 'video', {'tools': ['trim_video'], 'args': {0: {'start_sec': 2, 'end_sec': 5}}}),
    ('what is lufs?', 'audio', {'tools': [], 'reply': 'LUFS'}),
    ('what can you do?', None, {'tools': [], 'reply': 'Vision'}),
    ('transcribe as-is, don\'t denoise', 'audio', {'tools': ['transcribe_audio']}),
    ('Inspect loudness and sample rate', 'audio', {'tools': ['inspect_media']}),
    ('Inspect this audio file and check its loudness and sample rate.', 'audio', {'tools': ['inspect_media']}),
    ('remove background portrait', 'image',
     {'tools': ['remove_background'], 'args': {0: {'quality_profile': 'portrait', 'refine': False}}}),
    # Follow-ups referring to the previous request / result
    ('0.2 to 0.8 seconds', 'audio', {'tools': ['trim_audio'], 'history': H('Trim audio', 'Which part should I keep?')}),
    ('1.5x', 'audio', {'tools': ['adjust_audio_speed'], 'args': {0: {'speed': 1.5}},
                       'history': H('Make it faster', 'What playback speed should I use?')}),
    ('now make it louder', 'audio', {'tools': ['adjust_volume'], 'args': {0: {'gain_db': 6}},
                                      'history': H('Normalize loudness', 'Done.')}),
    ('Do it again', 'audio', {'tools': ['normalize_audio', 'apply_audio_fade'],
                              'history': H('normalize then fade out over 2 seconds', 'Done.')}),
    ('again with cleaner edges', 'image',
     {'tools': ['remove_background'], 'args': {0: {'refine': True, 'quality_profile': 'detail'}},
      'history': H('remove the background', 'Done.')}),
    ('Run exactly in my order', 'audio', {'tools': ['normalize_audio', 'reduce_noise'],
                                          'history': H('normalize then denoise', 'Plan reordered.')}),
    # Held-out phrasings found during review (removal-style trims, casual wording)
    ('could you chop off everything after 12 seconds', 'video',
     {'tools': ['trim_video'], 'args': {0: {'start_sec': 0, 'end_sec': 12}}}),
    ('lose the first 3 seconds', 'audio', {'tools': ['trim_audio'], 'args': {0: {'start_sec': 3, 'end_sec': None}}}),
    ('remove the last 2 seconds', 'audio',
     {'tools': ['trim_audio'], 'args': {0: {'start_sec': 0, 'end_sec': 18}}, 'context': {'duration': 20}}),
    ('remove the noise from 2 to 5 seconds', 'audio', {'tools': ['reduce_noise']}),
    ('clean up the hiss and bump the volume a little, then give me a wav', 'video',
     {'tools': ['reduce_noise', 'adjust_volume', 'extract_audio'], 'args': {1: {'gain_db': 3}, 2: {'format': 'wav'}}}),
    ('make the voice clearer and cut the dead air', 'audio', {'tools': ['enhance_speech', 'auto_trim_silence']}),
    ('get rid of the music', 'audio', {'tools': ['isolate_voice']}),
    ('brighten the photo', 'image', {'tools': [], 'note': 'brightness'}),
    ('make it twice as fast then slow it back down to 0.5x', 'video',
     {'tools': ['adjust_video_speed', 'adjust_video_speed'], 'args': {0: {'speed': 2}, 1: {'speed': 0.5}}}),
    ('1) mute 2) thumbnail @ 2s 3) compress hard', 'video',
     {'tools': ['mute_video', 'extract_frame', 'compress_video'], 'args': {1: {'time_sec': 2}, 2: {'level': 'high'}}}),
    ('remove bg, then 4x upscale, then save as webp', 'image',
     {'tools': ['remove_background', 'upscale_image', 'convert_image_format'],
      'args': {1: {'scale': 4}, 2: {'target_format': 'webp'}}}),
    ('normalize to -16 lufs for apple podcasts and export flac', 'audio',
     {'tools': ['normalize_audio', 'convert_audio_format'], 'args': {0: {'target_lufs': -16}, 1: {'target_format': 'flac'}}}),
]


def _close(actual, expected):
    if expected is None or actual is None:
        return actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool):
        return math.isclose(actual, expected, abs_tol=1e-3)
    return actual == expected


def check_case(prompt, media_type, expectation):
    """Return a list of failure strings (empty = pass)."""
    plan = plan_for(prompt, media_type, expectation.get('history'), **expectation.get('context', {}))
    problems = []
    if names(plan) != expectation['tools']:
        problems.append(f"tools {names(plan)} != {expectation['tools']}")
    if bool(plan['clarification_needed']) != expectation.get('clarify', False):
        problems.append(f"clarification {plan['clarification_needed']} != {expectation.get('clarify', False)}")
    for index, exact in expectation.get('exact_args', {}).items():
        if index < len(plan['tools']) and plan['tools'][index]['args'] != exact:
            problems.append(f"step {index} args {plan['tools'][index]['args']} != exactly {exact}")
    for index, expected_args in expectation.get('args', {}).items():
        if index >= len(plan['tools']):
            problems.append(f'missing step {index}')
            continue
        actual = plan['tools'][index]['args']
        for key, value in expected_args.items():
            if not _close(actual.get(key), value):
                problems.append(f'step {index} {key}={actual.get(key)!r} != {value!r}')
    for index, expected_input in expectation.get('inputs', {}).items():
        if index < len(plan['tools']):
            actual_input = plan['tools'][index]['input']
            wanted = 'source' if expected_input == 'source' else plan['tools'][index - 1]['id']
            if actual_input != wanted:
                problems.append(f'step {index} input {actual_input} != {wanted}')
    if expectation.get('note') and not any(expectation['note'].lower() in note.lower() for note in plan['plan_notes']):
        problems.append(f"note containing {expectation['note']!r} not in {plan['plan_notes']}")
    if expectation.get('reply') and expectation['reply'].lower() not in plan['reply'].lower():
        problems.append(f"reply missing {expectation['reply']!r}")
    public = json.dumps({key: plan[key] for key in ('reply', 'thought', 'plan_notes', 'suggested_actions',
                                                    'clarification_options')}).lower()
    for term in FORBIDDEN_TERMS:
        if term in public:
            problems.append(f'privacy leak: {term}')
    for key in ('reply', 'thought'):
        if not isinstance(plan[key], str):
            problems.append(f'{key} is not a string')
    if not isinstance(plan['suggested_actions'], list) or not all(isinstance(item, str) for item in plan['suggested_actions']):
        problems.append('suggested_actions is not a list of strings')
    return problems


# ── Unseen-phrasing set: written BEFORE the generalization fixes (pre/post pass rates are reported) ──
D20 = {'duration': 20.0}
WEBM = {'target_format': 'webm'}
UNSEEN = [
    # Class 1: trimming both ends / "except the last N" (duration-relative ends)
    ('chop off the first 4 seconds and the last 2, then give me the audio as a wav', 'video',
     {'tools': ['trim_video', 'extract_audio'], 'args': {0: {'start_sec': 4, 'end_sec': 18}, 1: {'format': 'wav'}},
      'context': D20}),
    ('remove the first 3 seconds and the last 5 seconds', 'video',
     {'tools': ['trim_video'], 'args': {0: {'start_sec': 3, 'end_sec': 15}}, 'context': D20}),
    ('cut the last 4 seconds', 'audio', {'tools': ['trim_audio'], 'args': {0: {'start_sec': 0, 'end_sec': 16}}, 'context': D20}),
    ('drop the final 2.5 seconds', 'audio',
     {'tools': ['trim_audio'], 'args': {0: {'start_sec': 0, 'end_sec': 17.5}}, 'context': D20}),
    ('keep everything except the last 3 seconds', 'video',
     {'tools': ['trim_video'], 'args': {0: {'start_sec': 0, 'end_sec': 17}}, 'context': D20}),
    ('keep everything but the first 2 seconds', 'audio',
     {'tools': ['trim_audio'], 'args': {0: {'start_sec': 2, 'end_sec': None}}, 'context': D20}),
    ('keep from 5s until the end minus 3 seconds', 'video',
     {'tools': ['trim_video'], 'args': {0: {'start_sec': 5, 'end_sec': 17}}, 'context': D20}),
    ('trim 1 second off each end', 'audio',
     {'tools': ['trim_audio'], 'args': {0: {'start_sec': 1, 'end_sec': 19}}, 'context': D20}),
    ('shave 2 seconds off the start and 1 off the end', 'video',
     {'tools': ['trim_video'], 'args': {0: {'start_sec': 2, 'end_sec': 19}}, 'context': D20}),
    ('get rid of the last 10 seconds and normalize', 'audio',
     {'tools': ['trim_audio', 'normalize_audio'], 'args': {0: {'start_sec': 0, 'end_sec': 10}}, 'context': D20}),
    ('lose the opening 1.5s and the closing 2s', 'audio',
     {'tools': ['trim_audio'], 'args': {0: {'start_sec': 1.5, 'end_sec': 18}}, 'context': D20}),
    ('cut off the last 2 seconds then compress', 'video',
     {'tools': ['trim_video', 'compress_video'], 'args': {0: {'end_sec': 18}}, 'context': D20}),
    ('remove the last 00:05', 'video',
     {'tools': ['trim_video'], 'args': {0: {'start_sec': 0, 'end_sec': 15}}, 'context': D20}),
    ('drop the first 4 seconds and the last 2', 'video', {'tools': [], 'clarify': True}),
    ('cut the last 3 seconds', 'audio', {'tools': [], 'clarify': True}),
    ('remove the last 30 seconds', 'audio', {'tools': [], 'clarify': True, 'context': D20}),
    # Class 2: "cut"/"isolate" verbs on an image mean a cutout, never a trim
    ('cut me out from this pic and make it 4x bigger', 'image',
     {'tools': ['remove_background', 'upscale_image'], 'args': {1: {'scale': 4}}}),
    ('cut out the subject', 'image', {'tools': ['remove_background']}),
    ('cut me out', 'image', {'tools': ['remove_background'], 'args': {0: {'quality_profile': 'portrait'}}}),
    ('isolate me from the photo', 'image', {'tools': ['remove_background'], 'args': {0: {'quality_profile': 'portrait'}}}),
    ('extract the subject and save as png', 'image', {'tools': ['remove_background', 'convert_image_format']}),
    ('can you cut the dog out?', 'image', {'tools': ['remove_background']}),
    ('cut out the product and upscale 2x', 'image',
     {'tools': ['remove_background', 'upscale_image'], 'args': {0: {'quality_profile': 'studio'}}}),
    ('clip the person out of this image', 'image', {'tools': ['remove_background']}),
    ('make a cutout of her', 'image', {'tools': ['remove_background']}),
    ('separate me from the background', 'image', {'tools': ['remove_background']}),
    ('isolate the subject then make it sharper', 'image', {'tools': ['remove_background', 'enhance_photo_clarity']}),
    ('cut it', 'image', {'tools': [], 'clarify': True}),
    ('trim from 2 to 5 seconds', 'image', {'tools': [], 'note': 'image'}),
    ('cut out the car and make it a jpg', 'image', {'tools': ['remove_background', 'convert_image_format']}),
    ('pull the subject out of the photo and boost clarity', 'image',
     {'tools': ['remove_background', 'enhance_photo_clarity']}),
    ('cut me out and give me a transparent png', 'image', {'tools': ['remove_background', 'convert_image_format']}),
    # Class 3: conversions keep full length unless a range is asked for (no leaked GIF defaults)
    ('grab a still at 00:03 and also convert the clip to webm', 'video',
     {'tools': ['extract_frame', 'convert_video_format'], 'args': {0: {'time_sec': 3}}, 'exact_args': {1: WEBM}}),
    ('convert to webm', 'video', {'tools': ['convert_video_format'], 'exact_args': {0: WEBM}}),
    ('export as mkv', 'video', {'tools': ['convert_video_format'], 'exact_args': {0: {'target_format': 'mkv'}}}),
    ('make it an mp4', 'video', {'tools': ['convert_video_format'], 'exact_args': {0: {'target_format': 'mp4'}}}),
    ('save the video in webm format', 'video', {'tools': ['convert_video_format'], 'exact_args': {0: WEBM}}),
    ('mute it and convert to webm', 'video', {'tools': ['mute_video', 'convert_video_format'], 'exact_args': {1: WEBM}}),
    ('convert to webm and compress', 'video',
     {'tools': ['compress_video', 'convert_video_format'], 'exact_args': {1: WEBM}}),
    ('trim 2 to 6 and convert to webm', 'video',
     {'tools': ['trim_video', 'convert_video_format'], 'exact_args': {1: WEBM}, 'context': D20}),
    ('convert 2 to 6 seconds into webm', 'video',
     {'tools': ['trim_video', 'convert_video_format'], 'args': {0: {'start_sec': 2, 'end_sec': 6}},
      'exact_args': {1: WEBM}}),
    ('make a gif', 'video',
     {'tools': ['convert_video_format'], 'exact_args': {0: {'target_format': 'gif', 'start': 0.0, 'duration': 5.0}}}),
    ('make a gif from 2 to 4 seconds', 'video',
     {'tools': ['convert_video_format'], 'exact_args': {0: {'target_format': 'gif', 'start': 2.0, 'duration': 2.0}}}),
    ('thumbnail at 2s then export as webm', 'video',
     {'tools': ['extract_frame', 'convert_video_format'], 'exact_args': {1: WEBM}}),
    ('extract the audio as mp3', 'video', {'tools': ['extract_audio'], 'exact_args': {0: {'format': 'mp3'}}}),
    ('export as jpg', 'image', {'tools': ['convert_image_format'], 'exact_args': {0: {'target_format': 'jpg'}}}),
    ('convert to flac', 'audio', {'tools': ['convert_audio_format'], 'exact_args': {0: {'target_format': 'flac'}}}),
    ('compress it', 'video', {'tools': ['compress_video'], 'exact_args': {0: {'level': 'balanced'}}}),
    # Class 4: removing an object/person (inpainting) is not a background cutout
    ('remove the guy in the background', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('remove the people in the background', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('erase the car from the background', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('get rid of the tree behind me', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('remove the photobomber', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('delete the person on the left', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('remove the watermark', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('take out the trash can in the background', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('remove the text from the photo', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('erase the power lines in the sky', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('remove the man behind her and upscale 2x', 'image', {'tools': [], 'clarify': True, 'reply': 'specific object'}),
    ('remove the background', 'image', {'tools': ['remove_background']}),
    ('remove the busy background behind me', 'image', {'tools': ['remove_background']}),
    ('remove background from the guy', 'image', {'tools': ['remove_background']}),
    ('make the background transparent', 'image', {'tools': ['remove_background']}),
    ('remove the background noise', 'audio', {'tools': ['reduce_noise']}),
    # Class 5: compatible fades merge into one step (one encode)
    ('fade in 1.5s, fade out 3 sec', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 1.5, 'fade_out_sec': 3.0}}}),
    ('fade in 2 seconds then fade out 4 seconds', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 2.0, 'fade_out_sec': 4.0}}}),
    ('add a fade in of 1s and a fade out of 2s', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 1.0, 'fade_out_sec': 2.0}}}),
    ('fade out 3s and fade in 0.5s', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 0.5, 'fade_out_sec': 3.0}}}),
    ('1. fade in 1s\n2. fade out 2s', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 1.0, 'fade_out_sec': 2.0}}}),
    ('fade in 1s, normalize, fade out 2s', 'audio',
     {'tools': ['apply_audio_fade', 'normalize_audio', 'apply_audio_fade']}),
    ('fade in over 2 seconds; also fade out over 5', 'video',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 2.0, 'fade_out_sec': 5.0}}}),
    ('fade-in 300ms, fade-out 700ms', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 0.3, 'fade_out_sec': 0.7}}}),
    ('trim 1 to 9 then fade in 1s and then fade out 1s', 'audio',
     {'tools': ['trim_audio', 'apply_audio_fade'], 'exact_args': {1: {'fade_in_sec': 1.0, 'fade_out_sec': 1.0}}}),
    ('fade in 2s and fade in 3s', 'audio', {'tools': ['apply_audio_fade'], 'args': {0: {'fade_in_sec': 3.0}}}),
    ('fade out 2s, fade in 1s, then export mp3', 'audio',
     {'tools': ['apply_audio_fade', 'convert_audio_format'],
      'exact_args': {0: {'fade_in_sec': 1.0, 'fade_out_sec': 2.0}}}),
    ('slow fade in of 4 seconds and a quick 1 second fade out', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 4.0, 'fade_out_sec': 1.0}}}),
    ('fade in 1s. fade out 1s.', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 1.0, 'fade_out_sec': 1.0}}}),
    ('fade in for 2s, then fade it out for 2s', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 2.0, 'fade_out_sec': 2.0}}}),
    ('give it a 2 second fade-in, and a 6 second fade-out', 'audio',
     {'tools': ['apply_audio_fade'], 'exact_args': {0: {'fade_in_sec': 2.0, 'fade_out_sec': 6.0}}}),
]


class UnseenPhrasingEval(unittest.TestCase):
    results = []

    def test_unseen_phrasings(self):
        for prompt, media_type, expectation in UNSEEN:
            problems = check_case(prompt, media_type, expectation)
            UnseenPhrasingEval.results.append((prompt, media_type, problems))
            with self.subTest(prompt=prompt, media=media_type):
                self.assertEqual(problems, [])

    @classmethod
    def tearDownClass(cls):
        if not cls.results:
            return
        passed = sum(1 for _, _, problems in cls.results if not problems)
        print(f'\n[Unseen-phrasing set] {passed}/{len(cls.results)} correct ({100.0 * passed / len(cls.results):.1f}%).')
        for prompt, media_type, problems in cls.results:
            if problems:
                print(f'  FAIL [{media_type}] {prompt!r}: {"; ".join(problems)}')


class GoldenPlanEval(unittest.TestCase):
    results = []

    def test_golden_set(self):
        self.assertGreaterEqual(len(GOLDEN), 50)
        for prompt, media_type, expectation in GOLDEN:
            problems = check_case(prompt, media_type, expectation)
            GoldenPlanEval.results.append((prompt, media_type, problems))
            with self.subTest(prompt=prompt, media=media_type):
                self.assertEqual(problems, [])

    @classmethod
    def tearDownClass(cls):
        if not cls.results:
            return
        passed = sum(1 for _, _, problems in cls.results if not problems)
        total = len(cls.results)
        print(f'\n[Golden set] {passed}/{total} prompts planned correctly ({100.0 * passed / total:.1f}%).')
        for prompt, media_type, problems in cls.results:
            if problems:
                print(f'  FAIL [{media_type}] {prompt!r}: {"; ".join(problems)}')


class SuggestionActionabilityEval(unittest.TestCase):
    """Every suggestion pill / clarification option the copilot emits must itself plan to an action."""

    def test_every_emitted_suggestion_is_actionable(self):
        sources = {media: list(options) for media, options in agent_planner.SUGGESTIONS_BY_TYPE.items() if media}
        for path, media in [('x.png', 'image'), ('x.mp4', 'video'), ('x.wav', 'audio')]:
            sources[media] += agent_processor.InspectorSubAgent.inspect([], None)['suggested_actions']
            with tempfile.TemporaryDirectory() as folder:
                sample = os.path.join(folder, path)
                with open(sample, 'wb') as handle:
                    handle.write(b'x')
                sources[media] += agent_processor.InspectorSubAgent.inspect([], sample)['suggested_actions']
        greeting = plan_for('hello')
        sources['video'] += [s for s in greeting['suggested_actions'] if 'video' in s.lower() or 'trim' in s.lower()]
        sources['image'] += [s for s in greeting['suggested_actions'] if 'image' in s.lower() or 'background' in s.lower()]
        sources['audio'] += [s for s in greeting['suggested_actions'] if 'audio' in s.lower() or 'transcribe' in s.lower()]
        for prompt, media in [('Trim the video', 'video'), ('Make it faster', 'audio'), ('export as m4a', 'audio'),
                              ('what is lufs?', 'audio'), ('enhance', 'video'), ('mp4 vs webm', 'video')]:
            plan = plan_for(prompt, media)
            sources[media] += plan['clarification_options'] + plan['suggested_actions']
        starter_cards = ['Trim video from 2.3 second to 5.3 second', 'Remove background and isolate subject',
                         'Transcribe speech to text and clean background noise', 'Upscale image to 4x high resolution']
        sources['video'].append(starter_cards[0])
        sources['image'] += [starter_cards[1], starter_cards[3]]
        sources['audio'].append(starter_cards[2])
        for media, options in sources.items():
            for option in set(options):
                with self.subTest(media=media, suggestion=option):
                    plan = plan_for(option, media)
                    self.assertTrue(plan['tools'], f'{option!r} on {media} plans nothing: {plan["reply"][:120]}')

    def test_run_in_my_order_pill_is_actionable(self):
        plan = plan_for('normalize then denoise', 'audio')
        self.assertIn('Run exactly in my order', plan['suggested_actions'])
        follow = plan_for('Run exactly in my order', 'audio', H('normalize then denoise', plan['reply']))
        self.assertEqual(names(follow), ['normalize_audio', 'reduce_noise'])


# ═══════════════════════════════════════════════════════════════════════════
#  2. VALIDATION / REPAIR LAYER
# ═══════════════════════════════════════════════════════════════════════════

class ValidationLayerTests(unittest.TestCase):

    def finalize(self, tools, media_type='audio', prompt='', **context):
        return agent_planner.finalize_plan({'tools': tools}, prompt, {'type': media_type, **context}, None, 'reasoning')

    def test_unknown_tool_is_dropped_with_explanation_and_rest_runs(self):
        plan = self.finalize([{'name': 'normalize_audio', 'args': {}}, {'name': 'teleport', 'args': {}},
                              {'name': 'apply_audio_fade', 'args': {'fade_in_sec': 1}}])
        self.assertEqual(names(plan), ['normalize_audio', 'apply_audio_fade'])
        self.assertTrue(any('teleport' in note for note in plan['plan_notes']))

    def test_aliases_and_string_values_are_coerced(self):
        plan = self.finalize([{'name': 'trim', 'args': {'start': '0:02', 'end': '00:10'}},
                              {'name': 'speed', 'args': {'speed': '1.5x'}},
                              {'name': 'fade', 'args': {'fade_in': '500ms', 'fade_out': '2s'}},
                              {'name': 'normalize', 'args': {'lufs': '-14 LUFS'}}])
        self.assertEqual(names(plan), ['trim_audio', 'adjust_audio_speed', 'apply_audio_fade', 'normalize_audio'])
        self.assertEqual(plan['tools'][0]['args'], {'start_sec': 2.0, 'end_sec': 10.0})
        self.assertEqual(plan['tools'][1]['args']['speed'], 1.5)
        self.assertEqual(plan['tools'][2]['args'], {'fade_in_sec': 0.5, 'fade_out_sec': 2.0})
        self.assertEqual(plan['tools'][3]['args']['target_lufs'], -14.0)

    def test_media_type_is_tracked_across_steps(self):
        plan = self.finalize([{'name': 'extract_audio', 'args': {'format': 'wav'}},
                              {'name': 'trim_video', 'args': {'start_sec': 1, 'end_sec': 2}},
                              {'name': 'adjust_video_speed', 'args': {'speed': 2}},
                              {'name': 'upscale_image', 'args': {'scale': 2}}], 'video')
        self.assertEqual(names(plan), ['extract_audio', 'trim_audio', 'adjust_audio_speed'])
        self.assertTrue(any('works on image' in note for note in plan['plan_notes']))

    def test_soundtrack_tracking_after_mute(self):
        plan = self.finalize([{'name': 'mute_video', 'args': {}}, {'name': 'normalize_audio', 'args': {}},
                              {'name': 'trim_video', 'args': {'start_sec': 0, 'end_sec': 1}}], 'video')
        self.assertEqual(names(plan), ['mute_video', 'trim_video'])

    def test_missing_required_or_out_of_range_values_ask_for_clarification(self):
        for tools in ([{'name': 'adjust_audio_speed', 'args': {}}], [{'name': 'adjust_audio_speed', 'args': {'speed': 9}}],
                      [{'name': 'convert_audio_format', 'args': {'target_format': '../../etc'}}],
                      [{'name': 'trim_audio', 'args': {}}], [{'name': 'adjust_volume', 'args': {'gain_db': 'loud'}}]):
            with self.subTest(tools=tools):
                plan = self.finalize(tools)
                self.assertTrue(plan['clarification_needed'])
                self.assertEqual(plan['tools'], [])

    def test_unsafe_values_are_clamped_with_notes(self):
        plan = self.finalize([{'name': 'apply_audio_fade', 'args': {'fade_in_sec': 90, 'fade_out_sec': -3}},
                              {'name': 'adjust_volume', 'args': {'gain_db': 80}},
                              {'name': 'normalize_audio', 'args': {'target_lufs': 3}}], duration=None)
        self.assertEqual(plan['tools'][0]['args'], {'fade_in_sec': 30.0, 'fade_out_sec': 0.0})
        self.assertEqual(plan['tools'][1]['args']['gain_db'], 30.0)
        self.assertEqual(plan['tools'][2]['args']['target_lufs'], -5.0)
        self.assertGreaterEqual(len(plan['plan_notes']), 3)

    def test_fades_are_fitted_to_known_duration(self):
        plan = self.finalize([{'name': 'apply_audio_fade', 'args': {'fade_in_sec': 2, 'fade_out_sec': 2}}], duration=1.0)
        args = plan['tools'][0]['args']
        self.assertAlmostEqual(args['fade_in_sec'] + args['fade_out_sec'], 1.0, places=3)

    def test_internal_and_unknown_args_are_stripped(self):
        plan = self.finalize([{'name': 'remove_background', 'args': {'model_name': 'secret-net', 'engine': 'x',
                                                                    'quality_profile': 'person'}}], 'image')
        self.assertEqual(plan['tools'][0]['args'], {'quality_profile': 'portrait', 'refine': False})

    def test_plan_has_explicit_ids_inputs_and_dependencies(self):
        plan = self.finalize([{'name': 'trim_video', 'args': {'start_sec': 0, 'end_sec': 1}},
                              {'name': 'transcribe_audio', 'args': {}},
                              {'name': 'extract_frame', 'args': {'time_sec': 0.2}},
                              {'name': 'compress_video', 'args': {}}], 'video')
        steps = plan['tools']
        self.assertEqual([step['id'] for step in steps], ['s1', 's2', 's3', 's4'])
        self.assertEqual(steps[0]['depends_on'], [])
        self.assertEqual(steps[1]['input'], 's1')           # analysis reads the trimmed result
        self.assertEqual(steps[2]['input'], 'source')       # thumbnail time refers to the original timeline
        self.assertTrue(steps[2]['branch'])
        self.assertEqual(steps[3]['input'], 's1')           # analysis/branch steps do not advance the chain

    def test_malformed_model_fields_are_normalized(self):
        raw = {'tools': [{'name': 'normalize_audio', 'args': {}}], 'clarification_needed': 'false',
               'reply': {'text': 'x'}, 'suggested_actions': 'Do more', 'clarification_options': 'A', 'thought': ['t']}
        plan = agent_planner.finalize_plan(raw, '', {'type': 'audio'}, None, 'reasoning')
        self.assertFalse(plan['clarification_needed'])
        self.assertEqual(names(plan), ['normalize_audio'])
        self.assertIsInstance(plan['reply'], str)
        self.assertIsInstance(plan['thought'], str)
        self.assertTrue(all(isinstance(item, str) for item in plan['suggested_actions']))
        weird = agent_planner.finalize_plan({'tools': 'normalize', 'steps': None}, '', {'type': 'audio'}, None, 'reasoning')
        self.assertEqual(weird['tools'], [])

    def test_strict_execution_validation(self):
        for step, media in [({'name': 'trim_video', 'args': {'start_sec': 0.8, 'end_sec': 0.2}}, 'video'),
                            ({'name': 'adjust_video_speed', 'args': {'speed': 0}}, 'video'),
                            ({'name': 'convert_video_format', 'args': {'target_format': '../../unsafe'}}, 'video'),
                            ({'name': 'upscale_image', 'args': {'scale': 2}}, 'audio'),
                            ({'name': 'normalize_audio', 'args': {}}, 'image'),
                            ({'name': 'not_a_tool', 'args': {}}, 'audio')]:
            with self.subTest(step=step):
                with self.assertRaises(agent_processor.StepInputError):
                    agent_processor._strict_check(step, {'type': media, 'has_audio': media != 'image'})

    def test_execution_steps_infer_dependencies_for_raw_plans(self):
        steps = agent_planner.execution_steps([{'name': 'normalize_audio', 'args': {}}, {'name': 'inspect_media', 'args': {}},
                                               {'name': 'bogus', 'args': {}}, {'name': 'apply_audio_fade', 'args': {}}])
        self.assertEqual([step['input'] for step in steps], ['source', 'step1', 'step1', 'step3'])

    def test_null_media_type_from_studio_drawer_is_safe(self):
        self.assertEqual(names(plan_for('Trim from 1 to 2 seconds', None, type=None)), ['trim_video'])
        from app import app
        app.config['TESTING'] = True
        response = app.test_client().post('/api/agent/chat', json={'message': 'Trim from 1 to 2 seconds',
                                                                   'context': {'type': None, 'duration': None}})
        self.assertEqual(response.status_code, 400)          # "upload media first", never a server error
        self.assertIn('Upload', response.get_json()['error'])

    def test_refined_cutout_only_escalates_from_the_general_profile(self):
        cases = [({'quality_profile': 'detail', 'refine': False}, 'auto'),
                 ({'quality_profile': 'detail', 'refine': True}, 'portrait'),
                 ({'quality_profile': 'portrait', 'refine': True}, 'portrait'),
                 ({'quality_profile': 'studio', 'refine': True}, 'studio')]
        for args, expected_model in cases:
            with self.subTest(args=args), tempfile.TemporaryDirectory() as folder, \
                    patch('agent_processor.image_processor.remove_bg') as remove_bg:
                agent_processor.VisionSubAgent.execute('remove_background', args, 'in.png', folder)
                self.assertEqual(remove_bg.call_args.kwargs['model_name'], expected_model)

    def test_public_registry_and_prompt_are_private(self):
        blob = (json.dumps(agent_processor.TOOL_DEFINITIONS) + agent_processor.SYSTEM_PROMPT).lower()
        for term in FORBIDDEN_TERMS:
            self.assertNotIn(term, blob)
        self.assertIn('never name or describe the underlying models', blob)
        self.assertIn('"tools"', blob)


# ═══════════════════════════════════════════════════════════════════════════
#  3. REASONING-SERVICE OUTPUT HANDLING (same validation as the local parser)
# ═══════════════════════════════════════════════════════════════════════════

class ReasoningPathTests(unittest.TestCase):

    def run_with(self, content, prompt, media_type='audio', history=None):
        with patch('agent_processor._local_reasoning_url', return_value='http://127.0.0.1:9/v1/chat/completions'), \
                patch('agent_processor._request_reasoning_content', return_value=content):
            return agent_processor.query_agent_orchestrator(prompt, {'type': media_type}, history)

    def test_fenced_json_inside_prose_and_reasoning_tags(self):
        body = json.dumps({'tools': [{'name': 'trim_audio', 'args': {'start_sec': 1, 'end_sec': 2}}], 'reply': 'ok'})
        for content in (f'Sure! Here is the plan:\n```json\n{body}\n```\nHope it helps.',
                        f'<think>user wants a trim {{maybe}}</think>{body}', f'<think>unterminated reasoning {body}',
                        body):
            with self.subTest(content=content[:30]):
                plan = self.run_with(content, 'trim 1 to 2')
                self.assertEqual(names(plan), ['trim_audio'])
                self.assertTrue(plan['validated'])

    def test_prose_without_json_falls_back_to_local_plan_for_edit_requests(self):
        plan = self.run_with('I will trim it for you now!', 'Trim from 0.2 to 0.8 seconds')
        self.assertEqual(names(plan), ['trim_audio'])

    def test_conversational_reply_is_kept_for_questions_and_scrubbed(self):
        plan = self.run_with('Loudness is measured in LUFS. Our whisper engine also helps.', 'what is loudness?')
        self.assertEqual(plan['tools'], [])
        self.assertIn('LUFS', plan['reply'])
        self.assertNotIn('whisper', plan['reply'].lower())

    def test_invalid_reasoning_plan_uses_local_plan(self):
        content = json.dumps({'tools': [{'name': 'hyperdrive', 'args': {}}], 'reply': 'done'})
        plan = self.run_with(content, 'normalize then fade out over 2 seconds')
        self.assertEqual(names(plan), ['normalize_audio', 'apply_audio_fade'])

    def test_wrong_media_type_from_model_is_repaired(self):
        content = json.dumps({'tools': [{'name': 'trim_video', 'args': {'start_sec': '0:01', 'end_sec': '0:03'}},
                                        {'name': 'adjust_video_speed', 'args': {'speed': '2x'}}]})
        plan = self.run_with(content, 'trim 1 to 3 and double speed')
        self.assertEqual(names(plan), ['trim_audio', 'adjust_audio_speed'])

    def test_string_false_clarification_does_not_drop_valid_tools(self):
        content = json.dumps({'tools': [{'name': 'normalize_audio', 'args': {}}], 'clarification_needed': 'false',
                              'suggested_actions': 'Something', 'reply': {'bad': 1}})
        plan = self.run_with(content, 'normalize')
        self.assertEqual(names(plan), ['normalize_audio'])
        self.assertIsInstance(plan['reply'], str)

    def test_service_failure_falls_back(self):
        with patch('agent_processor._local_reasoning_url', return_value='http://127.0.0.1:9/v1/chat/completions'), \
                patch('agent_processor._request_reasoning_content', side_effect=OSError('down')):
            plan = agent_processor.query_agent_orchestrator('Fade edges', {'type': 'audio'})
        self.assertEqual(names(plan), ['apply_audio_fade'])

    def test_llm_and_local_paths_agree_on_golden_chains(self):
        cases = [
            ('Trim from 2s to 8s, remove the background noise, normalize loudness, then export as mp3', 'video',
             [{'name': 'trim', 'args': {'start': 2, 'end': 8}}, {'name': 'denoise', 'args': {}},
              {'name': 'normalize', 'args': {}}, {'name': 'export', 'args': {'format': 'mp3'}}]),
            ('Extract the audio as wav, speed it up 1.25x and add a 1 second fade in and out', 'video',
             [{'name': 'extract_audio', 'args': {'format': 'wav'}}, {'name': 'speed', 'args': {'speed': 1.25}},
              {'name': 'fade', 'args': {'fade_in': 1, 'fade_out': 1}}]),
        ]
        for prompt, media, model_tools in cases:
            with self.subTest(prompt=prompt):
                via_model = self.run_with(json.dumps({'tools': model_tools}), prompt, media)
                via_rules = plan_for(prompt, media)
                self.assertEqual([(t['name'], t['args']) for t in via_model['tools']],
                                 [(t['name'], t['args']) for t in via_rules['tools']])


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHETIC MEDIA
# ═══════════════════════════════════════════════════════════════════════════

SR = 16000


def speech_like(seconds=4.0, amplitude=0.4):
    """Syllable-rate pulses of a harmonic voice-like tone separated by short gaps."""
    t = np.arange(int(SR * seconds)) / SR
    y = np.zeros_like(t)
    position, k = 0.0, 0
    while position + 0.22 < seconds:
        start, end = int(position * SR), int((position + 0.22) * SR)
        tt = t[start:end] - position
        f0 = 120 + 30 * math.sin(k)
        y[start:end] = sum(np.sin(2 * np.pi * f0 * h * tt) / h for h in range(1, 12)) * np.hanning(end - start)
        position += 0.36
        k += 1
    return amplitude * y / np.max(np.abs(y))


def write_wav(path, samples, rate=SR, channels=1):
    data = (np.clip(samples, -1, 1) * 32767).astype('<i2')
    with wave.open(path, 'wb') as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return path


class SyntheticMedia:
    folder = None

    @classmethod
    def build(cls):
        if cls.folder:
            return cls
        cls.folder = tempfile.mkdtemp(prefix='agent-eval-')
        rng = np.random.default_rng(7)
        speech = speech_like()
        t = np.arange(speech.size) / SR
        chord = 0.06 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t) + np.sin(2 * np.pi * 330 * t))
        cls.clean = write_wav(os.path.join(cls.folder, 'clean.wav'), speech)
        cls.noisy = write_wav(os.path.join(cls.folder, 'noisy.wav'), speech + 0.05 * rng.standard_normal(speech.size))
        cls.music = write_wav(os.path.join(cls.folder, 'music.wav'), speech + chord)
        cls.quiet = write_wav(os.path.join(cls.folder, 'quiet.wav'), speech * 0.03)
        cls.silent = write_wav(os.path.join(cls.folder, 'silent.wav'), np.zeros(SR))
        cls.tone = write_wav(os.path.join(cls.folder, 'tone.wav'), 0.4 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR), 44100 // 1)

        image = Image.new('RGB', (400, 300), (30, 40, 60))
        draw = ImageDraw.Draw(image)
        for x in range(0, 400, 20):
            draw.line([(x, 0), (399 - x, 299)], fill=(220, 200, 90), width=2)
        cls.sharp = os.path.join(cls.folder, 'sharp.png')
        image.save(cls.sharp)
        cls.blurry = os.path.join(cls.folder, 'blurry.png')
        image.filter(ImageFilter.GaussianBlur(5)).save(cls.blurry)
        grain = np.asarray(image).astype(float) + rng.normal(0, 18, (300, 400, 3))
        cls.grainy = os.path.join(cls.folder, 'grainy.png')
        Image.fromarray(np.clip(grain, 0, 255).astype('uint8')).save(cls.grainy)
        cutout = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(cutout).ellipse([50, 50, 150, 150], fill=(200, 60, 60, 255))
        cls.transparent = os.path.join(cls.folder, 'cutout.png')
        cutout.save(cls.transparent)

        cls.video = os.path.join(cls.folder, 'clip.mp4')
        subprocess.run([video_processor.FFMPEG, '-y', '-f', 'lavfi', '-i', 'testsrc=duration=1.5:size=160x120:rate=15',
                        '-f', 'lavfi', '-i', 'sine=frequency=660:duration=1.5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', '-shortest', cls.video], capture_output=True, check=True)
        cls.stereo = os.path.join(cls.folder, 'stereo.wav')
        left = 0.3 * np.sin(2 * np.pi * 300 * np.arange(SR) / SR)
        voice = speech_like(1.0, 0.3)
        write_wav(cls.stereo, np.column_stack([left + voice, -left + voice]).reshape(-1), channels=2)
        return cls

    @classmethod
    def cleanup(cls):
        if cls.folder:
            shutil.rmtree(cls.folder, ignore_errors=True)
            cls.folder = None


def context_for(path):
    return agent_processor._media_context_for_file(path, {})


# ═══════════════════════════════════════════════════════════════════════════
#  4. CONDITION-AWARE PLANNING
# ═══════════════════════════════════════════════════════════════════════════

class ConditionAwarePlanningTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.media = SyntheticMedia.build()

    def plan(self, prompt, path, history=None):
        return agent_processor._fallback_intent_parser(prompt, context_for(path), history)

    def test_perception_classifies_synthetic_audio(self):
        expectations = {self.media.clean: 'clean', self.media.noisy: 'noise', self.media.music: 'music',
                        self.media.quiet: 'clean'}
        for path, background in expectations.items():
            with self.subTest(path=os.path.basename(path)):
                report = media_inspector.perceive(path, 'audio')['audio']
                self.assertEqual(report['background'], background)
                self.assertTrue(report['speech_likely'])
        self.assertTrue(media_inspector.perceive(self.media.quiet, 'audio')['audio']['quiet'])
        self.assertTrue(media_inspector.perceive(self.media.silent, 'audio')['audio']['silent'])

    def test_transcription_preparation_depends_on_condition(self):
        cases = {self.media.clean: ['transcribe_audio'],
                 self.media.noisy: ['reduce_noise', 'transcribe_audio'],
                 self.media.music: ['isolate_voice', 'transcribe_audio'],
                 self.media.quiet: ['normalize_audio', 'transcribe_audio']}
        for path, expected in cases.items():
            with self.subTest(path=os.path.basename(path)):
                plan = self.plan('Transcribe this recording', path)
                self.assertEqual(names(plan), expected)
                if len(expected) > 1:
                    prep = plan['tools'][0]
                    self.assertEqual(prep['role'], 'prep')
                    self.assertTrue(prep['branch'])                       # the user's media is not altered
                    self.assertEqual(plan['tools'][1]['input'], prep['id'])
                    self.assertTrue(prep['reason'])
                    self.assertIn(prep['reason'], plan['reply'])

    def test_music_reason_quantifies_the_condition(self):
        plan = self.plan('give me subtitles', self.media.music)
        self.assertRegex(plan['tools'][0]['reason'], r'Background music detected \(\d+ dB under the speech\)')
        self.assertEqual(plan['tools'][0]['fallback']['name'], 'enhance_speech')

    def test_user_overrides_beat_heuristics(self):
        for prompt in ('Transcribe as-is', "Transcribe it but don't denoise", 'transcribe without any processing'):
            with self.subTest(prompt=prompt):
                self.assertEqual(names(self.plan(prompt, self.media.noisy)), ['transcribe_audio'])

    def test_explicit_cleanup_is_not_duplicated(self):
        plan = self.plan('Remove the noise and then transcribe', self.media.noisy)
        self.assertEqual(names(plan), ['reduce_noise', 'transcribe_audio'])
        self.assertEqual(plan['tools'][0]['role'], 'requested')

    def test_silent_audio_skips_transcription(self):
        plan = self.plan('Transcribe', self.media.silent)
        self.assertEqual(plan['tools'], [])
        self.assertTrue(any('nothing to transcribe' in note for note in plan['plan_notes']))

    def test_video_subtitles_prepare_a_soundtrack_copy_when_needed(self):
        with patch('media_inspector.analyze_audio', return_value={**media_inspector.analyze_audio(self.media.noisy)}):
            media_inspector.clear_cache()
            plan = self.plan('Generate subtitles', self.media.video)
        media_inspector.clear_cache()
        self.assertEqual(names(plan), ['extract_audio', 'reduce_noise', 'transcribe_audio'])
        self.assertTrue(all(step['branch'] for step in plan['tools'][:2]))

    def test_image_plans_depend_on_focus_and_grain(self):
        self.assertEqual(names(self.plan('Upscale 2x', self.media.sharp)), ['upscale_image'])
        blurry = self.plan('Upscale 2x', self.media.blurry)
        self.assertEqual(names(blurry), ['upscale_image', 'enhance_photo_clarity'])
        self.assertEqual(blurry['tools'][1]['role'], 'post')
        grainy = self.plan('Upscale 2x', self.media.grainy)
        self.assertEqual(names(grainy), ['enhance_photo_clarity', 'upscale_image'])
        self.assertIn('grain', grainy['tools'][0]['reason'])
        self.assertEqual(names(self.plan("Upscale 2x, don't sharpen", self.media.blurry)), ['upscale_image'])
        stronger = self.plan('Boost clarity', self.media.blurry)
        self.assertEqual(stronger['tools'][0]['args']['sharpen_strength'], 1.8)
        self.assertEqual(self.plan('Boost clarity', self.media.sharp)['tools'][0]['args']['sharpen_strength'], 1.2)

    def test_no_op_detection(self):
        lufs = media_inspector.perceive(self.media.clean, 'audio')['audio']['loudness_lufs']
        on_target = self.plan(f'normalize loudness to {lufs:.1f} LUFS', self.media.clean)
        self.assertEqual(on_target['tools'], [])
        self.assertTrue(any('already on target' in note for note in on_target['plan_notes']))
        self.assertEqual(names(self.plan('normalize loudness to -30 LUFS', self.media.clean)), ['normalize_audio'])
        self.assertEqual(self.plan('remove the background', self.media.transparent)['tools'], [])
        self.assertEqual(names(self.plan('refine the cutout', self.media.transparent)), ['remove_background'])

    def test_job_memory_swaps_a_repeatedly_failing_preparation_step(self):
        memory = [{'recommendation': 'avoid', 'failed_tool': 'isolate_voice', 'runs': 3, 'successes': 0}]
        with patch('agent_memory.recall_similar_jobs', return_value=memory):
            plan = self.plan('Transcribe', self.media.music)
            requested = self.plan('isolate the voice then transcribe', self.media.music)
        self.assertEqual(names(plan), ['enhance_speech', 'transcribe_audio'])
        self.assertTrue(any('did not work on similar media' in note for note in plan['plan_notes']))
        self.assertEqual(names(requested)[0], 'isolate_voice')                 # explicit user steps are never changed
        with patch('agent_memory.recall_similar_jobs', side_effect=RuntimeError('store offline')):
            self.assertEqual(names(self.plan('Transcribe', self.media.music)), ['isolate_voice', 'transcribe_audio'])

    def test_reasoning_plans_get_the_same_condition_policies(self):
        content = json.dumps({'tools': [{'name': 'transcribe_audio', 'args': {}}]})
        with patch('agent_processor._local_reasoning_url', return_value='http://127.0.0.1:9/v1/chat/completions'), \
                patch('agent_processor._request_reasoning_content', return_value=content) as request:
            plan = agent_processor.query_agent_orchestrator('transcribe', context_for(self.media.noisy))
        self.assertEqual(names(plan), ['reduce_noise', 'transcribe_audio'])
        sent = request.call_args.args[1]['messages'][1]['content']
        self.assertIn('Measured condition', sent)
        self.assertNotIn(self.media.folder, sent)               # paths never reach the model


# ═══════════════════════════════════════════════════════════════════════════
#  5. EXECUTION ON SYNTHETIC MEDIA
# ═══════════════════════════════════════════════════════════════════════════

class ExecutionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.media = SyntheticMedia.build()

    def setUp(self):
        self.out = tempfile.mkdtemp(prefix='agent-eval-out-')

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def run_prompt(self, prompt, path):
        context = context_for(path)
        plan = agent_processor._fallback_intent_parser(prompt, context)
        self.assertFalse(plan['clarification_needed'], plan['reply'])
        return plan, agent_processor.execute_agent_plan(path, plan['tools'], self.out, context)

    def test_audio_chain_executes_in_order_with_timing(self):
        plan, results = self.run_prompt('Trim from 0.5s to 2.5s, normalize, fade in 0.1s and out 0.2s, then export as mp3',
                                        self.media.clean)
        self.assertEqual([step['tool'] for step in results],
                         ['trim_audio', 'normalize_audio', 'apply_audio_fade', 'convert_audio_format'])
        self.assertTrue(all(step['status'] == 'success' for step in results), results)
        self.assertTrue(all(isinstance(step['duration_ms'], int) and step['attempts'] == 1 for step in results))
        final = agent_processor.final_output_of(results)
        self.assertTrue(final.endswith('.mp3'))
        self.assertAlmostEqual(video_processor.probe_media(final)['duration'], 2.0, delta=0.15)

    def test_video_chain_with_side_output_thumbnail(self):
        plan, results = self.run_prompt('Mute the video, then cut 0.2 to 1.0 seconds, and also give me a thumbnail at 0.5s',
                                        self.media.video)
        self.assertTrue(all(step['status'] == 'success' for step in results), results)
        final = agent_processor.final_output_of(results)
        info = video_processor.probe_media(final)
        self.assertFalse(info['has_audio'])
        self.assertAlmostEqual(info['duration'], 0.8, delta=0.2)
        artifacts = agent_processor.side_outputs_of(results)
        self.assertEqual(len(artifacts), 1)
        with Image.open(artifacts[0]['output_file']) as frame:
            self.assertEqual(frame.size, (160, 120))

    def test_container_conversion_keeps_full_length(self):
        plan, results = self.run_prompt('grab a still at 0.5s and also convert the clip to webm', self.media.video)
        self.assertEqual(plan['tools'][1]['args'], {'target_format': 'webm'})
        self.assertTrue(all(step['status'] == 'success' for step in results), results)
        final = agent_processor.final_output_of(results)
        self.assertTrue(final.endswith('.webm'))
        self.assertAlmostEqual(video_processor.probe_media(final)['duration'], 1.5, delta=0.2)
        self.assertEqual(len(agent_processor.side_outputs_of(results)), 1)

    def test_trim_both_ends_executes(self):
        plan, results = self.run_prompt('chop off the first 0.3 seconds and the last 0.2, then give me the audio as a wav',
                                        self.media.video)
        self.assertEqual([step['tool'] for step in results], ['trim_video', 'extract_audio'])
        final = agent_processor.final_output_of(results)
        self.assertAlmostEqual(video_processor.probe_media(final)['duration'], 1.0, delta=0.12)

    def test_consecutive_soundtrack_edits_are_fused(self):
        tools = [{'name': 'normalize_audio', 'args': {}}, {'name': 'apply_audio_fade', 'args': {'fade_in_sec': 0.2, 'fade_out_sec': 0.2}},
                 {'name': 'adjust_volume', 'args': {'gain_db': -3}}]
        with patch('agent_processor.video_processor.extract_audio', wraps=video_processor.extract_audio) as extract, \
                patch('agent_processor._remux_soundtrack', wraps=agent_processor._remux_soundtrack) as remux:
            results = agent_processor.execute_agent_plan(self.media.video, tools, self.out, {'type': 'video'})
        self.assertTrue(all(step['status'] == 'success' for step in results), results)
        self.assertEqual(extract.call_count, 1)
        self.assertEqual(remux.call_count, 1)
        self.assertEqual(len({step['output_file'] for step in results}), 1)
        self.assertTrue(video_processor.probe_media(results[-1]['output_file'])['has_video'])
        self.assertFalse([name for name in os.listdir(self.out) if name.startswith('agent-audio-')])

    def test_failure_blocks_only_dependent_steps(self):
        tools = [{'name': 'mute_video', 'args': {}}, {'name': 'trim_video', 'args': {'start_sec': 0.1, 'end_sec': 0.9}},
                 {'name': 'extract_frame', 'args': {'time_sec': 0.3}}]
        with patch('agent_processor.video_processor.quick_mute', side_effect=ValueError('Mute is broken for this file.')):
            results = agent_processor.execute_agent_plan(self.media.video, tools, self.out, {'type': 'video'})
        self.assertEqual([(step['tool'], step['status']) for step in results],
                         [('mute_video', 'error'), ('extract_frame', 'success')])
        self.assertEqual(results[0]['blocked_steps'], [{'step': 2, 'tool': 'trim_video'}])
        self.assertEqual(results[0]['message'], 'Mute is broken for this file.')    # app-authored ValueError shown
        reply = agent_processor.execution_reply({'plan_notes': []}, results, agent_processor.final_output_of(results))
        self.assertIn('stopped', reply)
        self.assertIn('Trim video', reply)

    def test_voiceover_stems_and_lyrics_tools(self):
        for prompt, tool, args in (('read this text aloud: hello world', 'generate_voiceover', {'text': 'hello world'}),
                                   ('make a voiceover "good morning everyone"', 'generate_voiceover', {'text': 'good morning everyone'}),
                                   ('split this into stems', 'separate_stems', {'mode': '4stem'}),
                                   ('make a karaoke version', 'separate_stems', {'mode': 'karaoke'}),
                                   ('separate the vocals and music', 'separate_stems', {'mode': 'vocals'}),
                                   ('get the lyrics', 'extract_lyrics', {})):
            with self.subTest(prompt=prompt):
                plan = agent_processor._fallback_intent_parser(prompt, context_for(self.media.music))
                self.assertEqual([t['name'] for t in plan['tools']], [tool], plan)
                for key, value in args.items():
                    self.assertEqual(plan['tools'][0]['args'][key], value)
        self.assertEqual(agent_planner.TOOL_REGISTRY['generate_voiceover'].params['speed'].maximum, 2.0)
        # existing phrasing keeps its meaning
        self.assertEqual([t['name'] for t in agent_processor._fallback_intent_parser(
            'isolate the vocals', context_for(self.media.music))['tools']], ['isolate_voice'])

    def test_separate_stems_returns_each_stem_as_artifact(self):
        results = agent_processor.execute_agent_plan(
            self.media.music, [{'name': 'separate_stems', 'args': {'mode': '4stem', 'quality': 'fast'}}],
            self.out, {'type': 'audio'})
        self.assertEqual(results[0]['status'], 'success', results)
        files = [results[0]['output_file']] + [e['output_file'] for e in results[0]['extra_outputs']]
        self.assertEqual(len(files), 4)
        self.assertTrue(all(os.path.isfile(f) for f in files))

    def test_voiceover_needs_no_media_content_and_validates_text(self):
        def fake(args, processed_dir):
            out = os.path.join(processed_dir, 'voiceover_test.wav')
            write_wav(out, np.ones(SR) * 0.1)
            return {'output_file': out, 'message': 'Voiceover ready.', 'data': {'duration': 1.0}}
        with patch('tts_processor.voiceover_tool', side_effect=fake):
            results = agent_processor.execute_agent_plan(self.media.video, [{'name': 'generate_voiceover', 'args': {'text': 'hi'}}],
                                                         self.out, {'type': 'video'})
        self.assertEqual(results[0]['status'], 'success', results)
        bad = agent_processor.execute_agent_plan(self.media.music, [{'name': 'generate_voiceover', 'args': {}}],
                                                 self.out, {'type': 'audio'})
        self.assertEqual(bad[0]['status'], 'error')

    def test_lyrics_export(self):
        fake = {'status': 'success', 'text': 'la la', 'lines': [{'start': 0, 'end': 1, 'text': 'la la'}],
                'srt': '1\n00:00:00,000 --> 00:00:01,000\nla la\n', 'vtt': 'WEBVTT\n', 'language': 'en'}
        with patch('separation_processor.lyrics', return_value=fake):
            results = agent_processor.execute_agent_plan(self.media.music, [{'name': 'extract_lyrics', 'args': {'format': 'txt'}}],
                                                         self.out, {'type': 'audio'})
        self.assertEqual(results[0]['status'], 'success', results)
        self.assertTrue(results[0]['data']['exports'][0]['url'].endswith('.txt'))

    def test_transient_failure_is_retried_once(self):
        real = video_processor.quick_mute
        calls = []

        def flaky(src, out):
            calls.append(out)
            if len(calls) == 1:
                raise RuntimeError('temporary encoder hiccup')
            return real(src, out)
        with patch('agent_processor.video_processor.quick_mute', side_effect=flaky):
            results = agent_processor.execute_agent_plan(self.media.video, [{'name': 'mute_video', 'args': {}}],
                                                         self.out, {'type': 'video'})
        self.assertEqual(results[0]['status'], 'success')
        self.assertEqual(results[0]['attempts'], 2)

    def test_internal_errors_are_not_exposed(self):
        with patch('agent_processor.video_processor.quick_mute', side_effect=RuntimeError('ffmpeg exploded in libx264')):
            results = agent_processor.execute_agent_plan(self.media.video, [{'name': 'mute_video', 'args': {}}],
                                                         self.out, {'type': 'video'})
        self.assertEqual(results[0]['status'], 'error')
        self.assertEqual(results[0]['attempts'], 2)
        self.assertNotIn('ffmpeg', results[0]['message'].lower())
        with patch('agent_processor.video_processor.quick_mute',
                   side_effect=ValueError(r'cannot read C:\Users\me\uploads\clip.mp4 with librosa')):
            results = agent_processor.execute_agent_plan(self.media.video, [{'name': 'mute_video', 'args': {}}],
                                                         self.out, {'type': 'video'})
        self.assertEqual(results[0]['attempts'], 1)                              # validation errors are not retried
        self.assertNotIn('Users', results[0]['message'])
        self.assertNotIn('librosa', results[0]['message'].lower())

    def test_planned_fallback_when_separation_is_unavailable(self):
        plan = agent_processor._fallback_intent_parser('isolate the voice', context_for(self.media.music))
        with patch('agent_processor.ai_processor.separate_stems', side_effect=RuntimeError('missing')), \
                patch('agent_processor.ai_processor.enhance_speech_studio',
                      side_effect=lambda src, out: shutil.copyfile(src, out)) as enhance:
            results = agent_processor.execute_agent_plan(self.media.music, plan['tools'], self.out, {'type': 'audio'})
        self.assertEqual(results[0]['status'], 'success', results)
        self.assertTrue(results[0]['adapted'])
        self.assertIn('not available here', results[0]['message'])
        enhance.assert_called_once()

    def test_stereo_vocal_removal_without_separation(self):
        with patch('agent_processor.ai_processor.separate_stems', side_effect=RuntimeError('missing')):
            results = agent_processor.execute_agent_plan(self.media.stereo, [{'name': 'remove_vocals', 'args': {}}],
                                                         self.out, {'type': 'audio'})
        self.assertEqual(results[0]['status'], 'success', results)

    def test_silent_output_is_caught_by_verification(self):
        def silence(src, out, *args, **kwargs):
            write_wav(out, np.zeros(SR))
            return out
        with patch('agent_processor.ai_processor.reduce_noise', side_effect=silence):
            results = agent_processor.execute_agent_plan(self.media.noisy, [{'name': 'reduce_noise', 'args': {}},
                                                                           {'name': 'normalize_audio', 'args': {}}],
                                                         self.out, {'type': 'audio'})
        self.assertEqual(results[0]['status'], 'error')
        self.assertIn('silent', results[0]['message'])
        self.assertEqual(results[0]['blocked_steps'], [{'step': 2, 'tool': 'normalize_audio'}])

    def test_noisy_transcription_runs_on_a_cleaned_copy(self):
        plan = agent_processor._fallback_intent_parser('Transcribe', context_for(self.media.noisy))
        seen = []

        def fake_stt(path):
            seen.append(path)
            return {'available': True, 'full_text': 'hello world', 'segments': [{'start': 0.0, 'end': 1.0, 'text': 'hello world'}]}
        with patch('agent_processor.ai_processor.transcribe_audio', side_effect=fake_stt):
            results = agent_processor.execute_agent_plan(self.media.noisy, plan['tools'], self.out, {'type': 'audio'})
        self.assertEqual([step['tool'] for step in results], ['reduce_noise', 'transcribe_audio'])
        self.assertTrue(all(step['status'] == 'success' for step in results))
        self.assertNotEqual(seen[0], self.media.noisy)                       # transcribed the cleaned copy
        self.assertEqual(agent_processor.final_output_of(results), self.media.noisy)   # user's media untouched
        self.assertIn('Background noise detected', results[0]['message'])

    def test_empty_transcript_retries_once_with_voice_enhancement(self):
        answers = [{'available': True, 'full_text': '', 'segments': []},
                   {'available': True, 'full_text': 'second pass', 'segments': [{'start': 0, 'end': 1, 'text': 'second pass'}]}]
        with patch('agent_processor.ai_processor.transcribe_audio', side_effect=answers) as stt, \
                patch('agent_processor.ai_processor.enhance_speech_studio',
                      side_effect=lambda src, out: shutil.copyfile(src, out)):
            results = agent_processor.execute_agent_plan(self.media.clean, [{'name': 'transcribe_audio', 'args': {}}],
                                                         self.out, {'type': 'audio'})
        self.assertEqual(stt.call_count, 2)
        self.assertTrue(results[0]['adapted'])
        self.assertEqual(results[0]['data']['text'], 'second pass')
        with patch('agent_processor.ai_processor.transcribe_audio', return_value=answers[0]) as stt:
            agent_processor.execute_agent_plan(self.media.tone, [{'name': 'transcribe_audio', 'args': {}}],
                                               self.out, {'type': 'audio'})
        self.assertEqual(stt.call_count, 1)                                    # no speech-like signal: no retry

    def test_image_chain_and_jpeg_export(self):
        plan, results = self.run_prompt('Boost clarity and export as jpg', self.media.transparent)
        self.assertTrue(all(step['status'] == 'success' for step in results), results)
        final = agent_processor.final_output_of(results)
        with Image.open(final) as image:
            self.assertEqual(image.format, 'JPEG')

    def test_chat_route_reports_artifacts_notes_and_skips(self):
        from app import app
        app.config['TESTING'] = True
        client = app.test_client()
        upload = os.path.join(app.config['UPLOAD_FOLDER'], 'agent_eval_clip.mp4')
        shutil.copyfile(self.media.video, upload)
        try:
            plan = agent_processor._fallback_intent_parser(
                'rotate it, mute the video and give me a thumbnail at 0.5s', context_for(upload))
            with patch('agent_processor.query_agent_orchestrator', return_value=plan):
                data = client.post('/api/agent/chat', json={'message': 'x', 'filename': 'agent_eval_clip.mp4',
                                                            'context': {'type': 'video', '_path': 'C:/secret'}}).get_json()
            self.assertEqual(data['status'], 'success', data)
            self.assertTrue(data['output_url'].startswith('/processed/muted_'))
            self.assertEqual(len(data['artifacts']), 1)
            self.assertTrue(data['artifacts'][0]['output_url'].endswith('.jpg'))
            self.assertIn('Rotating', data['reply'])
            self.assertTrue(data['plan_notes'])
        finally:
            os.remove(upload)


_MEMORY_PATCH = patch('agent_memory.recall_similar_jobs', return_value=[])


def setUpModule():
    # Keep the evaluation deterministic: past jobs stored on this machine must not influence plans here.
    _MEMORY_PATCH.start()


def tearDownModule():
    _MEMORY_PATCH.stop()
    SyntheticMedia.cleanup()


if __name__ == '__main__':
    unittest.main(verbosity=1)
