"""
Evaluation for assistant skills (skills/*/SKILL.md + agent_skills.py).

    ./venv/Scripts/python.exe agent_skills_eval.py

1. Loader: every built-in skill validates, every registered tool is covered, invalid or unsafe
   skill files are rejected and skipped without crashing.
2. Invocation: explicit "@id args", composition with free text, several skills in one message,
   unknown skills, wrong media type, missing settings, automatic triggers with opt-out.
3. Reasoning-model path: top-k skill hints only; {"skill": ...} steps expand and validate.
4. API + privacy: catalog, details, import / rename / delete, "save this as @name".
5. End-to-end: EVERY visible skill runs on synthetic media through the real planner + executor.
Prints the tool -> skills coverage matrix and the end-to-end pass rate.
"""

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
from PIL import Image, ImageDraw

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

import agent_planner  # noqa: E402
import agent_processor  # noqa: E402
import agent_skills  # noqa: E402
import video_processor  # noqa: E402
from agent_orchestration_eval import FORBIDDEN_TERMS, speech_like, write_wav  # noqa: E402

USER_DIR = tempfile.mkdtemp(prefix='agent-user-skills-')
_MEMORY_PATCH = patch('agent_memory.recall_similar_jobs', return_value=[])


def setUpModule():
    _MEMORY_PATCH.start()
    agent_skills.configure(agent_skills.SKILLS_DIR, USER_DIR)


def tearDownModule():
    _MEMORY_PATCH.stop()
    agent_skills.configure(agent_skills.SKILLS_DIR, agent_skills.USER_SKILLS_DIR)
    shutil.rmtree(USER_DIR, ignore_errors=True)
    Media.cleanup()


def plan(prompt, media_type=None, history=None, **context):
    media_context = dict(context)
    if media_type:
        media_context['type'] = media_type
    return agent_processor._fallback_intent_parser(prompt, media_context or None, history)


def names(result):
    return [step['name'] for step in result['tools']]


VALID = """---
name: {name}
title: Test skill
description: Use to check that skill validation works as expected.
category: {category}
media_types: [audio]
steps:
{steps}
---

Body text.
"""


def skill_text(name='test-skill', steps='- tool: normalize_audio', category='workflows'):
    return VALID.format(name=name, steps=steps, category=category)


# ═══════════════════════════════════════════════════════════════════════════
#  1. LOADER
# ═══════════════════════════════════════════════════════════════════════════

class LoaderTests(unittest.TestCase):

    def test_all_builtin_skills_are_valid_and_cover_every_tool(self):
        registry = agent_skills.registry()
        self.assertEqual(registry.errors, [])
        self.assertGreaterEqual(len(registry.visible()), 30)
        matrix = agent_skills.coverage()
        uncovered = [tool for tool, ids in matrix.items() if not ids]
        self.assertEqual(uncovered, [], 'every registered tool needs at least one skill')
        print('\n[Coverage] tool -> skills')
        for tool, ids in matrix.items():
            print(f'  {tool:22s} {", ".join(ids)}')

    def test_capability_skills_requested_by_product_exist(self):
        for skill_id in ('image-enhance', 'cutout', 'video-enhance', 'noise-removal', 'transcribe', 'video-to-audio',
                         'video-to-text', 'podcast-polish', 'spotify-master', 'youtube-ready', 'subtitles', 'gif-preview',
                         'thumbnail', 'hd-photo', 'profile-pic', 'web-ready', 'voice-clean', 'remove-silence'):
            with self.subTest(skill=skill_id):
                self.assertIsNotNone(agent_skills.registry().get(skill_id))
                self.assertTrue(agent_skills.skill_body(skill_id))

    def test_voice_and_separation_skills_are_available(self):
        for skill_id in ('voiceover', 'vocals', 'karaoke', 'four-stems', 'lyrics'):
            with self.subTest(skill=skill_id):
                skill = agent_skills.registry().get(skill_id)
                self.assertIsNotNone(skill)
                self.assertFalse(skill.hidden)
                self.assertTrue(agent_skills.skill_body(skill_id))
                self.assertNotRegex(open(skill.path, encoding='utf-8').read().lower(), r'copilot|omni|demucs|whisper|torch')
        self.assertEqual(names(plan('@voiceover "hello there"', 'audio')), ['generate_voiceover'])
        self.assertEqual(plan('@karaoke', 'audio')['tools'][0]['args']['mode'], 'karaoke')
        self.assertEqual(plan('@four-stems', 'audio')['tools'][0]['args']['mode'], '4stem')
        self.assertEqual(plan('@vocals', 'audio')['tools'][0]['args']['mode'], 'vocals')
        self.assertEqual(plan('@lyrics vtt', 'audio')['tools'][0]['args']['format'], 'vtt')

    def test_invalid_skills_are_rejected(self):
        bad = {
            'unknown tool': skill_text(steps='- tool: launch_rockets'),
            'unknown arg': skill_text(steps='- tool: normalize_audio\n  args: {model_name: x}'),
            'bad param ref': skill_text(steps='- tool: adjust_volume\n  args: {gain_db: "{{nope}}"}'),
            'private name': skill_text().replace('expected.', 'expected with whisper.'),
            'python tag': skill_text().replace('title: Test skill', 'title: !!python/object/apply:os.system ["echo"]'),
            'no front matter': 'name: x\nsteps: []',
            'bad id': skill_text(name='Bad_Name'),
            'too many steps': skill_text(steps='\n'.join(['- tool: normalize_audio'] * 13)),
            'bad category': skill_text(category='hacks'),
            'oversize': skill_text() + 'x' * (agent_skills.MAX_SKILL_BYTES + 1),
        }
        for label, text in bad.items():
            with self.subTest(case=label):
                with self.assertRaises(agent_skills.SkillError):
                    agent_skills.parse_skill(text, 'user')

    def test_loader_skips_invalid_files_without_crashing(self):
        with tempfile.TemporaryDirectory() as builtin:
            for name, text in (('good-one', skill_text('good-one')), ('broken', skill_text('broken', '- tool: nope')),
                               ('mismatch', skill_text('other-name'))):
                os.makedirs(os.path.join(builtin, name))
                with open(os.path.join(builtin, name, 'SKILL.md'), 'w', encoding='utf-8') as handle:
                    handle.write(text)
            registry = agent_skills.SkillRegistry(builtin, "")
            self.assertEqual(list(registry.skills), ['good-one'])
            self.assertEqual(sorted(name for name, _ in registry.errors), ['broken', 'mismatch'])


# ═══════════════════════════════════════════════════════════════════════════
#  2. INVOCATION & COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════

class InvocationTests(unittest.TestCase):

    def test_explicit_skill_with_positional_and_named_settings(self):
        result = plan('@loudness podcast', 'audio')
        self.assertEqual(names(result), ['normalize_audio'])
        self.assertEqual(result['tools'][0]['args']['preset'], 'podcast')
        self.assertEqual(result['skills_used'], [{'id': 'loudness', 'title': 'Loudness for a platform'}])
        self.assertIn('Using Loudness for a platform (@loudness).', result['reply'])
        fades = plan('@fade in=1 out=2.5', 'audio')
        self.assertEqual(fades['tools'][0]['args'], {'fade_in_sec': 1.0, 'fade_out_sec': 2.5})
        cutout = plan('use @cutout product please', 'image')
        self.assertEqual(cutout['tools'][0]['args']['quality_profile'], 'studio')

    def test_skill_steps_get_the_same_validation_and_media_routing(self):
        self.assertEqual(names(plan('@trim 2 8', 'video', duration=20.0)), ['trim_video'])
        self.assertEqual(names(plan('@trim 2 8', 'audio', duration=20.0)), ['trim_audio'])
        self.assertEqual(plan('@trim-outro 3', 'video', duration=20.0)['tools'][0]['args']['end_sec'], 17.0)
        self.assertTrue(plan('@trim-outro 3', 'video')['clarification_needed'], 'unknown length needs a question')
        podcast_from_video = plan('@podcast-polish 320k', 'video')
        self.assertEqual(names(podcast_from_video), ['extract_audio', 'enhance_speech', 'normalize_audio', 'convert_audio_format'])
        self.assertEqual(podcast_from_video['tools'][-1]['args'], {'target_format': 'mp3', 'bitrate': '320k'})
        self.assertEqual(names(plan('@speed 2', 'video')), ['adjust_video_speed'])
        self.assertEqual(names(plan('@speed 2', 'audio')), ['adjust_audio_speed'])

    def test_composition_with_free_text_and_several_skills(self):
        result = plan('@youtube-ready then trim the first 5s', 'video', duration=20.0)
        self.assertEqual(names(result), ['normalize_audio', 'trim_video', 'convert_video_format'])
        self.assertEqual(result['tools'][1]['args'], {'start_sec': 5.0, 'end_sec': None})
        self.assertIn('Run exactly in my order', result['suggested_actions'])
        two = plan('@loudness podcast and @fade 1 3, then export as flac', 'audio')
        self.assertEqual(names(two), ['normalize_audio', 'apply_audio_fade', 'convert_audio_format'])
        self.assertEqual([step.get('skill', {}).get('id') for step in two['tools']], ['loudness', 'fade', None])
        before = plan('trim 1 to 9 and then @podcast-polish', 'audio', duration=20.0)
        self.assertEqual(names(before)[0], 'trim_audio')

    def test_unknown_skill_lists_close_matches(self):
        result = plan('@podcst-polish', 'audio')
        self.assertTrue(result['clarification_needed'])
        self.assertIn('@podcast-polish', result['reply'])
        self.assertIn('@podcast-polish', result['clarification_options'])

    def test_wrong_media_type_explains_and_suggests(self):
        result = plan('@cutout', 'audio')
        self.assertEqual(result['tools'], [])
        self.assertIn('works on image', result['reply'])
        self.assertTrue(result['suggested_actions'])
        for suggestion in result['suggested_actions']:
            skill = agent_skills.registry().get(suggestion.lstrip('@'))
            self.assertIn('audio', skill.media_types)

    def test_missing_or_invalid_settings_ask(self):
        for prompt in ('@speed', '@speed 9', '@loudness loud', '@trim 2'):
            with self.subTest(prompt=prompt):
                result = plan(prompt, 'audio')
                self.assertTrue(result['clarification_needed'])
                self.assertEqual(result['tools'], [])

    def test_automatic_skill_with_opt_out(self):
        auto = plan('polish my podcast', 'audio')
        self.assertEqual(auto['auto_skill'], 'podcast-polish')
        self.assertEqual(names(auto), ['enhance_speech', 'normalize_audio', 'convert_audio_format'])
        self.assertIn('polish my podcast (no skill)', auto['suggested_actions'])
        manual = plan('polish my podcast (no skill)', 'audio')
        self.assertNotIn('auto_skill', manual)
        self.assertEqual(names(plan('make it youtube ready', 'video')), ['normalize_audio', 'convert_video_format'])
        # Triggers never fire for the wrong media type, and ordinary requests are untouched.
        self.assertNotIn('auto_skill', plan('polish my podcast', 'image'))
        self.assertNotIn('auto_skill', plan('normalize for youtube', 'audio'))

    def test_condition_policies_still_apply_inside_skills(self):
        from agent_orchestration_eval import SyntheticMedia, context_for
        media = SyntheticMedia.build()
        noisy = agent_processor._fallback_intent_parser('@transcribe', context_for(media.noisy))
        self.assertEqual(names(noisy), ['reduce_noise', 'transcribe_audio'])
        asis = agent_processor._fallback_intent_parser('@transcribe as-is', context_for(media.noisy))
        self.assertEqual(names(asis), ['transcribe_audio'])
        music = agent_processor._fallback_intent_parser('@noise-removal', context_for(media.music))
        self.assertEqual(names(music), ['isolate_voice'])
        clean = agent_processor._fallback_intent_parser('@noise-removal', context_for(media.clean))
        self.assertEqual(clean['tools'], [])
        self.assertTrue(any('already clean' in note for note in clean['plan_notes']))
        hd = agent_processor._fallback_intent_parser('@hd-photo', context_for(media.grainy))
        self.assertEqual(names(hd), ['enhance_photo_clarity', 'upscale_image'])


# ═══════════════════════════════════════════════════════════════════════════
#  3. REASONING-MODEL PATH
# ═══════════════════════════════════════════════════════════════════════════

class ReasoningSkillTests(unittest.TestCase):

    def run_with(self, content, prompt, media_type='audio'):
        with patch('agent_processor._local_reasoning_url', return_value='http://127.0.0.1:9/v1/chat/completions'), \
                patch('agent_processor._request_reasoning_content', return_value=content) as request:
            result = agent_processor.query_agent_orchestrator(prompt, {'type': media_type})
        return result, request

    def test_model_skill_steps_expand_and_validate(self):
        content = json.dumps({'tools': [{'skill': 'loudness', 'args': {'target': 'broadcast'}},
                                        {'name': 'apply_audio_fade', 'args': {'fade_out_sec': 2}}]})
        result, request = self.run_with(content, 'level it for tv and fade the end')
        self.assertEqual(names(result), ['normalize_audio', 'apply_audio_fade'])
        self.assertEqual(result['tools'][0]['args']['preset'], 'broadcast')
        sent = request.call_args.args[1]['messages'][1]['content']
        self.assertLessEqual(sent.count('\n- @'), 3, 'only the top-k skills reach the prompt')

    def test_explicit_skill_never_needs_the_model(self):
        with patch('agent_processor._request_reasoning_content') as request:
            result = agent_processor.query_agent_orchestrator('@podcast-polish', {'type': 'audio'})
        request.assert_not_called()
        self.assertEqual(names(result)[0], 'enhance_speech')

    def test_search_helper_for_prompt_budgeting(self):
        found = agent_skills.search_skills('make subtitles for my video', 'video', 3)
        self.assertLessEqual(len(found), 3)
        self.assertIn('subtitles', [item['id'] for item in found])
        self.assertTrue(all(set(item) == {'id', 'description'} for item in found))


# ═══════════════════════════════════════════════════════════════════════════
#  4. API, PRIVACY, PERSONAL SKILLS
# ═══════════════════════════════════════════════════════════════════════════

class ApiTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from app import app
        app.config['TESTING'] = True
        cls.app = app
        cls.client = app.test_client()

    def test_catalog_lists_every_visible_skill_with_a_description(self):
        data = self.client.get('/api/agent/skills').get_json()
        self.assertEqual(data['status'], 'success')
        ids = {item['id'] for item in data['skills']}
        self.assertEqual(ids, {skill.id for skill in agent_skills.registry().visible()})
        for item in data['skills']:
            self.assertTrue(item['description'].strip())
            self.assertTrue(item['title'].strip())
            self.assertTrue(item['steps'])
        payload = json.dumps(data).lower()
        for term in FORBIDDEN_TERMS + ['copilot', 'omni']:
            self.assertNotIn(term, payload, f'skills catalog leaked "{term}"')
        audio = self.client.get('/api/agent/skills?media_type=audio').get_json()['skills']
        self.assertTrue(audio and all('audio' in item['media_types'] for item in audio))

    def test_skill_details_include_the_procedure(self):
        data = self.client.get('/api/agent/skills/transcribe').get_json()
        self.assertIn('Verification', data['skill']['procedure'])
        self.assertEqual(self.client.get('/api/agent/skills/voiceover').status_code, 200)
        self.assertEqual(self.client.get('/api/agent/skills/nope').status_code, 404)

    def test_import_rename_delete_round_trip(self):
        text = skill_text('my-level', '- tool: normalize_audio\n  args: {preset: podcast}', category='workflows')
        imported = self.client.post('/api/agent/skills/import', json={'content': text}).get_json()
        self.assertEqual(imported['skill']['category'], 'mine')
        self.assertTrue(imported['skill']['user'])
        self.assertEqual(self.client.post('/api/agent/skills/import',
                                          json={'content': skill_text('loudness')}).status_code, 400)
        self.assertEqual(self.client.post('/api/agent/skills/import',
                                          json={'content': skill_text('evil', '- tool: rm_rf')}).status_code, 400)
        renamed = self.client.post('/api/agent/skills/my-level/rename', json={'new_id': 'pod-level'}).get_json()
        self.assertEqual(renamed['skill']['id'], 'pod-level')
        self.assertEqual(names(plan('@pod-level', 'audio')), ['normalize_audio'])
        self.assertEqual(self.client.delete('/api/agent/skills/loudness').status_code, 400)
        self.assertEqual(self.client.delete('/api/agent/skills/pod-level').status_code, 200)
        self.assertIsNone(agent_skills.registry().get('pod-level'))
        self.assertFalse(any(name.startswith('pod-level') for name in os.listdir(USER_DIR)))

    def test_save_last_successful_edit_as_a_skill(self):
        media = Media.build()
        upload = os.path.join(self.app.config['UPLOAD_FOLDER'], 'skills_eval_clip.wav')
        shutil.copyfile(media.audio, upload)
        try:
            session = 'skills_eval_session'
            nothing = self.client.post('/api/agent/chat', json={'message': 'save this as @my-chain',
                                                                'session_id': 'skills_eval_other'}).get_json()
            self.assertIn('no finished edit', nothing['reply'])
            edit = self.client.post('/api/agent/chat', json={
                'message': 'trim from 1 to 6 seconds then fade out over 1 second', 'filename': 'skills_eval_clip.wav',
                'context': {'type': 'audio'}, 'session_id': session}).get_json()
            self.assertEqual(edit['status'], 'success', edit)
            saved = self.client.post('/api/agent/chat', json={'message': 'save this as @my-chain',
                                                              'session_id': session}).get_json()
            self.assertIn('Saved as @my-chain', saved['reply'])
            self.assertEqual(saved['suggested_actions'], ['@my-chain'])
            replay = plan('@my-chain', 'audio', duration=10.0)
            self.assertEqual(names(replay), ['trim_audio', 'apply_audio_fade'])
            listed = self.client.get('/api/agent/skills').get_json()['skills']
            mine = [item for item in listed if item['id'] == 'my-chain']
            self.assertTrue(mine and mine[0]['user'] and mine[0]['category'] == 'mine')
            deleted = self.client.post('/api/agent/chat', json={'message': 'delete skill @my-chain',
                                                                'session_id': session}).get_json()
            self.assertIn('Deleted', deleted['reply'])
            self.assertIsNone(agent_skills.registry().get('my-chain'))
            builtin = self.client.post('/api/agent/chat', json={'message': 'save this as @loudness',
                                                                'session_id': session}).get_json()
            self.assertIn('built-in', builtin['reply'])
        finally:
            os.remove(upload)

    def test_chat_response_reports_skills_used(self):
        media = Media.build()
        upload = os.path.join(self.app.config['UPLOAD_FOLDER'], 'skills_eval_chat.wav')
        shutil.copyfile(media.audio, upload)
        try:
            data = self.client.post('/api/agent/chat', json={'message': '@fade 0.5 0.5', 'filename': 'skills_eval_chat.wav',
                                                             'context': {'type': 'audio'}}).get_json()
            self.assertEqual(data['status'], 'success', data)
            self.assertEqual(data['skills_used'], [{'id': 'fade', 'title': 'Fade in and out'}])
            self.assertIsNone(data['auto_skill'])
        finally:
            os.remove(upload)


# ═══════════════════════════════════════════════════════════════════════════
#  5. END-TO-END: EVERY VISIBLE SKILL
# ═══════════════════════════════════════════════════════════════════════════

class Media:
    folder = None

    @classmethod
    def build(cls):
        if cls.folder:
            return cls
        cls.folder = tempfile.mkdtemp(prefix='skills-eval-')
        rng = np.random.default_rng(3)
        speech = speech_like(10.0)
        cls.audio = write_wav(os.path.join(cls.folder, 'speech.wav'), speech + 0.01 * rng.standard_normal(speech.size))
        left = speech_like(6.0, 0.3)
        bed = 0.25 * np.sin(2 * np.pi * 300 * np.arange(left.size) / 16000)
        cls.stereo = os.path.join(cls.folder, 'song.wav')
        write_wav(cls.stereo, np.column_stack([left + bed, left - bed]).reshape(-1), channels=2)
        cls.video = os.path.join(cls.folder, 'clip.mp4')
        subprocess.run([video_processor.FFMPEG, '-y', '-f', 'lavfi', '-i', 'testsrc=duration=6:size=320x180:rate=15',
                        '-i', cls.audio, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest',
                        cls.video], capture_output=True, check=True)
        image = Image.new('RGB', (240, 180), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        draw.ellipse([60, 30, 180, 150], fill=(200, 80, 60))
        cls.image = os.path.join(cls.folder, 'photo.png')
        image.save(cls.image)
        return cls

    @classmethod
    def cleanup(cls):
        if cls.folder:
            shutil.rmtree(cls.folder, ignore_errors=True)
            cls.folder = None


def _fake_cutout(src, out, **kwargs):
    with Image.open(src) as image:
        rgba = image.convert('RGBA')
    mask = Image.new('L', rgba.size, 0)
    ImageDraw.Draw(mask).ellipse([rgba.width // 4, rgba.height // 6, rgba.width * 3 // 4, rgba.height * 5 // 6], fill=255)
    rgba.putalpha(mask)
    rgba.save(out)
    return out


def _fake_upscale(src, out, scale=2, **kwargs):
    with Image.open(src) as image:
        image.resize((image.width * scale, image.height * scale)).save(out)
    return {'size': 'ok'}


def _fake_stt(path):
    return {'available': True, 'full_text': 'hello there', 'avg_word_confidence': 0.9,
            'segments': [{'start': 0.0, 'end': 1.0, 'text': 'hello there'}]}


def _fake_enhance(src, out):
    shutil.copyfile(src, out)
    return {'status': 'success'}


EXPECTED_SUFFIX = {
    'podcast-polish': '.mp3', 'spotify-master': '.flac', 'thumbnail': '.jpg', 'convert-audio': '.flac', 'video-to-audio': '.wav',
    'web-ready': '.webp', 'product-cutout': '.png', 'profile-pic': '.png', 'convert-image': '.jpg', 'gif-preview': '.gif',
    'youtube-ready': '.mp4', 'convert-video': '.webm', 'video-enhance': '.mp4', 'mute': '.mp4',
}


class EndToEndSkillTests(unittest.TestCase):
    results = []

    @classmethod
    def setUpClass(cls):
        cls.media = Media.build()

    def source_for(self, skill):
        if 'audio' in skill.media_types and skill.id != 'instrumental':
            return self.media.audio
        if skill.id == 'instrumental':
            return self.media.stereo
        return self.media.video if 'video' in skill.media_types else self.media.image

    def test_every_visible_skill_runs_end_to_end(self):
        skills = agent_skills.registry().visible()
        for skill in skills:
            with self.subTest(skill=skill.id), tempfile.TemporaryDirectory() as out_dir, \
                    patch('agent_processor.image_processor.remove_bg', side_effect=_fake_cutout), \
                    patch('agent_processor.image_processor.upscale', side_effect=_fake_upscale), \
                    patch('agent_processor.ai_processor.transcribe_audio', side_effect=_fake_stt), \
                    patch('agent_processor.ai_processor.enhance_speech_studio', side_effect=_fake_enhance), \
                    patch('agent_processor.ai_processor.separate_stems', side_effect=RuntimeError('not installed')),                     patch('agent_processor.image_processor.restore_faces', side_effect=_fake_enhance):
                source = self.source_for(skill)
                context = agent_processor._media_context_for_file(source, {})
                prompt = skill.example
                result = agent_processor._fallback_intent_parser(prompt, context)
                problem = None
                if result['clarification_needed'] or not result['tools']:
                    problem = f'no plan: {result["reply"][:120]}'
                else:
                    steps = agent_processor.execute_agent_plan(source, result['tools'], out_dir, context)
                    failed = [step for step in steps if step['status'] != 'success']
                    final = agent_processor.final_output_of(steps)
                    if failed:
                        problem = f'failed steps: {[(step["tool"], step["message"]) for step in failed]}'
                    elif not final or not os.path.isfile(final):
                        problem = 'no output'
                    elif skill.id in EXPECTED_SUFFIX and not final.endswith(EXPECTED_SUFFIX[skill.id]):
                        problem = f'unexpected output {os.path.basename(final)}'
                EndToEndSkillTests.results.append((skill.id, prompt, problem))
                self.assertIsNone(problem, f'{skill.id} ({prompt})')

    @classmethod
    def tearDownClass(cls):
        if cls.results:
            passed = sum(1 for _, _, problem in cls.results if not problem)
            print(f'\n[Skills end-to-end] {passed}/{len(cls.results)} skills ran successfully on synthetic media.')
            for skill_id, prompt, problem in cls.results:
                if problem:
                    print(f'  FAIL @{skill_id} ({prompt}): {problem}')


if __name__ == '__main__':
    unittest.main(verbosity=1)
