"""
═══════════════════════════════════════════════════════════════════════════════
COMPREHENSIVE END-TO-END AUTOMATED TEST SUITE
Platform: Autonomous AI Audio, Video & Image Studio
Scope:
  1. Frontend Web Views & Content Verification (Landing, Agent, Studio, Audio, Image, Video)
  2. Zero Internal Model / Implementation Name Leakage Audit
  3. Static Asset Integrity (CSS, JS, Tool scripts)
  4. Diagnostics & System Health Monitoring
  5. AI Multi-Agent Copilot Chat & Media Orchestration
  6. Studio Multitrack Project Persistence (Save, Load, Traversal Guard)
  7. Audio Processing (Multi-Region Cut, Separate Zip, LUFS, EQ, Silence, Beats, VAD, Pitch, Multitrack Mix)
  8. Image Processing (Super-Resolution, Clarity, Color Match, Background Removal, Inpaint)
  9. Video Processing (Upload, Probing, Filmstrip Thumbnails, Quick Tools, Audio Extract, Scenes, Timeline Export)
  10. Security Hardening (Path Traversal, File Extension Whitelists, Error Formats)
  11. Platform Memory Reclaim & Debounced Storage Purging
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import io
import wave
import math
import struct
import json
import time
import zipfile
import subprocess
import unittest
import logging
import threading
import tempfile
import warnings
import gc
from unittest.mock import patch
from PIL import Image, ImageDraw

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from app import (
    app,
    allowed_file,
    _cleanup_old_temp_files,
    _maybe_cleanup_temp_files,
    BrandPrivacyLogFilter
)
from model_manager import ModelLifecycleManager
import video_processor
import agent_processor

class ComprehensivePlatformE2ETests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()
        cls.test_dir = tempfile.mkdtemp(prefix='e2e_media_')

        # 1. Synthetic 1-second 44.1kHz mono WAV file (440Hz sine wave)
        cls.sample_wav_path = os.path.join(cls.test_dir, 'sample_audio.wav')
        sample_rate = 44100
        duration = 1.0
        num_samples = int(sample_rate * duration)
        with wave.open(cls.sample_wav_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            raw_data = bytearray()
            for i in range(num_samples):
                val = int(32767.0 * 0.4 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
                raw_data.extend(struct.pack('<h', val))
            wav_file.writeframes(raw_data)

        # 2. Synthetic 160x160 PNG image with distinct shapes
        cls.sample_png_path = os.path.join(cls.test_dir, 'sample_image.png')
        img = Image.new('RGB', (160, 160), color=(25, 35, 60))
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 140, 140], fill=(210, 95, 45), outline=(255, 255, 255))
        draw.ellipse([40, 40, 120, 120], fill=(45, 180, 140))
        img.save(cls.sample_png_path, format='PNG')

        # 3. Synthetic 160x160 Grayscale Mask for Inpainting
        cls.sample_mask_path = os.path.join(cls.test_dir, 'sample_mask.png')
        mask_img = Image.new('L', (160, 160), color=0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.rectangle([60, 60, 100, 100], fill=255)
        mask_img.save(cls.sample_mask_path, format='PNG')

        # 4. Synthetic 1-second MP4 video with both video and audio streams via ffmpeg
        cls.sample_mp4_path = os.path.join(cls.test_dir, 'sample_video.mp4')
        cmd = [
            video_processor.FFMPEG, '-y',
            '-f', 'lavfi', '-i', 'testsrc=duration=1:size=160x120:rate=15',
            '-f', 'lavfi', '-i', 'sine=frequency=880:duration=1',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            cls.sample_mp4_path
        ]
        res = subprocess.run(cmd, capture_output=True)
        if not os.path.exists(cls.sample_mp4_path) or os.path.getsize(cls.sample_mp4_path) == 0:
            raise RuntimeError(f"Failed to generate synthetic video for tests: {res.stderr.decode('utf-8', errors='ignore')}")

    @classmethod
    def tearDownClass(cls):
        # Per-run private directory: removing it can never disturb a concurrent run.
        import shutil
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    # =========================================================================
    # SECTION 1: FRONTEND WEB ROUTES & BRAND PRIVACY LEAK AUDIT
    # =========================================================================

    def test_01_all_web_routes_render_and_privacy_audit(self):
        """Verify all platform HTML views return 200 and leak no confidential internal model names."""
        forbidden_strings = [
            'gemma', 'whisper', 'vllm', 'isnet', 'u2net', 'ffmpeg',
            'ffprobe', 'opencv', 'librosa', 'pydub', 'pytorch', 'torch',
            'ctranslate2', 'onnx', 'edsr', 'fsrcnn', 'rembg'
        ]
        routes = [
            ('/', 'Landing Page'),
            ('/agent', 'AI Agent Workspace'),
            ('/chat', 'AI Agent Chat Alias'),
            ('/studio', 'Pro Multitrack Studio'),
            ('/audio', 'Dedicated Audio Editor'),
            ('/image', 'Dedicated Image Studio'),
            ('/video', 'Dedicated Video Editor'),
        ]

        for route, name in routes:
            with self.subTest(route=route, page=name):
                resp = self.client.get(route)
                self.assertEqual(resp.status_code, 200, f"Route {route} failed with status {resp.status_code}")
                content = resp.get_data(as_text=True)
                normalized_content = content.lower()
                self.assertIn('<html', normalized_content, f"{name} is missing HTML root element")
                for external_origin in (
                    'cdnjs.cloudflare.com',
                    'unpkg.com',
                    'fonts.googleapis.com',
                    'fonts.gstatic.com'
                ):
                    self.assertNotIn(
                        external_origin,
                        normalized_content,
                        f"Runtime network dependency '{external_origin}' found in {name} ({route})"
                    )

                # Verify privacy guarantee across entire page content
                for forbidden in forbidden_strings:
                    self.assertNotIn(
                        forbidden,
                        normalized_content,
                        f"Privacy violation: '{forbidden}' found in {name} ({route})"
                    )

    def test_02_landing_page_dual_choice_architecture(self):
        """Verify the landing page strictly presents the clean 2-choice model without visual noise."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        html = resp.get_data(as_text=True)

        # 1. Exactly two primary choices present
        import branding
        self.assertIn(f'Go with {branding.assistant_name()}', html)
        # The final product name isn't chosen yet: no hard-coded placeholder brands.
        for retired_name in ('Copilot', 'Omni'):
            self.assertNotIn(retired_name, html)
        self.assertIn('Go Manually', html)

        # 2. Interactive drawer for manual studios
        self.assertIn('manualToolsDrawer', html)
        self.assertIn('/image', html)
        self.assertIn('/audio', html)
        self.assertIn('/video', html)
        self.assertIn('/studio', html)

    def test_03_static_asset_availability(self):
        """Ensure all required CSS and JS bundles are served properly by Flask."""
        assets = [
            '/static/landing.css',
            '/static/agent.css',
            '/static/studio.css',
            '/static/image.css',
            '/static/video.css',
            '/static/style.css',
            '/static/agent.js',
            '/static/js/ai_agent.js',
            '/static/js/image_studio.js',
            '/static/image.js',
            '/static/video.js',
            '/static/script.js',
            '/static/processing_overlay.js',
            '/static/vendor/fontawesome/css/all.min.css',
            '/static/vendor/fontawesome/webfonts/fa-solid-900.woff2',
            '/static/vendor/wavesurfer/wavesurfer.min.js',
            '/static/vendor/wavesurfer/regions.min.js',
            '/static/vendor/wavesurfer/timeline.min.js',
            '/static/vendor/jszip/jszip.min.js'
        ]
        for asset in assets:
            with self.subTest(asset=asset):
                resp = self.client.get(asset)
                self.assertEqual(resp.status_code, 200, f"Asset {asset} returned {resp.status_code}")
                self.assertGreater(len(resp.data), 50, f"Asset {asset} appears unexpectedly empty")
                if not (
                    resp.mimetype.startswith('text/')
                    or resp.mimetype in {'application/javascript', 'application/json'}
                ):
                    continue
                content = resp.get_data(as_text=True).lower()
                for forbidden in [
                    'gemma', 'whisper', 'vllm', 'isnet', 'u2net', 'ffmpeg',
                    'ffprobe', 'opencv', 'librosa', 'pydub', 'pytorch',
                    'ctranslate2', 'onnx', 'edsr', 'fsrcnn', 'rembg'
                ]:
                    self.assertNotIn(
                        forbidden,
                        content,
                        f"Privacy violation: '{forbidden}' found in {asset}"
                    )

    # =========================================================================
    # SECTION 2: HEALTH, DIAGNOSTICS & AGENT TOOLS SCHEMA
    # =========================================================================

    def test_04_health_check_endpoint(self):
        """Verify /health diagnostic returns healthy status and active storage paths."""
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertIn('storage', data)
        self.assertTrue(data['storage'].get('uploads_exists'))
        self.assertTrue(data['storage'].get('processed_exists'))
        self.assertTrue(data['storage'].get('projects_exists'))

    def test_05_agent_tool_registry_schema(self):
        """Verify the AI Agent tool schemas and sub-agent registration."""
        resp = self.client.get('/api/agent/tools')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('subagents', data)
        expected_subagents = {'VisionSubAgent', 'VideoSubAgent', 'AudioSubAgent', 'InspectorSubAgent', 'MasterOrchestrator'}
        self.assertTrue(expected_subagents.issubset(set(data['subagents'])))

        self.assertIn('tools', data)
        tool_names = [t.get('name') for t in data['tools']]
        self.assertIn('remove_background', tool_names)
        self.assertIn('upscale_image', tool_names)
        self.assertIn('enhance_speech', tool_names)
        public_payload = json.dumps(data).lower()
        for forbidden in [
            'gemma', 'whisper', 'vllm', 'isnet', 'u2net', 'ffmpeg',
            'ffprobe', 'opencv', 'librosa', 'pydub', 'pytorch',
            'ctranslate2', 'onnx', '"engine"', '"model_name"', '"model_used"'
        ]:
            self.assertNotIn(
                forbidden,
                public_payload,
                f"Agent tool registry leaked internal detail: '{forbidden}'"
            )

    # =========================================================================
    # SECTION 3: MULTI-AGENT COPILOT WORKSPACE (UPLOAD & CHAT E2E)
    # =========================================================================

    def test_06_agent_media_uploads_all_types(self):
        """Verify uploading audio, image, and video to /api/agent/upload returns valid metadata."""
        uploads = [
            (self.sample_wav_path, 'test_audio.wav', 'audio'),
            (self.sample_png_path, 'test_image.png', 'image'),
            (self.sample_mp4_path, 'test_video.mp4', 'video'),
        ]

        with tempfile.TemporaryDirectory() as isolated_uploads, patch.dict(app.config, {'UPLOAD_FOLDER': isolated_uploads}):
            for invalid_name, content in [
                ('unsafe.html', b'<script>bad()</script>'), ('empty.wav', b''),
                ('not-audio.wav', b'not media'), ('corrupt.png', b'not an image'),
                ('not-video.mp4', b'not video'),
            ]:
                invalid = self.client.post('/api/agent/upload',
                    data={'file': (io.BytesIO(content), invalid_name)}, content_type='multipart/form-data')
                self.assertEqual(invalid.status_code, 400)
                self.assertEqual(os.listdir(isolated_uploads), [])

        for filepath, fname, expected_type in uploads:
            with self.subTest(file_type=expected_type):
                with open(filepath, 'rb') as f:
                    resp = self.client.post(
                        '/api/agent/upload',
                        data={'file': (io.BytesIO(f.read()), fname)},
                        content_type='multipart/form-data'
                    )
                self.assertEqual(resp.status_code, 200)
                data = resp.get_json()
                self.assertEqual(data.get('status'), 'success')
                self.assertTrue(data.get('id'), "Agent upload should return unique id")
                self.assertEqual(data.get('type'), expected_type)
                self.assertIn('/media/', data.get('url'))
                if expected_type == 'image':
                    self.assertGreater(data['width'], 0)
                    self.assertGreater(data['height'], 0)
                else:
                    self.assertGreater(data['duration'], 0)

        with tempfile.TemporaryDirectory() as temporary:
            for extension, codec in [('webm', 'libopus'), ('mp4', 'aac')]:
                audio_container = os.path.join(temporary, 'voice.' + extension)
                video_processor._run([video_processor.FFMPEG, '-y', '-i', self.sample_wav_path,
                                      '-vn', '-c:a', codec, audio_container])
                with open(audio_container, 'rb') as audio:
                    uploaded = self.client.post('/api/agent/upload',
                        data={'file': (io.BytesIO(audio.read()), 'voice.' + extension)},
                        content_type='multipart/form-data').get_json()
                self.assertEqual(uploaded['type'], 'audio')
                self.assertFalse(uploaded['has_video'])
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded['filename'])
                self.assertEqual(agent_processor._media_context_for_file(file_path)['type'], 'audio')

    def test_07_agent_chat_conversational_execution(self):
        """Verify natural language conversation with the Agent Copilot."""
        routing_cases = [
            ('Clean background noise', 'audio', ['reduce_noise']),
            ('Remove background from photo', 'image', ['remove_background']),
            ('Make GIF clip', 'video', ['convert_video_format']),
            ('Enhance voice clarity', 'audio', ['enhance_speech']),
            ('Inspect loudness and sample rate', 'audio', ['inspect_media']),
            ('Trim from 0.2 to 0.8 seconds', 'audio', ['trim_audio']),
            ('Normalize then fade in 0.1 seconds and fade out 0.2 seconds', 'audio',
             ['normalize_audio', 'apply_audio_fade']),
            ('Upscale 2x then remove background', 'image', ['upscale_image', 'remove_background']),
            ('Adjust speed to 0.75x', 'video', ['adjust_video_speed']),
            ('Adjust speed to 1.25x', 'audio', ['adjust_audio_speed']),
            ('Convert to MP3', 'audio', ['convert_audio_format']),
            ('Export video as WebM', 'video', ['convert_video_format']),
            ('Compress this video', 'video', ['compress_video']),
            ('Extract audio as WAV then trim from 0.2 to 0.8 seconds', 'video', ['extract_audio', 'trim_audio']),
        ]
        for prompt, media_type, expected in routing_cases:
            with self.subTest(local_prompt=prompt):
                routed = agent_processor._fallback_intent_parser(prompt, {'type': media_type})
                self.assertEqual([tool['name'] for tool in routed['tools']], expected)
                self.assertFalse(routed.get('clarification_needed'))
        ambiguous = agent_processor._fallback_intent_parser('Normalize then do something unknown', {'type': 'audio'})
        self.assertTrue(ambiguous['clarification_needed'])
        self.assertEqual(ambiguous['tools'], [])
        fades = agent_processor._fallback_intent_parser('Fade edges', {'type': 'audio'})
        self.assertGreater(fades['tools'][0]['args']['fade_in_sec'], 0)
        self.assertGreater(fades['tools'][0]['args']['fade_out_sec'], 0)
        followup = agent_processor._fallback_intent_parser('0.2 to 0.8 seconds', {'type': 'audio'},
            [{'role': 'user', 'content': 'Trim audio'}, {'role': 'assistant', 'content': 'Which timestamps?'}])
        self.assertEqual(followup['tools'][0]['name'], 'trim_audio')
        for multiplier in [0.25, 0.75, 1.25, 3.5, 4]:
            speed_plan = agent_processor._fallback_intent_parser(f'Adjust speed to {multiplier}x', {'type': 'audio'})
            self.assertEqual(speed_plan['tools'][0]['args']['speed'], multiplier)
        for invalid_speed in ['Adjust speed to 0x', 'Adjust speed to -1x', 'Adjust speed to 5x', 'Make it faster']:
            self.assertTrue(agent_processor._fallback_intent_parser(invalid_speed, {'type': 'audio'})['clarification_needed'])
        for invalid_payload in [[], {'message': 123}, {'message': 'Hi', 'history': ['invalid']}]:
            self.assertEqual(self.client.post('/api/agent/chat', json=invalid_payload).status_code, 400)
        original_reasoning_url = agent_processor.REASONING_API_BASE
        try:
            agent_processor.REASONING_API_BASE = 'https://example.com/v1'
            with patch('agent_processor.urllib.request.urlopen') as remote_request:
                local_result = agent_processor.query_agent_orchestrator('hello')
            remote_request.assert_not_called()
            self.assertIn('reply', local_result)
        finally:
            agent_processor.REASONING_API_BASE = original_reasoning_url

        # 1. Text-only conversational prompt
        resp = self.client.post(
            '/api/agent/chat',
            data=json.dumps({"message": "Hello! What can you help me with?"}),
            content_type='application/json'
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('reply', data)
        self.assertIsInstance(data['reply'], str)
        self.assertGreater(len(data['reply']), 0)

        # 2. Contextual inspection prompt with an uploaded audio file
        with open(self.sample_wav_path, 'rb') as f:
            up_resp = self.client.post(
                '/api/agent/upload',
                data={'file': (io.BytesIO(f.read()), 'clip.wav')},
                content_type='multipart/form-data'
            )
        up_data = up_resp.get_json()

        with self.subTest('audio speed and conversion produce real audio at the requested duration'):
            for multiplier in [0.25, 1.25, 4]:
                speed_plan = agent_processor._fallback_intent_parser(
                    f'Adjust speed to {multiplier}x then export as FLAC', {'type': 'audio'})
                with patch('agent_processor.query_agent_orchestrator', return_value=speed_plan):
                    rendered = self.client.post('/api/agent/chat', json={
                        'message': 'Change speed and export', 'filename': up_data['filename'],
                        'context': {'type': 'audio'},
                    }).get_json()
                self.assertEqual(rendered['status'], 'success', rendered)
                output_path = os.path.join(app.config['PROCESSED_FOLDER'], rendered['output_file'])
                metadata = video_processor.probe_media(output_path)
                self.assertTrue(metadata['has_audio'])
                self.assertFalse(metadata['has_video'])
                self.assertAlmostEqual(metadata['duration'], 1 / multiplier, delta=0.15)
                self.assertTrue(rendered['output_file'].endswith('.flac'))
                with open(output_path, 'rb') as output:
                    self.assertEqual(output.read(4), b'fLaC')

        with self.subTest('video conversion and extracted audio trimming execute in order'):
            with open(self.sample_mp4_path, 'rb') as video:
                uploaded_video = self.client.post('/api/agent/upload',
                    data={'file': (io.BytesIO(video.read()), 'conversion.mp4')},
                    content_type='multipart/form-data').get_json()
            for prompt, expected_suffix in [
                ('Export video as WebM', '.webm'),
                ('Adjust speed to 0.75x', '.mp4'),
                ('Extract audio as WAV then trim from 0.2 to 0.8 seconds', '.wav'),
            ]:
                plan = agent_processor._fallback_intent_parser(prompt, {'type': 'video'})
                with patch('agent_processor.query_agent_orchestrator', return_value=plan):
                    result = self.client.post('/api/agent/chat', json={
                        'message': prompt, 'filename': uploaded_video['filename'], 'context': {'type': 'video'},
                    }).get_json()
                self.assertEqual(result['status'], 'success', result)
                self.assertTrue(result['output_file'].endswith(expected_suffix))
                info = video_processor.probe_media(os.path.join(app.config['PROCESSED_FOLDER'], result['output_file']))
                if expected_suffix == '.wav':
                    self.assertFalse(info['has_video'])
                    self.assertAlmostEqual(info['duration'], 0.6, delta=0.02)
                elif 'speed' in prompt:
                    self.assertAlmostEqual(info['duration'], 1 / 0.75, delta=0.15)
                else:
                    self.assertTrue(info['has_video'])

            for invalid_tool in [
                {'name': 'trim_video', 'args': {'start_sec': 0.8, 'end_sec': 0.2}},
                {'name': 'adjust_video_speed', 'args': {'speed': 0}},
                {'name': 'convert_video_format', 'args': {'target_format': '../../unsafe'}},
            ]:
                results = agent_processor.execute_agent_plan(
                    os.path.join(app.config['UPLOAD_FOLDER'], uploaded_video['filename']),
                    [invalid_tool], app.config['PROCESSED_FOLDER'], {'type': 'video'})
                self.assertEqual(results[0]['status'], 'error')

        with self.subTest('audio trim creates the requested precise interval'):
            trim_plan = agent_processor._fallback_intent_parser('Trim from 0.2 to 0.8 seconds', {'type': 'audio'})
            with patch('agent_processor.query_agent_orchestrator', return_value=trim_plan):
                trimmed = self.client.post('/api/agent/chat', json={
                    'message': 'Trim from 0.2 to 0.8 seconds', 'filename': up_data['filename'],
                    'context': {'type': 'audio'},
                }).get_json()
            self.assertEqual(trimmed['status'], 'success')
            download = self.client.get(trimmed['output_url'])
            try:
                with wave.open(io.BytesIO(download.data), 'rb') as audio:
                    self.assertAlmostEqual(audio.getnframes() / audio.getframerate(), 0.6, places=2)
            finally:
                download.close()

        with self.subTest('soundtrack effects preserve video and subsequent edits keep correct context'):
            source_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_video['filename'])
            source_frames = video_processor._run([
                video_processor.FFMPEG, '-v', 'error', '-i', source_path,
                '-map', '0:v:0', '-f', 'framemd5', '-',
            ]).stdout
            for audio_effect in [
                {'name': 'normalize_audio', 'args': {}},
                {'name': 'apply_audio_fade', 'args': {'fade_in_sec': 0.1, 'fade_out_sec': 0.1}},
            ]:
                with tempfile.TemporaryDirectory() as output_dir:
                    results = agent_processor.execute_agent_plan(source_path,
                        [audio_effect, {'name': 'inspect_media', 'args': {}}], output_dir, {'type': 'video'})
                    self.assertTrue(all(step['status'] == 'success' for step in results), results)
                    output_path = results[0]['output_file']
                    metadata = video_processor.probe_media(output_path)
                    self.assertTrue(metadata['has_video'])
                    self.assertTrue(metadata['has_audio'])
                    self.assertEqual(metadata['width'], 160)
                    self.assertEqual(metadata['height'], 120)
                    edited_frames = video_processor._run([
                        video_processor.FFMPEG, '-v', 'error', '-i', output_path,
                        '-map', '0:v:0', '-f', 'framemd5', '-',
                    ]).stdout
                    frame_hashes = lambda content: [line.split(',')[-1].strip()
                        for line in content.splitlines() if line and not line.startswith('#')]
                    self.assertEqual(frame_hashes(source_frames), frame_hashes(edited_frames))
                    self.assertIn('Original video preserved', results[0]['message'])
                    self.assertFalse(any(name.startswith('agent-audio-') for name in os.listdir(output_dir)))

            with tempfile.TemporaryDirectory() as output_dir:
                results = agent_processor.execute_agent_plan(source_path, [
                    {'name': 'extract_audio', 'args': {'format': 'wav'}},
                    {'name': 'inspect_media', 'args': {}},
                ], output_dir, {'type': 'video'})
                self.assertEqual(results[-1]['data']['sample_rate'], 44100)
                self.assertIn('Audio:', results[-1]['message'])

        with self.subTest('odd-height video conversions and speed edits render successfully'):
            with tempfile.TemporaryDirectory() as output_dir:
                odd_source = os.path.join(output_dir, 'odd.webm')
                video_processor._run([
                    video_processor.FFMPEG, '-y', '-f', 'lavfi', '-i',
                    'testsrc=duration=0.5:size=160x91:rate=15',
                    '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuv444p', odd_source,
                ])
                for name, render in [
                    ('conversion', lambda output: video_processor.quick_convert(odd_source, output)),
                    ('compression', lambda output: video_processor.quick_compress(odd_source, output, level='high')),
                    ('speed', lambda output: video_processor.process_timeline({'clips': [{'path': odd_source, 'speed': 1.5}]}, output)),
                ]:
                    output = os.path.join(output_dir, name + '.mp4')
                    render(output)
                    metadata = video_processor.probe_media(output)
                    self.assertTrue(metadata['has_video'])
                    self.assertEqual(metadata['height'] % 2, 0)
                    self.assertEqual(metadata['width'] % 2, 0)
                with patch('video_processor._run') as encoder:
                    video_processor.quick_compress(odd_source, os.path.join(output_dir, 'high.mp4'), level='high')
                    command = encoder.call_args.args[0]
                    self.assertEqual(command[command.index('-crf') + 1], '33')

        chat_payload = {
            "message": "Inspect this audio file and check its loudness and sample rate.",
            "filename": up_data['filename'],
            "context": {"type": "audio", "duration": 1.0},
            "history": []
        }
        resp2 = self.client.post(
            '/api/agent/chat',
            data=json.dumps(chat_payload),
            content_type='application/json'
        )
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.get_json()
        self.assertIn('reply', data2)
        # Verify no confidential model names leaked in the reply
        self.assertNotIn('Gemma', data2['reply'])
        self.assertEqual([step['tool'] for step in data2['execution_results']], ['inspect_media'])
        self.assertEqual(data2['output_url'], up_data['url'])
        self.assertIn('44100 Hz', data2['reply'])

        with self.subTest('silence trim and subsequent edits produce a downloadable result'):
            with wave.open(self.sample_wav_path, 'rb') as original_audio:
                audio_frames = original_audio.readframes(original_audio.getnframes())
                audio_params = original_audio.getparams()
            padded_audio = io.BytesIO()
            with wave.open(padded_audio, 'wb') as padded_file:
                padded_file.setparams(audio_params)
                silence = b'\0' * (audio_params.framerate // 2 * audio_params.sampwidth * audio_params.nchannels)
                padded_file.writeframes(silence + audio_frames + silence)
            uploaded = self.client.post('/api/agent/upload',
                data={'file': (io.BytesIO(padded_audio.getvalue()), 'padded.wav')},
                content_type='multipart/form-data').get_json()
            chain = {'reply': 'Finished editing.', 'tools': [
                {'name': 'auto_trim_silence', 'args': {'threshold': 40}},
                {'name': 'normalize_audio', 'args': {}},
                {'name': 'apply_audio_fade', 'args': {'fade_in_sec': 0.1, 'fade_out_sec': 0.1}},
            ]}
            with patch('agent_processor.query_agent_orchestrator', return_value=chain):
                edited = self.client.post('/api/agent/chat', json={
                    'message': 'Trim silence, normalize, then fade the edges',
                    'filename': uploaded['filename'], 'context': {'type': 'audio'},
                }).get_json()
            self.assertEqual(edited['status'], 'success')
            self.assertEqual(len(edited['execution_results']), 3)
            self.assertTrue(all(step['status'] == 'success' for step in edited['execution_results']))
            self.assertNotIn(os.sep, edited['output_file'])
            rendered = self.client.get(edited['output_url'])
            try:
                self.assertEqual(rendered.status_code, 200)
                with wave.open(io.BytesIO(rendered.data), 'rb') as output_audio:
                    duration = output_audio.getnframes() / output_audio.getframerate()
                    self.assertGreater(duration, 0.8)
                    self.assertLess(duration, 1.3)
            finally:
                rendered.close()

        with self.subTest('failed action stops dependent edits and reports partial completion'):
            partial_plan = {'reply': 'All edits completed.', 'tools': [
                {'name': 'normalize_audio', 'args': {}},
                {'name': 'unsupported_action', 'args': {}},
                {'name': 'apply_audio_fade', 'args': {}},
            ]}
            with patch('agent_processor.query_agent_orchestrator', return_value=partial_plan), \
                    patch('agent_processor.audio_processor.apply_fades') as fade:
                partial = self.client.post('/api/agent/chat', json={
                    'message': 'Run the edit chain', 'filename': up_data['filename'],
                }).get_json()
            fade.assert_not_called()
            self.assertEqual(partial['status'], 'partial')
            self.assertEqual(len(partial['execution_results']), 2)
            self.assertIn('stopped', partial['reply'])
            self.assertTrue(partial['output_url'])

        with self.subTest('missing output is a failed edit'):
            plan = {'reply': 'Done.', 'tools': [{'name': 'normalize_audio', 'args': {}}]}
            with patch('agent_processor.query_agent_orchestrator', return_value=plan), \
                    patch('agent_processor.audio_processor.normalize_audio', return_value=None):
                failed = self.client.post('/api/agent/chat', json={
                    'message': 'Normalize', 'filename': up_data['filename'],
                }).get_json()
            self.assertEqual(failed['status'], 'failed')
            self.assertIsNone(failed['output_url'])

        with self.subTest('unavailable transcription is not reported as successful'):
            plan = {'tools': [{'name': 'transcribe_audio', 'args': {}}]}
            with patch('agent_processor.query_agent_orchestrator', return_value=plan), \
                    patch('agent_processor.ai_processor.transcribe_audio', return_value={'available': False}):
                failed = self.client.post('/api/agent/chat', json={
                    'message': 'Transcribe', 'filename': up_data['filename'],
                }).get_json()
            self.assertEqual(failed['status'], 'failed')

        with self.subTest('analysis results link to the original upload'):
            with patch('agent_processor.query_agent_orchestrator', return_value=plan), \
                    patch('agent_processor.ai_processor.transcribe_audio', return_value={'available': True, 'segments': []}):
                analysis = self.client.post('/api/agent/chat', json={
                    'message': 'Transcribe', 'filename': up_data['filename'],
                }).get_json()
            self.assertEqual(analysis['output_url'], up_data['url'])

        with self.subTest('transcription includes downloadable UTF-8 transcript and timed subtitles'):
            transcript = {'available': True, 'full_text': "It's a creator's studio. नमस्ते!", 'model_name': 'private-backend',
                'segments': [
                    {'start': 0.125, 'end': 0.999, 'text': "It's a creator's studio."},
                    {'start': 59.9996, 'end': 60.125, 'text': 'नमस्ते!'},
                ]}
            with patch('agent_processor.query_agent_orchestrator', return_value=plan), \
                    patch('agent_processor.ai_processor.transcribe_audio', return_value=transcript):
                transcription = self.client.post('/api/agent/chat', json={
                    'message': 'Generate subtitles', 'filename': up_data['filename'],
                }).get_json()
            self.assertEqual(transcription['status'], 'success')
            self.assertEqual(transcription['output_url'], up_data['url'])
            data = transcription['execution_results'][0]['data']
            self.assertEqual(data['text'], transcript['full_text'])
            self.assertNotIn('model_name', data)
            self.assertEqual({item['format'] for item in data['exports']}, {'TXT', 'SRT', 'VTT'})
            for export in data['exports']:
                downloaded = self.client.get(export['url'])
                try:
                    self.assertEqual(downloaded.status_code, 200)
                    content = downloaded.data.decode('utf-8')
                    self.assertIn('नमस्ते!', content)
                    if export['format'] == 'SRT':
                        self.assertIn('00:00:00,125 --> 00:00:00,999', content)
                        self.assertIn('00:01:00,000 --> 00:01:00,125', content)
                    elif export['format'] == 'VTT':
                        self.assertTrue(content.startswith('WEBVTT\n\n'))
                        self.assertIn('00:00:00.125 --> 00:00:00.999', content)
                    else:
                        self.assertEqual(content.strip(), transcript['full_text'])
                finally:
                    downloaded.close()
            with tempfile.TemporaryDirectory() as export_dir:
                self.assertEqual(agent_processor._transcript_exports({'segments': []}, export_dir, 'silent'), [])
                exports = agent_processor._transcript_exports({'text': 'Untimed speech'}, export_dir, 'untimed')
                self.assertEqual([item['format'] for item in exports], ['TXT'])

        with self.subTest('editing requires media and clarification prevents execution'):
            with patch('agent_processor.query_agent_orchestrator', return_value=chain):
                missing_media = self.client.post('/api/agent/chat', json={'message': 'Trim silence'})
            self.assertEqual(missing_media.status_code, 400)
            with patch('agent_processor.query_agent_orchestrator', return_value={
                **chain, 'clarification_needed': True,
            }), patch('agent_processor.execute_agent_plan') as execute:
                clarification = self.client.post('/api/agent/chat', json={
                    'message': 'Edit this', 'filename': up_data['filename'],
                }).get_json()
            execute.assert_not_called()
            self.assertTrue(clarification['clarification_needed'])

    def test_08_agent_chat_error_validation(self):
        """Verify empty message payload returns 400 JSON error."""
        resp = self.client.post(
            '/api/agent/chat',
            data=json.dumps({"message": ""}),
            content_type='application/json'
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('error', data)

    # =========================================================================
    # SECTION 4: PRO STUDIO MULTITRACK PERSISTENCE (SAVE & LOAD)
    # =========================================================================

    def test_09_studio_project_save_and_load_roundtrip(self):
        """Verify full project state serialization, saving, and reloading."""
        project_state = {
            "name": "Cinematic Trailer Master",
            "tracks": [
                {
                    "id": "v1", "name": "B-Roll Video", "type": "video",
                    "volume": 1.0, "pan": 0.0, "muted": False,
                    "clips": [{
                        "id": "video_clip_1", "mediaId": "video_1", "start": 0.0,
                        "duration": 8.5, "offset": 1.25, "volume": 0.9, "pan": -0.35
                    }]
                },
                {
                    "id": "t1", "name": "Text / Subs", "type": "text",
                    "volume": 1.0, "pan": 0.0, "muted": False,
                    "clips": [{
                        "id": "text_clip_1", "start": 2.0, "duration": 4.0,
                        "text": "Opening title", "fontSize": 48, "color": "#ffffff"
                    }]
                },
                {
                    "id": "a1", "name": "Dialogue Track", "type": "audio",
                    "volume": 0.7, "pan": 0.25, "muted": False,
                    "clips": [{
                        "id": "audio_clip_1", "mediaId": "audio_1", "start": 0.0,
                        "duration": 12.0, "volume": 0.8, "pan": 0.4
                    }]
                }
            ],
            "playhead": 14.85,
            "bpm": 120,
            "zoom": 1.5,
            "version": "2.2"
        }

        # Save
        save_resp = self.client.post(
            '/studio/project/save',
            data=json.dumps(project_state),
            content_type='application/json'
        )
        self.assertEqual(save_resp.status_code, 200)
        save_data = save_resp.get_json()
        self.assertEqual(save_data.get('status'), 'success')
        proj_id = save_data.get('id')
        self.assertTrue(proj_id)
        saved_project_path = os.path.join(app.config['PROJECTS_FOLDER'], f"{proj_id}.aviproject")
        self.addCleanup(lambda: os.path.exists(saved_project_path) and os.remove(saved_project_path))

        # Load
        load_resp = self.client.get(f'/studio/project/load/{proj_id}')
        self.assertEqual(load_resp.status_code, 200)
        loaded = load_resp.get_json()
        self.assertEqual(loaded.get('name'), "Cinematic Trailer Master")
        self.assertEqual(len(loaded.get('tracks', [])), 3)
        self.assertEqual(loaded.get('playhead'), 14.85)
        self.assertEqual(loaded.get('bpm'), 120)
        self.assertEqual(loaded['tracks'][0]['clips'][0]['pan'], -0.35)
        self.assertEqual(loaded['tracks'][1]['clips'][0]['text'], "Opening title")
        self.assertEqual(loaded['tracks'][2]['pan'], 0.25)

        studio_html = self.client.get('/studio').get_data(as_text=True)
        core_js = self.client.get('/static/js/studio_core.js').get_data(as_text=True)
        video_js = self.client.get('/static/js/video_studio.js').get_data(as_text=True)
        self.assertIn('propClipVolume', studio_html)
        self.assertIn('propClipPan', studio_html)
        self.assertIn('downloadProjectFile', core_js)
        self.assertIn('syncProjectState', video_js)
        self.assertIn('texts:', video_js)
        playback_check = subprocess.run(
            ['node', os.path.join(WORKSPACE_DIR, 'studio_playback_test.js')],
            cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(playback_check.returncode, 0,
                         playback_check.stdout + playback_check.stderr)
        image_check = subprocess.run(
            ['node', os.path.join(WORKSPACE_DIR, 'image_studio_test.js')],
            cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(image_check.returncode, 0, image_check.stdout + image_check.stderr)
        navigation_check = subprocess.run(
            ['node', os.path.join(WORKSPACE_DIR, 'theme_navigation_test.js')],
            cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(navigation_check.returncode, 0,
                         navigation_check.stdout + navigation_check.stderr)
        for regression_script in ['studio_agent_test.js', 'agent_session_test.js', 'local_voice_input_test.js']:
            agent_check = subprocess.run(
                ['node', os.path.join(WORKSPACE_DIR, regression_script)],
                cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(agent_check.returncode, 0, agent_check.stdout + agent_check.stderr)

    def test_10_studio_project_security_and_not_found(self):
        """Verify project loader blocks path traversal and handles 404 cleanly."""
        # 1. Invalid Project ID format
        bad_resp = self.client.get('/studio/project/load/invalid..traversal')
        self.assertEqual(bad_resp.status_code, 400)
        self.assertIn('error', bad_resp.get_json())

        bad_save_resp = self.client.post(
            '/studio/project/save',
            data=json.dumps({"id": "../../outside", "name": "Unsafe", "tracks": []}),
            content_type='application/json'
        )
        self.assertEqual(bad_save_resp.status_code, 400)
        self.assertIn('error', bad_save_resp.get_json())

        # 2. Non-existent project
        not_found_resp = self.client.get('/studio/project/load/nonexistent-uuid-12345678')
        self.assertEqual(not_found_resp.status_code, 404)
        self.assertIn('error', not_found_resp.get_json())

    # =========================================================================
    # SECTION 5: AUDIO PROCESSING PIPELINE E2E
    # =========================================================================

    def test_11_audio_cut_merged_export(self):
        """Verify multi-region audio cutting with merged export mode."""
        regions = [
            {"start": 0.05, "end": 0.45, "name": "Intro Segment"},
            {"start": 0.55, "end": 0.95, "name": "Outro Segment"}
        ]
        with open(self.sample_wav_path, 'rb') as f:
            resp = self.client.post(
                '/cut',
                data={
                    'file': (io.BytesIO(f.read()), 'audio_to_cut.wav'),
                    'regions': json.dumps(regions),
                    'export_mode': 'merged',
                    'format': 'wav',
                    'normalize': 'true',
                    'fade_in': 'true',
                    'fade_out': 'true'
                },
                content_type='multipart/form-data'
            )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.data), 500)
        self.assertIn('attachment', resp.headers.get('Content-Disposition', ''))

    def test_12_audio_cut_separate_zip_export(self):
        """Verify multi-region audio cutting with individual separate ZIP export."""
        regions = [
            {"start": 0.1, "end": 0.4, "name": "Clip A"},
            {"start": 0.5, "end": 0.8, "name": "Clip B"}
        ]
        with open(self.sample_wav_path, 'rb') as f:
            resp = self.client.post(
                '/cut',
                data={
                    'file': (io.BytesIO(f.read()), 'audio_to_zip.wav'),
                    'regions': json.dumps(regions),
                    'export_mode': 'separate',
                    'format': 'wav'
                },
                content_type='multipart/form-data'
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, 'application/zip')
        
        # Verify valid ZIP archive containing both exported segments
        with zipfile.ZipFile(io.BytesIO(resp.data)) as zf:
            namelist = zf.namelist()
            self.assertEqual(len(namelist), 2)
            self.assertTrue(any('Clip_A' in name for name in namelist))
            self.assertTrue(any('Clip_B' in name for name in namelist))

    def test_13_audio_lufs_and_equalizer(self):
        """Verify integrated LUFS loudness analysis and 10-band parametric EQ."""
        # 1. LUFS Analysis
        with open(self.sample_wav_path, 'rb') as f:
            lufs_resp = self.client.post(
                '/audio/lufs',
                data={'file': (io.BytesIO(f.read()), 'sample.wav')},
                content_type='multipart/form-data'
            )
        self.assertEqual(lufs_resp.status_code, 200)
        lufs_data = lufs_resp.get_json()
        self.assertIn('lufs', lufs_data)
        self.assertIn('peak_db', lufs_data)

        # 2. Parametric EQ
        eq_bands = {"31": 2.0, "62": 1.5, "1000": -2.0, "16000": 3.0}
        with open(self.sample_wav_path, 'rb') as f:
            eq_resp = self.client.post(
                '/audio/eq',
                data={
                    'file': (io.BytesIO(f.read()), 'sample.wav'),
                    'eq_bands': json.dumps(eq_bands)
                },
                content_type='multipart/form-data'
            )
        self.assertEqual(eq_resp.status_code, 200)
        self.assertGreater(len(eq_resp.data), 500)

    def test_14_ai_audio_analysis_tools(self):
        """Verify silence detection, beat tracking, and voice activity detection endpoints."""
        # 1. Silence Detection
        with open(self.sample_wav_path, 'rb') as f:
            silence_resp = self.client.post(
                '/ai/detect-silence',
                data={
                    'file': (io.BytesIO(f.read()), 'audio.wav'),
                    'min_silence_len': '0.2',
                    'silence_thresh': '40'
                },
                content_type='multipart/form-data'
            )
        self.assertEqual(silence_resp.status_code, 200)
        self.assertIsInstance(silence_resp.get_json(), list)

        # 2. Beat Detection
        with open(self.sample_wav_path, 'rb') as f:
            beats_resp = self.client.post(
                '/ai/detect-beats',
                data={'file': (io.BytesIO(f.read()), 'audio.wav')},
                content_type='multipart/form-data'
            )
        self.assertEqual(beats_resp.status_code, 200)
        beats_data = beats_resp.get_json()
        self.assertIn('bpm', beats_data)
        self.assertIn('beat_times', beats_data)

        # 3. Voice Activity Detection (VAD)
        with open(self.sample_wav_path, 'rb') as f:
            vad_resp = self.client.post(
                '/ai/detect-vad',
                data={'file': (io.BytesIO(f.read()), 'audio.wav'), 'threshold_db': '-40.0'},
                content_type='multipart/form-data'
            )
        self.assertEqual(vad_resp.status_code, 200)
        self.assertIsInstance(vad_resp.get_json(), list)

    def test_15_audio_pitch_speed_and_multitrack_mix(self):
        """Verify pitch-preserved audio speed alteration and multi-track audio mixing."""
        # 1. Pitch Preserved Speed
        with open(self.sample_wav_path, 'rb') as f:
            speed_resp = self.client.post(
                '/ai/pitch-speed',
                data={'file': (io.BytesIO(f.read()), 'audio.wav'), 'speed': '1.25'},
                content_type='multipart/form-data'
            )
        self.assertEqual(speed_resp.status_code, 200)
        self.assertGreater(len(speed_resp.data), 500)

        # 2. Multi-track Audio Mix
        # Upload media file to get valid media_id
        with open(self.sample_wav_path, 'rb') as f:
            up_resp = self.client.post(
                '/video/upload',
                data={'file': (io.BytesIO(f.read()), 'audio_track.wav')},
                content_type='multipart/form-data'
            )
        media_id = up_resp.get_json()['id']

        mix_spec = {
            "format": "wav",
            "master_volume": 0.95,
            "tracks": [
                {"media_id": media_id, "start_time": 0.0, "volume": 0.8, "pan": -0.2},
                {"media_id": media_id, "start_time": 0.2, "volume": 0.7, "pan": 0.2}
            ]
        }
        mix_resp = self.client.post(
            '/audio/multitrack-mix',
            data=json.dumps(mix_spec),
            content_type='application/json'
        )
        self.assertEqual(mix_resp.status_code, 200)
        self.assertGreater(len(mix_resp.data), 500)

        pan_command = video_processor._build_clip_command(
            {
                "in": 0,
                "out": 1,
                "speed": 1,
                "volume": 0.8,
                "pan": 0.75,
                "hasAudio": True
            },
            self.sample_mp4_path,
            os.path.join(self.test_dir, 'pan_preview.mp4'),
            (160, 120),
            True,
            True,
            "none"
        )
        self.assertTrue(any("stereotools=balance_out=0.750" in part for part in pan_command))

    # =========================================================================
    # SECTION 6: IMAGE PROCESSING PIPELINE E2E
    # =========================================================================

    def test_16_image_enhance_super_resolution(self):
        """Verify image upscale/super-resolution processing."""
        with open(self.sample_png_path, 'rb') as f:
            resp = self.client.post(
                '/image/enhance',
                data={
                    'file': (io.BytesIO(f.read()), 'sample.png'),
                    'scale': '2',
                    'model': 'fast'
                },
                content_type='multipart/form-data'
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, 'image/png')
        self.assertGreater(len(resp.data), 100)

        # Verify output image dimensions were doubled
        out_img = Image.open(io.BytesIO(resp.data))
        self.assertEqual(out_img.size, (320, 320))

    def test_17_image_clarity_and_color_match(self):
        """Verify photo clarity enhancement and palette color matching."""
        # 1. Image Clarity
        with open(self.sample_png_path, 'rb') as f:
            clarity_resp = self.client.post(
                '/image/clarity',
                data={'file': (io.BytesIO(f.read()), 'sample.png')},
                content_type='multipart/form-data'
            )
        self.assertEqual(clarity_resp.status_code, 200)
        self.assertEqual(clarity_resp.mimetype, 'image/png')

        # 2. Color Match
        with open(self.sample_png_path, 'rb') as f:
            col_resp = self.client.post(
                '/image/color-match',
                data={
                    'file': (io.BytesIO(f.read()), 'sample.png'),
                    'palette': 'teal_orange'
                },
                content_type='multipart/form-data'
            )
        self.assertEqual(col_resp.status_code, 200)
        self.assertEqual(col_resp.mimetype, 'image/png')

    def test_18_image_remove_bg_and_inpaint(self):
        """Verify background removal and inpainting pipelines."""
        # 1. Background removal
        with open(self.sample_png_path, 'rb') as f:
            resp = self.client.post(
                '/image/remove-bg',
                data={'file': (io.BytesIO(f.read()), 'sample.png')},
                content_type='multipart/form-data'
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.mimetype, 'image/png')
        out_img = Image.open(io.BytesIO(resp.data))
        self.assertIn(out_img.mode, ('RGBA', 'LA', 'P'))

        # 2. Inpainting
        with open(self.sample_png_path, 'rb') as img_f, open(self.sample_mask_path, 'rb') as mask_f:
            inp_resp = self.client.post(
                '/image/inpaint',
                data={
                    'file': (io.BytesIO(img_f.read()), 'sample.png'),
                    'mask': (io.BytesIO(mask_f.read()), 'mask.png'),
                    'method': 'telea'
                },
                content_type='multipart/form-data'
            )
        self.assertEqual(inp_resp.status_code, 200)
        self.assertEqual(inp_resp.mimetype, 'image/png')

    # =========================================================================
    # SECTION 7: VIDEO PROCESSING PIPELINE E2E
    # =========================================================================

    def test_19_video_upload_and_metadata_probe(self):
        """Verify video upload, stream probing, and filmstrip generation."""
        with open(self.sample_mp4_path, 'rb') as f:
            resp = self.client.post(
                '/video/upload',
                data={'file': (io.BytesIO(f.read()), 'clip.mp4')},
                content_type='multipart/form-data'
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data.get('id'))
        self.assertEqual(data.get('has_video'), True)
        self.assertEqual(data.get('has_audio'), True)
        self.assertGreater(data.get('duration', 0), 0.5)
        self.assertIn('/media/', data.get('url'))

        # Return media_id for subsequent quick-tool tests
        return data['id']

    def test_20_video_quick_operations_and_media_serving(self):
        """Verify video quick operations: frame extraction, mute, extract audio, and media stream serving."""
        # First upload video to get media_id
        media_id = self.test_19_video_upload_and_metadata_probe()

        # 1. Media Range Request Serving (/media/<media_id>)
        media_resp = self.client.get(f"/media/{media_id}")
        self.assertIn(media_resp.status_code, [200, 206])

        # 2. Quick Frame Extraction
        frame_resp = self.client.post(
            '/video/quick',
            data={'media_id': media_id, 'op': 'frame', 't': '0.1'}
        )
        self.assertEqual(frame_resp.status_code, 200)
        self.assertGreater(len(frame_resp.data), 100)

        # 3. Quick Mute Audio
        mute_resp = self.client.post(
            '/video/quick',
            data={'media_id': media_id, 'op': 'mute'}
        )
        self.assertEqual(mute_resp.status_code, 200)
        self.assertGreater(len(mute_resp.data), 500)

        # 4. Extract Audio Track as MP3
        extract_resp = self.client.post(
            '/video/extract-audio',
            data={'media_id': media_id, 'format': 'mp3'}
        )
        self.assertEqual(extract_resp.status_code, 200)
        self.assertGreater(len(extract_resp.data), 500)

        export_resp = self.client.post('/video/export', json={
            'format': 'mp4',
            'resolution': 'original',
            'clips': [{
                'source': media_id, 'in': 0.0, 'out': 0.8,
                'speed': 1.0, 'volume': 0.8, 'pan': 0.5,
                'muted': False, 'hasAudio': True,
            }],
            'texts': [{
                'text': 'Studio timeline', 'start': 0.1, 'end': 0.7,
                'size': 30, 'color': '#ffffff', 'position': 'bc',
                'bg': True,
            }],
        })
        self.assertEqual(export_resp.status_code, 200)
        self.assertGreater(len(export_resp.data), 1000)

        with open(self.sample_wav_path, 'rb') as audio_file:
            audio_upload = self.client.post(
                '/api/agent/upload',
                data={'file': (io.BytesIO(audio_file.read()), 'timeline.wav')},
                content_type='multipart/form-data',
            )
        self.assertEqual(audio_upload.status_code, 200)
        positioned_resp = self.client.post('/video/export', json={
            'format': 'mp4', 'resolution': 'original',
            'clips': [
                {'source': media_id, 'in': 0, 'out': 0.8, 'start': 0.5},
                {'source': audio_upload.get_json()['id'],
                 'in': 0, 'out': 0.8, 'start': 0.5, 'pan': -0.5},
                {'source': media_id, 'in': 0, 'out': 0.5, 'start': 1.8},
            ],
        })
        self.assertEqual(positioned_resp.status_code, 200)
        positioned_path = os.path.join(self.test_dir, 'positioned.mp4')
        self.addCleanup(lambda: os.path.exists(positioned_path) and os.remove(positioned_path))
        with open(positioned_path, 'wb') as rendered_file:
            rendered_file.write(positioned_resp.data)
        rendered_info = video_processor.probe_media(positioned_path)
        self.assertAlmostEqual(rendered_info['duration'], 2.3, delta=0.15)
        self.assertTrue(rendered_info['has_video'])
        self.assertTrue(rendered_info['has_audio'])

        audio_resp = self.client.post('/video/export', json={
            'format': 'wav',
            'clips': [{'source': audio_upload.get_json()['id'],
                       'in': 0, 'out': 0.8, 'start': 0.5, 'pan': 0.5}],
        })
        self.assertEqual(audio_resp.status_code, 200)
        with wave.open(io.BytesIO(audio_resp.data), 'rb') as rendered_audio:
            self.assertEqual(rendered_audio.getnchannels(), 2)
            self.assertAlmostEqual(rendered_audio.getnframes() /
                                   rendered_audio.getframerate(), 1.3, delta=0.03)
            initial_samples = rendered_audio.readframes(int(
                rendered_audio.getframerate() * 0.3))
            self.assertEqual(set(initial_samples), {0})
            rendered_audio.setpos(int(rendered_audio.getframerate() * 0.6))
            active_samples = rendered_audio.readframes(1000)
            self.assertGreater(len(set(active_samples)), 1)

    def test_21_video_scene_detection(self):
        """Verify scene cut detection endpoint runs smoothly."""
        with open(self.sample_mp4_path, 'rb') as f:
            resp = self.client.post(
                '/video/detect-scenes',
                data={'file': (io.BytesIO(f.read()), 'clip.mp4'), 'threshold': '20.0'},
                content_type='multipart/form-data'
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('scenes', data)

    # =========================================================================
    # SECTION 8: SECURITY, FILE INTEGRITY & CLEANUP MANAGEMENT
    # =========================================================================

    def test_22_directory_traversal_hardening(self):
        """Verify path traversal attacks across all file serving routes are blocked."""
        traversal_attempts = [
            '/processed/../../app.py',
            '/processed/..%2F..%2Fapp.py',
            '/processed/subfolder/../../app.py',
            '/media/../../app.py',
            '/video/thumb/../../0'
        ]
        for url in traversal_attempts:
            with self.subTest(url=url):
                res = self.client.get(url)
                self.assertIn(res.status_code, [400, 403, 404])

    def test_23_file_extension_whitelist_validator(self):
        """Verify allowed_file helper strictly validates supported formats."""
        self.assertTrue(allowed_file('recording.mp3', ['audio']))
        self.assertTrue(allowed_file('camera.mp4', ['video']))
        self.assertTrue(allowed_file('artwork.png', ['image']))
        self.assertTrue(allowed_file('lossless.flac'))

        # Forbidden extensions
        self.assertFalse(allowed_file('malicious.exe'))
        self.assertFalse(allowed_file('script.sh'))
        self.assertFalse(allowed_file('payload.py'))
        self.assertFalse(allowed_file(''))
        self.assertFalse(allowed_file('no_extension'))

    def test_24_temp_cleanup_and_memory_reclaim(self):
        """Verify background disk cleaner and memory reclaim execution."""
        with tempfile.TemporaryDirectory() as storage_dir:
            storage_config = {
                'UPLOAD_FOLDER': os.path.join(storage_dir, 'uploads'),
                'PROCESSED_FOLDER': os.path.join(storage_dir, 'processed'),
                'PROJECTS_FOLDER': os.path.join(storage_dir, 'projects'),
            }
            for folder in storage_config.values():
                os.makedirs(folder)
            with patch.dict(app.config, storage_config):
                old_time = time.time() - 5000

                def create_storage_file(folder, name, aged=True):
                    path = os.path.join(storage_config[folder], name)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, 'w') as storage_file:
                        storage_file.write('test media')
                    if aged:
                        os.utime(path, (old_time, old_time))
                    return path

                orphan_upload = create_storage_file('UPLOAD_FOLDER', 'orphan.wav')
                saved_upload = create_storage_file('UPLOAD_FOLDER', 'saved-video.mp4')
                recent_upload = create_storage_file('UPLOAD_FOLDER', 'recent.wav', aged=False)
                saved_stem = create_storage_file('PROCESSED_FOLDER', 'stems_saved/vocals.wav')
                orphan_stem = create_storage_file('PROCESSED_FOLDER', 'stems_orphan/vocals.wav')
                saved_thumb = create_storage_file('PROCESSED_FOLDER', 'thumbs/saved-video/thumb_0.jpg')
                orphan_thumb = create_storage_file('PROCESSED_FOLDER', 'thumbs/orphan/thumb_0.jpg')
                forensic_upload = create_storage_file('UPLOAD_FOLDER', 'cases/case_01/evidence.wav')
                forensic_audit = create_storage_file('PROCESSED_FOLDER', 'evidence_hash.audit')
                for path in (orphan_stem, orphan_thumb, forensic_upload):
                    os.utime(os.path.dirname(path), (old_time, old_time))
                project_state = {
                    'id': 'storage-test', 'name': 'Retained media',
                    'mediaBin': [{'id': 'saved-video.mp4', 'url': '/media/saved-video.mp4'}],
                    'tracks': [{'clips': [{'source': 'saved-video.mp4'}]}],
                    'stems': [{'url': '/processed/stems_saved/vocals.wav'}],
                }
                save_response = self.client.post('/studio/project/save', json=project_state)
                self.assertEqual(save_response.status_code, 200)
                _cleanup_old_temp_files(max_age_seconds=3600)
                self.assertFalse(os.path.exists(orphan_upload))
                self.assertFalse(os.path.exists(orphan_stem))
                self.assertFalse(os.path.exists(orphan_thumb))
                self.assertFalse(os.path.exists(os.path.dirname(orphan_stem)))
                for retained in (saved_upload, recent_upload, saved_stem, saved_thumb, forensic_upload, forensic_audit):
                    self.assertTrue(os.path.exists(retained), retained)

                with patch('app.json.dump', side_effect=OSError('Interrupted save')):
                    failed_save = self.client.post('/studio/project/save', json={
                        **project_state, 'name': 'Should not replace existing project',
                    })
                self.assertEqual(failed_save.status_code, 500)
                loaded_project = self.client.get('/studio/project/load/storage-test').get_json()
                self.assertEqual(loaded_project['name'], 'Retained media')
                self.assertEqual(os.listdir(storage_config['PROJECTS_FOLDER']), ['storage-test.aviproject'])

                deferred_orphan = create_storage_file('UPLOAD_FOLDER', 'deferred.wav')
                unreadable_project = os.path.join(storage_config['PROJECTS_FOLDER'], 'broken.aviproject')
                with open(unreadable_project, 'w') as project_file:
                    project_file.write('{')
                _cleanup_old_temp_files(max_age_seconds=3600)
                self.assertTrue(os.path.exists(deferred_orphan))
                os.remove(unreadable_project)
                _cleanup_old_temp_files(max_age_seconds=3600)
                self.assertFalse(os.path.exists(deferred_orphan))

        with self.subTest('cleanup is debounced across concurrent requests'):
            scan_started = threading.Event()
            release_scan = threading.Event()
            cleanup_errors = []

            def controlled_scan(max_age_seconds):
                scan_started.set()
                if not release_scan.wait(2):
                    raise RuntimeError('Cleanup was not released')

            def cleanup_task():
                try:
                    _maybe_cleanup_temp_files(interval_seconds=300)
                except Exception as error:
                    cleanup_errors.append(error)

            with patch('app._last_cleanup_time', 0), \
                    patch('app.time.monotonic', return_value=10000), \
                    patch('app._cleanup_old_temp_files', side_effect=controlled_scan) as scan:
                cleanup_worker = threading.Thread(target=cleanup_task)
                cleanup_worker.start()
                try:
                    self.assertTrue(scan_started.wait(1))
                    _maybe_cleanup_temp_files(interval_seconds=300)
                finally:
                    release_scan.set()
                    cleanup_worker.join(3)
                _maybe_cleanup_temp_files(interval_seconds=300)
                self.assertFalse(cleanup_worker.is_alive())
                self.assertEqual(cleanup_errors, [])
                scan.assert_called_once()

        log_stream = io.StringIO()
        log_handler = logging.StreamHandler(log_stream)
        model_logger = logging.getLogger("model_manager")
        model_logger.addHandler(log_handler)
        try:
            manager = ModelLifecycleManager(idle_timeout_sec=0.0)
            with manager.session(
                "private_u2net_model_identifier",
                lambda: (object(), {}),
                auto_offload=True
            ):
                pass
        finally:
            model_logger.removeHandler(log_handler)

        log_output = log_stream.getvalue().lower()
        self.assertNotIn("u2net", log_output)
        self.assertNotIn("private_", log_output)

        with self.subTest('concurrent capability sessions remain exclusive'):
            manager = ModelLifecycleManager()
            first_running = threading.Event()
            release_first = threading.Event()
            second_requested = threading.Event()
            second_loaded = threading.Event()
            unloaded = []
            worker_errors = []

            def first_task():
                try:
                    with manager.session('first', lambda: (object(), {}),
                                         lambda instance: unloaded.append('first')):
                        first_running.set()
                        if not release_first.wait(3):
                            raise RuntimeError('First task was not released')
                except Exception as error:
                    worker_errors.append(error)

            def second_loader():
                second_loaded.set()
                self.assertEqual(unloaded, ['first'])
                return object(), {}

            def second_task():
                try:
                    second_requested.set()
                    with manager.session('second', second_loader):
                        pass
                except Exception as error:
                    worker_errors.append(error)

            with patch.object(manager, '_flush_system_memory'):
                first_worker = threading.Thread(target=first_task)
                second_worker = threading.Thread(target=second_task)
                first_worker.start()
                try:
                    self.assertTrue(first_running.wait(1))
                    second_worker.start()
                    self.assertTrue(second_requested.wait(1))
                    self.assertFalse(second_loaded.wait(0.1))
                    self.assertEqual(unloaded, [])
                finally:
                    release_first.set()
                    first_worker.join(3)
                    if second_worker.ident is not None:
                        second_worker.join(3)
                self.assertFalse(first_worker.is_alive())
                self.assertFalse(second_worker.is_alive())
                self.assertEqual(worker_errors, [])
                self.assertTrue(second_loaded.is_set())
                self.assertIsNone(manager._active_model_instance)

        with self.subTest('nested sessions retain outer ownership'):
            manager = ModelLifecycleManager()
            instance = object()
            with patch.object(manager, '_flush_system_memory'):
                with manager.session('shared', lambda: (instance, {})):
                    with manager.session('shared', lambda: self.fail('Unexpected reload')):
                        self.assertEqual(manager._session_depth, 2)
                    self.assertIs(manager._active_model_instance, instance)
                    with self.assertRaises(RuntimeError):
                        with manager.session('other', lambda: (object(), {})):
                            pass
                    with self.assertRaises(RuntimeError):
                        manager.unload_active_model()
                self.assertIsNone(manager._active_model_instance)
                self.assertEqual(manager._session_depth, 0)

        with self.subTest('failed tasks still release capability'):
            manager = ModelLifecycleManager()
            with patch.object(manager, '_flush_system_memory') as flush:
                with self.assertRaises(ValueError):
                    with manager.session('failure', lambda: (object(), {})):
                        raise ValueError('Task failed')
                self.assertIsNone(manager._active_model_instance)
                flush.assert_called_once()

        with self.subTest('stale timer cannot unload newer capability'):
            manager = ModelLifecycleManager(idle_timeout_sec=0.03)
            released = threading.Event()
            with patch.object(manager, '_flush_system_memory'):
                with manager.session('previous', lambda: (object(), {}), auto_offload=False):
                    pass
                previous_generation = manager._idle_generation
                with manager.session('current', lambda: (object(), {}),
                                     lambda instance: released.set(), auto_offload=False):
                    manager._unload_if_idle(previous_generation)
                    self.assertEqual(manager._active_model_id, 'current')
                self.assertTrue(released.wait(1))
                with manager._lock:
                    self.assertIsNone(manager._active_model_instance)

        with self.subTest('failed loading and cleanup preserve empty state'):
            manager = ModelLifecycleManager()

            def failed_loader():
                raise ValueError('Loading failed')

            def failed_cleanup(instance):
                raise ValueError('Cleanup failed')

            with patch.object(manager, '_flush_system_memory') as flush:
                with self.assertRaises(ValueError):
                    with manager.session('failed-loading', failed_loader):
                        pass
                self.assertIsNone(manager._active_model_instance)
                with manager.session('failed-cleanup', lambda: (object(), {}), failed_cleanup):
                    pass
                self.assertIsNone(manager._active_model_instance)
                self.assertEqual(manager._session_depth, 0)
                self.assertEqual(flush.call_count, 2)

        with self.subTest('memory reclamation does not load inference runtime'):
            reclaim_check = subprocess.run(
                [sys.executable, '-c',
                 'import sys; from model_manager import ModelLifecycleManager; '
                 'manager = ModelLifecycleManager(); manager._flush_system_memory(); '
                 'assert "torch" not in sys.modules'],
                cwd=WORKSPACE_DIR, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(reclaim_check.returncode, 0,
                             reclaim_check.stdout + reclaim_check.stderr)

        private_record = logging.LogRecord(
            "pyscenedetect",
            logging.INFO,
            __file__,
            0,
            "FFmpeg and OpenCV processing completed",
            (),
            None
        )
        BrandPrivacyLogFilter().filter(private_record)
        self.assertEqual(private_record.name, "media_engine")
        self.assertNotIn("ffmpeg", private_record.getMessage().lower())
        self.assertNotIn("opencv", private_record.getMessage().lower())


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(ComprehensivePlatformE2ETests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
