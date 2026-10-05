"""
Platform hardening regression suite.

Covers fixes for: stored XSS via media upload/serving, security headers,
studio export formats (WebM / GIF / 4K), in-use media surviving storage
purges, privacy sanitizer coverage for responses and processor logs, and
safe file resolution in the Copilot chat route.

Run:  python test_platform_hardening.py
"""

import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

import app as app_module
import video_processor
from app import app, _sanitize_public_text, _sanitize_public_payload, BrandPrivacyLogFilter


class HardeningTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()
        cls.tmp = tempfile.mkdtemp(prefix='hardening_')
        cls.original_folders = (app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'])
        cls.upload_dir = os.path.join(cls.tmp, 'uploads')
        cls.processed_dir = os.path.join(cls.tmp, 'processed')
        os.makedirs(cls.upload_dir)
        os.makedirs(cls.processed_dir)
        app.config['UPLOAD_FOLDER'] = cls.upload_dir
        app.config['PROCESSED_FOLDER'] = cls.processed_dir

        cls.video_path = os.path.join(cls.tmp, 'clip.mp4')
        subprocess.run([
            video_processor.FFMPEG, '-y',
            '-f', 'lavfi', '-i', 'testsrc=duration=1:size=160x120:rate=15',
            '-f', 'lavfi', '-i', 'sine=frequency=660:duration=1',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', cls.video_path
        ], capture_output=True, check=True)

    @classmethod
    def tearDownClass(cls):
        app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'] = cls.original_folders
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _upload_video(self, name='clip.mp4'):
        with open(self.video_path, 'rb') as handle:
            data = handle.read()
        return self.client.post('/video/upload', data={'file': (io.BytesIO(data), name)},
                                content_type='multipart/form-data')

    # ── Upload / serving security ─────────────────────────────────────────

    def test_upload_rejects_active_content_extensions(self):
        with open(self.video_path, 'rb') as handle:
            payload = b'<script>alert(1)</script>' + handle.read()
        before = set(os.listdir(self.upload_dir))
        for name in ('evil.html', 'evil.svg', 'evil.htm', 'evil.js', 'noext'):
            with self.subTest(name=name):
                resp = self.client.post('/video/upload', data={'file': (io.BytesIO(payload), name)},
                                        content_type='multipart/form-data')
                self.assertEqual(resp.status_code, 400)
        self.assertEqual(set(os.listdir(self.upload_dir)) - before, set())

    def test_upload_error_does_not_leak_paths_or_engines(self):
        resp = self.client.post('/video/upload', data={'file': (io.BytesIO(b'not a video'), 'x.mp4')},
                                content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 400)
        body = resp.get_data(as_text=True).lower()
        for leak in (self.upload_dir.lower().replace('\\', '\\\\'), 'uploads', 'mov,mp4', 'ffprobe', 'ffmpeg'):
            self.assertNotIn(leak, body)

    def test_media_serving_refuses_non_media_extensions_and_sets_nosniff(self):
        planted = os.path.join(self.upload_dir, 'planted.html')
        with open(planted, 'w', encoding='utf-8') as handle:
            handle.write('<script>alert(1)</script>')
        self.assertEqual(self.client.get('/media/planted.html').status_code, 404)

        resp = self._upload_video()
        self.assertEqual(resp.status_code, 200)
        media = self.client.get(resp.get_json()['url'])
        self.assertEqual(media.status_code, 200)
        self.assertEqual(media.headers.get('X-Content-Type-Options'), 'nosniff')
        media.close()

    def test_every_response_has_nosniff(self):
        for route in ('/', '/health', '/api/system/resources'):
            with self.subTest(route=route):
                self.assertEqual(self.client.get(route).headers.get('X-Content-Type-Options'), 'nosniff')

    # ── Storage hygiene ───────────────────────────────────────────────────

    def test_media_in_use_survives_age_based_purge(self):
        resp = self._upload_video()
        media_id = resp.get_json()['id']
        path = os.path.join(self.upload_dir, media_id)
        stale = time.time() - 7200
        os.utime(path, (stale, stale))

        # Touching through the serving route marks it as in use.
        self.client.get(f'/media/{media_id}').close()
        app_module._cleanup_old_temp_files(max_age_seconds=3600)
        self.assertTrue(os.path.exists(path))

        # Untouched stale files are still purged.
        os.utime(path, (stale, stale))
        app_module._cleanup_old_temp_files(max_age_seconds=3600)
        self.assertFalse(os.path.exists(path))

    # ── Export formats ────────────────────────────────────────────────────

    def _export(self, fmt, resolution='720p'):
        media_id = self._upload_video().get_json()['id']
        spec = {
            'format': fmt,
            'resolution': resolution,
            'clips': [{'source': media_id, 'start': 0, 'in': 0, 'out': 1, 'speed': 1,
                       'volume': 1, 'pan': 0, 'muted': False, 'hasAudio': True}],
            'texts': [],
        }
        return self.client.post('/video/export', json=spec)

    def test_export_webm_is_real_webm(self):
        resp = self._export('webm')
        self.assertEqual(resp.status_code, 200, resp.data[:200])
        self.assertEqual(resp.mimetype, 'video/webm')
        self.assertEqual(resp.data[:4], b'\x1a\x45\xdf\xa3')  # EBML header

    def test_export_gif_is_real_gif(self):
        resp = self._export('gif')
        self.assertEqual(resp.status_code, 200, resp.data[:200])
        self.assertEqual(resp.mimetype, 'image/gif')
        self.assertTrue(resp.data.startswith(b'GIF8'))

    def test_export_4k_preset_is_supported(self):
        self.assertEqual(video_processor.RESOLUTIONS['4k'], (3840, 2160))

    def test_export_unknown_format_is_rejected(self):
        resp = self.client.post('/video/export', json={'format': 'exe', 'clips': []})
        self.assertEqual(resp.status_code, 400)

    # ── Privacy ───────────────────────────────────────────────────────────

    def test_sanitizer_covers_engine_families(self):
        samples = ['Real-ESRGAN x4', 'onnxruntime', 'torchvision', 'cv2.error', 'DeepFilterNet3',
                   'PySceneDetect', 'demucs', 'faster-whisper large-v3', 'google/gemma-3-4b',
                   'u2net_human_seg', 'birefnet-general', 'rembg', 'EDSR', 'fsrcnn', 'lama inpaint']
        for sample in samples:
            with self.subTest(sample=sample):
                cleaned = _sanitize_public_text(sample).lower()
                for term in ('esrgan', 'onnx', 'torch', 'cv2', 'deepfilter', 'scenedetect', 'demucs',
                             'whisper', 'gemma', 'u2net', 'birefnet', 'rembg', 'edsr', 'fsrcnn', 'lama'):
                    self.assertNotIn(term, cleaned)
        self.assertEqual(_sanitize_public_text('llama farm and Lamarck'), 'llama farm and Lamarck')

    def test_private_payload_keys_are_stripped(self):
        payload = {'text': 'hi', 'model': 'large-v3-turbo', 'device': 'GPU', 'note': 'x',
                   'model_accuracy': '97%', 'nested': [{'engine': 'edsr', 'ok': 1}]}
        self.assertEqual(_sanitize_public_payload(payload), {'text': 'hi', 'nested': [{'ok': 1}]})

    def test_processor_loggers_are_filtered(self):
        stt_logger = logging.getLogger('transcribe')
        self.assertTrue(stt_logger.handlers, 'speech logger should have its own handler')
        for handler in stt_logger.handlers:
            self.assertTrue(any(isinstance(f, BrandPrivacyLogFilter) for f in handler.filters))
        record = logging.LogRecord('transcribe', logging.INFO, __file__, 1,
                                   'loading %s via %s', ('large-v3 faster-whisper', 'ctranslate2'), None)
        BrandPrivacyLogFilter().filter(record)
        self.assertNotIn('whisper', record.getMessage().lower())
        self.assertNotIn('ctranslate2', record.getMessage().lower())

    # ── Copilot file resolution ───────────────────────────────────────────

    def test_chat_does_not_resolve_directories_as_media(self):
        for filename in ('', '/', 'uploads/', '..'):
            with self.subTest(filename=filename):
                resp = self.client.post('/api/agent/chat', json={
                    'message': 'trim from 0 to 0.5 seconds', 'filename': filename,
                    'context': {'type': 'video'}})
                self.assertIn(resp.status_code, (200, 400))
                self.assertNotEqual(resp.status_code, 500)


if __name__ == '__main__':
    unittest.main(verbosity=2)
