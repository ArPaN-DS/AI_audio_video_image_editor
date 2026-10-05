"""
End-to-End Test Suite verifying all bug fixes and platform stabilization.
"""
import os
import sys
import unittest
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class PlatformVerificationTests(unittest.TestCase):

    def test_01_logger_import(self):
        """Verify logger loads safely without requiring PostgreSQL."""
        import logger
        self.assertTrue(hasattr(logger, 'log_upload_details'))

    def test_02_video_processor_fixes(self):
        """Verify FFMPEG is properly referenced and process_timeline is defined."""
        import video_processor
        self.assertTrue(hasattr(video_processor, 'FFMPEG'))
        self.assertTrue(hasattr(video_processor, 'burn_subtitles'))
        self.assertTrue(callable(video_processor.process_timeline))

    def test_03_audio_processor_new_functions(self):
        """Verify normalize_audio and apply_fades exist in audio_processor."""
        import audio_processor
        self.assertTrue(hasattr(audio_processor, 'normalize_audio'))
        self.assertTrue(callable(audio_processor.normalize_audio))
        self.assertTrue(hasattr(audio_processor, 'apply_fades'))
        self.assertTrue(callable(audio_processor.apply_fades))

    def test_04_ai_processor_youtube_chapters(self):
        """Verify no circular import in generate_youtube_chapters."""
        import ai_processor
        self.assertTrue(hasattr(ai_processor, 'generate_youtube_chapters'))
        self.assertTrue(callable(ai_processor.generate_youtube_chapters))

    def test_05_flask_app_health_and_json_errors(self):
        """Verify Flask /health endpoint and JSON error standardization."""
        from app import app
        app.config['TESTING'] = True
        client = app.test_client()

        # 1. Health check
        res = client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'healthy')

        # 2. /cut without file returns JSON error (not plain text)
        res_cut = client.post('/cut')
        self.assertEqual(res_cut.status_code, 400)
        self.assertTrue(res_cut.is_json)
        cut_data = res_cut.get_json()
        self.assertIn('error', cut_data)

        # 3. /ai/detect-silence returns JSON error
        res_silence = client.post('/ai/detect-silence')
        self.assertEqual(res_silence.status_code, 400)
        self.assertTrue(res_silence.is_json)
        silence_data = res_silence.get_json()
        self.assertIn('error', silence_data)

        # 4. /processed with traversal attempt is denied
        res_traversal = client.get('/processed/../../app.py')
        self.assertIn(res_traversal.status_code, [400, 403, 404])

    def test_06_agent_tools_and_clean_landing(self):
        """Verify agent tools schema endpoint and landing page render."""
        from app import app
        app.config['TESTING'] = True
        client = app.test_client()

        res_tools = client.get('/api/agent/tools')
        self.assertEqual(res_tools.status_code, 200)
        tools_data = res_tools.get_json()
        self.assertEqual(tools_data.get('status'), 'success')

        res_landing = client.get('/')
        self.assertEqual(res_landing.status_code, 200)
        html = res_landing.get_data(as_text=True)
        # Verify no raw model names leak on the landing page
        self.assertNotIn('Gemma 4 E2B', html)
        self.assertNotIn('Whisper AI', html)

if __name__ == '__main__':
    unittest.main()
