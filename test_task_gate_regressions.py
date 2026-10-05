"""Regressions discovered while checking the remaining task-card quality gate."""
import unittest

import agent_nlu
import agent_processor
from app import app


class GateRegressionTests(unittest.TestCase):
    def test_shared_head_makes_no_external_font_requests(self):
        with app.test_client() as client:
            for route in ('/', '/agent', '/chat', '/audio', '/video', '/image', '/studio'):
                with self.subTest(route=route):
                    page = client.get(route).get_data(as_text=True).lower()
                    self.assertNotIn('fonts.googleapis.com', page)
                    self.assertNotIn('fonts.gstatic.com', page)

    def test_edit_commands_take_precedence_over_media_knowledge(self):
        for prompt, kind, tools in (
            ('Convert to MP3', 'audio', ['convert_audio_format']),
            ('Export video as WebM', 'video', ['convert_video_format']),
            ('Inspect loudness and sample rate', 'audio', ['inspect_media']),
            ('Upscale 2x then remove background', 'image', ['upscale_image', 'remove_background']),
            ('Extract audio as WAV then trim from 0.2 to 0.8 seconds', 'video', ['extract_audio', 'trim_audio']),
        ):
            with self.subTest(prompt=prompt):
                plan = agent_processor._fallback_intent_parser(prompt, {'type': kind})
                self.assertEqual([tool['name'] for tool in plan['tools']], tools)

    def test_media_questions_still_receive_answers(self):
        for prompt in ('What is MP3?', 'What is LUFS?', 'MP4 vs WebM'):
            with self.subTest(prompt=prompt):
                plan = agent_nlu.parse_request(prompt, {'type': 'audio'})
                self.assertFalse(plan['tools'])
                self.assertEqual(plan['thought'], 'Media knowledge question.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
