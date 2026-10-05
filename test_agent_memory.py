"""
Copilot Memory regression + retrieval-quality suite.

Run:  ./venv/Scripts/python.exe test_agent_memory.py

Covers storage CRUD, schema/migrations, eviction, concurrency, hybrid
retrieval quality on a golden set (recall@3 / MRR), token-budget packing,
embedding-space isolation, cross-session preference recall, orchestration
(job) memory, forget routes, and failure isolation.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

import agent_memory  # noqa: E402
from agent_memory import MemoryStore, estimate_tokens, condition_signature  # noqa: E402

BANNED_PUBLIC_TERMS = ('onnx', 'minilm', 'bert', 'sentence-transformers', 'sqlite', 'fts5',
                       'hnsw', 'numpy', 'huggingface', 'gemma', 'vllm', 'whisper')


class _Clock:
    def __init__(self, start=1_700_000_000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _FakeEmbedder:
    """Deterministic dense embedder used to exercise embedding-space handling."""

    def __init__(self, space, dim, offset=0):
        self.space = space
        self.dim = dim
        self.offset = offset
        self.calls = 0
        self.embedded = 0

    def available(self):
        return True

    def embed(self, texts):
        self.calls += 1
        self.embedded += len(texts)
        rows = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in agent_memory.tokenize(text):
                vec[(hash_token(token) + self.offset) % self.dim] += 1.0
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm else vec)
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)


def hash_token(token):
    import zlib
    return zlib.crc32(token.encode('utf-8'))


class _TempStoreMixin:
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='copilot-memory-test-')
        self.path = os.path.join(self.tmp, 'memory.db')
        self.clock = _Clock()
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_store(self, **kwargs):
        kwargs.setdefault('dense', 'off')
        kwargs.setdefault('clock', self.clock)
        store = MemoryStore(kwargs.pop('path', self.path), **kwargs)
        self.stores.append(store)
        return store


# ─────────────────────────────────────────────────────────────────────────
#  Storage
# ─────────────────────────────────────────────────────────────────────────

class StorageTests(_TempStoreMixin, unittest.TestCase):

    def test_crud_roundtrip_and_persistence(self):
        store = self.make_store()
        mid = store.add('conversation', 'Trim the interview from 2 to 5 seconds.', session_id='s1', role='user')
        row = store.get(mid)
        self.assertEqual(row['kind'], 'conversation')
        self.assertEqual(row['session_id'], 's1')
        self.assertIn('interview', row['text'])
        self.assertEqual(store.count(session_id='s1'), 1)
        store.close()
        reopened = self.make_store()
        self.assertEqual(reopened.get(mid)['text'], row['text'])
        self.assertTrue(reopened.delete(mid))
        self.assertIsNone(reopened.get(mid))
        self.assertEqual(reopened.count(), 0)

    def test_schema_version_and_wal(self):
        store = self.make_store()
        self.assertEqual(store.schema_version(), agent_memory.SCHEMA_VERSION)
        self.assertEqual(store.journal_mode(), 'wal')

    def test_newer_schema_disables_memory_without_raising(self):
        store = self.make_store()
        store.add('conversation', 'hello world', session_id='s')
        store._force_schema_version(agent_memory.SCHEMA_VERSION + 5)
        store.close()
        newer = self.make_store()
        self.assertFalse(newer.enabled)
        self.assertEqual(newer.search('hello', session_id='s'), [])
        self.assertEqual(newer.build_context('s', 'hello', token_budget=100), '')

    def test_text_is_sanitized_and_bounded(self):
        store = self.make_store()
        noisy = 'Clean C:\\Users\\me\\secret\\podcast.wav now\x00\x07 ' + 'x' * 10_000
        mid = store.add('conversation', noisy, session_id='s', role='user')
        text = store.get(mid)['text']
        self.assertLessEqual(len(text), agent_memory.MAX_TEXT_CHARS)
        self.assertNotIn('\x00', text)
        self.assertNotIn('Users\\me', text)
        self.assertIn('podcast.wav', text)

    def test_near_duplicate_preferences_are_merged_on_write(self):
        store = self.make_store()
        first = store.extract_preferences('s1', 'Always export podcasts at -16 LUFS.')
        second = store.extract_preferences('s1', 'Always export podcasts at -16 LUFS!')
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0], second[0])
        self.assertEqual(store.count(kind='preference'), 1)

    def test_newer_preference_supersedes_same_slot(self):
        store = self.make_store()
        store.extract_preferences('s1', 'Always export podcasts at -16 LUFS.')
        self.clock.advance(60)
        store.extract_preferences('s2', 'From now on export podcasts at -14 LUFS.')
        values = store.preference_values()
        self.assertEqual(values['loudness_lufs:podcast'], -14.0)
        self.assertEqual(store.count(kind='preference'), 1)

    def test_non_preferences_are_not_stored(self):
        store = self.make_store()
        for text in ('Never mind.', 'Does it always take this long?', 'Trim from 1 to 2 seconds'):
            self.assertEqual(store.extract_preferences('s', text), [])
        self.assertEqual(store.count(kind='preference'), 0)

    def test_corrupt_database_is_quarantined_and_recreated(self):
        with open(self.path, 'wb') as handle:
            handle.write(b'definitely not a database' * 200)
        store = self.make_store()
        self.assertTrue(store.enabled)
        store.add('conversation', 'works after recovery', session_id='s')
        self.assertEqual(store.count(), 1)
        quarantined = [name for name in os.listdir(self.tmp) if '.corrupt-' in name]
        self.assertTrue(quarantined)


class EvictionTests(_TempStoreMixin, unittest.TestCase):

    def test_ttl_expiry_removes_old_rows(self):
        store = self.make_store()
        store.add('conversation', 'temporary note about trimming', session_id='s', ttl=10)
        keep = store.add('conversation', 'long lived note about upscaling', session_id='s', ttl=10_000)
        self.clock.advance(100)
        store.evict()
        self.assertEqual(store.count(), 1)
        self.assertIsNotNone(store.get(keep))

    def test_size_cap_evicts_low_importance_but_keeps_preferences(self):
        store = self.make_store(max_rows=40)
        store.extract_preferences('s', 'I prefer MP3 at 320k for exports.')
        for index in range(80):
            self.clock.advance(1)
            store.add('conversation', f'message number {index} about clip {index * 7}', session_id='s',
                      role='user', importance=0.2 if index % 2 else 0.6)
        self.assertLessEqual(store.count(), 40)
        self.assertEqual(store.count(kind='preference'), 1)
        stats = store.stats()
        self.assertGreater(stats['evicted_total'], 0)


class ConcurrencyTests(_TempStoreMixin, unittest.TestCase):

    def test_parallel_writes_and_reads_are_safe(self):
        store = self.make_store()
        errors = []

        def writer(worker):
            try:
                for index in range(40):
                    store.add('conversation', f'worker {worker} edit {index} normalize trim fade',
                              session_id=f's{worker}', role='user')
                    if index % 5 == 0:
                        store.search('normalize the clip', session_id=f's{worker}', k=3)
                        store.build_context(f's{worker}', 'fade the clip', token_budget=120)
            except Exception as error:  # pragma: no cover - surfaced by assertion
                errors.append(error)

        threads = [threading.Thread(target=writer, args=(worker,)) for worker in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(store.count(kind='conversation'), 320)


# ─────────────────────────────────────────────────────────────────────────
#  Retrieval quality (golden set)
# ─────────────────────────────────────────────────────────────────────────

GOLDEN_MEMORIES = {
    'c1': 'Trim the interview video from 12.5 seconds to 47 seconds.',
    'c2': 'Remove the background from my product photo and keep the shadow.',
    'c3': 'The podcast has a loud air conditioner hum in the background.',
    'c4': 'Transcribe the lecture recording and give me subtitles.',
    'c5': 'Upscale the old family portrait to 4x resolution.',
    'c6': 'Make the drone clip play at half speed for slow motion.',
    'c7': 'Convert the screen recording to an animated GIF for the docs.',
    'c8': 'Compress the wedding video so I can email it.',
    'c9': 'Add a two second fade in and fade out to the intro music.',
    'c10': 'Extract the audio track from the concert video as WAV.',
    'c11': 'Cut the long silences out of the voice memo.',
    'c12': 'Restore the faces in the scanned graduation photo.',
    'c13': 'Make the vocals clearer, the speech sounds muffled.',
    'c14': 'Sharpen the blurry street photo, it is out of focus.',
    'c15': 'Convert the episode to FLAC for archiving.',
    'c16': 'The client wants the trailer in WebM for their website.',
    'c17': 'My channel intro is 1080p, keep that resolution.',
    'c18': 'The interview guest speaks very quietly compared to the host.',
    'c19': 'Remove background noise like keyboard clicks from the stream recording.',
    'c20': 'Speed up the tutorial screencast to 1.5x.',
    'd1': 'Hello there!',
    'd2': 'Thanks, that looks great.',
    'd3': 'ok sounds good',
}

GOLDEN_QUERIES = [
    ('what loudness target do I use for podcasts?', {'p_lufs'}),
    ('which bitrate should audio exports use', {'p_mp3'}),
    ('what timestamps did I want for the interview cut', {'c1'}),
    ('product photo cutout', {'c2'}),
    ('hum in the podcast', {'c3'}),
    ('captions for the lecture', {'c4'}),
    ('enlarge the family portrait', {'c5'}),
    ('slow motion drone footage', {'c6'}),
    ('gif of the screen recording', {'c7'}),
    ('shrink the wedding video file size', {'c8'}),
    ('fade on the intro music', {'c9'}),
    ('pull the soundtrack out of the concert video', {'c10'}),
    ('remove pauses from the voice memo', {'c11'}),
    ('fix the faces on the graduation scan', {'c12'}),
    ('muffled speech clarity', {'c13'}),
    ('blurry street picture', {'c14'}),
    ('archive the episode lossless', {'c15'}),
    ("trailer format for the client's website", {'c16'}),
    ('what resolution is my channel intro', {'c17'}),
    ('quiet guest in the interview', {'c18'}),
    ('keyboard clicks noise on the stream', {'c19'}),
    ('speed of the tutorial screencast', {'c20'}),
    ('podcast noise cleanup that worked before', {'c3', 'o1'}),
]


def _seed_golden(store, clock):
    ids = {}
    ids['p_lufs'] = store.extract_preferences('golden', 'Always export podcasts at -16 LUFS.')[0]
    ids['p_mp3'] = store.extract_preferences('golden', 'I prefer MP3 at 320k for audio exports.')[0]
    for key, text in GOLDEN_MEMORIES.items():
        clock.advance(30)
        ids[key] = store.add('conversation', text, session_id='golden', role='user')
    ids['o1'] = store.record_outcome(
        'golden', 'clean up the podcast noise',
        [{'name': 'reduce_noise', 'args': {}}, {'name': 'normalize_audio', 'args': {}}],
        [{'tool': 'reduce_noise', 'status': 'success'}, {'tool': 'normalize_audio', 'status': 'success'}])
    return {value: key for key, value in ids.items()}


def _evaluate(store, id_to_key, k=3):
    hits, reciprocal_ranks, misses = 0, [], []
    for query, expected in GOLDEN_QUERIES:
        results = store.search(query, session_id='golden', k=k)
        keys = [id_to_key.get(item['id']) for item in results]
        rank = next((index + 1 for index, key in enumerate(keys) if key in expected), None)
        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append((query, keys))
    return hits / len(GOLDEN_QUERIES), sum(reciprocal_ranks) / len(reciprocal_ranks), misses


class RetrievalQualityTests(_TempStoreMixin, unittest.TestCase):

    def test_lexical_hybrid_recall_at_3_and_mrr(self):
        store = self.make_store()
        id_to_key = _seed_golden(store, self.clock)
        recall, mrr, misses = _evaluate(store, id_to_key)
        print(f'\n[golden lexical] recall@3={recall:.3f} MRR={mrr:.3f} misses={misses}')
        self.assertGreaterEqual(recall, 0.90, misses)
        self.assertGreaterEqual(mrr, 0.75)

    def test_lexical_fallback_without_full_text_index_still_meets_bar(self):
        store = self.make_store(full_text_index=False)
        id_to_key = _seed_golden(store, self.clock)
        recall, mrr, misses = _evaluate(store, id_to_key)
        print(f'\n[golden lexical, built-in ranking] recall@3={recall:.3f} MRR={mrr:.3f}')
        self.assertGreaterEqual(recall, 0.90, misses)

    def test_on_device_semantic_tier_when_available(self):
        embedder = agent_memory.OnDeviceSemanticEmbedder.discover()
        if embedder is None:
            self.skipTest('No cached on-device semantic index available.')
        from model_manager import AdaptiveQualityGovernor, ResourceSnapshot
        governor = AdaptiveQualityGovernor(probe=lambda: ResourceSnapshot(16, 32, 16, 0, 0))
        store = self.make_store(dense='auto', embedders=[embedder], governor=governor)
        id_to_key = _seed_golden(store, self.clock)
        recall, mrr, misses = _evaluate(store, id_to_key)
        print(f'\n[golden semantic hybrid] recall@3={recall:.3f} MRR={mrr:.3f} misses={misses}')
        self.assertGreaterEqual(recall, 0.90, misses)
        self.assertIn(embedder.space, store.stats()['vectors_by_space_internal'])

    def test_mmr_removes_near_duplicates_from_results(self):
        store = self.make_store()
        for variant in ('Normalize the podcast loudness.', 'normalize the podcast loudness please',
                        'Please normalize the podcast loudness!', 'Normalize the podcast loudness, please.'):
            self.clock.advance(5)
            store.add('conversation', variant, session_id='s', role='user')
        store.add('conversation', 'Normalize the podcast and also reduce the hiss.', session_id='s', role='user')
        results = store.search('normalize podcast loudness', session_id='s', k=3)
        texts = [item['text'] for item in results]
        self.assertTrue(any('hiss' in text for text in texts), texts)

    def test_metadata_filters_isolate_sessions(self):
        store = self.make_store()
        store.add('conversation', 'Secret trailer cut for project falcon.', session_id='a', role='user')
        store.add('conversation', 'Trim the trailer to ten seconds.', session_id='b', role='user')
        results = store.search('falcon trailer', session_id='b', k=5)
        self.assertTrue(results)
        self.assertTrue(all('falcon' not in item['text'] for item in results))


# ─────────────────────────────────────────────────────────────────────────
#  Embedding spaces
# ─────────────────────────────────────────────────────────────────────────

class EmbeddingSpaceTests(_TempStoreMixin, unittest.TestCase):

    def test_vectors_are_never_compared_across_spaces(self):
        space_a = _FakeEmbedder('dense:test-a', 32)
        store = self.make_store(dense='auto', embedders=[space_a])
        for text in ('Remove the photo background.', 'Normalize podcast loudness.', 'Trim the clip.'):
            store.add('conversation', text, session_id='s', role='user')
        self.assertTrue(store.search('podcast loudness', session_id='s', k=2))
        self.assertGreater(space_a.embedded, 0)
        store.close()

        # A different space with a different dimensionality: old vectors must be ignored, not mixed.
        space_b = _FakeEmbedder('dense:test-b', 48, offset=7)
        reopened = self.make_store(dense='auto', embedders=[space_b])
        results = reopened.search('podcast loudness', session_id='s', k=2)
        self.assertIn('podcast', results[0]['text'].lower())
        self.assertEqual(space_b.embedded, 4)  # query + three lazy re-embeds in its own space
        spaces = reopened.stats()['vectors_by_space_internal']
        self.assertEqual(spaces['dense:test-a'], 3)
        self.assertEqual(spaces['dense:test-b'], 3)

    def test_unavailable_semantic_tier_falls_back_to_lexical(self):
        class Broken(_FakeEmbedder):
            def embed(self, texts):
                raise MemoryError('simulated pressure')

        from model_manager import AdaptiveQualityGovernor, ResourceSnapshot
        governor = AdaptiveQualityGovernor(probe=lambda: ResourceSnapshot(16, 32, 16, 0, 0))
        store = self.make_store(dense='auto', embedders=[Broken('dense:broken', 16)], governor=governor)
        store.add('conversation', 'Upscale the portrait 4x.', session_id='s', role='user')
        results = store.search('upscale portrait', session_id='s', k=1)
        self.assertEqual(len(results), 1)

    def test_dense_off_never_calls_embedders(self):
        embedder = _FakeEmbedder('dense:never', 8)
        store = self.make_store(dense='off', embedders=[embedder])
        store.add('conversation', 'Upscale the portrait 4x.', session_id='s', role='user')
        store.search('upscale', session_id='s')
        self.assertEqual(embedder.calls, 0)


# ─────────────────────────────────────────────────────────────────────────
#  Context packing
# ─────────────────────────────────────────────────────────────────────────

class ContextPackingTests(_TempStoreMixin, unittest.TestCase):

    def _busy_session(self, store):
        store.extract_preferences('s', 'Always export podcasts at -16 LUFS.')
        for index in range(60):
            self.clock.advance(10)
            store.record_turn('s', 'user', f'Please normalize podcast episode {index} and trim the intro '
                                           f'of segment {index} because it drags on ' + 'and on ' * 30)
            store.record_turn('s', 'assistant', f'Done with episode {index}. ' + 'details ' * 40)

    def test_budget_is_strictly_respected(self):
        store = self.make_store()
        self._busy_session(store)
        for budget in (40, 80, 150, 300, 600, 1200):
            context = store.build_context('s', 'normalize the podcast episode', token_budget=budget)
            self.assertLessEqual(estimate_tokens(context), budget, budget)
            self.assertLessEqual(len(context) / 4.0, budget)

    def test_context_has_recent_turns_summary_and_preferences(self):
        store = self.make_store()
        self._busy_session(store)
        history = [{'role': 'user', 'content': 'Trim the intro of episode 59'},
                   {'role': 'assistant', 'content': 'Trimmed the intro.'}]
        context = store.build_context('s', 'export the podcast', token_budget=700, recent_history=history)
        self.assertIn('-16 LUFS', context)
        self.assertIn('Trimmed the intro.', context)
        self.assertIn('Session summary', context)
        self.assertTrue(context.rstrip().endswith('Trimmed the intro.'))

    def test_recent_turns_come_from_memory_without_client_history(self):
        store = self.make_store()
        store.record_turn('s', 'user', 'Remove the background from the logo.')
        store.record_turn('s', 'assistant', 'Background removed.')
        context = store.build_context('s', 'make it sharper', token_budget=300)
        self.assertIn('Background removed.', context)

    def test_tiny_budget_returns_empty_or_fitting_text(self):
        store = self.make_store()
        self._busy_session(store)
        self.assertLessEqual(estimate_tokens(store.build_context('s', 'x', token_budget=5)), 5)
        self.assertEqual(store.build_context('s', 'x', token_budget=0), '')

    def test_estimate_is_conservative(self):
        self.assertGreaterEqual(estimate_tokens('a' * 400), 100)
        self.assertGreaterEqual(estimate_tokens('é' * 100), 50)
        self.assertEqual(estimate_tokens(''), 0)


class CrossSessionTests(_TempStoreMixin, unittest.TestCase):

    def test_preferences_cross_sessions_but_conversations_do_not(self):
        store = self.make_store()
        store.extract_preferences('s1', 'Always export podcasts at -16 LUFS.')
        store.record_turn('s1', 'user', 'The secret interview with the mayor needs trimming.')
        context = store.build_context('s2', 'export my podcast interview', token_budget=500)
        self.assertIn('-16 LUFS', context)
        self.assertNotIn('mayor', context)

    def test_media_facts_follow_the_content_fingerprint(self):
        store = self.make_store()
        store.record_media_facts('fp-123', lines=['Background noise detected (signal-to-noise about 12 dB).'],
                                 condition=['type:audio', 'bg:noise'])
        context = store.build_context('another', 'clean this', media_fingerprint='fp-123', token_budget=300)
        self.assertIn('Background noise detected', context)
        self.assertNotIn('Background noise', store.build_context('another', 'clean this', token_budget=300))


# ─────────────────────────────────────────────────────────────────────────
#  Orchestration memory
# ─────────────────────────────────────────────────────────────────────────

class OrchestrationMemoryTests(_TempStoreMixin, unittest.TestCase):
    NOISY = ['type:audio', 'bg:noise', 'speech']
    CLEAN = ['type:audio', 'bg:clean', 'speech']
    CLEAN_PLAN = [{'name': 'reduce_noise', 'args': {}}, {'name': 'normalize_audio', 'args': {}}]

    def test_jobs_merge_and_recall_with_outcomes(self):
        store = self.make_store()
        ok = [{'tool': 'reduce_noise', 'status': 'success'}, {'tool': 'normalize_audio', 'status': 'success'}]
        store.record_job('a', 'clean up this noisy podcast', self.CLEAN_PLAN, ok, condition=self.NOISY, elapsed_ms=900)
        store.record_job('b', 'clean the podcast noise', self.CLEAN_PLAN, ok, condition=self.NOISY, elapsed_ms=1100)
        bad_plan = [{'name': 'enhance_speech', 'args': {}}]
        store.record_job('a', 'clean up this noisy podcast', bad_plan,
                         [{'tool': 'enhance_speech', 'status': 'error'}], condition=self.NOISY)
        self.assertEqual(store.count(kind='job'), 2)
        jobs = store.recall_similar_jobs('please clean my noisy podcast', self.NOISY, k=3)
        self.assertEqual(jobs[0]['tools'], ['reduce_noise', 'normalize_audio'])
        self.assertEqual(jobs[0]['runs'], 2)
        self.assertEqual(jobs[0]['recommendation'], 'reuse')
        self.assertAlmostEqual(jobs[0]['avg_ms'], 1000, delta=1)
        avoid = [job for job in jobs if job['tools'] == ['enhance_speech']]
        self.assertEqual(avoid[0]['recommendation'], 'avoid')

    def test_condition_similarity_prefers_matching_media(self):
        store = self.make_store()
        ok = [{'tool': 'normalize_audio', 'status': 'success'}]
        store.record_job('a', 'make the podcast sound better', [{'name': 'normalize_audio', 'args': {}}],
                         ok, condition=self.CLEAN)
        store.record_job('a', 'make the podcast sound better', self.CLEAN_PLAN,
                         ok * 2, condition=self.NOISY)
        jobs = store.recall_similar_jobs('make the podcast sound better', self.NOISY, k=2)
        self.assertEqual(jobs[0]['condition'], sorted(self.NOISY))

    def test_condition_signature_from_inspector_report(self):
        report = {'type': 'audio', 'audio': {'background': 'music', 'speech_likely': True, 'quiet': True,
                                             'noisy': False, 'music': True, 'clipping': False, 'silent': False}}
        self.assertEqual(condition_signature(report), ['bg:music', 'quiet', 'speech', 'type:audio'])
        image = {'type': 'image', 'image': {'blurry': True, 'noisy': True, 'has_alpha': False, 'small': False}}
        self.assertEqual(condition_signature(image), ['img:blurry', 'img:noisy', 'type:image'])
        self.assertEqual(condition_signature('bg:noise type:audio'), ['bg:noise', 'type:audio'])
        self.assertEqual(condition_signature(None), [])

    def test_repeat_plan_reuses_last_successful_edit(self):
        store = self.make_store()
        self.assertIsNone(store.repeat_plan('s', 'do the same as before'))
        store.record_outcome('s', 'normalize it', [{'name': 'normalize_audio', 'args': {}}],
                             [{'tool': 'normalize_audio', 'status': 'success'}])
        store.record_outcome('s', 'transcribe', [{'name': 'transcribe_audio', 'args': {}}],
                             [{'tool': 'transcribe_audio', 'status': 'error'}])
        for phrase in ('do the same as before', 'Same again please', 'repeat the last edit', 'do that again'):
            plan = store.repeat_plan('s', phrase)
            self.assertIsNotNone(plan, phrase)
            self.assertEqual(plan['tools'], [{'name': 'normalize_audio', 'args': {}}])
        for phrase in ('normalize the same file to -14', 'trim 1 to 2', 'the same thing happened'):
            self.assertIsNone(store.repeat_plan('s', phrase), phrase)
        self.assertIsNone(store.repeat_plan('other', 'do the same as before'))

    def test_apply_preferences_fills_only_missing_arguments(self):
        store = self.make_store()
        store.extract_preferences('s', 'I prefer FLAC for audio exports.')
        plan = {'tools': [{'name': 'convert_audio_format', 'args': {}}]}
        self.assertEqual(store.apply_preferences(plan, 'convert the audio')['tools'][0]['args'],
                         {'target_format': 'flac'})
        explicit = {'tools': [{'name': 'convert_audio_format', 'args': {'target_format': 'wav'}}]}
        self.assertEqual(store.apply_preferences(explicit, 'convert to wav')['tools'][0]['args'],
                         {'target_format': 'wav'})
        untouched = {'tools': [{'name': 'convert_audio_format', 'args': {}}]}
        self.assertEqual(store.apply_preferences(untouched, 'convert it to mp3')['tools'][0]['args'], {})

    def test_context_includes_similar_job_hints_for_edit_requests(self):
        store = self.make_store()
        store.record_job('a', 'clean up this noisy podcast', self.CLEAN_PLAN,
                         [{'tool': 'reduce_noise', 'status': 'success'}], condition=self.NOISY)
        context = store.build_context('b', 'clean the noisy podcast', token_budget=400,
                                      media_condition=self.NOISY)
        self.assertIn('reduce_noise', context)


# ─────────────────────────────────────────────────────────────────────────
#  Privacy
# ─────────────────────────────────────────────────────────────────────────

class PrivacyTests(_TempStoreMixin, unittest.TestCase):

    def test_forget_session_and_all(self):
        store = self.make_store()
        store.record_turn('a', 'user', 'Trim the clip.')
        store.extract_preferences('a', 'Always export at 320k MP3.')
        store.record_turn('b', 'user', 'Upscale the photo.')
        removed = store.forget_session('a')
        self.assertEqual(removed, 2)
        self.assertEqual(store.count(session_id='a'), 0)
        self.assertEqual(store.count(session_id='b'), 1)
        self.assertEqual(store.search('trim clip', session_id='a'), [])
        store.forget_all()
        self.assertEqual(store.count(), 0)

    def test_public_surfaces_have_no_engine_names(self):
        store = self.make_store(embedders=[_FakeEmbedder('dense:x', 8)])
        store.record_turn('s', 'user', 'Normalize it.')
        public = json.dumps(store.public_stats()).lower()
        context = store.build_context('s', 'normalize', token_budget=200).lower()
        for term in BANNED_PUBLIC_TERMS:
            self.assertNotIn(term, public)
            self.assertNotIn(term, context)


# ─────────────────────────────────────────────────────────────────────────
#  Integration: orchestrator + Flask routes
# ─────────────────────────────────────────────────────────────────────────

class IntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from app import app
        import agent_processor
        cls.app = app
        cls.agent_processor = agent_processor
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='copilot-memory-int-')
        self.store = agent_memory.configure(os.path.join(self.tmp, 'memory.db'), dense='off')

    def tearDown(self):
        agent_memory.flush()
        agent_memory.configure(None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _chat(self, message, session_id='sess_int', **extra):
        return self.client.post('/api/agent/chat', json={'message': message, 'session_id': session_id, **extra})

    def test_orchestrator_prompt_uses_packed_memory(self):
        self.store.extract_preferences('s1', 'Always export podcasts at -16 LUFS.')
        captured = {}

        def fake_request(url, payload):
            captured['payload'] = payload
            return json.dumps({'tools': [], 'reply': 'ok'})

        history = [{'role': 'user', 'content': f'old message {i}'} for i in range(12)]
        with patch.object(self.agent_processor, '_request_reasoning_content', side_effect=fake_request):
            self.agent_processor.query_agent_orchestrator('export the podcast', {'type': 'audio'}, history,
                                                          session_id='s2')
        user_content = captured['payload']['messages'][1]['content']
        self.assertIn('-16 LUFS', user_content)
        self.assertIn('old message 11', user_content)
        self.assertNotIn('old message 3', user_content)

    def test_orchestrator_falls_back_to_raw_history_when_memory_fails(self):
        captured = {}

        def fake_request(url, payload):
            captured['payload'] = payload
            return json.dumps({'tools': [], 'reply': 'ok'})

        history = [{'role': 'user', 'content': 'earlier turn'}]
        with patch.object(agent_memory, 'build_context', side_effect=RuntimeError('boom')), \
                patch.object(self.agent_processor, '_request_reasoning_content', side_effect=fake_request):
            result = self.agent_processor.query_agent_orchestrator('hello there', None, history, session_id='s')
        self.assertEqual(result['reply'], 'ok')
        self.assertIn('earlier turn', captured['payload']['messages'][1]['content'])

    def test_chat_records_turns_and_preferences(self):
        resp = self._chat('I prefer MP3 at 320k for audio exports.')
        self.assertEqual(resp.status_code, 200)
        agent_memory.flush()
        self.assertEqual(self.store.count(session_id='sess_int', kind='conversation'), 2)
        self.assertEqual(self.store.preference_values()['audio_format'], 'mp3')

    def test_preferences_are_kept_even_when_the_chat_cannot_run_an_edit(self):
        resp = self._chat('Always export podcasts at -16 LUFS.')  # an edit request without media
        self.assertIn(resp.status_code, (200, 400))
        agent_memory.flush()
        self.assertEqual(self.store.preference_values().get('loudness_lufs:podcast'), -16.0)

    def test_repeat_request_replays_last_successful_edit_on_new_media(self):
        import math
        import struct
        import wave
        uploaded = []
        for index in range(2):
            path = os.path.join(self.tmp, f'tone{index}.wav')
            with wave.open(path, 'w') as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b''.join(struct.pack('<h', int(8000 * math.sin(i / (5 + index))))
                                            for i in range(16000)))
            with open(path, 'rb') as source:
                resp = self.client.post('/api/agent/upload', data={'file': (source, f'tone{index}.wav')},
                                        content_type='multipart/form-data')
            self.assertEqual(resp.status_code, 200, resp.get_json())
            uploaded.append(resp.get_json()['filename'])
        first = self._chat('Normalize loudness to -18 LUFS', filename=uploaded[0])
        self.assertEqual(first.get_json().get('status'), 'success', first.get_json())
        agent_memory.flush()
        again = self._chat('do the same as before', filename=uploaded[1]).get_json()
        self.assertEqual(again.get('status'), 'success', again)
        self.assertEqual([tool['name'] for tool in again['tools_planned']], ['normalize_audio'])
        self.assertEqual(again['tools_planned'][0]['args'].get('target_lufs'), -18.0)
        agent_memory.flush()
        jobs = self.store.recall_similar_jobs('normalize loudness', ['type:audio'], k=1)
        self.assertEqual(jobs[0]['tools'], ['normalize_audio'])

    def test_invalid_session_ids_are_rejected_or_ignored(self):
        self.assertEqual(self.client.post('/api/agent/chat', json={'message': 'hi', 'session_id': 5}).status_code, 400)
        self.assertEqual(self._chat('hi', session_id='../../etc').status_code, 200)
        agent_memory.flush()
        self.assertEqual(self.store.count(kind='conversation'), 0)

    def test_forget_routes(self):
        self._chat('I prefer MP3 at 320k for audio exports.')
        agent_memory.flush()
        self.assertGreater(self.store.count(session_id='sess_int'), 0)
        self.assertEqual(self.client.delete('/api/agent/memory').status_code, 400)
        resp = self.client.delete('/api/agent/memory?session_id=sess_int')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.store.count(session_id='sess_int'), 0)
        self._chat('Trim it', session_id='other')
        agent_memory.flush()
        resp = self.client.delete('/api/agent/memory?scope=all')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.store.count(), 0)
        status = self.client.get('/api/agent/memory').get_json()
        import branding
        self.assertIn(f'{branding.assistant_name()} memory', json.dumps(status))
        self.assertNotIn('Copilot', json.dumps(status))
        for term in BANNED_PUBLIC_TERMS:
            self.assertNotIn(term, json.dumps(status).lower())

    def test_corrupt_database_never_breaks_chat(self):
        bad_path = os.path.join(self.tmp, 'broken.db')
        with open(bad_path, 'wb') as handle:
            handle.write(os.urandom(4096))
        agent_memory.configure(bad_path, dense='off')
        resp = self._chat('hello')
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json().get('reply'))

    def test_memory_exceptions_never_break_chat(self):
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Capture()
        logging.getLogger().addHandler(handler)
        try:
            with patch.object(agent_memory, 'get_store', side_effect=RuntimeError('disk on fire')):
                resp = self._chat('hello')
                agent_memory.flush()
        finally:
            logging.getLogger().removeHandler(handler)
        self.assertEqual(resp.status_code, 200)
        for message in records:
            for term in BANNED_PUBLIC_TERMS:
                self.assertNotIn(term, message.lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)
