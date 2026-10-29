"""Chapter recovery preserves source text and finishes before loading the TTS."""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from epub_audio.adapters.ollama import ChapterLLMError, OllamaChapters
from epub_audio.cli.main import main, plan_books
from epub_audio.core.models import BookMetadata, BookPlan, Chapter
from epub_audio.io.llm_chapters import improve_chapters, chapter_report, _validate
from epub_audio.io.preferences import load_preferences, save_preferences


def source_plan():
    return BookPlan([
        Chapter('title.xhtml', ['Mon roman', 'Mon auteur']),
        Chapter('front.xhtml', ['Une dédicace à conserver.']),
        Chapter('split1.xhtml', ['CHAPITRE PREMIER', 'Le récit commence.']),
        Chapter('split2.xhtml', ['La suite dans un autre fichier.', 'II', 'Le récit se poursuit.']),
    ], metadata=BookMetadata('Mon roman', ['Mon auteur']), mode='legacy', fallback_reason='Navigation invalide.')


def fake_client(chapter_ids=(3, 6), part_ids=()):
    def classify(system, content, schema):
        entries = json.loads(content)['candidates']
        return {'decisions': {str(e['paragraph']): {'confidence': 0.99,
                              'kind': ('chapter' if e['paragraph'] in chapter_ids else
                                       'part' if e['paragraph'] in part_ids else 'other')}
                              for e in entries}}
    return Mock(model='test-local-model', classify=Mock(side_effect=classify))


class LLMChapterTests(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        self.cache = Path(self.folder) / 'cache'
        self.plan = source_plan()

    def improve(self, client=None, plan=None):
        return improve_chapters(plan or self.plan, client or fake_client(),
                                cache_dir=self.cache, message=lambda _: None)

    def test_recovers_chapters_and_groups_front_matter_without_dropping_text(self):
        result = self.improve()
        self.assertEqual(result.mode, 'LLM local')
        self.assertEqual([c.title for c in result.chapters], ['Préambule', 'CHAPITRE PREMIER', 'II'])
        self.assertEqual([p for c in result.chapters for p in c.paragraphs],
                         [p for c in self.plan.chapters for p in c.paragraphs])
        self.assertEqual(result.chapters[1].paragraphs[-1], 'La suite dans un autre fichier.')
        self.assertEqual(result.chapters[2].heading_indices, (0,))
        self.assertEqual(self.plan.mode, 'legacy')
        self.assertIs(result.metadata, self.plan.metadata)

    def test_cached_plan_is_revalidated_and_does_not_contact_the_model(self):
        original = self.improve()
        client = fake_client()
        client.classify.side_effect = AssertionError('Cached plan should not call the LLM')
        cached = self.improve(client)
        self.assertEqual(original.chapters, cached.chapters)
        self.assertTrue(cached.chapter_analysis['cached'])

    def test_book_changes_invalidate_cache(self):
        self.improve()
        changed = source_plan()
        changed.chapters[-1].paragraphs[-1] = 'Une autre fin, conservée intégralement.'
        client = fake_client()
        result = self.improve(client, changed)
        client.classify.assert_called()
        self.assertFalse(result.chapter_analysis['cached'])
        self.assertEqual(result.chapters[-1].paragraphs[-1], changed.chapters[-1].paragraphs[-1])

    def test_corrupt_cache_is_recomputed(self):
        self.improve()
        path = next(self.cache.glob('*.json'))
        data = json.loads(path.read_text(encoding='utf-8'))
        data['decisions'][0]['paragraph'] = 99999
        path.write_text(json.dumps(data), encoding='utf-8')
        client = fake_client()
        self.improve(client)
        client.classify.assert_called()

    def test_model_change_invalidates_cache(self):
        self.improve()
        client = fake_client()
        client.model = 'a-different-local-model'
        self.improve(client)
        client.classify.assert_called()

    def test_low_confidence_keeps_the_original_plan(self):
        client = fake_client()
        classify = client.classify.side_effect
        def uncertain(*args):
            result = classify(*args)
            for d in result['decisions'].values():
                d['confidence'] = 0.5
            return result
        client.classify.side_effect = uncertain
        result = self.improve(client)
        self.assertEqual(result.chapters, self.plan.chapters)
        self.assertEqual(result.chapter_analysis['status'], 'unchanged')

    def test_part_and_immediate_chapter_are_kept_together(self):
        plan = BookPlan([Chapter('all.xhtml', ['PREMIÈRE PARTIE', 'I', 'Le récit commence.',
                                               'II', 'La suite.'])], mode='legacy')
        result = self.improve(fake_client((1, 3), (0,)), plan)
        self.assertEqual([c.title for c in result.chapters], ['PREMIÈRE PARTIE — I', 'II'])
        self.assertEqual(result.chapters[0].paragraphs, plan.chapters[0].paragraphs[:3])

    def test_invalid_or_incomplete_responses_are_rejected(self):
        valid = {'paragraph': 3, 'kind': 'chapter', 'confidence': 0.9}
        for decisions in ([valid, valid], [], [{**valid, 'paragraph': True}],
                          [{**valid, 'paragraph': 100}], [{**valid, 'confidence': float('nan')}],
                          [{**valid, 'confidence': True}], [{**valid, 'kind': 'rewrite'}]):
            with self.subTest(decisions=decisions), self.assertRaises(ChapterLLMError):
                _validate({'decisions': decisions}, [3])

    def test_late_batch_failure_never_returns_a_partial_plan_or_cache(self):
        client = fake_client()
        classify = client.classify.side_effect
        count = 0
        def fail_second(*args):
            nonlocal count
            count += 1
            if count == 2:
                raise ChapterLLMError('Interrupted generation')
            return classify(*args)
        client.classify.side_effect = fail_second
        with patch('epub_audio.io.llm_chapters.BATCH_SIZE', 2), self.assertRaises(ChapterLLMError):
            self.improve(client)
        self.assertFalse(list(self.cache.glob('*.json')))
        self.assertEqual(self.plan.chapters, source_plan().chapters)

    def test_report_tracks_the_exact_preserved_text(self):
        before = chapter_report(self.plan, 'book.epub')
        after = chapter_report(self.improve(), 'book.epub')
        self.assertEqual(before['text_sha256'], after['text_sha256'])
        self.assertEqual(before['paragraphs'], after['paragraphs'])
        self.assertEqual([c['start_paragraph'] for c in after['chapters']], [0, 3, 6])

    def test_ocr_sentence_split_before_a_lowercase_continuation_is_not_a_title(self):
        plan = BookPlan([Chapter('all.xhtml', ['CHAPITRE I', 'Le voyage commence.',
                                               'Un jour ils retrouveraient', 'la trace du voyageur.',
                                               'CHAPITRE II', 'La suite.'])], mode='legacy')
        # Even an over-eager model never gets offered the cut-off sentence as a boundary.
        result = self.improve(fake_client((0, 2, 4)), plan)
        self.assertEqual(len(result.chapters), 2)
        self.assertEqual(result.chapters[0].paragraphs, plan.chapters[0].paragraphs[:4])


class OllamaLifecycleTests(unittest.TestCase):
    def test_remote_or_cloud_endpoints_are_rejected(self):
        for url in ('https://ollama.com', 'http://192.168.0.27:11434',
                    'http://localhost.evil.com', 'http://user:pass@localhost:11434',
                    'http://127.0.0.1:11434/api/chat'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaChapters(url=url)
        with self.assertRaises(ValueError):
            OllamaChapters(model='qwen:cloud')

    def test_structured_analysis_and_unload_before_exit(self):
        client = OllamaChapters()
        client._ready = True
        client._request = Mock(side_effect=[{'done': True, 'message': {'content': '{"decisions":[]}'}}, {}])
        with client:
            self.assertEqual(client.classify('system', 'book', {'type': 'object'}), {'decisions': []})
        payload = client._request.call_args_list[0].args[1]
        self.assertFalse(payload['stream'])
        self.assertFalse(payload['think'])
        self.assertEqual(payload['format'], {'type': 'object'})
        self.assertEqual(client._request.call_args.args, ('/api/generate', {'model': client.model, 'keep_alive': 0}))

    def test_invalid_json_still_releases_video_memory(self):
        client = OllamaChapters()
        client._ready = True
        client._request = Mock(side_effect=[{'done': True, 'message': {'content': 'not JSON'}}, {}])
        with self.assertRaises(ChapterLLMError), client:
            client.classify('system', 'book', {})
        self.assertEqual(client._request.call_args.args[0], '/api/generate')

    def test_failed_unload_stops_processing_before_tts(self):
        client = OllamaChapters()
        client._loaded = True
        client._request = Mock(side_effect=ChapterLLMError('Offline'))
        with self.assertRaisesRegex(ChapterLLMError, 'mémoire GPU'), client:
            pass

    def test_a_remote_model_alias_on_a_local_server_is_refused(self):
        client = OllamaChapters(model='my-alias')
        tags = {'models': [{'name': 'my-alias'}]}
        client._request = Mock(side_effect=[tags, tags, {'remote_model': 'a-cloud-model'}])
        with self.assertRaisesRegex(ChapterLLMError, 'sans relais cloud'):
            client._ensure_ready()


class ChapterCLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.root / 'book.epub'
        self.source.write_bytes(b'Mocked EPUB')
        self.output = self.root / 'out'
        self.prefs = self.root / 'prefs.json'
        stack.enter_context(patch('epub_audio.io.preferences.PREFERENCES_FILE', self.prefs))
        stack.enter_context(patch('epub_audio.cli.main.require_tools'))
        stack.enter_context(patch('sys.stdout', new=io.StringIO()))

    def test_analysis_only_never_loads_tts_and_never_overwrites_reports(self):
        plans = {self.source.resolve(): source_plan()}
        with patch('epub_audio.cli.main.plan_books', return_value=(plans, False)), \
             patch('epub_audio.cli.main.QwenEngine') as engine, \
             patch('epub_audio.cli.main.process_book') as process:
            for _ in range(2):
                self.assertEqual(main([str(self.source), '--output', str(self.output),
                                       '--chapters-only', '--llm-chapters', '--yes']), 0)
        engine.assert_not_called()
        process.assert_not_called()
        self.assertEqual(len(list(self.output.rglob('*.chapters.json'))), 2)

    def test_analysis_finishes_before_gpu_tts_is_constructed(self):
        events = []
        plan = source_plan()
        def prepare(*args, **kwargs):
            events.append('llm_done_and_unloaded')
            return {self.source.resolve(): plan}, False
        def engine(*args, **kwargs):
            events.append('tts_constructed')
            return Mock()
        with patch('epub_audio.cli.main.plan_books', side_effect=prepare), \
             patch('epub_audio.cli.main.QwenEngine', side_effect=engine), \
             patch('epub_audio.cli.main.process_book') as process:
            self.assertEqual(main([str(self.source), '--llm-chapters', '--yes']), 0)
        self.assertEqual(events, ['llm_done_and_unloaded', 'tts_constructed'])
        self.assertIs(process.call_args.kwargs['book_plan'], plan)

    def test_unavailable_llm_falls_back_with_visible_error_and_nonzero_analysis_status(self):
        client = fake_client()
        client.classify.side_effect = ChapterLLMError('Modèle absent')
        context = Mock()
        context.__enter__ = Mock(return_value=client)
        context.__exit__ = Mock(return_value=False)
        messages = []
        with patch('epub_audio.adapters.ollama.OllamaChapters', return_value=context), \
             patch('epub_audio.io.epub.read_book', return_value=source_plan()), \
             patch('epub_audio.io.llm_chapters.CACHE_DIR', self.root / 'cache'):
            plans, failed = plan_books([self.source], use_llm=True, model=client.model,
                                       url='http://127.0.0.1:11434', message=messages.append)
        self.assertTrue(failed)
        self.assertEqual(plans[self.source].mode, 'legacy')
        self.assertEqual(plans[self.source].chapters, source_plan().chapters)
        self.assertIn('Découpage habituel conservé', ' '.join(messages))

    def test_disabled_mode_does_not_contact_llm(self):
        save_preferences('expr_chap_s01_0004-7e5a77eb', 1, llm_chapters=True)
        with patch('epub_audio.cli.main.plan_books') as analyze, \
             patch('epub_audio.cli.main.QwenEngine'), patch('epub_audio.cli.main.process_book'):
            self.assertEqual(main([str(self.source), '--no-llm-chapters', '--yes']), 0)
        analyze.assert_not_called()
        self.assertFalse(load_preferences()['llm_chapters'])

    def test_interactive_option_and_preference_are_remembered(self):
        with patch('epub_audio.cli.main.plan_books', return_value=({self.source.resolve(): source_plan()}, False)), \
             patch('epub_audio.cli.main.QwenEngine'), patch('epub_audio.cli.main.process_book'), \
             patch('builtins.input', side_effect=['', '2', '', '']) as ask:
            self.assertEqual(main([str(self.source), '--voice', 'expr_chap_s01_0004-7e5a77eb',
                                   '--batch-size', '6', '--segmentation', 'paragraph', '--format', 'wav', '--no-normalize']), 0)
        self.assertTrue(load_preferences()['llm_chapters'])
        self.assertIn('Améliorer le découpage', ask.call_args_list[1].args[0])
        save_preferences('expr_chap_s01_0004-7e5a77eb', 1)
        self.assertTrue(load_preferences()['llm_chapters'])


if __name__ == '__main__':
    unittest.main()
