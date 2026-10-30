"""Interrupted work must retain complete speech, and never certify a partial WAV."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch

from epub_audio.cli.main import main
from epub_audio.core.models import BookMetadata, BookPlan, Chapter
from epub_audio.core.options import AudioOptions
from epub_audio.core.pipeline import process_book
from epub_audio.io.llm_chapters import chapter_report
from epub_audio.io.fs import retry_windows_io
from epub_audio.io.resume import (BookCheckpoint, discover_incomplete, file_hash,
                                   import_legacy, legacy_completed, valid_wav)


def write_wav(path, value=1000, frames=2400):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(struct.pack('<h', value) * frames)
    return path


class Voice:
    batch_size = 1
    max_segment_characters = None

    def __init__(self, fail=None):
        self.texts = []
        self.fail = fail

    def synthesize(self, text, path):
        self.texts.append(text)
        if text == self.fail:
            # A well-formed WAV containing only a prefix is still incomplete.
            write_wav(path, frames=100)
            raise OSError('interrupted')
        return write_wav(path, value=1000 + ord(text[0]))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'livre.epub'
        self.source.write_bytes(b'fixture')
        self.output = self.root / 'output'
        self.options = AudioOptions(audio_format='wav', paragraph_pause_ms=0)
        self.texts = [letter * 210 for letter in 'ABCDEF']
        self.plan = BookPlan([Chapter('Premier', self.texts[:2]), Chapter('Second', self.texts[2:])])

    def process(self, voice, **kwargs):
        with patch('epub_audio.core.pipeline.read_book', return_value=self.plan):
            return process_book(self.source, voice, output_dir=self.output,
                                options=kwargs.pop('options', self.options), message=lambda _: None, **kwargs)

    def interrupt(self):
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.process(Voice(fail=self.texts[3]))
        job, = discover_incomplete(self.output)
        return job['directory']

    def test_resume_keeps_completed_chapter_and_saved_passages_without_repetition(self):
        directory = self.interrupt()
        first = (directory / '.chapter-1.wav').read_bytes()
        state = BookCheckpoint.load(directory)
        self.assertEqual(set(state.data['chapters']), {'1'})
        self.assertTrue(valid_wav(directory / '.chapter-2.building.wav'))
        voice = Voice()
        result = self.process(voice, resume_dir=directory)
        self.assertNotIn(self.texts[0], voice.texts)
        self.assertNotIn(self.texts[1], voice.texts)
        self.assertNotIn(self.texts[2], voice.texts)
        self.assertIn(self.texts[3], voice.texts)
        tracks = sorted(result.glob('Chapitre *.wav'))
        self.assertEqual(tracks[0].read_bytes(), first)
        with wave.open(str(tracks[1]), 'rb') as audio:
            expected = b''.join(struct.pack('<h', 1000 + ord(t[0])) * 2400 for t in self.texts[2:])
            self.assertEqual(audio.readframes(audio.getnframes()), expected)
        self.assertFalse(discover_incomplete(self.output))
        self.assertFalse(list(result.glob('.segments-*')))
        self.assertFalse(list(result.glob('.*.wav')))

    def test_a_modified_but_valid_completed_wav_is_regenerated(self):
        directory = self.interrupt()
        write_wav(directory / '.chapter-1.wav', frames=100)
        voice = Voice()
        self.process(voice, resume_dir=directory)
        self.assertIn(self.texts[0], voice.texts)
        self.assertIn(self.texts[1], voice.texts)

    def test_corrupted_saved_passage_is_regenerated(self):
        directory = self.interrupt()
        write_wav(directory / '.segments-2' / '000000.wav', frames=10)
        voice = Voice()
        self.process(voice, resume_dir=directory)
        self.assertIn(self.texts[2], voice.texts)

    def test_changed_epub_or_options_are_rejected_before_synthesis(self):
        directory = self.interrupt()
        voice = Voice()
        with self.assertRaisesRegex(ValueError, 'réglages'):
            self.process(voice, resume_dir=directory, options=AudioOptions(audio_format='wav', paragraph_pause_ms=9))
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'EPUB a changé'):
            self.process(voice, resume_dir=directory)
        self.assertEqual(voice.texts, [])

    def test_changed_segment_plan_is_rejected(self):
        directory = self.interrupt()
        state = BookCheckpoint.load(directory)
        state.data['segment_plan_sha256'] = 'old algorithm'
        state.save()
        with self.assertRaisesRegex(ValueError, 'découpage des passages a changé'):
            self.process(Voice(), resume_dir=directory)

    def test_export_failure_resumes_without_any_gpu_synthesis(self):
        options = AudioOptions(audio_format='mp3', workers=1)
        with patch('epub_audio.core.pipeline.encode_chapter', side_effect=OSError('encoder stopped')):
            with self.assertRaisesRegex(OSError, 'encoder stopped'):
                self.process(Voice(), options=options)
        job, = discover_incomplete(self.output)
        voice = Voice()
        calls = []

        def encode(source, destination, **kwargs):
            calls.append(kwargs['track'])
            destination.write_bytes(b'encoded')
            return 1.0

        with patch('epub_audio.core.pipeline.encode_chapter', side_effect=encode):
            result = self.process(voice, options=options, resume_dir=job['directory'])
        self.assertEqual(voice.texts, [])
        self.assertCountEqual(calls, [1, 2])
        self.assertEqual(len(list(result.glob('*.mp3'))), 2)

    def test_batched_out_of_order_saved_results_survive_a_failed_group(self):
        class BatchVoice(Voice):
            batch_size = 2
            individual_progress = True

            def synthesize_batch(self, texts, paths, *, on_progress):
                for i in reversed(range(len(texts))):
                    self.synthesize(texts[i], paths[i])
                    on_progress(i, 'ready')
                return paths

        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.process(BatchVoice(fail=self.texts[2]))
        job, = discover_incomplete(self.output)
        directory = job['directory']
        # Each certified passage maps to its original reading position.
        saved = []
        for path in directory.glob('.segments-*/*.ready'):
            chapter = int(path.parent.name.split('-')[-1])
            saved.append(self.plan.chapters[chapter - 1].paragraphs[int(path.stem)])
        self.assertTrue(saved)
        voice = BatchVoice()
        result = self.process(voice, resume_dir=directory)
        self.assertFalse(set(saved) & set(voice.texts))
        self.assertEqual(len(list(result.glob('Chapitre *.wav'))), 2)

    def test_resumed_introduction_and_scene_pauses_match_uninterrupted_pcm(self):
        self.plan.metadata = BookMetadata('Titre ' + 'x' * 210, ['Auteur'])
        self.plan.chapters[1].paragraphs.insert(1, '***')
        options = AudioOptions(audio_format='wav', segmentation='sentence')
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.process(Voice(fail=self.texts[3]), options=options)
        job, = discover_incomplete(self.output)
        resumed = self.process(Voice(), options=options, resume_dir=job['directory'])
        original = self.process(Voice(), options=options)
        self.assertEqual([p.read_bytes() for p in sorted(resumed.glob('Chapitre *.wav'))],
                         [p.read_bytes() for p in sorted(original.glob('Chapitre *.wav'))])

    def test_completed_exports_are_kept_when_final_folder_rename_fails(self):
        import os
        rename = os.rename
        for fail_on in (1, 2):
            with self.subTest(rename=fail_on):
                count = 0

                def fail(source, destination):
                    nonlocal count
                    count += 1
                    if count == fail_on:
                        raise OSError('rename interrupted')
                    return rename(source, destination)

                with patch('epub_audio.core.pipeline.os.rename', side_effect=fail):
                    with self.assertRaisesRegex(OSError, 'rename interrupted'):
                        self.process(Voice())
                job, = discover_incomplete(self.output)
                voice = Voice()
                with patch('epub_audio.core.pipeline.encode_chapter', side_effect=AssertionError('encoded twice')):
                    result = self.process(voice, resume_dir=job['directory'])
                self.assertEqual(voice.texts, [])
                self.assertEqual(len(list(result.glob('Chapitre *.wav'))), 2)
                self.assertEqual(len(list((self.output / 'master' / result.name).glob('*.wav'))), 2)
                self.assertFalse(discover_incomplete(self.output))

    def test_list_only_and_startup_menu_do_not_load_the_model(self):
        self.interrupt()
        with patch('epub_audio.cli.main.QwenEngine') as engine, \
             patch('epub_audio.cli.main.resume_books', return_value=0) as resume, \
             patch('epub_audio.cli.args.OUTPUT_DIR', self.output), \
             patch('sys.stdout', new=io.StringIO()) as output:
            self.assertEqual(main(['--list-incomplete']), 0)
            self.assertIn('1/2 chapitres', output.getvalue())
            engine.assert_not_called()
            resume.assert_not_called()
            with patch('builtins.input', return_value='1'):
                self.assertEqual(main([]), 0)
            resume.assert_called_once()
            engine.assert_not_called()

    def test_resume_flag_restores_saved_options_without_any_questions(self):
        self.interrupt()
        voice = Voice()
        with patch('epub_audio.cli.resume.QwenEngine', return_value=voice), \
             patch('builtins.input', side_effect=AssertionError('unexpected prompt')), \
             patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(main(['--resume', '--output', str(self.output)]), 0)
        self.assertNotIn(self.texts[0], voice.texts)
        self.assertFalse(discover_incomplete(self.output))


class LegacyResumeTests(unittest.TestCase):
    def test_transient_windows_locks_are_retried_but_permanent_errors_propagate(self):
        from unittest.mock import Mock
        locked = PermissionError('temporarily locked')
        locked.winerror = 32
        operation = Mock(side_effect=[locked, 'ok'])
        with patch('epub_audio.io.fs.time.sleep') as sleep:
            self.assertEqual(retry_windows_io(operation, 'path'), 'ok')
            sleep.assert_called_once()
            permanent = Mock(side_effect=locked)
            with self.assertRaises(PermissionError):
                retry_windows_io(permanent)
            self.assertEqual(permanent.call_count, 6)
        with patch('epub_audio.io.fs.time.sleep') as sleep:
            with self.assertRaises(FileNotFoundError):
                retry_windows_io(Mock(side_effect=FileNotFoundError('missing')))
            sleep.assert_not_called()

    def test_certification_needs_journal_texts_and_pcm_coverage_not_just_valid_headers(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / 'book.epub'
                source.write_bytes(b'fixture')
                directory = root / '.ebook-old'
                directory.mkdir()
                texts = ['A' * 210, 'B' * 210, 'C' * 210]
                plan = BookPlan([Chapter('Un', texts[:2]), Chapter('Deux', texts[2:])])
                (directory / 'decoupage.json').write_text(json.dumps(chapter_report(plan, source)), encoding='utf-8')
                write_wav(directory / '.chapter-1.wav', frames=2400 if truncated else 4800)
                write_wav(directory / '.chapter-2.wav', frames=100)
                (directory / '.segments-1').mkdir()
                anchor = write_wav(directory / '.segments-2' / '000000.wav')
                # This last file has a valid header but no evidence that assembly finished.
                self.assertEqual(legacy_completed(directory), [1])
                stamp = datetime.fromtimestamp(anchor.stat().st_mtime, timezone.utc).isoformat()
                entries = [{'text_sha256': hashlib.sha256(t.encode()).hexdigest(), 'run_id': 'original',
                            'finished': True, 'voice': 'test', 'finished_at': stamp, 'duration_seconds': 0.1}
                           for t in texts]
                journal = root / 'journal.jsonl'
                journal.write_text('\n'.join(map(json.dumps, entries)), encoding='utf-8')
                voice = Voice()
                voice.voice = 'test'
                with patch('epub_audio.io.epub.read_book', return_value=plan):
                    if truncated:
                        with self.assertRaisesRegex(ValueError, 'Aucun chapitre ancien'):
                            import_legacy(directory, AudioOptions(audio_format='wav', paragraph_pause_ms=0),
                                          voice, root / 'master', journal=journal, message=lambda _: None)
                        self.assertFalse((directory / 'progression.json').exists())
                    else:
                        state = import_legacy(directory, AudioOptions(audio_format='wav', paragraph_pause_ms=0),
                                              voice, root / 'master', journal=journal, message=lambda _: None)
                        self.assertEqual(set(state.data['chapters']), {'1'})
                        self.assertEqual(state.data['legacy_validation']['run_id'], 'original')
                        self.assertEqual(state.data['chapters']['1'], file_hash(directory / '.chapter-1.wav'))

    def test_truncated_pcm_is_rejected_even_if_header_claims_complete_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_wav(Path(folder) / 'audio.wav')
            path.write_bytes(path.read_bytes()[:-20])
            self.assertFalse(valid_wav(path))


if __name__ == '__main__':
    unittest.main()
