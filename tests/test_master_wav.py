"""Master PCM preservation in a mirrored output tree, without a GPU."""
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import patch

from epub_audio.adapters.ffmpeg import audio_info, encode_chapter
from epub_audio.core.models import BookMetadata, BookPlan, Chapter
from epub_audio.core.pipeline import process_book
from epub_audio.core.export_progress import ExportProgress
from epub_audio.core.tts_qwen import QwenEngine


class MasterWavTests(unittest.TestCase):
    def setUp(self):
        self.originals = {}
        self.plan = BookPlan([Chapter('Premier', ['Un texte.']), Chapter('Second', ['La suite.'])],
                             metadata=BookMetadata('Le livre'))

    def render(self, engine, chapters, directory, **kwargs):
        for index in range(1, len(chapters) + 1):
            chunks = directory / f'.segments-{index}'
            chunks.mkdir()
            path = directory / f'.chapter-{index}.wav'
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\xe8\x03\x18\xfc' * (12000 * index))
            self.originals[index] = path.read_bytes()
            yield index, path, chunks

    def test_each_compressed_format_keeps_byte_exact_native_masters(self):
        for codec, extension, rate in (('mp3', 'mp3', 44100), ('aac', 'm4a', 48000),
                                       ('opus', 'opus', 24000)):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / 'livre.epub'
                source.write_bytes(b'EPUB fixture')
                with patch('epub_audio.core.pipeline.read_book', return_value=self.plan), \
                     patch('epub_audio.core.pipeline.render_chapters', side_effect=self.render) as render, \
                     patch('epub_audio.core.pipeline.encode_chapter', wraps=encode_chapter) as encode:
                    result = process_book(source, object(), output_dir=root / 'output',
                                          audio_format=codec, sample_rate=rate, workers=2,
                                          message=lambda text: None)
                render.assert_called_once()
                self.assertEqual(encode.call_count, 2)
                self.assertEqual(len(list(result.glob('*.' + extension))), 2)
                master_dir = root / 'output' / 'master' / result.name
                masters = sorted(master_dir.glob('*.wav'))
                self.assertEqual(len(masters), 2)
                self.assertFalse(list(result.glob('*.wav')))
                self.assertFalse(list(result.glob('.chapter-*')))
                self.assertFalse(list(result.glob('.segments-*')))
                for index, master in enumerate(masters, 1):
                    self.assertEqual(master.read_bytes(), self.originals[index])
                    with wave.open(str(master), 'rb') as audio:
                        self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()),
                                         (1, 2, 24000))
                    encoded = next(result.glob(f'Chapitre {index} - *.{extension}'))
                    info = audio_info(encoded)
                    self.assertEqual(info['codec_name'], codec)
                    self.assertEqual(int(info['sample_rate']), 48000 if codec == 'opus' else rate)
                self.assertEqual(next(result.glob('*.epub')).read_bytes(), source.read_bytes())
                self.assertEqual(next(master_dir.glob('*.epub')).read_bytes(), source.read_bytes())
                self.assertFalse((result / 'Master WAV').exists())

    def test_gpu_synthesis_stays_serial_while_final_exports_run_concurrently(self):
        from epub_audio.cli.args import arguments
        self.assertEqual(arguments(['--yes']).workers, 20)
        self.assertEqual(arguments(['--workers', '3', '--yes']).workers, 3)
        for requested, count in ((None, 21), (3, 6), (20, 2)):
            with self.subTest(workers=requested, chapters=count), tempfile.TemporaryDirectory() as folder:
                self.plan = BookPlan([Chapter(f'Chapitre {n}', ['Texte.']) for n in range(count)])
                expected = min(20 if requested is None else requested, count)
                barrier = threading.Barrier(expected, timeout=5)
                lock = threading.Lock()
                active = peak = 0

                def encode(source, destination, **kwargs):
                    nonlocal active, peak
                    with lock:
                        active += 1
                        peak = max(peak, active)
                    try:
                        if kwargs['track'] <= expected:
                            barrier.wait()
                        destination.write_bytes(b'encoded fixture')
                        return 1.0
                    finally:
                        with lock:
                            active -= 1

                root = Path(folder)
                source = root / 'livre.epub'
                source.write_bytes(b'EPUB fixture')
                options = {} if requested is None else {'workers': requested}
                messages = []
                with patch('epub_audio.core.pipeline.read_book', return_value=self.plan), \
                     patch('epub_audio.core.pipeline.render_chapters', side_effect=self.render) as render, \
                     patch('epub_audio.core.pipeline.encode_chapter', side_effect=encode):
                    result = process_book(source, SimpleNamespace(serial_gpu=True),
                                          output_dir=root / 'output', audio_format='mp3',
                                          message=messages.append, **options)
                self.assertEqual(render.call_args.kwargs['workers'], 1)
                self.assertEqual(peak, expected)
                self.assertTrue(any(f'({expected} workers)' in line for line in messages))
                self.assertEqual(len(list(result.glob('*.mp3'))), count)
                self.assertEqual(len(list((root / 'output' / 'master' / result.name).glob('*.wav'))), count)

    def test_wav_choice_keeps_separate_byte_exact_master_without_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'livre.epub'
            source.write_bytes(b'EPUB fixture')
            with patch('epub_audio.core.pipeline.read_book', return_value=self.plan), \
                 patch('epub_audio.core.pipeline.render_chapters', side_effect=self.render) as render, \
                 patch('epub_audio.core.pipeline.encode_chapter', side_effect=AssertionError('Unexpected encoding')):
                result = process_book(source, object(), output_dir=root / 'output',
                                      audio_format='wav', workers=1, message=lambda text: None)
            render.assert_called_once()
            self.assertFalse((result / 'Master WAV').exists())
            tracks = sorted(result.rglob('*.wav'))
            self.assertEqual(len(tracks), 2)
            for index, track in enumerate(tracks, 1):
                self.assertEqual(track.read_bytes(), self.originals[index])
            master_dir = root / 'output' / 'master' / result.name
            masters = sorted(master_dir.glob('*.wav'))
            self.assertEqual(len(masters), 2)
            for index, master in enumerate(masters, 1):
                self.assertEqual(master.read_bytes(), self.originals[index])

    def test_nested_master_tree_preserves_sidecars_and_reserves_matching_folder_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'livre.epub'
            source.write_bytes(b'EPUB fixture')
            output = root / 'output' / 'Auteur' / 'Collection'
            masters = root / 'output' / 'master' / 'Auteur' / 'Collection'
            cover = SimpleNamespace(media_type='image/jpeg', get_content=lambda: b'cover fixture')
            self.plan.cover = cover
            with patch('epub_audio.core.pipeline.read_book', return_value=self.plan), \
                 patch('epub_audio.core.pipeline.render_chapters', side_effect=self.render):
                first = process_book(source, object(), output_dir=output, master_output_dir=masters,
                                     audio_format='wav', workers=1, message=lambda _: None)
                # Simulate a moved audiobook whose original masters still exist.
                first.rename(output / 'archived')
                second = process_book(source, object(), output_dir=output, master_output_dir=masters,
                                      audio_format='wav', workers=1, message=lambda _: None)
            self.assertNotEqual(first.name, second.name)
            for name in (first.name, second.name):
                self.assertEqual((masters / name / 'cover.jpg').read_bytes(), b'cover fixture')
                self.assertEqual(next((masters / name).glob('*.epub')).read_bytes(), source.read_bytes())
                for index, track in enumerate(sorted((masters / name).glob('*.wav')), 1):
                    self.assertEqual(track.read_bytes(), self.originals[index])

    def test_exports_report_actual_concurrency_at_the_end_of_the_book(self):
        count = 10
        self.plan = BookPlan([Chapter(f'Chapitre {n}', ['Texte.']) for n in range(count)])
        barrier = threading.Barrier(count, timeout=5)
        events = []

        def encode(source, destination, **kwargs):
            barrier.wait()
            destination.write_bytes(b'encoded fixture')
            return 1.0

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'livre.epub'
            source.write_bytes(b'EPUB fixture')
            with patch('epub_audio.core.pipeline.read_book', return_value=self.plan), \
                 patch('epub_audio.core.pipeline.render_chapters', side_effect=self.render), \
                 patch('epub_audio.core.pipeline.encode_chapter', side_effect=encode):
                result = process_book(source, QwenEngine(),
                                      output_dir=root / 'output', audio_format='opus', workers=20,
                                      message=lambda _: None, progress=events.append)
            exports = [event for event in events if isinstance(event, ExportProgress)]
            self.assertEqual(max(event.active for event in exports), count)
            self.assertEqual(exports[0].pending, count)
            self.assertEqual(exports[-1].completed, count)
            self.assertEqual((exports[-1].active, exports[-1].pending, exports[-1].failed), (0, 0, 0))
            self.assertTrue(all(event.workers == count for event in exports))
            self.assertEqual(len(list(result.glob('*.opus'))), count)

    def test_failed_exports_are_not_reported_as_completed(self):
        from epub_audio.core.export_progress import ExportTracker
        events = []
        tracker = ExportTracker(2, 2, 'opus', events.append)
        with tracker.task():
            pass
        with self.assertRaises(RuntimeError), tracker.task():
            raise RuntimeError('Encoding failed')
        self.assertEqual(events[-1].completed, 1)
        self.assertEqual(events[-1].failed, 1)
        self.assertEqual(events[-1].active, 0)
        self.assertIn('1 échec(s)', events[-1])

    def test_encoding_failure_preserves_prepared_pcm_and_reports_work_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'livre.epub'
            source.write_bytes(b'EPUB fixture')
            output = root / 'output'
            messages = []
            with patch('epub_audio.core.pipeline.read_book', return_value=self.plan), \
                 patch('epub_audio.core.pipeline.render_chapters', side_effect=self.render), \
                 patch('epub_audio.core.pipeline.encode_chapter', side_effect=RuntimeError('Encoding failed')):
                with self.assertRaisesRegex(RuntimeError, 'Encoding failed'):
                    process_book(source, object(), output_dir=output, audio_format='mp3',
                                 workers=1, message=messages.append)
            work, = output.iterdir()
            self.assertTrue(work.name.startswith('.ebook-'))
            self.assertIn('Encoding failed', (work / 'erreur.txt').read_text(encoding='utf-8'))
            self.assertTrue(any(str(work) in message for message in messages))
            self.assertEqual((work / 'Master WAV' / 'Chapitre 1 - 00.01.wav').read_bytes(),
                             self.originals[1])
            self.assertCountEqual([path.read_bytes() for path in work.rglob('*.wav')],
                                  self.originals.values())


if __name__ == '__main__':
    unittest.main()
