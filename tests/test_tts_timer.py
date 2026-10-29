from pathlib import Path
import io
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

from epub_audio.core.tts_progress import TtsProgress
from epub_audio.cli.ui import ConsoleOutput
from epub_audio.core.pipeline import process_book
from epub_audio.core.models import BookPlan, BookMetadata, Chapter


class TtsTimerTests(unittest.TestCase):
    def test_clock_updates_while_a_segment_is_still_generating_then_stops(self):
        clock = [0.0]
        tick_seen = threading.Event()
        lines, messages = [], []

        def show(text):
            lines.append(text)
            if '00:00:07' in text:
                tick_seen.set()

        with TtsProgress(show, messages.append, interval=0.01, clock=lambda: clock[0]) as timer:
            self.assertIn('00:00:00', lines[0])
            timer.progress('Chapitre 1/1 : voix 0/15 segments')
            clock[0] = 7.0
            self.assertTrue(tick_seen.wait(1), 'No live refresh during a long segment')
            clock[0] = 12.0
        self.assertEqual(messages, ['Durée génération TTS : 00:00:12.'])
        self.assertEqual(timer.elapsed, 12.0)
        self.assertFalse(timer._thread.is_alive())
        clock[0] = 999
        self.assertIn('00:00:12', lines[-1])

    def test_interruption_freezes_and_displays_duration_without_swallowing_error(self):
        clock = [0.0]
        messages = []
        timer = TtsProgress(lambda text: None, messages.append, clock=lambda: clock[0])
        with self.assertRaises(KeyboardInterrupt):
            with timer:
                clock[0] = 65.0
                raise KeyboardInterrupt()
        self.assertEqual(messages, ['Durée génération TTS : 00:01:05 (interrompue).'])
        self.assertFalse(timer._thread.is_alive())

    def test_pipeline_times_synthesis_and_stops_before_export(self):
        import wave
        clock = [100.0]
        messages, progress = [], []
        timer_factory = TtsProgress
        instances = []

        def make_timer(show, message):
            timer = timer_factory(show, message, clock=lambda: clock[0])
            instances.append(timer)
            return timer

        def synthesize(engine, chapters, directory, **kwargs):
            self.assertIn('00:00:00', progress[-1])
            clock[0] += 12.0
            kwargs['progress']('Chapitre 1/1 : voix 1/1 segments')
            chunks = directory / '.segments-1'
            chunks.mkdir()
            path = directory / '.chapter-1.wav'
            with wave.open(str(path), 'wb') as audio:
                audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                audio.writeframes(b'\0\0' * 2400)
            yield 1, path, chunks

        def encode(source, destination, **kwargs):
            self.assertIn('Durée génération TTS : 00:00:12.', messages)
            self.assertFalse(instances[0]._thread.is_alive())
            clock[0] += 90.0
            destination.write_bytes(b'encoded fixture')
            return 0.1

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'livre.epub'
            source.write_bytes(b'book fixture')
            plan = BookPlan([Chapter('Premier', ['Un texte.'])],
                            metadata=BookMetadata('Titre'))
            with patch('epub_audio.core.pipeline.read_book', return_value=plan), \
                 patch('epub_audio.core.pipeline.render_chapters', side_effect=synthesize), \
                 patch('epub_audio.core.pipeline.TtsProgress', side_effect=make_timer), \
                 patch('epub_audio.core.pipeline.encode_chapter', side_effect=encode):
                process_book(source, object(), output_dir=root/'output', message=messages.append,
                             progress=progress.append, audio_format='mp3', workers=1)
        self.assertEqual(instances[0].elapsed, 12.0)
        self.assertEqual(sum('Durée génération TTS :' in message for message in messages), 1)

    def test_live_timer_keeps_terminal_progress_on_one_line(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        stream = Terminal()
        console = ConsoleOutput(stream)
        clock = [0.0]
        with TtsProgress(console.progress, console.message, clock=lambda: clock[0]) as timer:
            timer.progress('Chapitre 1/1 : voix 1/2 segments')
            clock[0] = 10.0
            timer.progress('Chapitre 1/1 : voix 2/2 segments')
            self.assertNotIn('\n', stream.getvalue())
        self.assertTrue(stream.getvalue().endswith('Durée génération TTS : 00:00:10.\n'))
