"""Verify visible terminal contents, not only the emitted escape sequences."""
import io
import os
import re
import unittest
from unittest.mock import patch

from epub_audio.cli.ui import ConsoleOutput
from epub_audio.core.tts_progress import ChapterProgress, TtsProgress


class Terminal(io.StringIO):
    def __init__(self):
        super().__init__()
        self.lines = ['']
        self.row = self.column = 0

    def isatty(self):
        return True

    def write(self, text):
        result = super().write(text)
        for token in re.findall(r'\x1b\[\d+[FK]|[^\x1b]', text):
            if token.startswith('\x1b['):
                number = int(token[2:-1])
                if token.endswith('F'):
                    self.row -= number
                    self.column = 0
                    if self.row < 0:
                        raise AssertionError('Cursor moved outside the live area')
                elif token.endswith('K'):
                    assert number == 2
                    self.lines[self.row] = ''
            elif token == '\r':
                self.column = 0
            elif token == '\n':
                self.row += 1
                self.column = 0
                if self.row == len(self.lines):
                    self.lines.append('')
            else:
                line = self.lines[self.row].ljust(self.column)
                self.lines[self.row] = line[:self.column] + token + line[self.column + 1:]
                self.column += 1
        return result

    def visible(self):
        return [line.rstrip() for line in self.lines if line.strip()]


class ChapterConsoleTests(unittest.TestCase):
    def test_parallel_chapters_update_in_order_and_warnings_preserve_each_line(self):
        stream = Terminal()
        console = ConsoleOutput(stream, terminal_controls=True)
        clock = [0.0]
        with TtsProgress(console.progress, console.message, clock=lambda: clock[0]) as timer:
            timer.progress(ChapterProgress(1, 2, 0, 2))
            timer.progress(ChapterProgress(2, 2, 0, 1))
            timer.progress(ChapterProgress(2, 2, 1, 1, validated=True))
            self.assertEqual(stream.visible()[:2], [
                'Chapitre 1/2 - 0/2 segments - 0 %',
                'Chapitre 2/2 - 1/1 segments - 100 %'])
            self.assertEqual(console._frozen, 0)
            console.message('Réessai du passage interrompu.')
            timer.progress(ChapterProgress(1, 2, 1, 2))
            clock[0] = 9.0
            timer.progress(ChapterProgress(1, 2, 2, 2))
            self.assertIn('(validation audio)', '\n'.join(stream.visible()))
            timer.progress(ChapterProgress(1, 2, 2, 2, validated=True))
            self.assertEqual(console._frozen, 2)
            clock[0] = 11.0
        visible = stream.visible()
        self.assertEqual(sum(line.startswith('Chapitre 1/') for line in visible), 1)
        self.assertEqual(sum(line.startswith('Chapitre 2/') for line in visible), 1)
        self.assertEqual(sum('Réessai' in line for line in visible), 1)
        self.assertEqual(visible[-1], 'Durée génération TTS : 00:00:11.')
        self.assertFalse(any(line.startswith('Durée TTS :') for line in visible))
        self.assertFalse(timer._thread.is_alive())

    def test_redirected_log_has_no_cursor_codes_and_no_timer_spam(self):
        stream = io.StringIO()
        console = ConsoleOutput(stream)
        console.progress(ChapterProgress(1, 2, 0, 2, elapsed=1))
        console.progress(ChapterProgress(2, 2, 1, 1, validated=True, elapsed=2))
        console.message('Réessai nécessaire.')
        before = stream.getvalue()
        for elapsed in range(3, 20):
            console.progress(ChapterProgress(2, 2, 1, 1, validated=True, elapsed=elapsed))
        self.assertEqual(stream.getvalue(), before)
        console.progress(ChapterProgress(1, 2, 2, 2, validated=True, elapsed=20))
        console.message('Durée génération TTS : 00:00:20.')
        log = stream.getvalue()
        self.assertNotIn('\x1b', log)
        self.assertEqual(log.count('Chapitre 2/2'), 1)
        self.assertEqual(log.count('100 %'), 2)
        self.assertTrue(log.endswith('Durée génération TTS : 00:00:20.\n'))
        console.progress(ChapterProgress(1, 1, 0, 1, elapsed=0))
        console.progress(ChapterProgress(1, 1, 1, 1, validated=True, elapsed=5))
        console.message('Durée génération TTS : 00:00:05.')
        self.assertIn('Chapitre 1/1 - 1/1 segments - 100 %', stream.getvalue())

    def test_live_area_fits_small_terminals_without_wrapping_or_losing_state(self):
        stream = Terminal()
        console = ConsoleOutput(stream, terminal_controls=True)
        with patch('epub_audio.cli.ui.shutil.get_terminal_size', return_value=os.terminal_size((38, 5))):
            for chapter in range(1, 7):
                console.progress(ChapterProgress(chapter, 6, 0, 10, elapsed=65))
            self.assertLessEqual(console._live_rows, 4)
            self.assertTrue(all(len(line) <= 37 for line in stream.visible()))
            self.assertEqual(len(console._chapters), 6)
            for chapter in (2, 3, 4, 5, 6, 1):
                console.progress(ChapterProgress(chapter, 6, 10, 10, validated=True, elapsed=80))
            self.assertEqual(console._frozen, 6)
        console.message('Durée génération TTS : 00:01:20.')
        self.assertEqual(sum(line.startswith('Chapitre') for line in stream.visible()), 6)

    def test_interruption_keeps_partial_chapters_and_final_duration(self):
        stream = Terminal()
        console = ConsoleOutput(stream, terminal_controls=True)
        with self.assertRaises(KeyboardInterrupt):
            with TtsProgress(console.progress, console.message) as timer:
                timer.progress(ChapterProgress(1, 2, 3, 10))
                timer.progress(ChapterProgress(2, 2, 2, 8))
                raise KeyboardInterrupt()
        self.assertEqual(sum(line.startswith('Chapitre') for line in stream.visible()), 2)
        self.assertIn('(interrompue)', stream.visible()[-1])
        self.assertEqual(console._live_rows, 0)


if __name__ == '__main__':
    unittest.main()
