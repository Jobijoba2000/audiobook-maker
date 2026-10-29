from pathlib import Path
import io
import tempfile
import unittest
from unittest.mock import Mock, patch
from epub_audio.cli.main import main
from epub_audio.cli.ui import ConsoleOutput


class ConsoleTests(unittest.TestCase):
    def test_interactive_run_does_not_ask_speed_or_reuse_old_tempo(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'livre.epub'
            source.write_bytes(b'test EPUB routing')
            with patch('epub_audio.cli.main.QwenEngine') as engine, \
                 patch('epub_audio.cli.main.process_book') as process, \
                 patch('epub_audio.cli.main.load_preferences', return_value={'rate':1.7}), \
                 patch('epub_audio.cli.main.save_preferences'), \
                 patch('builtins.input', side_effect=['','','']) as prompt, \
                 patch('sys.stdout', new=io.StringIO()) as output:
                self.assertEqual(main([str(source),'--segmentation','paragraph','--codec','wav','--voice','expr_chap_s01_0004-7e5a77eb','--no-llm-chapters','--no-normalize']),0)
            self.assertIn('Temps total écoulé',output.getvalue())
            self.assertEqual(prompt.call_count,3)
            self.assertFalse(any('Vitesse' in call.args[0] for call in prompt.call_args_list))
            self.assertEqual(engine.call_args.kwargs['speed'],1.0)
            self.assertEqual(process.call_args.kwargs['options'].speed,1.0)
            self.assertEqual(process.call_args.kwargs['options'].workers,20)

    def test_progress_replaces_the_same_terminal_line(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        stream=Terminal()
        console=ConsoleOutput(stream)
        console.progress('Chapitre 1/1 : voix 1/39 segments')
        console.progress('Chapitre 1/1 : voix 2/39 segments')
        self.assertNotIn('\n',stream.getvalue())
        self.assertEqual(stream.getvalue().count('\r'),2)
        console.message('Terminé')
        self.assertTrue(stream.getvalue().endswith('\nTerminé\n'))

if __name__=='__main__':
    unittest.main()
