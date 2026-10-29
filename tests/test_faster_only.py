"""Faster 0.6B is fixed; old engine preferences preserve the user's voice."""
from pathlib import Path
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from epub_audio.cli.args import arguments
from epub_audio.cli.main import main
from epub_audio.core.tts_qwen import QwenEngine, configuration
from epub_audio.io.preferences import load_preferences, PROMPT_NAMES

VOICE = 'expr_chap_s01_0004-7e5a77eb'


class FasterOnlyTests(unittest.TestCase):
    def test_old_preferences_keep_faster_voice_batching_and_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'book.epub'
            source.write_bytes(b'Routing fixture')
            prefs = root / 'prefs.json'
            prefs.write_text(json.dumps({
                'engine': 'nvigi-qwen-q8', 'model_size': '1.7B', 'voice': 'old-voice',
                'engine_voices': {'faster-qwen': VOICE}, 'audio_format': 'opus',
                'bitrate': 48, 'sample_rate': 24000, 'faster_small_batching': True,
                'faster_small_batch_size': 7, 'prompts': {'engine': True, 'model_size': True},
            }), encoding='utf-8')
            with patch('epub_audio.io.preferences.PREFERENCES_FILE', prefs), \
                 patch('epub_audio.cli.main.require_tools'), patch('epub_audio.cli.main.QwenEngine') as engine, \
                 patch('epub_audio.cli.main.process_book'), patch('sys.stdout', new=io.StringIO()) as output:
                self.assertEqual(main([str(source), '--yes']), 0)
                self.assertEqual(engine.call_args.args[0], VOICE)
                self.assertEqual(engine.call_args.kwargs['batch_size'], 7)
                self.assertNotIn('engine', engine.call_args.kwargs)
                self.assertNotIn('model_size', engine.call_args.kwargs)
                self.assertIn('Faster-Qwen3-TTS 0.6B', output.getvalue())
                saved = load_preferences()
                self.assertEqual((saved['voice'], saved['audio_format'], saved['bitrate']), (VOICE, 'opus', 48))
            raw = json.loads(prefs.read_text(encoding='utf-8'))
            for key in ('engine', 'model_size', 'engine_voices'):
                self.assertNotIn(key, raw)
            self.assertNotIn('engine', PROMPT_NAMES)
            self.assertNotIn('model_size', PROMPT_NAMES)

    def test_removed_selector_flags_are_rejected(self):
        for flags in (['--engine', 'faster-qwen'], ['--engine', 'microsoft'],
                      ['--engine', 'nvigi-qwen'], ['--model-size', '0.6B'], ['--model-size', '1.7B']):
            with self.subTest(flags=flags), patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                arguments(flags)

    def test_engine_exposes_no_model_selection_parameters(self):
        engine = QwenEngine()
        self.assertEqual((engine.engine_id, engine.model_size), ('faster-qwen', '0.6B'))
        self.assertEqual(configuration()['model_dir'], 'models/Qwen3-TTS-0.6B-Base')
        with self.assertRaises(TypeError):
            QwenEngine(model_size='1.7B')
        with self.assertRaises(TypeError):
            QwenEngine(engine='qwen')
