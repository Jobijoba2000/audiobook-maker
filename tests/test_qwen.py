from pathlib import Path
from types import SimpleNamespace
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import wave

from epub_audio.core.tts_qwen import QwenEngine, configuration, list_available_voices
from epub_audio.core.tts_generate import render_chapters
from epub_audio.io.fs import ROOT


class SerialVoice:
    serial_gpu = True
    max_segment_characters = 150

    def __init__(self):
        self.texts = []
        self.threads = []

    def synthesize(self, text, destination):
        self.texts.append(text)
        self.threads.append(threading.get_ident())
        time.sleep(0.01)
        with wave.open(str(destination), 'wb') as wav:
            wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\x00\x10' * 2400)
        return Path(destination)


class QwenTests(unittest.TestCase):
    def test_default_reference_matches_validated_siwis_selection(self):
        c = configuration()
        self.assertEqual(c['engine'], 'faster-qwen')
        self.assertEqual(c['voice'], 'expr_chap_s01_0004-7e5a77eb')
        self.assertEqual((c['seed'], c['temperature'], c['max_new_tokens']), (42, 0.9, 2048))
        self.assertIsNone(c['max_segment_characters'])
        self.assertIn(c['voice'], [v['id'] for v in list_available_voices()])

    def test_single_gpu_queue_reuses_engine_across_chapters_and_keeps_words(self):
        voice = SerialVoice()
        paragraphs = ['Voici une phrase entière. ' * 20, 'Dernier passage du livre.']
        with tempfile.TemporaryDirectory() as folder:
            result = list(render_chapters(voice, [[paragraphs[0]], [paragraphs[1]]], Path(folder), workers=20, progress=lambda s: None))
            self.assertEqual(len(result), 2)
            for _, path, _ in result:
                with wave.open(str(path), 'rb') as audio:
                    self.assertGreater(audio.getnframes(), 0)
        self.assertEqual(' '.join(voice.texts), ' '.join(p.strip() for p in paragraphs))
        self.assertTrue(all(len(t) <= 150 for t in voice.texts))
        self.assertEqual(len(set(voice.threads)), 1)

    def test_faster_loads_once_and_reuses_the_same_reference_prompt(self):
        import numpy as np
        fake = Mock()
        fake.model.tts_model_type = 'base'
        fake.model.tts_model_size = '0b6'
        fake.model.talker.generation_config = SimpleNamespace(pad_token_id=None)
        fake.model.config.talker_config.codec_eos_token_id = 2150
        prompt = [object()]
        fake.create_voice_clone_prompt.return_value = prompt
        fast = Mock(model=fake)
        references = []
        def generate(**kwargs):
            references.append(kwargs['voice_clone_prompt'])
            fast.last_generation_timing = dict(ended_with_eos=True, steps=30)
            return [np.full(2400, 0.1, dtype=np.float32)], 24000
        fast.generate_voice_clone.side_effect = generate
        with tempfile.TemporaryDirectory() as folder, \
             patch('epub_audio.core.tts_qwen.ROOT', Path(folder)), \
             patch.dict(os.environ), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='Fake GPU'), \
             patch('faster_qwen3_tts.FasterQwen3TTS.from_pretrained', return_value=fast) as loader:
            config = configuration()
            fixture = Path(folder)
            ref = fixture / 'voices/references_fr' / (config['voice'] + '.wav')
            ref.parent.mkdir(parents=True)
            ref.write_bytes(b'fixture-audio')
            ref.with_suffix('.txt').write_text('Une référence de test.', encoding='utf-8')
            for name in ('config.json', 'model.safetensors', 'speech_tokenizer/model.safetensors'):
                model_file = fixture / config['model_dir'] / name
                model_file.parent.mkdir(parents=True, exist_ok=True)
                model_file.write_bytes(b'fixture-model')
            engine = QwenEngine(message=lambda s: None)
            engine.prepare()
            engine.prepare()
            engine.synthesize('Premier passage.', fixture / 'a.wav')
            engine.synthesize('Second passage.', fixture / 'b.wav')
            self.assertEqual(engine.calls, 2)
            loader.assert_called_once()
            self.assertEqual(loader.call_args.kwargs['attn_implementation'], 'eager')
            self.assertEqual(fake.model.talker.generation_config.pad_token_id, 2150)
            fast.warmup.assert_called_once()
            fake.create_voice_clone_prompt.assert_called_once()
            self.assertTrue(all(p is prompt for p in references))
            self.assertEqual(fast.generate_voice_clone.call_args.kwargs['temperature'], 0.9)
            self.assertTrue(fast.generate_voice_clone.call_args.kwargs['non_streaming_mode'])
            report = json.loads((fixture / '.cache/qwen-last-run.json').read_text(encoding='utf-8'))
            self.assertEqual((report['engine'], report['model_size']), ('faster-qwen', '0.6B'))
            self.assertEqual(Path(loader.call_args.args[0]).name, 'Qwen3-TTS-0.6B-Base')
            entries = [json.loads(line) for line in (fixture / '.cache/qwen-segments.jsonl').read_text().splitlines()]
            self.assertEqual([entry['engine'] for entry in entries], ['faster-qwen'] * 2)
            self.assertEqual([entry['model_size'] for entry in entries], ['0.6B'] * 2)

    def test_truncated_generation_does_not_write_audio(self):
        import numpy as np
        engine = QwenEngine(message=lambda s: None)
        fast = Mock()
        def generate(**kwargs):
            fast.last_generation_timing = dict(ended_with_eos=False, steps=2048)
            return [np.zeros(2400, dtype=np.float32)], 24000
        fast.generate_voice_clone.side_effect = generate
        engine.model, engine.prompt, engine.fast_model = Mock(), [object()], fast
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audio.wav'
            with self.assertRaisesRegex(RuntimeError, 'limite'):
                engine.synthesize('Un passage.', path)
            self.assertFalse(path.exists())

    def test_generation_error_is_propagated_without_writing_audio(self):
        engine = QwenEngine(message=lambda s: None)
        engine.model, engine.prompt, engine.fast_model = Mock(), [object()], Mock()
        engine.fast_model.generate_voice_clone.side_effect = RuntimeError('test failure')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audio.wav'
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                engine.synthesize('Un passage.', path)
            self.assertFalse(path.exists())

    def test_gpu_worker_initializes_cpu_settings_in_its_own_thread(self):
        from epub_audio.workers import _initialize_worker
        engine=Mock(serial_gpu=True)
        _initialize_worker(engine)
        engine.initialize_worker.assert_called_once_with()

    def test_module_and_bat_are_independent_of_working_directory(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as folder:
            commands = [[sys.executable, '-B', '-m', 'epub_audio', '--list-voices']]
            if os.name == 'nt':
                commands.append(['cmd.exe', '/d', '/c', str(ROOT / 'start_ebook.bat'), '--list-voices'])
            for command in commands:
                result = subprocess.run(command, cwd=folder, env=env, capture_output=True, text=True, encoding='utf-8', timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(configuration()['voice'], result.stdout)


if __name__ == '__main__':
    unittest.main()
