"""Optional final chapter loudness: real PCM/FFmpeg exports, no TTS models."""
from contextlib import ExitStack
import io
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from unittest.mock import patch

from epub_audio.adapters.ffmpeg import audio_info, chapter_loudness_filter, encode_chapter, require_tools
from epub_audio.cli.args import arguments
from epub_audio.cli.main import main
from epub_audio.core.models import BookMetadata, BookPlan, Chapter
from epub_audio.core.options import AudioOptions
from epub_audio.core.pipeline import process_book
from epub_audio.io.preferences import load_preferences, save_preferences


def write_tone(path, amplitude=1000):
    samples = [round(amplitude * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(96000)]
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(struct.pack('<' + 'h' * len(samples), *samples))


def measure(path):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-nostdin', '-i', str(path),
                             '-af', 'loudnorm=print_format=json', '-f', 'null', '-'],
                            capture_output=True, text=True, check=True)
    text = result.stderr
    return json.loads(text[text.rindex('{'):text.rindex('}') + 1])


class NormalizationChoiceTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.root / 'book.epub'
        self.source.write_bytes(b'Fixture; synthesis mocked')
        self.prefs = self.root / 'preferences.json'
        stack.enter_context(patch('epub_audio.io.preferences.PREFERENCES_FILE', self.prefs))
        self.tools = stack.enter_context(patch('epub_audio.cli.main.require_tools'))
        stack.enter_context(patch('epub_audio.cli.main.QwenEngine'))
        self.process = stack.enter_context(patch('epub_audio.cli.main.process_book'))
        stack.enter_context(patch('sys.stdout', new=io.StringIO()))
        self.flags = [str(self.source), '--voice', 'expr_chap_s01_0004-7e5a77eb', '--no-batching',
                      '--segmentation', 'paragraph', '--no-llm-chapters', '--format', 'wav']

    def test_menu_remembers_both_choices_and_hidden_question_reuses_them(self):
        for answer, expected in (('2', True), ('1', False)):
            with self.subTest(answer=answer):
                with patch('builtins.input', side_effect=['', 'invalid', answer, '']) as prompt:
                    self.assertEqual(main(self.flags), 0)
                self.assertIn('Normaliser le volume final', prompt.call_args_list[-2].args[0])
                self.assertIs(self.process.call_args.kwargs['options'].normalize, expected)
                self.assertIs(load_preferences()['normalize'], expected)
                data = json.loads(self.prefs.read_text(encoding='utf-8'))
                data['prompts'] = {'normalize': False}
                self.prefs.write_text(json.dumps(data), encoding='utf-8')
                with patch('builtins.input', side_effect=['', '']) as prompt:
                    self.assertEqual(main(self.flags), 0)
                self.assertEqual(prompt.call_count, 2)
                self.assertIs(self.process.call_args.kwargs['options'].normalize, expected)
                data['prompts'] = {}
                self.prefs.write_text(json.dumps(data), encoding='utf-8')

    def test_flags_override_saved_choice_without_questions_in_yes_mode(self):
        for flag, expected in (('--normalize', True), ('--no-normalize', False)):
            with self.subTest(flag=flag):
                self.prefs.write_text(json.dumps({'normalize': not expected}), encoding='utf-8')
                with patch('builtins.input', side_effect=AssertionError('Unexpected question')):
                    self.assertEqual(main(self.flags + ['--yes', flag]), 0)
                self.assertIs(self.process.call_args.kwargs['options'].normalize, expected)
                self.assertIs(load_preferences()['normalize'], expected)
                self.tools.assert_called_with('wav', normalize=expected)
        self.assertIsNone(arguments([]).normalize)
        self.assertFalse(AudioOptions().normalize)

    def test_explicit_flag_skips_enabled_menu_and_yes_reuses_saved_choice(self):
        self.prefs.write_text(json.dumps({'normalize': False, 'prompts': {'normalize': True}}), encoding='utf-8')
        with patch('builtins.input', side_effect=['', '']) as prompt:
            self.assertEqual(main(self.flags + ['--normalize']), 0)
        self.assertEqual(prompt.call_count, 2)
        self.assertTrue(self.process.call_args.kwargs['options'].normalize)
        with patch('builtins.input', side_effect=AssertionError('Unexpected question')):
            self.assertEqual(main(self.flags + ['--yes']), 0)
        self.assertTrue(self.process.call_args.kwargs['options'].normalize)

    def test_preferences_preserve_boolean_and_reject_other_types(self):
        save_preferences('voice', 1, normalize=True)
        save_preferences('voice', 1)
        self.assertTrue(load_preferences()['normalize'])
        for bad in ('false', 0, 1, [], {}):
            with self.subTest(value=bad):
                self.prefs.write_text(json.dumps({'normalize': bad}), encoding='utf-8')
                self.assertNotIn('normalize', load_preferences())
                with self.assertRaises(ValueError):
                    AudioOptions(normalize=bad)
                with self.assertRaises(ValueError):
                    save_preferences('voice', 1, normalize=bad)

    def test_normalized_wav_requires_ffmpeg_but_original_wav_does_not(self):
        with patch('epub_audio.adapters.ffmpeg.tool_executable', return_value=None):
            require_tools('wav')
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg'):
                require_tools('wav', normalize=True)


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class NormalizationExportTests(unittest.TestCase):
    def test_different_input_levels_reach_same_target_without_changing_rate_or_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for amplitude in (500, 6000):
                source = root / f'source-{amplitude}.wav'
                destination = root / f'normalized-{amplitude}.wav'
                write_tone(source, amplitude)
                original = source.read_bytes()
                encode_chapter(source, destination, audio_format='wav', normalize=True, sample_rate=24000)
                self.assertEqual(source.read_bytes(), original)
                with wave.open(str(destination), 'rb') as audio:
                    self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()),
                                     (24000, 1, 2, 96000))
                stats = measure(destination)
                self.assertAlmostEqual(float(stats['input_i']), -18, delta=.2)
                self.assertLessEqual(float(stats['input_tp']), -2.9)

    def test_every_compressed_codec_supports_normalization_at_selected_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.wav'
            write_tone(source)
            for codec, suffix, rate in (('mp3', 'mp3', 44100), ('aac', 'm4a', 24000), ('opus', 'opus', 24000)):
                with self.subTest(codec=codec):
                    target = root / ('normalized.' + suffix)
                    encode_chapter(source, target, audio_format=codec, bitrate=64, sample_rate=rate, normalize=True)
                    info = audio_info(target)
                    self.assertEqual(info['codec_name'], codec)
                    self.assertEqual(int(info['sample_rate']), 48000 if codec == 'opus' else rate)
                    # MP3's encoder attenuation alone can lower playback by ~0.45 dB.
                    self.assertAlmostEqual(float(measure(target)['input_i']), -18, delta=.6)

    def test_silent_chapter_stays_silent(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'silent.wav', Path(folder) / 'out.wav'
            write_tone(source, 0)
            self.assertIsNone(chapter_loudness_filter(source))
            encode_chapter(source, target, audio_format='wav', normalize=True, sample_rate=24000)
            with wave.open(str(target), 'rb') as audio:
                self.assertEqual(audio.readframes(audio.getnframes()), b'\0' * 192000)

    def test_very_short_chapter_does_not_fail_or_get_amplified(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'short.wav', Path(folder) / 'out.wav'
            with wave.open(str(source), 'wb') as audio:
                audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                samples = [round(29000 * math.sin(2 * math.pi * 440 * i / 24000)) for i in range(2400)]
                audio.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
            encode_chapter(source, target, audio_format='wav', normalize=True, sample_rate=24000)
            with wave.open(str(target), 'rb') as audio:
                self.assertEqual(audio.getnframes(), 2400)
                values = struct.unpack('<' + 'h' * 2400, audio.readframes(2400))
                self.assertGreater(max(abs(value) for value in values), 1000)
            self.assertLessEqual(float(measure(target)['input_tp']), -2.9)

    def test_unreadable_loudness_stats_fail_explicitly(self):
        for data in ('no measurement', '{}', json.dumps({'input_i': 'nan', 'input_tp': '-3',
                      'input_lra': '4', 'input_thresh': '-30', 'target_offset': '0'})):
            with self.subTest(data=data), patch('epub_audio.adapters.ffmpeg.run_tool', return_value=data):
                with self.assertRaisesRegex(RuntimeError, 'mesurer le volume'):
                    chapter_loudness_filter('chapter.wav')

    def test_pipeline_normalizes_final_chapters_and_keeps_original_wav_for_all_formats(self):
        plan = BookPlan([Chapter('Premier', ['Un texte.']), Chapter('Second', ['La suite.'])],
                        metadata=BookMetadata('Le livre'))
        originals = {}

        def render(engine, chapters, directory, **kwargs):
            for index in range(1, 3):
                chunks = directory / f'.segments-{index}'
                chunks.mkdir()
                wav = directory / f'.chapter-{index}.wav'
                write_tone(wav, index * 500)
                originals[index] = wav.read_bytes()
                yield index, wav, chunks

        for codec, suffix in (('wav', 'wav'), ('mp3', 'mp3'), ('aac', 'm4a'), ('opus', 'opus')):
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / 'book.epub'
                source.write_bytes(b'Fixture')
                with patch('epub_audio.core.pipeline.render_chapters', side_effect=render) as synthesis:
                    result = process_book(source, object(), book_plan=plan, output_dir=root / 'out',
                                          options=AudioOptions(audio_format=codec, normalize=True, workers=2),
                                          message=lambda text: None)
                synthesis.assert_called_once()
                masters = sorted((root / 'out' / 'master' / result.name).glob('*.wav'))
                outputs = sorted(result.glob('*.' + suffix))
                self.assertEqual(len(masters), 2)
                self.assertEqual(len(outputs), 2)
                for index, (master, output) in enumerate(zip(masters, outputs), 1):
                    self.assertEqual(master.read_bytes(), originals[index])
                    self.assertAlmostEqual(float(measure(output)['input_i']), -18, delta=.6)


if __name__ == '__main__':
    unittest.main()
