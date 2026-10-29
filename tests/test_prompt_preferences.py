"""Questions can be hidden independently without losing remembered settings."""
from contextlib import ExitStack
from pathlib import Path
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from epub_audio.cli.main import main
from epub_audio.io.preferences import PROMPT_NAMES, load_preferences, save_preferences


class PromptPreferencesTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.root / 'book.epub'
        self.source.write_bytes(b'Routing fixture; synthesis mocked')
        self.prefs = self.root / 'preferences.json'
        stack.enter_context(patch('epub_audio.io.preferences.PREFERENCES_FILE', self.prefs))
        stack.enter_context(patch('epub_audio.cli.main.require_tools'))
        self.engine = stack.enter_context(patch('epub_audio.cli.main.QwenEngine'))
        self.process = stack.enter_context(patch('epub_audio.cli.main.process_book'))
        stack.enter_context(patch('sys.stdout', new=io.StringIO()))
        self.settings = dict(voice='expr_chap_s01_0004-7e5a77eb', audio_format='opus',
                             bitrate=48, sample_rate=24000, segmentation='paragraph',
                             faster_small_batching=True, faster_small_batch_size=6)

    def write(self, prompts):
        self.prefs.write_text(json.dumps(dict(self.settings, prompts=prompts)), encoding='utf-8')

    def test_four_hidden_audio_questions_use_saved_values_and_survive_multiple_runs(self):
        prompts = {name: False for name in ('segmentation', 'audio_format', 'bitrate', 'sample_rate', 'normalize')}
        self.write(prompts)
        for _ in range(2):
            with patch('builtins.input', side_effect=['', '', '', '', '']) as ask:
                self.assertEqual(main([str(self.source)]), 0)
            questions = [call.args[0] for call in ask.call_args_list]
            self.assertEqual(len(questions), 5)
            self.assertTrue(questions[0].startswith('Livres à traiter'))
            self.assertTrue(questions[1].startswith('Améliorer le découpage'))
            self.assertTrue(questions[2].startswith('Voix de référence'))
            self.assertTrue(questions[3].startswith('Batching'))
            self.assertTrue(questions[4].startswith('Entrée pour commencer'))
            options = self.process.call_args.kwargs['options']
            self.assertEqual((options.segmentation, options.audio_format, options.bitrate, options.sample_rate),
                             ('paragraph', 'opus', 48, 24000))
            self.assertEqual(load_preferences()['prompts'], prompts)

    def test_each_audio_question_can_be_enabled_independently(self):
        selectors = {'segmentation': ('select_segmentation', 'sentence'),
                     'audio_format': ('select_audio_format', 'mp3'),
                     'bitrate': ('select_bitrate', 64),
                     'sample_rate': ('select_sample_rate', 48000)}
        for enabled in selectors:
            with self.subTest(question=enabled), ExitStack() as stack:
                prompts = dict.fromkeys(PROMPT_NAMES, False)
                prompts[enabled] = True
                self.write(prompts)
                menus = {name: stack.enter_context(patch('epub_audio.cli.main.' + selector, return_value=value))
                         for name, (selector, value) in selectors.items()}
                ask = stack.enter_context(patch('builtins.input', side_effect=['']))
                self.assertEqual(main([str(self.source)]), 0)
                self.assertEqual(ask.call_count, 1)  # Book selection remains interactive.
                for name, menu in menus.items():
                    self.assertEqual(menu.call_count, int(name == enabled))
                options = self.process.call_args.kwargs['options']
                self.assertEqual(getattr(options, enabled), selectors[enabled][1])
                self.assertEqual(load_preferences()['prompts'], prompts)

    def test_explicit_options_override_hidden_questions(self):
        prompts = dict.fromkeys(PROMPT_NAMES, False)
        self.write(prompts)
        with patch('builtins.input', side_effect=['']) as ask:
            self.assertEqual(main([str(self.source), '--segmentation', 'sentence', '--format', 'aac',
                                   '--bitrate', '64', '--frequency', '48', '--batch-size', '3']), 0)
        self.assertEqual(ask.call_count, 1)
        options = self.process.call_args.kwargs['options']
        self.assertEqual((options.segmentation, options.audio_format, options.bitrate, options.sample_rate),
                         ('sentence', 'aac', 64, 48000))
        self.assertEqual(self.engine.call_args.kwargs['batch_size'], 3)
        self.assertEqual(load_preferences()['prompts'], prompts)

    def test_yes_suppresses_even_enabled_questions(self):
        self.write(dict.fromkeys(PROMPT_NAMES, True))
        with patch('builtins.input', side_effect=AssertionError('Unexpected question')):
            self.assertEqual(main([str(self.source), '--yes']), 0)

    def test_save_without_prompt_argument_preserves_existing_prompt_choices(self):
        prompts = {'segmentation': False, 'audio_format': True, 'confirmation': False}
        self.write(prompts)
        save_preferences(self.settings['voice'], 1, audio_format='opus', bitrate=48)
        self.assertEqual(load_preferences()['prompts'], prompts)

    def test_invalid_prompt_settings_are_ignored_on_load_and_rejected_on_save(self):
        for invalid in ('false', 0, 1, [], {}):
            with self.subTest(value=invalid):
                self.write({'segmentation': invalid, 'bitrate': False, 'unknown': True})
                self.assertEqual(load_preferences()['prompts'], {'bitrate': False})
                with self.assertRaises(ValueError):
                    save_preferences(self.settings['voice'], 1, prompts={'segmentation': invalid})

    def test_hidden_questions_without_saved_values_use_program_defaults(self):
        self.prefs.write_text(json.dumps(dict(prompts=dict.fromkeys(PROMPT_NAMES, False))), encoding='utf-8')
        with patch('builtins.input', side_effect=['']):
            self.assertEqual(main([str(self.source)]), 0)
        options = self.process.call_args.kwargs['options']
        self.assertEqual((options.segmentation, options.audio_format, options.sample_rate), ('paragraph', 'wav', 24000))

    def test_hidden_number_question_uses_saved_false_across_runs(self):
        self.settings['convert_numbers'] = False
        prompts = dict.fromkeys(PROMPT_NAMES, False)
        self.write(prompts)
        for _ in range(2):
            with patch('builtins.input', side_effect=['']) as ask:
                self.assertEqual(main([str(self.source)]), 0)
            self.assertEqual(ask.call_count, 1)
            self.assertFalse(self.process.call_args.kwargs['options'].convert_numbers)
            self.assertFalse(load_preferences()['convert_numbers'])
            self.assertFalse(load_preferences()['prompts']['convert_numbers'])

    def test_number_menu_selects_and_remembers_both_values(self):
        prompts = dict.fromkeys(PROMPT_NAMES, False)
        prompts['convert_numbers'] = True
        for answer, expected in (('1', False), ('2', True)):
            with self.subTest(answer=answer):
                self.write(prompts)
                with patch('builtins.input', side_effect=['', 'invalid', answer]) as ask:
                    self.assertEqual(main([str(self.source)]), 0)
                self.assertIn('Convertir les nombres en lettres', ask.call_args.args[0])
                self.assertIs(self.process.call_args.kwargs['options'].convert_numbers, expected)
                self.assertIs(load_preferences()['convert_numbers'], expected)

    def test_explicit_number_flags_override_preferences_and_skip_menu(self):
        prompts = dict.fromkeys(PROMPT_NAMES, False)
        prompts['convert_numbers'] = True
        for flag, expected in (('--convert-numbers', True), ('--no-convert-numbers', False)):
            with self.subTest(flag=flag):
                self.settings['convert_numbers'] = not expected
                self.write(prompts)
                with patch('builtins.input', side_effect=['']) as ask:
                    self.assertEqual(main([str(self.source), flag]), 0)
                self.assertEqual(ask.call_count, 1)
                self.assertIs(self.process.call_args.kwargs['options'].convert_numbers, expected)
                self.assertIs(load_preferences()['convert_numbers'], expected)

    def test_saving_other_preferences_preserves_number_setting(self):
        self.settings['convert_numbers'] = False
        self.write({'convert_numbers': False})
        save_preferences(self.settings['voice'], 1)
        self.assertFalse(load_preferences()['convert_numbers'])

    def test_number_setting_requires_a_boolean(self):
        from epub_audio.core.options import AudioOptions
        for invalid in ('false', 0, 1, [], {}):
            with self.subTest(value=invalid):
                self.settings['convert_numbers'] = invalid
                self.write({})
                self.assertNotIn('convert_numbers', load_preferences())
                with self.assertRaises(ValueError):
                    save_preferences(self.settings['voice'], 1, convert_numbers=invalid)
                with self.assertRaises(ValueError):
                    AudioOptions(convert_numbers=invalid)
