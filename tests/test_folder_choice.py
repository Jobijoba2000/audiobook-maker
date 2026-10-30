"""Browse input folders and books while preserving their output hierarchy."""
from pathlib import Path
from unittest.mock import patch
import io
import tempfile
import unittest

from epub_audio.cli.main import main


class FolderChoiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / 'input'
        self.output = self.root / 'output'
        self.collection = self.input / 'GUIEU Jimmy - Blade et Baker'
        self.other = self.input / 'Autre collection'
        self.collection.mkdir(parents=True)
        (self.collection / 'Sous-dossier').mkdir()
        self.other.mkdir()
        for path in (self.collection / 'A.epub', self.collection / 'Sous-dossier' / 'B.epub',
                     self.other / 'C.epub', self.input / 'D.epub'):
            path.write_bytes(b'Folder routing test; audio generation mocked')
        self.stdout = io.StringIO()
        for target, kwargs in (
                ('epub_audio.cli.args.INPUT_DIR', {'new': self.input}),
                ('epub_audio.cli.args.OUTPUT_DIR', {'new': self.output}),
                ('epub_audio.cli.main.INPUT_DIR', {'new': self.input}),
                ('epub_audio.cli.main.OUTPUT_DIR', {'new': self.output}),
                ('epub_audio.cli.main.require_tools', {}),
                ('epub_audio.cli.main.load_preferences', {'return_value': {}}),
                ('epub_audio.cli.main.save_preferences', {}),
                ('sys.stdout', {'new': self.stdout})):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch('epub_audio.cli.main.process_book')
        self.process = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch('epub_audio.cli.main.QwenEngine')
        self.engine = patcher.start()
        self.addCleanup(patcher.stop)
        self.flags = ['--no-llm-chapters', '--no-normalize', '--no-batching', '--segmentation', 'paragraph', '--voice', 'expr_chap_s01_0004-7e5a77eb', '--rate', '1.0',
                      '--codec', 'mp3', '--bitrate', '128', '--frequency', '44.1']

    def test_enter_defaults_to_all_input_and_normal_output(self):
        with patch('builtins.input', side_effect=['', '', '']) as prompt:
            self.assertEqual(main(self.flags), 0)
        self.assertIn('Numéro du dossier ou du livre', prompt.call_args_list[0].args[0])
        self.assertNotIn('Modifier les dossiers', self.stdout.getvalue())
        self.assertEqual(self.process.call_count, 4)
        self.assert_outputs_under(self.input, self.output)

    def pending_book(self):
        source = self.collection / 'A.epub'
        directory = self.output / self.collection.name / '.ebook-pending'
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / 'progression.json'
        marker.write_bytes(b'Preserve this interrupted book and all its audio')
        return {'source': source, 'directory': directory, 'completed': 1, 'total': 2, 'legacy': False}, marker

    def test_choose_another_book_defers_incomplete_books_for_the_whole_session(self):
        job, marker = self.pending_book()
        original = marker.read_bytes()
        for choice in ('', '0', 'n'):
            with self.subTest(choice=choice), \
                 patch('epub_audio.cli.main.discover_incomplete', return_value=[job]), \
                 patch('epub_audio.cli.main.resume_books') as resume, \
                 patch('builtins.input', side_effect=[choice, '', '', '']):
                self.process.reset_mock()
                self.assertEqual(main(self.flags), 0)
                resume.assert_not_called()
                self.assertEqual(self.process.call_count, 3)
                self.assertNotIn(job['source'], [call.args[0] for call in self.process.call_args_list])
                self.assertEqual(marker.read_bytes(), original)
        self.assertIn('0. Choisir un autre livre', self.stdout.getvalue())

    def test_invalid_startup_choice_can_return_to_other_books(self):
        job, _ = self.pending_book()
        with patch('epub_audio.cli.main.discover_incomplete', return_value=[job]), \
             patch('epub_audio.cli.main.resume_books') as resume, \
             patch('builtins.input', side_effect=['invalide', '0', '', '', '']):
            self.assertEqual(main(self.flags), 0)
        resume.assert_not_called()
        self.assertEqual(self.process.call_count, 3)

    def test_selected_batch_can_leave_interrupted_books_pending(self):
        job, marker = self.pending_book()
        original = marker.read_bytes()
        with patch('epub_audio.cli.main.discover_incomplete', return_value=[job]), \
             patch('epub_audio.cli.main.resume_books') as resume, \
             patch('builtins.input', side_effect=['', 'p', '']):
            self.assertEqual(main([str(self.input), '--output', str(self.output), *self.flags]), 0)
        resume.assert_not_called()
        self.assertEqual(self.process.call_count, 3)
        self.assertNotIn(job['source'], [call.args[0] for call in self.process.call_args_list])
        self.assertEqual(marker.read_bytes(), original)

    def assert_outputs_under(self, source_root, output_root):
        for call in self.process.call_args_list:
            expected = output_root / call.args[0].parent.relative_to(source_root)
            self.assertEqual(call.kwargs['output_dir'], expected)
            # Interactive folder selection keeps master at the configured root.
            master_root = self.output if expected.is_relative_to(self.output) else output_root
            self.assertEqual(call.kwargs['master_output_dir'],
                             master_root / 'master' / expected.relative_to(master_root))

    def test_named_collection_is_the_only_input_and_output_is_created(self):
        with patch('builtins.input', side_effect=[self.collection.name, '', '', '']):
            self.assertEqual(main(self.flags), 0)
        self.assertEqual(self.process.call_count, 2)
        self.assertTrue(all(call.args[0].is_relative_to(self.collection) for call in self.process.call_args_list))
        expected = self.output / self.collection.name
        self.assertTrue(expected.is_dir())
        self.assert_outputs_under(self.collection, expected)
        self.assertIn(str(expected), self.stdout.getvalue())

    def test_missing_and_outside_folders_are_rejected_before_valid_quoted_name(self):
        outside = self.root / 'outside'
        outside.mkdir()
        responses = ['Absent', '../outside', str(outside), f'"{self.collection.name}"', '', '', '']
        with patch('builtins.input', side_effect=responses):
            self.assertEqual(main(self.flags), 0)
        self.assertIn('Dossier ou fichier EPUB introuvable', self.stdout.getvalue())
        self.assertEqual(self.process.call_count, 2)
        self.assertFalse((outside / 'A.epub').exists())

    def test_next_launch_starts_at_input_again(self):
        with patch('builtins.input', side_effect=[self.collection.name, '', '', '']):
            self.assertEqual(main(self.flags), 0)
        self.process.reset_mock()
        with patch('builtins.input', side_effect=['', '', '']):
            self.assertEqual(main(self.flags), 0)
        self.assertEqual(self.process.call_count, 4)
        self.assert_outputs_under(self.input, self.output)

    def test_empty_folder_still_creates_the_corresponding_output(self):
        (self.input / 'Vide').mkdir()
        with patch('builtins.input', side_effect=['Vide', '']):
            self.assertEqual(main(self.flags), 0)
        self.assertTrue((self.output / 'Vide').is_dir())
        self.process.assert_not_called()
        self.engine.assert_not_called()

    def test_existing_output_file_is_preserved_and_another_folder_can_be_chosen(self):
        self.output.mkdir()
        conflict = self.output / self.collection.name
        conflict.write_bytes(b'preserve this file')
        with patch('builtins.input', side_effect=[self.collection.name, self.other.name, '', '', '']):
            self.assertEqual(main(self.flags), 0)
        self.assertEqual(conflict.read_bytes(), b'preserve this file')
        self.assertEqual(self.process.call_count, 1)
        self.assertEqual(self.process.call_args.kwargs['output_dir'], self.output / self.other.name)

    def test_yes_and_list_voices_do_not_ask_for_folders(self):
        with patch('builtins.input', side_effect=AssertionError('Unexpected question')):
            self.assertEqual(main([*self.flags, '--yes']), 0)
            self.assertEqual(main(['--list-voices']), 0)
        self.assertEqual(self.process.call_count, 4)

    def test_explicit_input_and_output_keep_their_command_line_meaning(self):
        custom = self.root / 'custom-output'
        with patch('builtins.input', side_effect=['', '']) as prompt:
            self.assertEqual(main([str(self.collection), '--output', str(custom), *self.flags]), 0)
        self.assertTrue(prompt.call_args_list[0].args[0].startswith('Livres à traiter'))
        self.assertEqual(self.process.call_count, 2)
        self.assert_outputs_under(self.collection, custom)

    def test_deep_subfolders_and_identical_filenames_remain_separate(self):
        nested = self.collection / 'Sous-dossier' / 'Tome 2'
        nested.mkdir()
        (nested / 'A.epub').write_bytes(b'another book with the same filename')
        with patch('builtins.input', side_effect=AssertionError('Unexpected question')):
            self.assertEqual(main([*self.flags, '--yes']), 0)
        self.assertEqual(self.process.call_count, 5)
        self.assert_outputs_under(self.input, self.output)
        destinations = {call.kwargs['output_dir'] for call in self.process.call_args_list
                        if call.args[0].name == 'A.epub'}
        self.assertEqual(destinations, {self.output / self.collection.name,
                                      self.output / self.collection.name / 'Sous-dossier' / 'Tome 2'})

    def test_selecting_one_book_keeps_its_full_relative_folder(self):
        selected = self.collection / 'Sous-dossier' / 'B.epub'
        with patch('builtins.input', side_effect=['2', '1', '1', '']) as prompt:
            self.assertEqual(main(self.flags), 0)
        self.assertFalse(any('Livres à traiter' in call.args[0] for call in prompt.call_args_list))
        self.process.assert_called_once()
        self.assertEqual(self.process.call_args.args[0], selected)
        self.assert_outputs_under(self.input, self.output)

    def test_immediate_listing_sorts_folders_first_then_epubs_ignoring_case_and_accents(self):
        (self.input / 'écrivains').mkdir()
        (self.input / 'zebra').mkdir()
        (self.input / 'alpha.EPUB').write_bytes(b'fixture')
        (self.input / 'Été.epub').write_bytes(b'fixture')
        (self.input / 'notes.txt').write_text('not a book', encoding='utf-8')
        with patch('builtins.input', side_effect=['', '', '']):
            self.assertEqual(main(self.flags), 0)
        listing = [line.split('. ', 1)[1] for line in self.stdout.getvalue().splitlines()
                   if line[:1].isdigit() and '. ' in line][:7]
        names = [line.split(' ', 1)[1] for line in listing]
        self.assertEqual(names, ['Autre collection', 'écrivains', self.collection.name, 'zebra',
                                'alpha.EPUB', 'D.epub', 'Été.epub'])
        self.assertNotIn('notes.txt', self.stdout.getvalue())

    def test_parent_navigation_returns_to_root_before_picking_a_file(self):
        with patch('builtins.input', side_effect=['2', '1', '0', '..', '3', '']):
            self.assertEqual(main(self.flags), 0)
        self.process.assert_called_once()
        self.assertEqual(self.process.call_args.args[0], self.input / 'D.epub')
        self.assertEqual(self.process.call_args.kwargs['output_dir'], self.output)

    def test_invalid_numbers_and_non_epub_files_are_rejected(self):
        (self.input / 'notes.txt').write_text('not a book', encoding='utf-8')
        with patch('builtins.input', side_effect=['99', 'notes.txt', '3', '']):
            self.assertEqual(main(self.flags), 0)
        self.assertIn('Choisissez un numéro', self.stdout.getvalue())
        self.process.assert_called_once()
        self.assertEqual(self.process.call_args.args[0], self.input / 'D.epub')

    def test_ascii_console_uses_text_labels_for_folders_and_files(self):
        class AsciiConsole(io.StringIO):
            encoding = 'ascii'
        stream = AsciiConsole()
        with patch('sys.stdout', stream), patch('builtins.input', side_effect=['3', '']):
            self.assertEqual(main(self.flags), 0)
        self.assertIn('1. [DOSSIER] Autre collection', stream.getvalue())
        self.assertIn('3. [EPUB] D.epub', stream.getvalue())

    def test_missing_input_uses_normal_empty_folder_message(self):
        missing = self.root / 'missing-input'
        with patch('epub_audio.cli.main.INPUT_DIR', missing), \
             patch('epub_audio.cli.args.INPUT_DIR', missing), \
             patch('builtins.input', return_value=''):
            self.assertEqual(main(self.flags), 0)
        self.assertTrue(missing.is_dir())
        self.assertIn('Aucun fichier .epub', self.stdout.getvalue())
        self.engine.assert_not_called()

    def test_overlapping_input_arguments_only_process_each_book_once(self):
        custom = self.root / 'custom-output'
        with patch('builtins.input', side_effect=AssertionError('Unexpected question')):
            self.assertEqual(main([str(self.collection), str(self.collection / 'Sous-dossier'),
                                   str(self.other / 'C.epub'), '--output', str(custom),
                                   *self.flags, '--yes']), 0)
        self.assertEqual(self.process.call_count, 3)
        destinations = {call.args[0]: call.kwargs['output_dir'] for call in self.process.call_args_list}
        self.assertEqual(destinations[self.collection / 'Sous-dossier' / 'B.epub'], custom / 'Sous-dossier')
        self.assertEqual(destinations[self.other / 'C.epub'], custom)


if __name__ == '__main__':
    unittest.main()
