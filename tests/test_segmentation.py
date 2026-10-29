"""Sentence selection preserves words, chapter boundaries and actual PCM pauses."""
from pathlib import Path
import io
import json
import re
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch

from epub_audio.cli.args import arguments
from epub_audio.cli.main import main
from epub_audio.core.models import BookPlan, BookMetadata, Chapter
from epub_audio.core.options import AudioOptions
from epub_audio.core.pipeline import process_book
from epub_audio.core.text import plan_segments, split_sentences
from epub_audio.core.tts_generate import render_chapter
from epub_audio.io.preferences import load_preferences, save_preferences


class CountingVoice:
    serial_gpu = True
    max_segment_characters = None

    def __init__(self):
        self.texts = []

    def synthesize(self, text, destination):
        self.texts.append(text)
        with wave.open(str(destination), 'wb') as audio:
            audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            audio.writeframes(struct.pack('<h', 1000 * len(self.texts)) * 2400)
        return Path(destination)


class SegmentationTests(unittest.TestCase):
    def test_french_sentences_keep_abbreviations_decimals_quotes_and_dialogue_tags(self):
        examples = [
            ('M. Aronnax rejoint Mme. Dupont. Ils partent.',
             ['M. Aronnax rejoint Mme. Dupont.', 'Ils partent.']),
            ('Le Dr. A. Martin mesure 3.14 mètres. Puis 2,5 mètres !',
             ['Le Dr. A. Martin mesure 3.14 mètres.', 'Puis 2,5 mètres !']),
            ('« Où allez-vous ? » demanda-t-il. « Ici ! »',
             ['« Où allez-vous ? » demanda-t-il.', '« Ici ! »']),
            ('Il hésite… puis repart. Ensuite, il court...',
             ['Il hésite… puis repart.', 'Ensuite, il court...']),
            ('Première phrase.\nSeconde phrase ?\tDernière sans point',
             ['Première phrase.', 'Seconde phrase ?', 'Dernière sans point']),
            ('Il répond : “Non.” Puis repart.', ['Il répond : “Non.”', 'Puis repart.']),
        ]
        for text, expected in examples:
            with self.subTest(text=text):
                actual = list(split_sentences(text))
                self.assertEqual(actual, expected)
                self.assertEqual(' '.join(actual), re.sub(r'\s+', ' ', text).strip())

    def test_only_original_paragraph_and_scene_boundaries_add_pauses(self):
        paragraphs = ['Première phrase. Deuxième phrase !', '', '***', 'Scène suivante. Encore.',
                      'Dernier paragraphe.']
        segments = list(plan_segments(paragraphs, segmentation='sentence'))
        self.assertEqual([s.text for s in segments],
                         ['Première phrase.', 'Deuxième phrase !', 'Scène suivante.', 'Encore.',
                          'Dernier paragraphe.'])
        self.assertEqual([s.pause_before_ms for s in segments], [0, 0, 1500, 0, 800])
        original = list(plan_segments(paragraphs))
        self.assertEqual([s.text for s in original],
                         [paragraphs[0], paragraphs[3], paragraphs[4]])
        self.assertEqual([s.pause_before_ms for s in original], [0, 1500, 800])

    def test_sentence_mode_exports_one_wav_per_chapter_with_exact_boundary_silences(self):
        # Keep this PCM-pause check outside the new short-prefix grouping case.
        title = 'Titre du livre' + 'x' * 200
        last = 'Dernier chapitre ' + 'x' * 200 + '.'
        plan = BookPlan([
            Chapter('Premier', ['Chapitre un', 'Première phrase. Deuxième phrase !',
                                '***', 'Scène nouvelle. Encore.']),
            Chapter('Second', [last + ' Fin.'])],
            metadata=BookMetadata(title, ['Jules Verne']))
        voice = CountingVoice()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'livre.epub'
            source.write_bytes(b'book fixture')
            with patch('epub_audio.core.pipeline.read_book', return_value=plan):
                result = process_book(source, voice, output_dir=root / 'output', message=lambda s: None,
                                      options=AudioOptions(audio_format='wav', segmentation='sentence'))
            self.assertEqual(voice.texts, [title + '.', 'Un livre de Jules Verne.', 'Chapitre un',
                                          'Première phrase.', 'Deuxième phrase !', 'Scène nouvelle.',
                                          'Encore.', last, 'Fin.'])
            tracks = sorted(result.glob('*.wav'))
            self.assertEqual(len(tracks), 2)
            pcm = lambda n: struct.pack('<h', 1000 * n) * 2400
            expected = [pcm(1) + pcm(2) + b'\0\0' * 48000 + pcm(3)
                        + b'\0\0' * 19200 + pcm(4) + pcm(5)
                        + b'\0\0' * 36000 + pcm(6) + pcm(7), pcm(8) + pcm(9)]
            for track, frames in zip(tracks, expected):
                with wave.open(str(track), 'rb') as audio:
                    self.assertEqual(audio.readframes(audio.getnframes()), frames)
            self.assertFalse(list(result.glob('.segments-*')))

    def test_single_chapter_entry_point_also_uses_sentence_mode(self):
        voice = CountingVoice()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            render_chapter(voice, ['Un début. Une fin.'], root / 'segments', root / 'chapter.wav',
                           workers=1, progress=lambda s: None, segmentation='sentence')
            self.assertEqual(voice.texts, ['Un début.\nUne fin.'])
            with wave.open(str(root / 'chapter.wav'), 'rb') as audio:
                self.assertEqual(audio.getnframes(), 2400)

    def test_menu_remembers_both_modes_and_routes_options(self):
        for number, mode in (('1', 'paragraph'), ('2', 'sentence')):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / 'livre.epub'
                source.write_bytes(b'routing fixture')
                flags = [str(source), '--no-llm-chapters', '--no-normalize', '--no-batching', '--codec', 'wav',
                         '--voice', 'expr_chap_s01_0004-7e5a77eb']
                with patch('epub_audio.io.preferences.PREFERENCES_FILE', root / 'preferences.json'), \
                     patch('epub_audio.cli.main.QwenEngine'), \
                     patch('epub_audio.cli.main.process_book') as process, \
                     patch('builtins.input', side_effect=['', 'invalid', number, '']), \
                     patch('sys.stdout', new=io.StringIO()) as output:
                    self.assertEqual(main(flags), 0)
                    self.assertEqual(process.call_args.kwargs['options'].segmentation, mode)
                    self.assertEqual(load_preferences()['segmentation'], mode)
                    self.assertIn('1. Par paragraphes', output.getvalue())
                    self.assertIn('2. Par phrases', output.getvalue())
                    with patch('builtins.input', side_effect=['', '', '']) as prompt:
                        self.assertEqual(main(flags), 0)
                    self.assertIn('[' + number + ']', prompt.call_args_list[1].args[0])
                    self.assertEqual(process.call_args.kwargs['options'].segmentation, mode)

    def test_batch_and_explicit_override_keep_existing_audio_preferences(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'livre.epub'
            source.write_bytes(b'routing fixture')
            with patch('epub_audio.io.preferences.PREFERENCES_FILE', root / 'preferences.json'), \
                 patch('epub_audio.cli.main.QwenEngine') as engine, \
                 patch('epub_audio.cli.main.process_book') as process, \
                 patch('builtins.input', side_effect=AssertionError('Unexpected question')), \
                 patch('sys.stdout', new=io.StringIO()):
                save_preferences('expr_chap_s01_0004-7e5a77eb', 1, audio_format='wav',
                                 segmentation='sentence')
                self.assertEqual(main([str(source), '--yes']), 0)
                self.assertEqual(process.call_args.kwargs['options'].segmentation, 'sentence')
                self.assertNotIn('model_size', engine.call_args.kwargs)
                self.assertEqual(main([str(source), '--yes', '--segmentation', 'paragraph']), 0)
                self.assertEqual(process.call_args.kwargs['options'].segmentation, 'paragraph')
                self.assertNotIn('model_size', load_preferences())
                self.assertNotIn('engine', load_preferences())

    def test_legacy_preferences_default_to_paragraph_and_invalid_modes_are_rejected(self):
        self.assertEqual(AudioOptions().segmentation, 'paragraph')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'preferences.json'
            with patch('epub_audio.io.preferences.PREFERENCES_FILE', path):
                for bad in (None, 'chapter', [], {}):
                    path.write_text(json.dumps(dict(segmentation=bad, bitrate=48)), encoding='utf-8')
                    self.assertNotIn('segmentation', load_preferences())
                    self.assertEqual(load_preferences()['bitrate'], 48)
        with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            arguments(['--segmentation', 'chapter'])
        with self.assertRaises(ValueError):
            AudioOptions(segmentation='chapter')
        with self.assertRaises(ValueError):
            list(plan_segments(['Un texte.'], segmentation='chapter'))


if __name__ == '__main__':
    unittest.main()
