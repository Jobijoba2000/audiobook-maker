"""Internet addresses must not reach narration or influence number conversion."""
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from epub_audio.core.models import BookMetadata, BookPlan, Chapter
from epub_audio.core.pipeline import process_book
from epub_audio.core.text import prepare_spoken_text


class SpokenUrlTests(unittest.TestCase):
    def test_schemes_www_queries_and_unicode_paths(self):
        for url in ('http://example.org', 'https://example.org/path?year=2026&n=42#note',
                    'HTTPS://EXAMPLE.ORG/2026', 'www.example.fr/page',
                    'ftp://example.org/file.txt', 'ftps://example.org/file.txt',
                    'https://exemple.fr/été', "https://example.org/l'histoire"):
            with self.subTest(url=url):
                self.assertEqual(prepare_spoken_text(url), 'Lien internet.')

    def test_sentence_endings_and_enclosing_punctuation(self):
        examples = {
            'Voir https://example.org. Puis partir.': 'Voir Lien internet. Puis partir.',
            'Voir www.example.org...': 'Voir Lien internet.',
            '(https://example.org)': '(Lien internet.)',
            '« https://example.org »': '« Lien internet. »',
            "'https://example.org'": "'Lien internet.'",
            'https://example.org/wiki/Titre_(suite)': 'Lien internet.',
            '(https://example.org/wiki/Titre_(suite))': '(Lien internet.)',
            'https://example.org, puis le récit.': 'Lien internet. puis le récit.',
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(prepare_spoken_text(text), expected)
                self.assertEqual(prepare_spoken_text(expected), expected)

    def test_four_wikisource_footer_links_become_four_labels(self):
        text = ('↑ http://fr.wikisource.org\n'
                '↑ http://creativecommons.org/licenses/by-sa/3.0/deed.fr\n'
                '↑ http://www.gnu.org/copyleft/fdl.html\n'
                '↑ http://fr.wikisource.org/wiki/Aide:Signaler_une_erreur')
        self.assertEqual(prepare_spoken_text(text), '\n'.join(['↑ Lien internet.'] * 4))

    def test_prose_numbers_and_email_addresses_are_preserved(self):
        self.assertEqual(prepare_spoken_text('En 1866, M. Aronnax lit le n° 42.'),
                         'En 1866, M. Aronnax lit le numéro 42.')
        for text in ('', 'Aucune adresse dans ce passage.', 'contact@www.example.fr', '3.14 mètres.'):
            with self.subTest(text=text):
                self.assertEqual(prepare_spoken_text(text), text)

    def test_pipeline_prepares_urls_everywhere_and_preserves_semicolons(self):
        for convert_numbers in (False, True):
            with self.subTest(convert_numbers=convert_numbers), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / 'source.epub'
                source.write_bytes(b'original EPUB bytes')
                prose = 'Un passage; sa suite. Consultez https://example.org/2026?ref=42.'
                paragraphs = ['IV', 'Un titre; sa suite', prose]
                title = 'Livre; suite https://example.org/123'
                plan = BookPlan([Chapter('IV', paragraphs, (0, 1))], metadata=BookMetadata(title))

                def render(engine, chapters, directory, **kwargs):
                    self.assertEqual(chapters, [['Chapitre quatre.', 'Un titre; sa suite.',
                                                 'Un passage; sa suite. Consultez Lien internet.']])
                    self.assertEqual(kwargs['introduction'], 'Livre; suite Lien internet.')
                    chunks = directory / 'segments'
                    chunks.mkdir()
                    audio = directory / 'chapter.wav'
                    with wave.open(str(audio), 'wb') as wav:
                        wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                        wav.writeframes(b'\0\0' * 2400)
                    yield 1, audio, chunks

                with patch('epub_audio.core.pipeline.read_book', return_value=plan), \
                     patch('epub_audio.core.pipeline.render_chapters', side_effect=render):
                    result = process_book(source, object(), output_dir=root / 'out',
                                          audio_format='wav', convert_numbers=convert_numbers,
                                          message=lambda _: None)
                self.assertEqual(plan.chapters[0].paragraphs, ['IV', 'Un titre; sa suite', prose])
                self.assertEqual(plan.metadata.title, title)
                self.assertEqual(source.read_bytes(), b'original EPUB bytes')
                self.assertEqual(next(result.glob('*.epub')).read_bytes(), source.read_bytes())
