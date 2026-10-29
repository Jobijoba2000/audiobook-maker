"""French narration numbers, before splitting, with unchanged EPUB sources."""
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch
from epub_audio.core.numbers_fr import spoken_numbers
from epub_audio.core.models import BookMetadata, BookPlan, Chapter
from epub_audio.core.options import AudioOptions
from epub_audio.core.pipeline import process_book


class NumberTests(unittest.TestCase):
    def test_cardinals_years_signs_and_grouped_thousands(self):
        cases={
            '1866':'mille huit cent soixante-six',
            '20.000 lieues':'vingt mille lieues',
            '20 000 lieues':'vingt mille lieues',
            '20\u00a0000':'vingt mille',
            '20\u202f000':'vingt mille',
            '1 234 567':'un million deux cent trente-quatre mille cinq cent soixante-sept',
            '−12 et +4':'moins douze et plus quatre',
            '0,05':'zéro virgule zéro cinq',
            '12.50':'douze virgule cinq zéro',
            '1 234,08':'mille deux cent trente-quatre virgule zéro huit',
            '007':'zéro zéro sept',
            '1er, 1re, 2e, 21ème':'premier, première, deuxième, vingt et unième',
        }
        for text,expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(spoken_numbers(text),expected)

    def test_prose_identifiers_urls_and_existing_words_are_preserved(self):
        text='Louis XIV, vingt mille, B612, chapitreA12. https://site.fr/2026 et a12@test.fr'
        self.assertEqual(spoken_numbers(text),text)
        self.assertEqual(spoken_numbers('42 ans, puis 3 jours.'),'quarante-deux ans, puis trois jours.')
        self.assertEqual(spoken_numbers('1, 2, 3.'),'un, deux, trois.')
        self.assertEqual(spoken_numbers('3-5'),'trois-cinq')
        self.assertEqual(spoken_numbers(''), '')

    def test_pipeline_normalizes_introduction_headings_and_prose_before_segmentation(self):
        class Voice:
            serial_synthesis=True
            max_segment_characters=None
            def __init__(self): self.texts=[]
            def synthesize(self,text,destination):
                self.texts.append(text)
                with wave.open(str(destination),'wb') as wav:
                    wav.setparams((1,2,24000,0,'NONE','not compressed'))
                    wav.writeframes(b'\0\0'*2400)
                return destination
        cases = [('faster-qwen', '0.6B')]
        for engine_id, model_size in cases:
            with self.subTest(engine=engine_id, model=model_size), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                source=root/'source.epub'
                source.write_bytes(b'unchanged source')
                plan=BookPlan([Chapter('I',['I','En 1866, il avait 42 ans.'],(0,))],
                              metadata=BookMetadata('20.000 lieues'))
                voice=Voice()
                voice.engine_id, voice.model_size = engine_id, model_size
                with patch('epub_audio.core.pipeline.read_book',return_value=plan), \
                     patch('epub_audio.core.numbers_fr.spoken_numbers', wraps=spoken_numbers) as convert:
                    result=process_book(source,voice,output_dir=root/'out',audio_format='wav',
                                        segmentation='sentence',message=lambda _:None)
                spoken=' '.join(voice.texts)
                self.assertIn('Chapitre un',spoken)
                self.assertIn('vingt mille lieues',spoken)
                self.assertIn('mille huit cent soixante-six',spoken)
                self.assertIn('quarante-deux ans',spoken)
                self.assertNotRegex(spoken,r'\d')
                self.assertTrue(convert.called)
                self.assertEqual(plan.chapters[0].paragraphs,['I','En 1866, il avait 42 ans.'])
                self.assertEqual(source.read_bytes(),b'unchanged source')
                self.assertEqual(next(result.glob('*.epub')).read_bytes(),source.read_bytes())

    def test_disabled_conversion_keeps_digits_but_still_announces_roman_chapters(self):
        prose = 'En 1866, Louis XIV avait 42 ans.'
        plan = BookPlan([Chapter('IV', ['IV', prose], (0,))],
                        metadata=BookMetadata('20.000 lieues'))
        for options in ({'convert_numbers': False, 'audio_format': 'wav'},
                        {'options': AudioOptions(convert_numbers=False, audio_format='wav')}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                source = root / 'source.epub'
                source.write_bytes(b'unchanged source')

                def render(engine, chapters, directory, **kwargs):
                    self.assertEqual(chapters, [['Chapitre quatre.', prose]])
                    self.assertEqual(kwargs['introduction'], '20.000 lieues.')
                    chunks = directory / 'segments'
                    chunks.mkdir()
                    wav_path = directory / 'chapter.wav'
                    with wave.open(str(wav_path), 'wb') as wav:
                        wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
                        wav.writeframes(b'\0\0' * 2400)
                    yield 1, wav_path, chunks

                with patch('epub_audio.core.pipeline.read_book', return_value=plan), \
                     patch('epub_audio.core.pipeline.render_chapters', side_effect=render) as synthesize, \
                     patch('epub_audio.core.numbers_fr.spoken_numbers') as numbers:
                    result = process_book(source, object(), output_dir=root / 'out',
                                          message=lambda _: None, **options)
                synthesize.assert_called_once()
                numbers.assert_not_called()
                self.assertEqual(plan.chapters[0].paragraphs, ['IV', prose])
                self.assertEqual(next(result.glob('*.epub')).read_bytes(), source.read_bytes())



if __name__=='__main__':
    unittest.main()
