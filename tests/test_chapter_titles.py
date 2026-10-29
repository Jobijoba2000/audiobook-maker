"""Roman chapter labels become spoken French without modifying narrative text."""
from pathlib import Path
import struct
import tempfile
import unittest
import wave
from epub_audio.core.chapter_titles import (roman_number, french_number,
                                            spoken_roman_heading, normalize_chapter_numbers,
                                            punctuate_chapter_headings, prepare_chapter_headings)
from epub_audio.core.text import plan_segments, merge_chapter_start
from epub_audio.core.pipeline import process_book
from epub_audio.io.epub import read_book


class TitleVoice:
    serial_gpu = True
    max_segment_characters = None

    def __init__(self):
        self.texts=[]

    def synthesize(self,text,destination):
        self.texts.append(text)
        with wave.open(str(destination),'wb') as audio:
            audio.setparams((1,2,24000,0,'NONE','not compressed'))
            audio.writeframes(struct.pack('<h',1000)*2400)
        return Path(destination)


class ChapterTitleTests(unittest.TestCase):
    def test_roman_numbers_are_strict_and_unicode_numerals_work(self):
        for numeral,expected in {'I':1,'IV':4,'IX':9,'XLII':42,'XCIX':99,'CD':400,
                                  'MCMXCIV':1994,'MMMCMXCIX':3999,'Ⅳ':4,'Ⅻ':12}.items():
            self.assertEqual(roman_number(numeral),expected)
        for invalid in ('','IIII','IC','VX','IIV','MMMM','ABC','IVX'):
            self.assertIsNone(roman_number(invalid))

    def test_french_cardinals_cover_chapter_numbers_and_irregular_forms(self):
        values={1:'un',16:'seize',17:'dix-sept',21:'vingt et un',42:'quarante-deux',
                70:'soixante-dix',71:'soixante et onze',79:'soixante-dix-neuf',80:'quatre-vingts',
                81:'quatre-vingt-un',90:'quatre-vingt-dix',91:'quatre-vingt-onze',
                100:'cent',101:'cent un',200:'deux cents',201:'deux cent un',
                280:'deux cent quatre-vingts',281:'deux cent quatre-vingt-un',
                1000:'mille',2000:'deux mille',2026:'deux mille vingt-six',
                3999:'trois mille neuf cent quatre-vingt-dix-neuf'}
        for number,expected in values.items():
            self.assertEqual(french_number(number),expected)
        for invalid in (0,-1,4000,True,'1'):
            with self.assertRaises(ValueError):
                french_number(invalid)

    def test_bare_and_existing_chapter_labels_preserve_the_title_suffix(self):
        examples={'I':'Chapitre un','II':'Chapitre deux','IV':'Chapitre quatre',
                  'Chapitre IV':'Chapitre quatre','CHAPITRE iv':'Chapitre quatre',
                  'I — Un écueil fuyant':'Chapitre un — Un écueil fuyant',
                  'XII. La traversée':'Chapitre douze. La traversée',
                  'Ⅳ':'Chapitre quatre','XLII':'Chapitre quarante-deux'}
        for heading,expected in examples.items():
            self.assertEqual(spoken_roman_heading(heading),expected)
        for unchanged in ('PREMIÈRE PARTIE','Louis XIV','Mix','i','IIII','Une idée.','II dans le récit'):
            self.assertEqual(spoken_roman_heading(unchanged),unchanged)

    def test_only_one_opening_heading_is_converted_and_prose_is_untouched(self):
        paragraphs=['PREMIÈRE PARTIE','I','UN ÉCUEIL FUYANT','Le roi Louis XIV arrive.','IV']
        original=paragraphs[:]
        self.assertEqual(normalize_chapter_numbers(paragraphs,(0,1,2)),
                         ['PREMIÈRE PARTIE','Chapitre un','UN ÉCUEIL FUYANT',paragraphs[3],'IV'])
        self.assertEqual(paragraphs,original)
        self.assertEqual(normalize_chapter_numbers(['I','IV'],()),['I','IV'])
        self.assertEqual(normalize_chapter_numbers(['Un récit.','IV'],(1,)),['Un récit.','IV'])
        self.assertEqual(normalize_chapter_numbers(['I','II'],(0,1)),['Chapitre un','II'])

    def test_number_expansion_happens_before_the_200_character_limit(self):
        paragraphs=normalize_chapter_numbers(['I','x'*188,'Stop'],(0,))
        segments=merge_chapter_start(plan_segments(paragraphs))
        self.assertEqual(len(segments),2)
        self.assertEqual(segments[0].text,'Chapitre un\n'+'x'*188)
        self.assertEqual(len(segments[0].text),200)
        self.assertEqual(segments[1].text,'Stop')

    def test_only_headings_get_a_period_and_existing_punctuation_is_preserved(self):
        paragraphs = ['PREMIÈRE PARTIE', 'Chapitre un', 'UN ÉCUEIL FUYANT',
                      'Déjà ponctué.', 'Une question ?', 'Une fin…', '« Un titre ! »',
                      'Un sous-titre :', '', '***', 'Le récit de 1866', 'IV']
        original = paragraphs[:]
        expected = ['PREMIÈRE PARTIE.', 'Chapitre un.', 'UN ÉCUEIL FUYANT.', *paragraphs[3:]]
        self.assertEqual(punctuate_chapter_headings(paragraphs, range(10)), expected)
        self.assertEqual(punctuate_chapter_headings(expected, range(10)), expected)
        self.assertEqual(punctuate_chapter_headings(paragraphs), paragraphs)
        self.assertEqual(paragraphs, original)

    def test_all_caps_headings_use_sentence_case_without_changing_prose_or_mixed_case(self):
        paragraphs = ['PREMIÈRE PARTIE', 'I', 'UN ÉCUEIL FUYANT', '« LE VOYAGE ! »',
                      'Le voyage de Jules Verne', 'LE RÉCIT EN 1866', 'Louis XIV arrive.', 'IV']
        original = paragraphs[:]
        expected = ['Première partie.', 'Chapitre un.', 'Un écueil fuyant.', '« Le voyage ! »',
                    'Le voyage de Jules Verne.', *paragraphs[5:]]
        self.assertEqual(prepare_chapter_headings(paragraphs, range(5)), expected)
        self.assertEqual(prepare_chapter_headings(expected, range(5)), expected)
        self.assertEqual(prepare_chapter_headings(paragraphs), paragraphs)
        self.assertEqual(paragraphs, original)

    def test_epub_heading_metadata_drives_spoken_text_for_each_chapter(self):
        from ebooklib import epub
        for mode in ('paragraph','sentence'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                book=epub.EpubBook()
                book.set_identifier('roman-heading-test')
                book.set_title('Livre test')
                book.set_language('fr')
                chapters=[]
                body='Un récit sur Louis XIV '+ 'x'*201
                for index,heading in enumerate(('I','CHAPITRE IV')):
                    item=epub.EpubHtml(title=heading,file_name=f'{index}.xhtml',lang='fr')
                    part='<h1>PREMIÈRE PARTIE</h1>' if index==0 else ''
                    item.content=f'<html><body>{part}<h2>{heading}</h2><h2>Le voyage</h2><p>{body}</p><p>IV</p></body></html>'
                    book.add_item(item)
                    chapters.append(item)
                book.toc=chapters
                book.spine=['nav']+chapters
                book.add_item(epub.EpubNcx())
                book.add_item(epub.EpubNav())
                source=root/'roman.epub'
                epub.write_epub(str(source),book)
                original=source.read_bytes()
                plan=read_book(source)
                self.assertEqual(plan.chapters[0].heading_indices,(0,1,2))
                self.assertEqual(plan.chapters[1].heading_indices,(0,1))
                self.assertEqual(plan.chapters[0].paragraphs[1],'I')
                voice=TitleVoice()
                destination=process_book(source,voice,output_dir=root/'out',audio_format='wav',
                                         segmentation=mode,message=lambda s:None)
                self.assertEqual('\n'.join(voice.texts), '\n'.join([
                    'Livre test.', 'Première partie.', 'Chapitre un.', 'Le voyage.', body, 'IV',
                    'Chapitre quatre.', 'Le voyage.', body, 'IV']))
                self.assertEqual(source.read_bytes(),original)
                self.assertEqual(next(destination.glob('*.epub')).read_bytes(),original)
                self.assertEqual(len(list(destination.glob('*.wav'))),2)


if __name__=='__main__':
    unittest.main()
