"""Sentence prefixes and paragraph context preserve book order and PCM pauses."""
from pathlib import Path
import struct
from collections import Counter
import tempfile
import unittest
import wave
from epub_audio.core.text import SpeechSegment, merge_chapter_start
from epub_audio.core.tts_generate import render_chapter, render_chapters


class PrefixVoice:
    serial_gpu = True
    max_segment_characters = None

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.texts = []
        self.sample_ids = {}

    def synthesize(self, text, destination):
        self.texts.append(text)
        with wave.open(str(destination), 'wb') as audio:
            audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            audio.writeframes(struct.pack('<h', self.sample_ids.get(text,len(self.texts))*1000)*2400)
        return Path(destination)

    def synthesize_batch(self, texts, destinations):
        return [self.synthesize(text, path) for text, path in zip(texts, destinations)]


class ChapterStartTests(unittest.TestCase):
    def test_initial_four_segments_merge_and_keep_first_pause_and_remaining_objects(self):
        texts=['Vingt Mille Lieues sous les Mers. Un livre de Jules Verne.',
               'PREMIÈRE PARTIE','I','UN ÉCUEIL FUYANT']
        original=[SpeechSegment(text, index*800) for index,text in enumerate(texts)]
        original.extend([SpeechSegment('x'*565,800),SpeechSegment('Suite.',1500)])
        merged=merge_chapter_start(original)
        self.assertEqual(merged[0].text,'\n'.join(texts))
        self.assertEqual(len(merged[0].text),93)
        self.assertEqual(merged[0].pause_before_ms,0)
        self.assertIs(merged[1],original[4])
        self.assertIs(merged[2],original[5])
        self.assertEqual(len(original),6)

    def test_200_is_inclusive_and_separator_and_unicode_characters_are_counted(self):
        original=[SpeechSegment('é'*98,17),SpeechSegment('x'*101,800),SpeechSegment('Z',1500)]
        merged=merge_chapter_start(original)
        self.assertEqual(len(merged),2)
        self.assertEqual(len(merged[0].text),200)
        self.assertEqual(merged[0].pause_before_ms,17)
        self.assertIs(merged[1],original[2])

    def test_stops_at_first_overflow_without_skipping_or_regrouping_later_segments(self):
        original=[SpeechSegment('a'*198),SpeechSegment('bb',800),SpeechSegment('c',800),SpeechSegment('d',1500)]
        merged=merge_chapter_start(original)
        self.assertEqual(merged,original)
        self.assertTrue(all(a is b for a,b in zip(merged,original)))

    def test_oversized_first_segment_is_left_whole_and_empty_and_single_inputs_work(self):
        self.assertEqual(merge_chapter_start([]),[])
        for length in (1,200,201,1000):
            first=SpeechSegment('x'*length)
            self.assertEqual(merge_chapter_start([first]),[first])
        original=[SpeechSegment('x'*201),SpeechSegment('I'),SpeechSegment('Titre')]
        self.assertEqual(merge_chapter_start(original),original)

    def test_each_chapter_merges_independently_in_both_modes_and_with_batching(self):
        for mode in ('paragraph','sentence'):
            for batch_size in (1,3):
                with self.subTest(mode=mode,batch_size=batch_size), tempfile.TemporaryDirectory() as folder:
                    voice=PrefixVoice(batch_size)
                    root=Path(folder)
                    chapters=[['I','Titre','x'*201,'Une phrase. Encore.'],['II','Suite','y'*201,'Fin.']]
                    if mode=='paragraph':
                        first=['Annonce.','I\nTitre\n'+'x'*201+'\nUne phrase. Encore.']
                        second=['II\nSuite\n'+'y'*201+'\nFin.']
                    else:
                        first=['Annonce.\nI\nTitre','x'*201,'Une phrase.','Encore.']
                        second=['II\nSuite','y'*201,'Fin.']
                    voice.sample_ids={text:index for index,text in enumerate(first+second,1)}
                    tracks=list(render_chapters(voice,chapters,root,workers=1,progress=lambda s:None,
                        introduction='Annonce.',segmentation=mode))
                    self.assertEqual(Counter(voice.texts),Counter(first+second))
                    pcm=lambda n:struct.pack('<h',n*1000)*2400
                    if mode=='paragraph':
                        first_frames=pcm(1)+b'\0\0'*48000+pcm(2)
                        second_frames=pcm(3)
                    else:
                        first_frames=pcm(1)+b'\0\0'*19200+pcm(2)+b'\0\0'*19200+pcm(3)+pcm(4)
                        offset=len(first)
                        second_frames=pcm(offset+1)+b'\0\0'*19200+pcm(offset+2)+b'\0\0'*19200+pcm(offset+3)
                    for (_,path,_),expected in zip(tracks,[first_frames,second_frames]):
                        with wave.open(str(path),'rb') as audio:
                            self.assertEqual(audio.readframes(audio.getnframes()),expected)

    def test_single_chapter_entry_point_attaches_short_paragraphs_to_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            voice=PrefixVoice()
            render_chapter(voice,['I','Titre','x'*201,'Fin','Suite'],root/'segments',root/'chapter.wav',
                           workers=1,progress=lambda s:None)
            self.assertEqual(voice.texts,['I\nTitre\n'+'x'*201+'\nFin\nSuite'])


if __name__=='__main__':
    unittest.main()
