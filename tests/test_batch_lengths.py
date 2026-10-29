"""Length grouping is bounded and preserves every passage, output slot and pause."""
from collections import deque,Counter
from pathlib import Path
from types import SimpleNamespace
import struct,tempfile,threading,unittest,wave
from concurrent.futures import CancelledError
from unittest.mock import patch
from epub_audio.core.tts_batch_queue import BatchedSynthesisPool,group_jobs_by_length
from epub_audio.core.tts_generate import _queue_segments,render_chapters


class QueueVoice:
    batch_size=2
    serial_gpu=True
    max_segment_characters=None

    def __init__(self):
        self.groups=[]

    def synthesize_batch(self,texts,paths):
        self.groups.append(list(texts))
        return list(paths)


class PcmVoice(QueueVoice):
    def synthesize_batch(self,texts,paths):
        self.groups.append(list(texts))
        for text,path in zip(texts,paths):
            identity=int(text[:3])+1
            with wave.open(str(path),'wb') as audio:
                audio.setparams((1,2,24000,0,'NONE','not compressed'))
                audio.writeframes(struct.pack('<h',identity*1000)*2400)
        return list(paths)


class LengthBatchingTests(unittest.TestCase):
    def test_similar_lengths_reduce_waiting_without_losing_jobs(self):
        lengths=[900,80,890,90,880,100,870,110]
        jobs=[('x'*length,Path(f'{index}.wav')) for index,length in enumerate(lengths)]
        batches=group_jobs_by_length(jobs,2)
        self.assertEqual([[index for index,_ in batch] for batch in batches],[[0,2],[1,3],[4,6],[5,7]])
        cost=sum(max(len(job[0]) for _,job in batch) for batch in batches)
        original_cost=sum(max(lengths[first:first+2]) for first in range(0,len(lengths),2))
        self.assertLess(cost,original_cost)
        self.assertEqual(sorted(index for batch in batches for index,_ in batch),list(range(8)))
        for size in (2,3,4,5,6,7,8):
            equal=[('same',Path(f'{index}.wav')) for index in range(13)]
            grouped=group_jobs_by_length(equal,size)
            self.assertEqual([index for batch in grouped for index,_ in batch],list(range(13)))
        self.assertEqual(group_jobs_by_length([],6),[])

    def test_futures_stay_in_source_order_including_identical_text_and_partial_groups(self):
        for size in (2,3,4,5,6,7,8):
            voice=QueueVoice()
            voice.batch_size=size
            jobs=[('same' if i%3==0 else 'x'*(50-i),Path(f'{i}.wav')) for i in range(13)]
            with self.subTest(size=size):
                pool=BatchedSynthesisPool(voice)
                try:
                    futures=pool.submit_window(jobs)
                    self.assertEqual([f.result(timeout=3) for f in futures],[job[1] for job in jobs])
                    self.assertEqual(Counter(text for group in voice.groups for text in group),Counter(job[0] for job in jobs))
                    self.assertTrue(all(1<=len(group)<=size for group in voice.groups))
                    self.assertEqual(pool.submit_window([]),[])
                finally:
                    pool.shutdown(wait=True,cancel_futures=True)

    def test_lookahead_reads_a_bounded_window_and_refills_without_losing_positions(self):
        voice=QueueVoice()
        voice.batch_size=6
        pool=BatchedSynthesisPool(voice)
        consumed=[]
        def jobs():
            for index in range(1000):
                consumed.append(index)
                yield ('x'*(index%97+1),Path(f'{index}.wav'))
        remaining=iter(jobs())
        pending=deque()
        try:
            _queue_segments(pool,remaining,pending,1)
            self.assertEqual(len(consumed),48)
            self.assertEqual(len(pending),48)
            for index in range(47):
                self.assertEqual(pending.popleft().result(timeout=3),Path(f'{index}.wav'))
                _queue_segments(pool,remaining,pending,1)
            self.assertEqual(len(consumed),96)
            self.assertEqual(len(pending),49)
            self.assertEqual([f.result(timeout=3) for f in pending],[Path(f'{index}.wav') for index in range(47,96)])
        finally:
            pool.shutdown(wait=True,cancel_futures=True)

    def test_errors_reach_window_futures_and_oversized_windows_are_rejected(self):
        voice=QueueVoice()
        voice.synthesize_batch=lambda *args:(_ for _ in ()).throw(OSError('disk failed'))
        pool=BatchedSynthesisPool(voice)
        try:
            jobs=[('x'*(index+1),Path(f'{index}.wav')) for index in range(7)]
            futures=pool.submit_window(jobs)
            for future in futures:
                with self.assertRaisesRegex(OSError,'disk failed'):
                    future.result(timeout=3)
            with self.assertRaises(ValueError):
                pool.submit_window(jobs*3)
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
        for invalid in (0,9,True,None):
            with self.assertRaises(ValueError):
                BatchedSynthesisPool(QueueVoice(),lookahead_batches=invalid)

    def test_cancellation_remains_attached_to_the_original_job(self):
        entered,release=threading.Event(),threading.Event()
        voice=QueueVoice()
        original=voice.synthesize_batch
        def blocked(texts,paths):
            entered.set()
            release.wait(3)
            return original(texts,paths)
        voice.synthesize_batch=blocked
        pool=BatchedSynthesisPool(voice)
        jobs=[(str(i),Path(f'{i}.wav')) for i in range(8)]
        try:
            futures=pool.submit_window(jobs)
            self.assertTrue(entered.wait(2))
            self.assertTrue(futures[7].cancel())
            release.set()
            for index,future in enumerate(futures[:-1]):
                self.assertEqual(future.result(timeout=3),Path(f'{index}.wav'))
            with self.assertRaises(CancelledError):
                futures[7].result(timeout=3)
            self.assertNotIn('7',[text for group in voice.groups for text in group])
        finally:
            release.set()
            pool.shutdown(wait=True,cancel_futures=True)

    def test_multiple_chapters_and_windows_preserve_exact_pcm_order_and_pauses(self):
        for size in (2,3,4,5,6,7,8):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as folder:
                voice=PcmVoice()
                voice.batch_size=size
                root=Path(folder)
                paragraphs=[f'{index:03} '+ 'x'*(500 if index%2==0 else 220) for index in range(27)]
                chapters=[paragraphs[first:first+9] for first in range(0,27,9)]
                tracks=list(render_chapters(voice,chapters,root,workers=1,progress=lambda s:None))
                self.assertEqual(len(tracks),3)
                self.assertEqual(Counter(text for group in voice.groups for text in group),Counter(paragraphs))
                for chapter_index,path,_ in tracks:
                    start=(chapter_index-1)*9
                    expected=(b'\0\0'*19200).join(struct.pack('<h',(i+1)*1000)*2400 for i in range(start,start+9))
                    with wave.open(str(path),'rb') as audio:
                        self.assertEqual(audio.readframes(audio.getnframes()),expected)
                self.assertFalse(list(root.glob('.segments-*/*.wav')))


if __name__=='__main__':
    unittest.main()
