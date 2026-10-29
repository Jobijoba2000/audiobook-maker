"""Progress follows real EOS and validated audio, independently of assembly order."""
from pathlib import Path
import re,struct,tempfile,unittest,wave
from epub_audio.core.tts_generate import render_chapter,render_chapters
from epub_audio.core.tts_batch_queue import BatchedSynthesisPool
from epub_audio.core.tts_progress import BatchProgress


class ProgressVoice:
    serial_gpu=True
    max_segment_characters=None
    batch_size=6

    def __init__(self):
        self.generated=[]
        self.groups=[]
        self.fail_group=None

    def synthesize(self,text,path):
        with wave.open(str(path),'wb') as audio:
            audio.setparams((1,2,24000,0,'NONE','not compressed'))
            audio.writeframes(struct.pack('<h',1000)*2400)
        self.generated.append(Path(path))
        return Path(path)

    def synthesize_batch(self,texts,paths):
        self.groups.append(list(paths))
        if len(self.groups)==self.fail_group:
            raise OSError('failed batch')
        return [self.synthesize(text,path) for text,path in zip(texts,paths)]


def counters(messages):
    return [int(re.search(r'(\d+)/\d+ segments',message).group(1)) for message in messages]


class LiveVoice(ProgressVoice):
    individual_progress = True

    def __init__(self):
        super().__init__()
        self.eos = []

    def synthesize_batch(self, texts, paths, *, on_progress):
        self.groups.append(list(paths))
        # Endings arrive out of order, before any decoder/WAV output exists.
        for row in reversed(range(len(texts))):
            self.eos.append(Path(paths[row]))
            on_progress(row, 'generated')
        results = []
        for row, (text, path) in enumerate(zip(texts, paths)):
            results.append(self.synthesize(text, path))
            on_progress(row, 'ready')
        return results


class BatchProgressTests(unittest.TestCase):
    def test_engines_without_individual_notifications_wait_for_real_group_completion(self):
        for single in (True,False):
            with self.subTest(single=single),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                voice=ProgressVoice()
                messages=[]
                paragraphs=[f'{index:03} '+'x'*(210+(index*31)%200) for index in range(30)]
                def progress(message):
                    done=counters([message])[0]
                    # The callback cannot announce WAVs that do not exist yet.
                    self.assertEqual(done,len(voice.generated))
                    messages.append(message)
                if single:
                    render_chapter(voice,paragraphs,root/'segments',root/'chapter.wav',workers=1,progress=progress)
                else:
                    list(render_chapters(voice,[paragraphs],root,workers=1,progress=progress))
                self.assertEqual(counters(messages),[0,6,12,18,24,30])
                self.assertTrue(all('/30 segments' in message for message in messages))
                # The groups are not consecutive in the source; assembly order
                # must therefore not be used to count synthesis completion.
                self.assertNotEqual([path.name for path in voice.generated],
                                    [f'{index:06d}.wav' for index in range(30)])

    def test_incomplete_groups_report_the_actual_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            voice=ProgressVoice()
            messages=[]
            paragraphs=[f'{index:03} '+'x'*(210+index*20) for index in range(11)]
            list(render_chapters(voice,[paragraphs],root,workers=1,progress=messages.append))
            self.assertEqual([len(group) for group in voice.groups],[5,6])
            self.assertEqual(counters(messages),[0,5,11])

    def test_next_chapter_progress_is_visible_before_previous_chapter_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            voice=ProgressVoice()
            messages=[]
            paragraphs=[f'{index:03} '+'x'*(400 if index%2==0 else 220) for index in range(12)]
            list(render_chapters(voice,[paragraphs[:7],paragraphs[7:]],root,workers=1,progress=messages.append))
            self.assertEqual([(m.chapter,m.done,m.total) for m in messages],
                             [(1,0,7),(2,0,5),(1,4,7),(2,2,5),(1,7,7),(2,5,5)])
            self.assertIn('57 %', messages[2])

    def test_failed_groups_are_not_counted_as_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            voice=ProgressVoice()
            voice.fail_group=2
            messages=[]
            paragraphs=[f'{index:03} '+'x'*210 for index in range(12)]
            with self.assertRaisesRegex(RuntimeError,'failed batch'):
                list(render_chapters(voice,[paragraphs],root,workers=1,progress=messages.append))
            self.assertEqual(counters(messages),[0,6])
            self.assertEqual(len(voice.generated),6)

    def test_without_batching_progress_still_counts_one_by_one(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            voice=ProgressVoice()
            voice.batch_size=1
            messages=[]
            paragraphs=[f'{index:03} '+'x'*210 for index in range(5)]
            list(render_chapters(voice,[paragraphs],root,workers=1,progress=messages.append))
            self.assertEqual(counters(messages),[0,1,2,3,4,5])

    def test_real_individual_notifications_arrive_before_the_group_is_decoded(self):
        for single in (True, False):
            with self.subTest(single=single), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                voice = LiveVoice()
                messages = []
                early = []
                paragraphs = [f'{index:03} ' + 'x'*(210+(index*31)%200) for index in range(30)]
                def show(snapshot):
                    self.assertEqual(snapshot.done, len(voice.eos))
                    if snapshot.done > len(voice.generated):
                        early.append(snapshot.done)
                    if snapshot.validated:
                        self.assertEqual(len(voice.generated), 30)
                    messages.append(snapshot)
                if single:
                    render_chapter(voice, paragraphs, root/'segments', root/'chapter.wav',
                                   workers=1, progress=show)
                else:
                    list(render_chapters(voice, [paragraphs], root, workers=1, progress=show))
                self.assertEqual(counters(messages), list(range(31)) + [30])
                self.assertIn(1, early)
                self.assertTrue(messages[-1].validated)
                self.assertIn('(validation audio)', messages[-2])

    def test_retries_and_duplicate_notifications_do_not_inflate_counts(self):
        messages = []
        tracker = BatchProgress(messages.append, [Path('chapter')], [3])
        jobs = [('text', Path('chapter')/f'{index}.wav') for index in range(3)]
        tracker.start(jobs)
        tracker.generated(jobs[:2])
        tracker.generated(jobs[1:2])
        self.assertEqual(counters(messages), [0, 2])
        tracker.retry(jobs[1:2])
        self.assertEqual(messages[-1].done, 1)
        self.assertEqual(messages[-1].retrying, 1)
        tracker(jobs[:1])
        tracker(jobs[1:2])
        tracker.generated(jobs[1:2])
        self.assertEqual(messages[-1].done, 2)
        self.assertFalse(messages[-1].validated)
        tracker(jobs[2:])
        tracker(jobs)
        tracker.retry(jobs[:1])
        self.assertEqual(counters(messages), [0, 2, 1, 2, 3])
        self.assertTrue(messages[-1].validated)
        self.assertEqual(messages[-1].retrying, 0)

    def test_callback_errors_reach_futures_without_blocking_assembly(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            voice=ProgressVoice()
            pool=BatchedSynthesisPool(voice)
            def fail(jobs):
                raise ValueError('progress failed')
            try:
                futures=pool.submit_window([('One',root/'1.wav'),('Two',root/'2.wav')],on_complete=fail)
                for future in futures:
                    with self.assertRaisesRegex(ValueError,'progress failed'):
                        future.result(timeout=3)
            finally:
                pool.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':
    unittest.main()
