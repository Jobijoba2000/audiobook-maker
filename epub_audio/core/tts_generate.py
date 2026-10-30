"""GPU scheduling and ordered chapter assembly across the whole book."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from itertools import islice
from pathlib import Path
from dataclasses import replace
from epub_audio.workers import _initialize_worker, _synthesize_segment
from epub_audio.core.audio import assemble_wav
from epub_audio.core.text import plan_segments, prepare_chapter_segments
from epub_audio.config.defaults import (DEFAULT_WORKERS, PARAGRAPH_PAUSE_MS, SCENE_PAUSE_MS,
                                        INTRODUCTION_PAUSE_MS, MAX_SEGMENT_CHARACTERS, DEFAULT_SEGMENTATION)
from epub_audio.core.text import SpeechSegment
from epub_audio.core.tts_progress import BatchProgress
from epub_audio.io.fs import retry_windows_io


def create_synthesis_pool(engine, workers):
    """One GPU worker; batches reuse the same loaded Faster engine."""
    if getattr(engine, 'batch_size', 1) > 1:
        from epub_audio.core.tts_batch_queue import BatchedSynthesisPool
        return BatchedSynthesisPool(engine)
    return ThreadPoolExecutor(max_workers=1, initializer=_initialize_worker, initargs=(engine,))


def _queue_segments(pool, remaining, pending, workers, *, batch_progress=None):
    """Queue bounded lookahead windows; futures always stay in reading order."""
    batch_size = getattr(pool, 'batch_size', 1)
    while len(pending) < workers * 2:
        jobs = list(islice(remaining, getattr(pool, 'window_size', batch_size)))
        if not jobs:
            break
        if batch_size > 1:
            pending.extend(pool.submit_window(jobs, on_complete=batch_progress))
        else:
            if batch_progress is not None:
                batch_progress.start(jobs)
                def generate(job):
                    result = _synthesize_segment(*job)
                    batch_progress([(job[0], result)])
                    return result
                pending.append(pool.submit(generate, jobs[0]))
            else:
                pending.append(pool.submit(_synthesize_segment, *jobs[0]))


def render_chapter(engine, paragraphs, directory, destination, workers=DEFAULT_WORKERS, progress=print, executor=None,
                   paragraph_pause_ms=PARAGRAPH_PAUSE_MS, scene_pause_ms=SCENE_PAUSE_MS,
                   segmentation=DEFAULT_SEGMENTATION):
    """Keep a bounded window of work; consume completed segments in book order."""
    directory = Path(directory)
    directory.mkdir()
    segments = list(plan_segments(paragraphs, maximum=getattr(engine, 'max_segment_characters', MAX_SEGMENT_CHARACTERS), paragraph_pause_ms=paragraph_pause_ms,
                                  scene_pause_ms=scene_pause_ms, segmentation=segmentation))
    if not segments:
        raise RuntimeError('Le chapitre ne contient aucun texte à lire.')
    segments = prepare_chapter_segments(segments, segmentation=segmentation,
                                        maximum=getattr(engine, 'max_segment_characters', MAX_SEGMENT_CHARACTERS))
    pending = deque()
    remaining = iter((segment.text, directory / f'{index:06d}.wav')
                     for index, segment in enumerate(segments))
    pool = executor if executor is not None else create_synthesis_pool(engine, workers)
    tracker = BatchProgress(progress, [directory], [len(segments)])
    batch_progress = tracker if getattr(pool, 'batch_size', 1) > 1 else None

    def ordered_paths():
        for done in range(len(segments)):
            if batch_progress is None:
                tracker.start([(segments[done].text, directory / f'{done:06d}.wav')])
            _queue_segments(pool, remaining, pending, workers, batch_progress=batch_progress)
            try:
                path = pending.popleft().result()
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                preview = segments[done].text.replace('\n', ' ')[:120]
                raise RuntimeError(f'Échec du segment {done + 1}/{len(segments)} '
                                   f'« {preview} » : {detail}') from exc
            if batch_progress is None:
                tracker([(segments[done].text, path)])
            try:
                yield path
            finally:
                Path(path).unlink(missing_ok=True)

    completed = False
    try:
        assemble_wav(ordered_paths(), destination,
                     pauses_before_ms=[segment.pause_before_ms for segment in segments])
        completed = True
    finally:
        if not completed:
            cancel = getattr(engine, 'cancel_synthesis', None)
            if cancel is not None:
                cancel()
        for future in pending:
            future.cancel()
        if executor is None:
            pool.shutdown(wait=True, cancel_futures=True)


def chapter_segment_groups(engine, chapters, *, paragraph_pause_ms=PARAGRAPH_PAUSE_MS,
                           scene_pause_ms=SCENE_PAUSE_MS, introduction='', segmentation=DEFAULT_SEGMENTATION):
    """The same deterministic passage plan for synthesis and recovery checks."""
    groups = [list(plan_segments(paragraphs, maximum=getattr(engine, 'max_segment_characters', MAX_SEGMENT_CHARACTERS), paragraph_pause_ms=paragraph_pause_ms,
                                scene_pause_ms=scene_pause_ms, segmentation=segmentation)) for paragraphs in chapters]
    if any(not segments for segments in groups):
        raise RuntimeError('Un chapitre ne contient aucun texte à lire.')
    if introduction and groups:
        groups[0][0] = replace(groups[0][0], pause_before_ms=INTRODUCTION_PAUSE_MS,
                               break_before=segmentation == 'paragraph' or groups[0][0].break_before)
        introduction_segments = (list(plan_segments([introduction], segmentation=segmentation))
                                 if segmentation == 'sentence' else [SpeechSegment(introduction)])
        groups[0][0:0] = introduction_segments
    return [prepare_chapter_segments(segments, segmentation=segmentation,
                                       maximum=getattr(engine, 'max_segment_characters', MAX_SEGMENT_CHARACTERS))
              for segments in groups]


def render_chapters(engine, chapters, directory, workers=DEFAULT_WORKERS, progress=print, executor=None,
                    paragraph_pause_ms=PARAGRAPH_PAUSE_MS, scene_pause_ms=SCENE_PAUSE_MS,
                    introduction='', segmentation=DEFAULT_SEGMENTATION, checkpoint=None):
    """Yield chapters in reading order, retaining validated work across interruptions."""
    directory = Path(directory)
    groups = chapter_segment_groups(engine, chapters, paragraph_pause_ms=paragraph_pause_ms,
                                    scene_pause_ms=scene_pause_ms, introduction=introduction,
                                    segmentation=segmentation)
    pending = deque()
    tracker = BatchProgress(progress,
                            [directory / f'.segments-{index}' for index in range(1, len(groups) + 1)],
                            [len(segments) for segments in groups])
    batch_progress = tracker if getattr(engine, 'batch_size', 1) > 1 else None

    complete, cached = {}, {}
    if checkpoint is not None:
        from epub_audio.io.resume import segment_plan_hash
        fingerprint = segment_plan_hash(groups)
        if checkpoint.data.get('segment_plan_sha256', fingerprint) != fingerprint:
            raise ValueError('Le découpage des passages a changé depuis l’interruption ; reprise refusée.')
        checkpoint.data['segment_plan_sha256'] = fingerprint
        class SavedProgress:
            def __getattr__(self, name):
                return getattr(tracker, name)

            def __call__(self, jobs):
                for _, path in jobs:
                    checkpoint.segment_ready(path)
                tracker(jobs)

        batch_progress = SavedProgress()
        for chapter_index, segments in enumerate(groups, 1):
            wav = checkpoint.chapter_path(chapter_index)
            chunks = directory / f'.segments-{chapter_index}'
            chunks.mkdir(exist_ok=True)
            if wav is not None:
                complete[chapter_index] = wav
                tracker([(s.text, chunks / f'{i:06d}.wav') for i, s in enumerate(segments)])
            else:
                checkpoint.data['chapters'].pop(str(chapter_index), None)
                checkpoint.data['exports'].pop(str(chapter_index), None)
                for i, segment in enumerate(segments):
                    path = chunks / f'{i:06d}.wav'
                    if checkpoint.segment_path(path) is not None:
                        cached[path] = path
                        tracker([(segment.text, path)])
        checkpoint.save()

    def jobs():
        for chapter_index, segments in enumerate(groups, 1):
            if chapter_index in complete:
                continue
            chunks = directory / f'.segments-{chapter_index}'
            chunks.mkdir(exist_ok=True)
            for segment_index, segment in enumerate(segments):
                path = chunks / f'{segment_index:06d}.wav'
                if path not in cached:
                    yield segment.text, path

    remaining = iter(jobs())
    if checkpoint is not None and any(
            index not in complete and any(directory / f'.segments-{index}' / f'{i:06d}.wav' not in cached
                                           for i in range(len(segments)))
            for index, segments in enumerate(groups, 1)):
        prepare = getattr(engine, 'prepare', None)
        if prepare is not None:
            prepare()
    pool = executor if executor is not None else create_synthesis_pool(engine, workers)

    def ordered_paths(chapter_index, segments):
        for index, segment in enumerate(segments, 1):
            saved_path = directory / f'.segments-{chapter_index}' / f'{index - 1:06d}.wav'
            if saved_path in cached:
                yield saved_path
                continue
            if batch_progress is None:
                tracker.start([(segment.text, directory / f'.segments-{chapter_index}' / f'{index - 1:06d}.wav')])
            _queue_segments(pool, remaining, pending, workers, batch_progress=batch_progress)
            try:
                path = pending.popleft().result()
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                preview = segment.text.replace('\n', ' ')[:120]
                raise RuntimeError(f'Échec du chapitre {chapter_index}/{len(groups)}, '
                                   f'segment {index}/{len(segments)} « {preview} » : {detail}') from exc
            if batch_progress is None:
                tracker([(segment.text, path)])
            try:
                yield path
            finally:
                if checkpoint is None:
                    Path(path).unlink(missing_ok=True)

    completed = False
    try:
        for chapter_index, segments in enumerate(groups, 1):
            if chapter_index in complete:
                yield chapter_index, complete[chapter_index], directory / f'.segments-{chapter_index}'
                continue
            wav = directory / f'.chapter-{chapter_index}.wav'
            assembled = wav.with_suffix('.building.wav') if checkpoint is not None else wav
            assemble_wav(ordered_paths(chapter_index, segments), assembled,
                         pauses_before_ms=[segment.pause_before_ms for segment in segments])
            if checkpoint is not None:
                retry_windows_io(assembled.replace, wav)
                checkpoint.chapter_ready(chapter_index, wav)
                chunks = directory / f'.segments-{chapter_index}'
                for pattern in ('*.wav', '*.ready'):
                    for path in chunks.glob(pattern):
                        path.unlink()
            yield chapter_index, wav, directory / f'.segments-{chapter_index}'
        completed = True
    finally:
        if not completed:
            cancel = getattr(engine, 'cancel_synthesis', None)
            if cancel is not None:
                cancel()
        for future in pending:
            future.cancel()
        if executor is None:
            pool.shutdown(wait=True, cancel_futures=True)
