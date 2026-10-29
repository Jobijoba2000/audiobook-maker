"""Live elapsed time for the synthesis phase, separate from loading and export."""
import threading
from pathlib import Path
import time


def elapsed_label(seconds):
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class ChapterProgress(str):
    """A printable chapter snapshot, retaining metadata for terminal redraws."""
    def __new__(cls, chapter, chapters, done, total, *, validated=False, retrying=0, elapsed=None):
        percent = 100 * done // total if total else 0
        row = f'Chapitre {chapter}/{chapters} - {done}/{total} segments - {percent} %'
        if retrying:
            row += ' (réessai)'
        elif done == total and not validated:
            row += ' (validation audio)'
        text = row if elapsed is None else f'{row} | Durée TTS : {elapsed_label(elapsed)}'
        value = super().__new__(cls, text)
        value.chapter, value.chapters = chapter, chapters
        value.done, value.total = done, total
        value.validated, value.retrying, value.elapsed = validated, retrying, elapsed
        value.row_text = row
        return value

    def with_elapsed(self, elapsed):
        return type(self)(self.chapter, self.chapters, self.done, self.total,
                          validated=self.validated, retrying=self.retrying, elapsed=elapsed)


class TtsProgress:
    def __init__(self, progress, message, *, interval=1.0, clock=None):
        self._progress = progress
        self._message = message
        self._interval = interval
        self._clock = clock if clock is not None else time.perf_counter
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._started = None
        self._finished = None
        self._text = 'Génération TTS'
        self.elapsed = 0.0

    def _draw(self):
        end = self._finished if self._finished is not None else self._clock()
        self.elapsed = max(0.0, end - self._started)
        if isinstance(self._text, ChapterProgress):
            self._progress(self._text.with_elapsed(self.elapsed))
        else:
            self._progress(f'{self._text.rstrip()} | Durée TTS : {elapsed_label(self.elapsed)}')

    def progress(self, text):
        with self._lock:
            self._text = text
            self._draw()

    def _tick(self):
        while not self._stop.wait(self._interval):
            with self._lock:
                if not self._stop.is_set():
                    self._draw()

    def __enter__(self):
        self._started = self._clock()
        with self._lock:
            self._draw()
        self._thread = threading.Thread(target=self._tick, name='tts-duration', daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._finished = self._clock()
        self._stop.set()
        self._thread.join()
        with self._lock:
            self._draw()
        status = ' (interrompue)' if exc_type is not None else ''
        self._message(f'Durée génération TTS : {elapsed_label(self.elapsed)}{status}.')
        return False


class BatchProgress:
    """Track actual generation and audio validation independently of assembly.

    EOS events arrive while other rows are still generating. A failed attempt
    is withdrawn until its retry succeeds; saving an already counted row never
    counts it twice. Chapters may progress concurrently in the lookahead window.
    """
    def __init__(self, progress, directories, totals, *, single=False):
        self.progress = progress
        self.directories = list(map(Path, directories))
        self.locations = {path: index for index, path in enumerate(self.directories)}
        self.totals = list(totals)
        self.finished = [set() for _ in self.totals]
        self.saved = [set() for _ in self.totals]
        self.retries = [set() for _ in self.totals]
        self.displayed = {}
        self.started = -1
        self._lock = threading.RLock()

    def _emit(self, index):
        state = (len(self.finished[index]), len(self.saved[index]) == self.totals[index],
                 len(self.retries[index]))
        if self.displayed.get(index) != state:
            self.progress(ChapterProgress(index + 1, len(self.totals), state[0],
                          self.totals[index], validated=state[1], retrying=state[2]))
            self.displayed[index] = state

    def start(self, jobs):
        with self._lock:
            last = max((self.locations[Path(path).parent] for _, path in jobs), default=-1)
            for index in range(self.started + 1, last + 1):
                self._emit(index)
            self.started = max(self.started, last)

    def _update(self, jobs, state):
        with self._lock:
            changed = set()
            for _, path in jobs:
                path = Path(path)
                index = self.locations[path.parent]
                if state == 'retry':
                    if path not in self.saved[index]:
                        self.finished[index].discard(path)
                        self.retries[index].add(path)
                else:
                    self.finished[index].add(path)
                    if state == 'ready':
                        self.saved[index].add(path)
                        self.retries[index].discard(path)
                changed.add(index)
            for index in sorted(changed):
                self._emit(index)

    def generated(self, jobs):
        self._update(jobs, 'generated')

    def retry(self, jobs):
        self._update(jobs, 'retry')

    def __call__(self, jobs):
        self._update(jobs, 'ready')
