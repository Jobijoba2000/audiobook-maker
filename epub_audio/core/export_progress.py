"""Report actual parallel chapter exports, independently of TTS progress."""
from contextlib import contextmanager
import threading


class ExportProgress(str):
    def __new__(cls, completed, total, active, workers, audio_format, failed=0):
        pending = total - completed - active - failed
        stage = 'Finalisation WAV' if audio_format == 'wav' else f'Encodage {audio_format.upper()}'
        text = (f'{stage} : {completed}/{total} terminés | {active} en cours | '
                f'{pending} en attente ({workers} workers)')
        if failed:
            text += f' | {failed} échec(s)'
        result = super().__new__(cls, text)
        result.completed, result.total, result.active = completed, total, active
        result.workers, result.pending, result.failed = workers, pending, failed
        return result


class ExportTracker:
    def __init__(self, total, workers, audio_format, progress):
        self.total, self.workers = total, workers
        self.audio_format, self.progress = audio_format, progress
        self.completed = self.active = self.failed = 0
        self._lock = threading.RLock()

    def show(self):
        with self._lock:
            self.progress(ExportProgress(self.completed, self.total, self.active,
                                         self.workers, self.audio_format, self.failed))

    @contextmanager
    def task(self):
        with self._lock:
            self.active += 1
            self.show()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            with self._lock:
                self.active -= 1
                if succeeded:
                    self.completed += 1
                else:
                    self.failed += 1
                self.show()
