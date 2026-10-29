"""Entry points for the single Faster synthesis worker."""
from epub_audio.core.tts_retry import synthesize_with_fallback
_WORKER_ENGINE = None


def _initialize_worker(engine):
    global _WORKER_ENGINE
    _WORKER_ENGINE = engine
    initialize = getattr(engine, 'initialize_worker', None)
    if initialize is not None:
        initialize()


def _synthesize_segment(text, destination):
    if _WORKER_ENGINE is None:
        raise RuntimeError('Le moteur du processus vocal n’a pas été initialisé.')
    return synthesize_with_fallback(_WORKER_ENGINE, text, destination)
