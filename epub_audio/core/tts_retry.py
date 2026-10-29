"""Retry a failed paragraph in smaller pieces without losing any words."""
from pathlib import Path
import re
import tempfile
from epub_audio.core.audio import assemble_wav
from epub_audio.core.tts_activity import segment_label, text_preview


def _halves(text):
    middle = len(text) // 2
    boundaries = [m.end() for m in re.finditer(r'[.!?…][»”\"\']?\s+', text)
                  if len(text) // 4 <= m.end() <= len(text) * 3 // 4]
    if not boundaries:
        boundaries = [m.end() for m in re.finditer(r'\s+', text)
                      if 0 < m.end() < len(text)]
    if not boundaries:
        return None
    position = min(boundaries, key=lambda value: abs(value - middle))
    return text[:position].strip(), text[position:].strip()


def synthesize_with_fallback(engine, text, destination, *, announce_retry=True):
    destination = Path(destination)
    failures = 0
    announced = False
    message = getattr(engine, 'message', None)
    label = segment_label(destination)
    def speak(part, output, depth=0):
        nonlocal failures, announced
        try:
            return engine.synthesize(part, output)
        except RuntimeError as exc:
            failures += 1
            halves = _halves(part) if failures < 32 and depth < 8 and len(part) > 48 else None
            if not halves or not all(halves):
                raise
            if announce_retry and not announced and callable(message):
                message(f'{label} — réessai en morceaux : {text_preview(str(exc), 160)}')
                announced = True
            # A directory unique to this attempt; never reuse another segment's audio.
            with tempfile.TemporaryDirectory(prefix='.retry-', dir=output.parent) as folder:
                paths = []
                for index, half in enumerate(halves):
                    path = Path(folder) / f'{index}.wav'
                    speak(half, path, depth + 1)
                    paths.append(path)
                # Technical cuts add no silence, and all PCM frames are retained.
                assemble_wav(paths, output)
            return output

    return speak(text, destination)


def retry_segment(engine, text, destination, reason):
    """Print one line per retried segment; nested retries remain quiet."""
    label = segment_label(destination)
    message = getattr(engine, 'message', None)
    if callable(message):
        message(f'{label} — Réessai individuel : {text_preview(str(reason), 160)}')
    return synthesize_with_fallback(engine, text, destination, announce_retry=False)
