"""Lossless PCM assembly and minimum pauses at reading boundaries."""
from pathlib import Path
from array import array
from itertools import chain
import sys
import wave


def _edge_silence_frames(source, limit, trailing=False):
    """Count quiet edge frames (-40 dBFS), without trimming any input audio."""
    if source.getsampwidth() != 2:
        raise RuntimeError('Les pauses nécessitent un WAV PCM 16 bits.')
    count = min(limit, source.getnframes())
    source.setpos(source.getnframes() - count if trailing else 0)
    samples = array('h', source.readframes(count))
    if sys.byteorder != 'little':
        samples.byteswap()
    if trailing:
        samples.reverse()
    quiet = 0
    for sample in samples:
        if abs(sample) > 327:
            break
        quiet += 1
    return quiet // source.getnchannels()


def assemble_wav(paths, destination, *, pauses_before_ms=None):
    """Stream PCM frames into one valid WAV, instead of concatenating RIFF headers."""
    iterator = iter(paths)
    try:
        first = next(iterator)
    except StopIteration:
        raise RuntimeError('Aucun segment audio à assembler.') from None
    params = None
    pause_values = list(pauses_before_ms) if pauses_before_ms is not None else None
    pauses = iter(pause_values) if pause_values is not None else None
    pause_limit_ms = max(pause_values, default=0) if pause_values is not None else 0
    previous_trailing = 0
    first_segment = True
    with wave.open(str(destination), 'wb') as output:
        for path in chain((first,), iterator):
            with wave.open(str(path), 'rb') as source:
                current = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                if params is None:
                    params = current
                    output.setparams((*current, 0, 'NONE', 'not compressed'))
                elif params != current:
                    raise RuntimeError('Les segments WAV ont des formats différents.')
                if not source.getnframes():
                    raise RuntimeError('Un segment WAV est vide.')
                if pauses is not None:
                    requested = int(max(0, next(pauses)) * source.getframerate() / 1000)
                    # Only add the missing portion of the boundary silence.
                    # Existing pauses and every speech sample remain untouched.
                    if not first_segment and requested:
                        leading = _edge_silence_frames(source, requested)
                        missing = max(0, requested - previous_trailing - leading)
                        output.writeframesraw(b'\0' * missing * current[0] * current[1])
                    limit = int(source.getframerate() * pause_limit_ms / 1000)
                    previous_trailing = _edge_silence_frames(source, limit, trailing=True)
                    source.rewind()
                while data := source.readframes(65536):
                    output.writeframesraw(data)
                first_segment = False
        if params is None:
            raise RuntimeError('Aucun segment audio à assembler.')
