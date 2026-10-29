"""Compact identifiers and excerpts for retry messages."""
from pathlib import Path
import re



def segment_label(destination, *, compact=False):
    path = Path(destination)
    chapter = re.fullmatch(r'\.segments-(\d+)', path.parent.name)
    segment = str(int(path.stem) + 1) if path.stem.isascii() and path.stem.isdigit() else path.stem
    if chapter:
        return f'C{chapter[1]}/S{segment}' if compact else f'Chapitre {chapter[1]}, segment {segment}'
    return f'Segment {segment}'


def text_preview(text, maximum=100):
    text = ' '.join(text.split())
    return text if len(text) <= maximum else text[:maximum - 1] + '…'
