"""Discover local voice references without importing an inference library."""
from pathlib import Path


def reference_voices(directory, *, transcript_required=True):
    return [dict(id=path.stem, display_name='SIWIS — ' + path.stem, lang='fr-FR')
            for path in sorted(Path(directory).glob('*'))
            if path.is_file() and path.suffix.lower() in ('.wav', '.flac')
            and (not transcript_required or path.with_suffix('.txt').is_file())]
