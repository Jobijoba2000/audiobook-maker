"""Source-based Windows filenames; no changes to the source EPUB."""
from pathlib import Path
import re
from epub_audio.helpers.time import format_duration_minutes_seconds


def chapter_filename(index, total, duration, audio_format):
    """Number audio files uniformly and append their duration."""
    width = len(str(total))
    return f'Chapitre {index:0{width}d} - {format_duration_minutes_seconds(duration)}.{audio_format}'


def safe_component(value):
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip(' .')
    if re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', value):
        value = '_' + value
    return value


def book_basename(source, output_root):
    name = safe_component(Path(source).stem)
    # The final path repeats this name in folder and EPUB; reserve space for
    # duration and collision suffixes, keeping ordinary Windows paths manageable.
    budget = max(24, min(140, (240 - len(str(Path(output_root).resolve())) - 35) // 2))
    return (name or 'Livre')[:budget].rstrip(' .')
