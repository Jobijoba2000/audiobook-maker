"""Paragraph boundaries, scene breaks and bounded synthesis segments."""
from dataclasses import dataclass, replace
import re
from epub_audio.config.defaults import (SCENE_BREAK, PARAGRAPH_PAUSE_MS, SCENE_PAUSE_MS,
                                        MAX_SEGMENT_CHARACTERS, SEGMENTATION_MODES, DEFAULT_SEGMENTATION,
                                        CHAPTER_START_MAX_CHARACTERS, MIN_PARAGRAPH_CHARACTERS,
                                        MAX_PARAGRAPH_CHARACTERS, PARAGRAPH_TARGET_MIN, PARAGRAPH_TARGET_MAX)

_NUMBER_ABBREVIATION = re.compile(r'(?<!\w)n[°º][^\S\r\n]*', re.IGNORECASE)
_INTERNET_URL = re.compile(r'''(?<![\w@])(?:https?://|ftps?://|www\.)[^\s<>«»“”"]+''', re.IGNORECASE)


def _spoken_url(match):
    url = match.group()
    end = len(url)
    brackets = {')': '(', ']': '[', '}': '{'}
    while end:
        last = url[end - 1]
        if (last in ".,;:!?…'’"
                or (last in brackets and url[:end].count(last) > url[:end].count(brackets[last]))):
            end -= 1
        else:
            break
    # Keep enclosing quotes/brackets; the spoken label already ends the sentence.
    return 'Lien internet.' + url[end:].lstrip('.,;:!?…')


def prepare_spoken_text(text):
    """Prepare narration URLs and number abbreviations before splitting."""
    text = _INTERNET_URL.sub(_spoken_url, text)
    return _NUMBER_ABBREVIATION.sub('numéro ', text)


def has_spoken_text(text):
    return any(character.isalnum() for character in text)


def is_scene_break(text):
    compact = ''.join(text.split())
    return bool(compact) and all(character in '*⁂' for character in compact)


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    pause_before_ms: int = 0
    break_before: bool = False


def merge_chapter_start(segments, maximum=CHAPTER_START_MAX_CHARACTERS):
    """Greedily merge only the first contiguous group; count newline separators."""
    segments = list(segments)
    if not segments:
        return segments
    merged = segments[0]
    consumed = 1
    for segment in segments[1:]:
        if segment.break_before:
            break
        candidate = merged.text + '\n' + segment.text
        if len(candidate) > maximum:
            break
        merged = replace(merged, text=candidate)
        consumed += 1
    return [merged, *segments[consumed:]]


# Keep common French titles and name initials with the words that follow.
_ABBREVIATIONS = {'m', 'mm', 'mme', 'mmes', 'mlle', 'mlles', 'mr', 'mrs', 'ms',
                  'dr', 'drs', 'pr', 'prof', 'me', 'mes', 'st', 'ste', 'sts', 'stes',
                  'p', 'pp', 'vol', 't', 'chap', 'fig', 'art', 'av', 'apr', 'env', 'cf'}
_SENTENCE_END = re.compile(r"""[.!?…]+(?:\s*[»”"’')\]])*(?=\s|$)""")


def split_sentences(text):
    """Keep punctuation, French dialogue endings, abbreviations and decimals."""
    start = 0
    for match in _SENTENCE_END.finditer(text):
        following = text[match.end():].lstrip()
        next_word = following.lstrip('«“"([—–- ')
        # A lowercase continuation usually belongs to an ellipsis or dialogue tag.
        if next_word and next_word[0].islower():
            continue
        if text[match.start()] == '.':
            word = re.search(r'(\w+)$', text[:match.start()])
            if word and (word[0].casefold() in _ABBREVIATIONS
                         or (len(word[0]) == 1 and word[0].isalpha())):
                continue
        sentence = text[start:match.end()].strip()
        if has_spoken_text(sentence):
            yield sentence
            start = match.end()
    remainder = text[start:].strip()
    if has_spoken_text(remainder):
        yield remainder


def _paragraph_chunks(segment, maximum):
    """Prefer whole sentences and 300–600 characters; avoid tiny remainder chunks."""
    if len(segment.text) <= maximum:
        return [segment]
    chunks = []
    current = ''
    for sentence in split_sentences(segment.text):
        candidate = current + ' ' + sentence if current else sentence
        if (current and len(current) >= MIN_PARAGRAPH_CHARACTERS
                and (len(candidate) > maximum
                     or (len(current) >= min(PARAGRAPH_TARGET_MIN, maximum)
                         and len(candidate) > min(PARAGRAPH_TARGET_MAX, maximum)))):
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        if chunks and len(current) < MIN_PARAGRAPH_CHARACTERS:
            chunks[-1] += ' ' + current
        else:
            chunks.append(current)
    # The first chunk retains the paragraph/scene pause. Continuations add none.
    return [replace(segment, text=text) if index == 0 else SpeechSegment(text)
            for index, text in enumerate(chunks)]


def prepare_chapter_segments(segments, *, segmentation=DEFAULT_SEGMENTATION, maximum=None):
    """Keep paragraph context while bounding long passages at sentence boundaries."""
    if segmentation == 'sentence':
        return merge_chapter_start(segments)
    limit = min(maximum, MAX_PARAGRAPH_CHARACTERS) if maximum is not None else MAX_PARAGRAPH_CHARACTERS
    prepared = [part for segment in segments for part in _paragraph_chunks(segment, limit)]
    index = 0
    while index < len(prepared):
        segment = prepared[index]
        if len(segment.text) >= MIN_PARAGRAPH_CHARACTERS:
            index += 1
            continue
        neighbors = []
        if index > 0 and not segment.break_before:
            neighbors.append(index - 1)
        if index + 1 < len(prepared) and not prepared[index + 1].break_before:
            neighbors.append(index + 1)
        # Prefer the preceding context when both neighbors fit. Otherwise use
        # the next, or rebalance the smaller combined passage by whole sentences.
        def neighbor_priority(other):
            combined = len(prepared[other].text) + 1 + len(segment.text)
            return combined > limit, combined if combined > limit else int(other > index)

        neighbors.sort(key=neighbor_priority)
        merged = False
        for other in neighbors:
            first, last = sorted((index, other))
            left, right = prepared[first], prepared[last]
            text = left.text + '\n' + right.text
            # An explicitly configured hard character limit still takes priority.
            if maximum is not None and len(text) > maximum:
                continue
            joined = replace(left, text=text)
            parts = _paragraph_chunks(joined, limit)
            # Sentence boundaries can make 800 a soft limit: keep an indivisible
            # sentence, or its very short neighbor, rather than breaking words.
            prepared[first:last + 1] = parts
            index = first
            merged = True
            break
        if not merged:
            # An isolated scene/chapter can legitimately contain only a few words.
            index += 1
    return prepared


def plan_segments(paragraphs, maximum=MAX_SEGMENT_CHARACTERS, paragraph_pause_ms=PARAGRAPH_PAUSE_MS,
                  scene_pause_ms=SCENE_PAUSE_MS, segmentation=DEFAULT_SEGMENTATION):
    """Preserve real paragraph boundaries independently of synthesis chunks."""
    if segmentation not in SEGMENTATION_MODES:
        raise ValueError('Découpage des segments inconnu : ' + str(segmentation))
    spoken = False
    pause = 0
    scene_break = False
    for paragraph in paragraphs:
        if is_scene_break(paragraph):
            pause = max(pause, scene_pause_ms)
            scene_break = True
            continue
        units = split_sentences(paragraph) if segmentation == 'sentence' else [paragraph]
        parts = list(split_text(units, maximum=maximum))
        if not parts:
            continue
        for index, text in enumerate(parts):
            before = max(paragraph_pause_ms, pause) if spoken and index == 0 else 0
            yield SpeechSegment(text, before, break_before=scene_break and index == 0)
        spoken = True
        pause = 0
        scene_break = False


def split_text(paragraphs, maximum=MAX_SEGMENT_CHARACTERS):
    """Split long paragraphs without dropping or reordering any words."""
    if maximum is not None and maximum < 100:
        raise ValueError('La taille des segments doit être au moins de 100 caractères.')
    for paragraph in paragraphs:
        text = re.sub(r'(?<!\w)_{3,}(?!\w)', '', paragraph).strip()
        if not has_spoken_text(text):
            continue
        if maximum is None:
            yield text
            continue
        while len(text) > maximum:
            limit = text.rfind(' ', 0, maximum + 1)
            if limit < maximum // 2:
                limit = maximum
            else:
                # Prefer a sentence boundary when one is reasonably close.
                stops = list(re.finditer(r'[.!?…][»”\"\']?\s+', text[:limit + 1]))
                if stops and stops[-1].end() >= maximum // 2:
                    limit = stops[-1].end()
            part = text[:limit].strip()
            if has_spoken_text(part):
                yield part
            text = text[limit:].lstrip()
        if has_spoken_text(text):
            yield text
