"""Durable book checkpoints and conservative recovery of older work folders."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import wave

from epub_audio.core.models import BookMetadata, BookPlan, Chapter
from epub_audio.io.llm_chapters import _write_json

STATE_FILE = 'progression.json'


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid_wav(path):
    try:
        with wave.open(str(path), 'rb') as audio:
            frames = audio.getnframes()
            if frames <= 0:
                return False
            # A killed writer can leave a header claiming more PCM than exists.
            audio.setpos(frames - 1)
            return len(audio.readframes(1)) == audio.getnchannels() * audio.getsampwidth()
    except (OSError, EOFError, wave.Error):
        return False


def engine_options(engine):
    return {key: getattr(engine, key, default) for key, default in (
        ('voice', None), ('batching', False), ('batch_size', 2),
        ('continuous_batching', False), ('shrinking_batching', True))}


def plan_data(plan):
    return {key: value for key, value in asdict(replace(plan, cover=None)).items() if key != 'cover'}


def segment_plan_hash(groups):
    data = [[(s.text, s.pause_before_ms) for s in group] for group in groups]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode('utf-8')).hexdigest()


class BookCheckpoint:
    def __init__(self, directory, data):
        self.directory = Path(directory)
        self.data = data

    @classmethod
    def create(cls, directory, source, plan, options, engine, master_root):
        state = cls(directory, {'version': 1, 'source': str(source), 'source_sha256': file_hash(source),
                               'plan': plan_data(plan), 'options': asdict(options),
                               'engine': engine_options(engine),
                               'engine_settings': getattr(engine, 'settings', None),
                               'master_root': str(master_root), 'chapters': {}, 'exports': {}})
        state.save()
        return state

    @classmethod
    def load(cls, directory):
        data = json.loads((Path(directory) / STATE_FILE).read_text(encoding='utf-8'))
        if data.get('version') != 1:
            raise ValueError('Version de sauvegarde de progression inconnue.')
        return cls(directory, data)

    def save(self):
        _write_json(self.directory / STATE_FILE, self.data)

    def plan(self):
        data = self.data['plan']
        return BookPlan(**{**data, 'chapters': [Chapter(**c) for c in data['chapters']],
                           'metadata': BookMetadata(**data['metadata'])})

    def validate(self, source, options, engine):
        if file_hash(source) != self.data['source_sha256']:
            raise ValueError('L’EPUB a changé depuis l’interruption ; reprise refusée.')
        if asdict(options) != self.data['options'] or engine_options(engine) != self.data['engine']:
            raise ValueError('La reprise nécessite les réglages enregistrés dans progression.json.')
        if getattr(engine, 'settings', None) != self.data.get('engine_settings'):
            raise ValueError('La configuration du moteur a changé depuis l’interruption.')

    def chapter_path(self, index):
        candidates = [self.directory / f'.chapter-{index}.wav']
        master_dir = self.directory / 'Master WAV'
        if not master_dir.exists() and self.data.get('master_destination'):
            master_dir = Path(self.data['master_destination'])
        candidates.extend(p for p in master_dir.glob('Chapitre *.wav')
                          if (match := re.match(r'^Chapitre (\d+) - ', p.name)) and int(match[1]) == index)
        expected = self.data['chapters'].get(str(index))
        return next((p for p in candidates if expected and valid_wav(p) and file_hash(p) == expected), None)

    def chapter_ready(self, index, path):
        if not valid_wav(path):
            raise RuntimeError('Un chapitre WAV est incomplet.')
        self.data['chapters'][str(index)] = file_hash(path)
        self.save()

    def segment_ready(self, path):
        path = Path(path)
        if not valid_wav(path):
            raise RuntimeError('Un segment WAV est incomplet.')
        _write_json(path.with_suffix('.ready'), {'sha256': file_hash(path)})

    def segment_path(self, path):
        try:
            saved = json.loads(path.with_suffix('.ready').read_text(encoding='utf-8'))
            return path if valid_wav(path) and saved['sha256'] == file_hash(path) else None
        except (OSError, ValueError, KeyError):
            return None

    def exported(self, index):
        entry = self.data['exports'].get(str(index))
        if entry:
            path = self.directory / entry['name']
            if path.is_file() and file_hash(path) == entry['sha256']:
                return entry
        return None


def recover_legacy_plan(directory):
    """Rebuild the exact old chapter boundaries without calling the LLM again."""
    from epub_audio.io.epub import read_book
    report = json.loads((Path(directory) / 'decoupage.json').read_text(encoding='utf-8'))
    source = Path(report['source'])
    plan = read_book(source)
    texts = [text for chapter in plan.chapters for text in chapter.paragraphs]
    digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode('utf-8')).hexdigest()
    if digest != report['text_sha256']:
        raise ValueError('Le texte de l’EPUB ne correspond plus au découpage sauvegardé.')
    headings, offset = set(), 0
    for chapter in plan.chapters:
        headings.update(offset + i for i in chapter.heading_indices)
        offset += len(chapter.paragraphs)
    analysis = report.get('analysis', {})
    headings.update(d['paragraph'] for d in analysis.get('decisions', [])
                    if d['kind'] != 'other' and d['confidence'] >= 0.85)
    chapters, position = [], 0
    for entry in report['chapters']:
        start, count = entry['start_paragraph'], entry['paragraphs']
        if start != position or count <= 0 or start + count > len(texts):
            raise ValueError('Le découpage sauvegardé est incomplet.')
        chapters.append(Chapter(entry['title'], texts[start:start + count],
                                tuple(i - start for i in sorted(headings) if start <= i < start + count)))
        position += count
    if position != len(texts):
        raise ValueError('Le découpage sauvegardé ne couvre pas tout le livre.')
    return source, replace(plan, chapters=chapters, mode=report['mode'],
                           fallback_reason=report.get('fallback_reason', ''), chapter_analysis=analysis)


def legacy_completed(directory):
    """The last assembled WAV may be partial, even with a valid WAV header."""
    directory = Path(directory)
    indices = sorted(int(m[1]) for p in directory.glob('.chapter-*.wav')
                     if (m := re.fullmatch(r'\.chapter-(\d+)\.wav', p.name)))
    completed = []
    for index in indices[:-1]:
        chunks = directory / f'.segments-{index}'
        if index != len(completed) + 1 or not chunks.is_dir() or any(chunks.iterdir()):
            break
        if not valid_wav(directory / f'.chapter-{index}.wav'):
            break
        completed.append(index)
    return completed


def import_legacy(directory, options, engine, master_root, *, journal, message=print):
    """Require execution evidence and PCM coverage before certifying old chapters."""
    from collections import defaultdict, deque
    from datetime import datetime
    from epub_audio.core.chapter_titles import prepare_chapter_headings
    from epub_audio.core.text import prepare_spoken_text
    from epub_audio.core.tts_generate import chapter_segment_groups
    source, plan = recover_legacy_plan(directory)
    chapters = [[prepare_spoken_text(t) for t in prepare_chapter_headings(c.paragraphs, c.heading_indices)]
                for c in plan.chapters]
    introduction = prepare_spoken_text(plan.metadata.introduction())
    if options.convert_numbers:
        from epub_audio.core.numbers_fr import spoken_numbers
        chapters = [[spoken_numbers(t) for t in c] for c in chapters]
        introduction = spoken_numbers(introduction)
    groups = chapter_segment_groups(engine, chapters, introduction=introduction,
                                    segmentation=options.segmentation,
                                    paragraph_pause_ms=options.paragraph_pause_ms,
                                    scene_pause_ms=options.scene_pause_ms)
    # Bind the journal to surviving passages using text, PCM duration and write
    # time. The last engine run can belong to an entirely different book.
    anchors = {}
    for index, segments in enumerate(groups, 1):
        for number, segment in enumerate(segments):
            path = Path(directory) / f'.segments-{index}' / f'{number:06d}.wav'
            if valid_wav(path):
                with wave.open(str(path), 'rb') as audio:
                    anchors[hashlib.sha256(segment.text.encode('utf-8')).hexdigest()] = (
                        path.stat().st_mtime, audio.getnframes(), audio.getframerate())
    candidates = []
    with Path(journal).open(encoding='utf-8') as stream:
        for line in stream:
            try:
                entry = json.loads(line)
                if entry.get('finished') is True and entry.get('voice') == engine.voice:
                    candidates.append(entry)
            except (ValueError, KeyError, TypeError):
                continue
    matched_runs = set()
    for entry in candidates:
        if entry.get('text_sha256') in anchors:
            stamp, frames, rate = anchors[entry['text_sha256']]
            if (abs(datetime.fromisoformat(entry['finished_at']).timestamp() - stamp) < 5
                    and frames == round(entry['duration_seconds'] * rate)):
                matched_runs.add(entry['run_id'])
    if len(matched_runs) != 1:
        raise ValueError('Impossible d’identifier avec certitude le journal de cet ancien traitement.')
    run_id, = matched_runs
    records = defaultdict(deque)
    for entry in candidates:
        if entry.get('run_id') == run_id:
            records[entry['text_sha256']].append(entry)
    evidence, segment_records = {}, {}
    for index, segments in enumerate(groups, 1):
        entries = []
        for segment in segments:
            key = hashlib.sha256(segment.text.encode('utf-8')).hexdigest()
            entries.append(records[key].popleft() if records[key] else None)
        segment_records[index] = entries
    for index in legacy_completed(directory):
        wav = Path(directory) / f'.chapter-{index}.wav'
        entries, segments = segment_records[index], groups[index - 1]
        if any(entry is None for entry in entries):
            message(f'Chapitre {index} : traces de passages insuffisantes, il sera régénéré.')
            continue
        with wave.open(str(wav), 'rb') as audio:
            rate, count = audio.getframerate(), audio.getnframes()
            expected = sum(round(e['duration_seconds'] * rate) for e in entries)
            pauses = sum(round(s.pause_before_ms * rate / 1000) for s in segments[1:])
            read = 0
            while data := audio.readframes(65536):
                read += len(data)
            if (read != count * audio.getnchannels() * audio.getsampwidth()
                    or not expected <= count <= expected + pauses):
                message(f'Chapitre {index} : durée ou données PCM incohérentes, il sera régénéré.')
                continue
            evidence[str(index)] = {'segments': len(segments), 'frames': count,
                                    'speech_frames': expected, 'sample_rate': rate}
        message(f'Chapitre {index} vérifié : {len(segments)} passages, WAV complet et durée cohérente.')
    if not evidence:
        raise ValueError('Aucun chapitre ancien ne peut être vérifié avec les traces de ce traitement.')
    state = BookCheckpoint.create(directory, source, plan, options, engine, master_root)
    state.data['legacy_validation'] = {'run_id': run_id, 'chapters': evidence}
    state.data['segment_plan_sha256'] = segment_plan_hash(groups)
    for index in evidence:
        state.data['chapters'][index] = file_hash(Path(directory) / f'.chapter-{index}.wav')
    # Only intact lookahead passages with matching generation traces are imported.
    for index, segments in enumerate(groups, 1):
        if str(index) in evidence:
            continue
        for number, entry in enumerate(segment_records[index]):
            path = Path(directory) / f'.segments-{index}' / f'{number:06d}.wav'
            if entry is not None and valid_wav(path):
                with wave.open(str(path), 'rb') as audio:
                    if audio.getnframes() != round(entry['duration_seconds'] * audio.getframerate()):
                        continue
                state.segment_ready(path)
    state.save()
    return state


def discover_incomplete(root):
    """Read-only discovery, including a crash during final folder renaming."""
    root = Path(root)
    jobs = []
    for path in sorted(root.rglob(STATE_FILE)):
        try:
            state = BookCheckpoint.load(path.parent)
            jobs.append({'directory': path.parent, 'source': Path(state.data['source']),
                         'completed': len(state.data['chapters']),
                         'total': len(state.data['plan']['chapters']), 'legacy': False})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    known = {job['directory'] for job in jobs}
    for directory in sorted(root.rglob('.ebook-*')):
        if not directory.is_dir() or directory in known or (directory / STATE_FILE).exists():
            continue
        try:
            report = json.loads((directory / 'decoupage.json').read_text(encoding='utf-8'))
            jobs.append({'directory': directory, 'source': Path(report['source']),
                         'completed': len(legacy_completed(directory)),
                         'total': len(report['chapters']), 'legacy': True})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return jobs
