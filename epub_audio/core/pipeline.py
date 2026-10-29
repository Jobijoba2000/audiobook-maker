"""Book processing: synthesize everything, then export chapters in parallel."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path
import os
import shutil
import tempfile
import traceback
import uuid
import wave
from epub_audio.adapters.ffmpeg import encode_chapter
from epub_audio.io.epub import read_book, save_cover
from epub_audio.core.models import Chapter
from epub_audio.io.naming import book_basename, chapter_filename
from epub_audio.core.options import AudioOptions
from epub_audio.core.tts_qwen import QwenEngine
from epub_audio.core.tts_generate import render_chapters
from epub_audio.core.chapter_titles import prepare_chapter_headings
from epub_audio.core.text import prepare_spoken_text
from epub_audio.core.tts_progress import TtsProgress
from epub_audio.core.export_progress import ExportTracker
from epub_audio.helpers.time import format_duration_full
from epub_audio.config.defaults import (DEFAULT_SPEED, DEFAULT_BITRATE, DEFAULT_WORKERS,
                                        PARAGRAPH_PAUSE_MS, SCENE_PAUSE_MS, DEFAULT_AUDIO_FORMAT,
                                        FORMAT_EXTENSIONS, DEFAULT_SEGMENTATION, DEFAULT_CONVERT_NUMBERS,
                                        DEFAULT_NORMALIZE)


def process_book(epub_path, engine, *, speed=DEFAULT_SPEED, bitrate=DEFAULT_BITRATE, workers=DEFAULT_WORKERS, message=print,
                 paragraph_pause_ms=PARAGRAPH_PAUSE_MS, scene_pause_ms=SCENE_PAUSE_MS, progress=None, options=None,
                 output_dir=None, master_output_dir=None, audio_format=DEFAULT_AUDIO_FORMAT, sample_rate=None,
                 segmentation=DEFAULT_SEGMENTATION, convert_numbers=DEFAULT_CONVERT_NUMBERS, book_plan=None,
                 normalize=DEFAULT_NORMALIZE):
    settings = options if options is not None else AudioOptions(
        speed=speed, bitrate=bitrate, workers=workers,
        paragraph_pause_ms=paragraph_pause_ms, scene_pause_ms=scene_pause_ms, audio_format=audio_format,
        sample_rate=sample_rate, segmentation=segmentation, convert_numbers=convert_numbers, normalize=normalize)
    speed, bitrate, workers = settings.speed, settings.bitrate, settings.workers
    audio_format = settings.audio_format
    sample_rate = settings.sample_rate
    synthesis_workers = 1
    extension = FORMAT_EXTENSIONS[audio_format]
    paragraph_pause_ms, scene_pause_ms = settings.paragraph_pause_ms, settings.scene_pause_ms
    if progress is None:
        progress = message
    epub_path = Path(epub_path).resolve()
    plan = book_plan if book_plan is not None else read_book(epub_path)
    chapters, cover = plan
    if plan.fallback_reason:
        message('Découpage de secours automatique (ancien système) pour ' + epub_path.name
                + ' : ' + plan.fallback_reason)
    message(f'Découpage : {plan.mode}, {len(chapters)} chapitres'
            + (f', {plan.excluded_pages} page(s) de présentation écartée(s).' if plan.excluded_pages else '.'))
    destination_root = Path(output_dir).resolve() if output_dir is not None else epub_path.parent
    master_root = (Path(master_output_dir).resolve() if master_output_dir is not None
                   else destination_root / 'master')
    destination_root.mkdir(parents=True, exist_ok=True)
    safe_name = book_basename(epub_path, destination_root)
    work = Path(tempfile.mkdtemp(prefix='.ebook-', dir=destination_root))
    message(f'\nTraitement : {epub_path.name} ({len(chapters)} chapitres)')
    try:
        save_cover(cover, work)
        if plan.chapter_analysis:
            from epub_audio.io.llm_chapters import chapter_report, _write_json
            _write_json(work / 'decoupage.json', chapter_report(plan, epub_path))
        if isinstance(engine, QwenEngine):
            engine.speed = speed
        # Exhaust synthesis first: its process pool closes before any final export.
        with TtsProgress(progress, message) as tts:
            spoken_chapters = [[prepare_spoken_text(text) for text in
                                prepare_chapter_headings(chapter.paragraphs, chapter.heading_indices)]
                               for chapter in chapters]
            introduction = prepare_spoken_text(plan.metadata.introduction())
            if settings.convert_numbers:
                from epub_audio.core.numbers_fr import spoken_numbers
                spoken_chapters = [[spoken_numbers(paragraph) for paragraph in chapter]
                                   for chapter in spoken_chapters]
                introduction = spoken_numbers(introduction)
            ready = render_chapters(engine, spoken_chapters, work,
                                    workers=synthesis_workers, progress=tts.progress, paragraph_pause_ms=paragraph_pause_ms,
                                    scene_pause_ms=scene_pause_ms, introduction=introduction,
                                    segmentation=settings.segmentation)
            with closing(ready):
                prepared = list(ready)
        export_workers = min(workers, len(prepared))
        message(f'Synthèse du livre terminée. Export des chapitres en {audio_format.upper()} ({export_workers} workers)…')

        master_dir = work / 'Master WAV'
        master_dir.mkdir()

        def encode_and_save_chapter(index, wav, chunks):
            # Preserve the assembled PCM without resampling or rewriting it.
            with wave.open(str(wav), 'rb') as audio:
                duration = audio.getnframes() / audio.getframerate()
            if duration <= 0:
                raise RuntimeError('Le chapitre WAV est vide.')
            master = master_dir / chapter_filename(index, len(chapters), duration, 'wav')
            if audio_format == 'wav' and not settings.normalize:
                shutil.copy2(wav, master)
                temporary_audio = wav
            else:
                wav.rename(master)
                # Encode from the master; it also survives a failed conversion.
                temporary_audio = work / f'.chapter-{index}.{extension}'
                export_options = {'normalize': True} if settings.normalize else {}
                duration = encode_chapter(master, temporary_audio, bitrate=bitrate, audio_format=audio_format,
                                          sample_rate=sample_rate, title=chapters[index - 1].title,
                                          album=plan.metadata.title or epub_path.stem, track=index, **export_options)
            name = chapter_filename(index, len(chapters), duration, extension)
            temporary_audio.rename(work / name)
            chunks.rmdir()
            return duration, name

        exports_tracker = ExportTracker(len(prepared), export_workers, audio_format, progress)
        exports_tracker.show()

        def export_chapter(index, wav, chunks):
            with exports_tracker.task():
                return encode_and_save_chapter(index, wav, chunks)

        total = 0.0
        with ThreadPoolExecutor(max_workers=export_workers) as encoder:
            exports = [encoder.submit(export_chapter, *chapter) for chapter in prepared]
            try:
                for future in as_completed(exports):
                    duration, name = future.result()
                    total += duration
                    message(f'  Généré : {name}')
            except BaseException:
                for future in exports:
                    future.cancel()
                raise
        shutil.copy2(epub_path, work / (safe_name + '.epub'))
        for sidecar in work.iterdir():
            if sidecar.is_file() and (sidecar.suffix == '.epub' or sidecar.stem == 'cover'
                                      or sidecar.name == 'decoupage.json'):
                shutil.copy2(sidecar, master_dir / sidecar.name)
        destination = destination_root / f'{safe_name} - {format_duration_full(total)}'
        # Keep the same folder name in both trees, without overwriting either.
        while destination.exists() or (master_root / destination.name).exists():
            destination = destination_root / f'{safe_name} - {format_duration_full(total)}_{uuid.uuid4().hex[:8]}'
        master_root.mkdir(parents=True, exist_ok=True)
        os.rename(work, destination)
        work = destination
        master_destination = master_root / destination.name
        os.rename(work / 'Master WAV', master_destination)
        message(f'Master WAV conservé : {master_destination}')
        message(f'Terminé : {destination}')
        return destination
    except BaseException:
        try:
            (work / 'erreur.txt').write_text(traceback.format_exc(), encoding='utf-8')
        except OSError:
            pass
        message(f'Livre interrompu. Les fichiers préparés restent dans : {work}')
        raise
