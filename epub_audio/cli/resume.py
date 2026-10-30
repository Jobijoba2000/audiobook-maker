"""Console discovery and resumption using each book's saved configuration."""
import json

from epub_audio.adapters.ffmpeg import require_tools
from epub_audio.cli.selectors import parse_selection
from epub_audio.core.options import AudioOptions
from epub_audio.core.pipeline import process_book
from epub_audio.core.tts_qwen import QwenEngine
from epub_audio.io.fs import ROOT
from epub_audio.io.preferences import load_preferences
from epub_audio.io.resume import BookCheckpoint, import_legacy


def show_incomplete(jobs):
    print('Livres interrompus détectés :')
    for index, job in enumerate(jobs, 1):
        status = 'à vérifier' if job['legacy'] else 'enregistrés, à revérifier'
        print(f"{index}. {job['source'].stem} — {job['completed']}/{job['total']} chapitres {status}")
        print(f"   {job['directory']}")


def select_incomplete(jobs):
    show_incomplete(jobs)
    print('0. Choisir un autre livre — garder les livres interrompus en attente')
    while True:
        raw = input('Choix [0] (numéros des livres à reprendre ; exemple : 1 ou 1-3) : ').strip().lower()
        if raw in ('', '0', 'n'):
            return []
        chosen = parse_selection(raw, jobs)
        if chosen:
            return chosen
        print('Choisissez un numéro de livre, ou 0 pour choisir un autre livre.')


def prepare_legacy(job, output_root, message=print):
    preferences = load_preferences()
    last_run = json.loads((ROOT / '.cache/qwen-last-run.json').read_text(encoding='utf-8'))
    options = AudioOptions(**{key: preferences[key] for key in (
        'bitrate', 'audio_format', 'sample_rate', 'segmentation', 'convert_numbers', 'normalize')
        if key in preferences}, speed=preferences.get('rate', 1.0))
    engine = QwenEngine(voice=last_run['voice'], batching=last_run.get('batching', False),
                        batch_size=last_run.get('batch_size', 2) if last_run.get('batching') else 2,
                        continuous_batching=last_run.get('continuous_batching', False),
                        shrinking_batching=last_run.get('shrinking_batching', True))
    engine.settings = last_run['settings']
    directory = job['directory'].resolve()
    master_root = output_root / 'master' / directory.parent.relative_to(output_root)
    message('Ancien traitement : vérification avec le découpage, les WAV et le journal TTS local.')
    return import_legacy(directory, options, engine, master_root,
                         journal=ROOT / '.cache/qwen-segments.jsonl', message=message)


def resume_books(jobs, output_root, *, offline=False):
    from epub_audio.cli.ui import ConsoleOutput
    console = ConsoleOutput()
    failed = False
    for job in jobs:
        try:
            state = (prepare_legacy(job, output_root, console.message) if job['legacy']
                     else BookCheckpoint.load(job['directory']))
            options = AudioOptions(**state.data['options'])
            engine_options = dict(state.data['engine'])
            if not engine_options['batching']:
                engine_options['batch_size'] = 2
            engine = QwenEngine(**engine_options, offline=offline, speed=options.speed, message=console.message)
            if state.data.get('engine_settings'):
                engine.settings = state.data['engine_settings']
            state.validate(job['source'], options, engine)
            require_tools(options.audio_format, normalize=options.normalize)
            # Preparation is lazy: a book awaiting only export never loads the GPU.
            process_book(job['source'], engine, options=options, resume_dir=job['directory'],
                         output_dir=job['directory'].parent,
                         master_output_dir=state.data['master_root'],
                         message=console.message, progress=console.progress)
        except Exception as exc:
            console.message(f"Reprise impossible pour {job['source'].name} : {exc}")
            failed = True
    return int(failed)
