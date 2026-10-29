"""Interactive and batch console orchestration."""
from pathlib import Path
from contextlib import nullcontext
import os
import sys
import time
from epub_audio.cli.args import arguments
from epub_audio.cli.selectors import (parse_selection, select_input_output,
                                      select_batching, select_shrinking_batching, select_segmentation, select_convert_numbers,
                                      select_llm_chapters,
                                      select_audio_format, select_bitrate, select_sample_rate,
                                      select_normalization)
from epub_audio.cli.ui import ConsoleOutput
from epub_audio.core.pipeline import process_book
from epub_audio.core.options import AudioOptions
from epub_audio.core.tts_qwen import QwenEngine, configuration
from epub_audio.adapters.ffmpeg import require_tools, tool_executable
from epub_audio.io.fs import INPUT_DIR, OUTPUT_DIR
from epub_audio.io.preferences import load_preferences, save_preferences
from epub_audio.config.defaults import (ENGINE_NAME, DEFAULT_MODEL_SIZE, DEFAULT_BATCHING,
                                        DEFAULT_BATCH_SIZE, FORMAT_SAMPLE_RATES, FORMAT_EXTENSIONS,
                                        SEGMENTATION_MODES, DEFAULT_SEGMENTATION, DEFAULT_CONVERT_NUMBERS,
                                        DEFAULT_NORMALIZE)


def available_voices(language):
    from epub_audio.core.tts_qwen import list_available_voices
    choices = list_available_voices()
    return choices, [voice for voice in choices if language.lower() == 'all'
                     or voice['lang'].lower() == language.lower()
                     or voice['lang'].lower().startswith(language.lower() + '-')]


def plan_books(epubs, *, use_llm, model, url, message):
    from epub_audio.adapters.ollama import OllamaChapters, ChapterLLMError
    from epub_audio.io.epub import read_book
    from epub_audio.io.llm_chapters import improve_chapters
    plans, failed = {}, False
    context = OllamaChapters(model, url) if use_llm else nullcontext(None)
    with context as client:
        for path in epubs:
            message(f'\nPréparation des chapitres : {path.name}')
            try:
                plan = read_book(path)
            except Exception as exc:
                message(f'Impossible de lire {path.name} : {exc}')
                failed = True
                continue
            if use_llm:
                try:
                    plan = improve_chapters(plan, client, message=message)
                except ChapterLLMError as exc:
                    failed = True
                    plan.chapter_analysis = {'status': 'failed', 'model': model, 'error': str(exc),
                                             'original_mode': plan.mode, 'original_chapters': len(plan.chapters)}
                    message(f'Analyse LLM impossible : {exc} Découpage habituel conservé.')
            plans[path] = plan
            message(f'Découpage retenu : {len(plan.chapters)} chapitres ({plan.mode}).')
    return plans, failed


def export_chapter_plans(epubs, destinations, plans, message):
    from epub_audio.io.llm_chapters import chapter_report, _write_json
    for path in epubs:
        if path not in plans:
            continue
        folder = destinations[path] / 'Découpages'
        target = folder / (path.stem + '.chapters.json')
        number = 1
        while target.exists():
            target = folder / (path.stem + f'.{number}.chapters.json')
            number += 1
        _write_json(target, chapter_report(plans[path], path))
        message(f'Plan des chapitres enregistré : {target.resolve()}')


def main(argv=None, default_directory=None):
    args = arguments(argv, default_directory)
    output_root = args.output.resolve()
    book_selected = False
    if (not args.yes and not args.list_voices and not args.paths_explicit
            and not args.output_explicit and default_directory is None):
        selected_input, selected_output = select_input_output(INPUT_DIR, OUTPUT_DIR)
        args.paths = selected_input if isinstance(selected_input, list) else [selected_input]
        args.output = selected_output
        book_selected = all(path.is_file() for path in args.paths)
        if selected_output != OUTPUT_DIR.resolve():
            selected_output.mkdir(parents=True, exist_ok=True)
    preferences = load_preferences()
    prompts = preferences.get('prompts', {})

    def ask(name):
        # The new number question is opt-in for existing console workflows.
        return not args.yes and prompts.get(name, name != 'convert_numbers')

    if args.list_voices:
        _, voices = available_voices(args.language)
        for voice in voices:
            print(f"{voice['id']} : {voice['display_name']}")
        return 0
    epubs = []
    destinations = {}
    for path in args.paths:
        if path == INPUT_DIR and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            root = path.resolve()
            sources = sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.epub'), key=str)
            for source in sources:
                epub = source.resolve()
                epubs.append(epub)
                destinations.setdefault(epub, args.output / source.parent.relative_to(root))
        elif path.is_file() and path.suffix.lower() == '.epub':
            epub = path.resolve()
            epubs.append(epub)
            destinations.setdefault(epub, args.output)
        else:
            raise ValueError(f'Fichier EPUB ou dossier introuvable : {path}')
    epubs = list(dict.fromkeys(epubs))
    if not epubs:
        print('Aucun fichier .epub trouvé dans les dossiers indiqués.')
        print('Dossiers recherchés : ' + ', '.join(str(path.resolve()) for path in args.paths))
        return 0
    if not args.yes and not book_selected:
        print('Fichiers EPUB trouvés :')
        for index, path in enumerate(epubs, 1):
            print(f'{index}. {path.name}')
        epubs = parse_selection(input('Livres à traiter (Entrée = tous ; exemples : 1-3 5 ou 1,3,5) : '), epubs)
        if not epubs:
            print('Aucun livre sélectionné.')
            return 0
    llm_chapters = args.llm_chapters if args.llm_chapters is not None else preferences.get('llm_chapters', False)
    if ask('llm_chapters') and args.llm_chapters is None:
        llm_chapters = select_llm_chapters(llm_chapters)
    if args.chapters_only:
        plans, failed = plan_books(epubs, use_llm=llm_chapters, model=args.llm_model,
                                   url=args.llm_url, message=print)
        export_chapter_plans(epubs, destinations, plans, print)
        return int(failed)
    all_voices, voices = available_voices(args.language)
    if args.voice and args.voice not in {item['id'] for item in voices}:
        raise ValueError('Cette référence vocale n’est pas disponible pour la langue choisie. Utilisez --list-voices.')
    saved_voice = preferences.get('voice')
    fallback_voice = configuration()['voice']
    default_voice = next((v['id'] for v in voices if v['id'] == saved_voice), None)
    default_voice = default_voice or next((v['id'] for v in voices if v['id'] == fallback_voice), None)
    default_voice = default_voice or (voices[0]['id'] if voices else None)
    voice = args.voice
    if not voice:
        if not ask('voice'):
            voice = default_voice
        else:
            for index, item in enumerate(voices, 1):
                print(f"{index}. {item['display_name']}")
            default_index = next((index for index, item in enumerate(voices, 1)
                                  if item['id'] == default_voice), None)
            while voices:
                raw = input(f'Voix de référence (1-{len(voices)}) [{default_index}] : ').strip()
                if not raw:
                    voice = default_voice
                    break
                if raw.isdigit() and 1 <= int(raw) <= len(voices):
                    voice = voices[int(raw) - 1]['id']
                    break
    if not voice:
        raise ValueError('Aucune voix pour la langue choisie. Utilisez --language all.')
    batching = args.batching if args.batching is not None else preferences.get('faster_small_batching', DEFAULT_BATCHING)
    batch_size = preferences.get('faster_small_batch_size', DEFAULT_BATCH_SIZE)
    if args.batch_size is not None:
        batch_size, batching = args.batch_size, True
    elif ask('batching') and args.batching is None:
        selection = select_batching(batch_size if batching else 1)
        batching = selection > 1
        if batching:
            batch_size = selection
    if args.continuous_batching is True and not batching:
        raise ValueError('Le mode continu nécessite le batching de Faster-Qwen3-TTS.')
    if args.shrinking_batching is True and (not batching or args.continuous_batching is True):
        raise ValueError('La réduction progressive nécessite le batching sans mode continu.')
    continuous = batching and args.continuous_batching is True
    shrinking_requested = (args.shrinking_batching if args.shrinking_batching is not None
                           else preferences.get('shrinking_batching', True))
    if batching and not continuous and args.shrinking_batching is None and ask('shrinking_batching'):
        shrinking_requested = select_shrinking_batching(shrinking_requested)
    shrinking = batching and not continuous and shrinking_requested
    segmentation = args.segmentation or preferences.get('segmentation', DEFAULT_SEGMENTATION)
    if ask('segmentation') and args.segmentation is None:
        segmentation = select_segmentation(segmentation)
    convert_numbers = (args.convert_numbers if args.convert_numbers is not None
                       else preferences.get('convert_numbers', DEFAULT_CONVERT_NUMBERS))
    if ask('convert_numbers') and args.convert_numbers is None:
        convert_numbers = select_convert_numbers(convert_numbers)
    # Les voix conservent leur débit naturel ; seul --rate change le tempo.
    rate = args.rate if args.rate_explicit else 1.0
    audio_format = args.audio_format if args.format_explicit else preferences.get('audio_format', args.audio_format)
    if ask('audio_format') and not args.format_explicit:
        audio_format = select_audio_format(audio_format)
    if audio_format == 'wav':
        if args.bitrate_explicit:
            raise ValueError('--bitrate ne s’applique pas au WAV PCM original.')
        if args.frequency_explicit and args.sample_rate != 24000:
            raise ValueError('WAV PCM original : la fréquence doit rester à 24 kHz.')
        bitrate, sample_rate = None, 24000
    else:
        bitrate = args.bitrate if args.bitrate_explicit else preferences.get('bitrate', args.bitrate)
        if ask('bitrate') and not args.bitrate_explicit:
            bitrate = select_bitrate(bitrate)
        sample_rate = args.sample_rate if args.frequency_explicit else preferences.get('sample_rate', args.sample_rate)
        if not args.frequency_explicit and sample_rate not in FORMAT_SAMPLE_RATES[audio_format]:
            sample_rate = 48000
        if ask('sample_rate') and not args.frequency_explicit:
            sample_rate = select_sample_rate(sample_rate, audio_format)
    normalize = args.normalize if args.normalize is not None else preferences.get('normalize', DEFAULT_NORMALIZE)
    if ask('normalize') and args.normalize is None:
        normalize = select_normalization(normalize)
    options = AudioOptions(speed=rate, workers=args.workers, audio_format=audio_format,
                           bitrate=bitrate, sample_rate=sample_rate, segmentation=segmentation,
                           convert_numbers=convert_numbers, normalize=normalize)
    if args.offline:
        os.environ['EBOOK_TTS_OFFLINE'] = '1'
    workers = 1
    if rate != 1:
        if not tool_executable('ffmpeg'):
            raise RuntimeError('FFmpeg est nécessaire pour modifier la vitesse de lecture.')
    require_tools(options.audio_format, normalize=normalize)
    if audio_format == 'wav':
        encoding = 'WAV'
    else:
        mode = {'mp3': 'CBR', 'aac': 'cible', 'opus': 'VBR cible'}[audio_format]
        frequency_label = f'{sample_rate / 1000:g}'.replace('.', ',')
        encoding = f'{audio_format.upper()} {mode} {bitrate} kbit/s, mono, {frequency_label} kHz'
        if audio_format == 'opus':
            encoding += ' en entrée encodeur, décodage à 48 kHz'
    voice_name = next(item['display_name'] for item in all_voices if item['id'] == voice)
    generation_label = (f'batching continu, jusqu’à {batch_size} passages simultanés' if continuous
                        else f'batching par {batch_size} passages, réduction progressive active' if shrinking
                        else f'batching par {batch_size} passages' if batching else f'{workers} génération à la fois')
    engine_label = f'{ENGINE_NAME} {DEFAULT_MODEL_SIZE}'
    print(f'\n{len(epubs)} livre(s), {engine_label}, {voice_name}, vitesse {rate:g}, {encoding} '
          f'(.{FORMAT_EXTENSIONS[audio_format]}), {generation_label}.')
    print(f'Découpage des segments : {SEGMENTATION_MODES[segmentation].lower()}.')
    print('Analyse LLM des chapitres : ' + (f'oui ({args.llm_model}).' if llm_chapters else 'non.'))
    print('Conversion des nombres en lettres : ' + ('oui.' if convert_numbers else 'non.'))
    print('Normalisation du volume final des chapitres : ' + ('oui.' if normalize else 'non.'))
    if audio_format != 'wav' or normalize:
        print(f'Encodage final : jusqu’à {options.workers} chapitres en parallèle.')
    print(f'Un master WAV original par chapitre sera conservé sous : {output_root / "master"}')
    print(f'Sortie : {args.output.resolve()}')
    if ask('confirmation'):
        input('Entrée pour commencer...')
    console = ConsoleOutput()
    try:
        save_preferences(voice, rate, bitrate, audio_format, sample_rate, segmentation=segmentation,
                         faster_small_batching=batching,
                         faster_small_batch_size=batch_size, prompts=prompts, convert_numbers=convert_numbers,
                         llm_chapters=llm_chapters, normalize=normalize, shrinking_batching=shrinking_requested)
    except OSError as exc:
        console.message(f'Impossible de mémoriser les réglages : {exc}')
    started = time.perf_counter()
    failed = False
    plans = None
    if llm_chapters:
        plans, failed = plan_books(epubs, use_llm=True, model=args.llm_model,
                                   url=args.llm_url, message=console.message)
    if plans is not None and not plans:
        return 1
    engine = QwenEngine(voice, offline=args.offline, message=console.message, speed=rate,
                        batching=batching, batch_size=batch_size, continuous_batching=continuous,
                        shrinking_batching=shrinking)
    engine.prepare()
    for path in epubs:
        if plans is not None and path not in plans:
            continue
        try:
            prepared = {'book_plan': plans[path]} if plans is not None else {}
            process_book(path, engine, output_dir=destinations[path],
                         master_output_dir=output_root / 'master' / destinations[path].resolve().relative_to(output_root),
                         options=options,
                         message=console.message, progress=console.progress, **prepared)
        except Exception as exc:
            failed = True
            detail = str(exc).strip() or type(exc).__name__
            print(f'Erreur pour {path.name} : {detail}', file=sys.stderr)
    elapsed = round(time.perf_counter() - started)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    included = 'analyse, chargement et export compris' if llm_chapters else 'chargement et export compris'
    console.message(f'Temps total écoulé : {hours:02d}:{minutes:02d}:{seconds:02d} ({included}).')
    return 1 if failed else 0
