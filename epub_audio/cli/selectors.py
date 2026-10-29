"""Interactive selection of books."""
import re
import math
import sys
import unicodedata
from epub_audio.config.defaults import SEGMENTATION_MODES, BATCH_SIZE_CHOICES
from pathlib import Path
from epub_audio.config.defaults import AUDIO_FORMATS, BITRATE_CHOICES, SAMPLE_RATE_CHOICES, FORMAT_SAMPLE_RATES


def _entry_sort_key(path):
    name = ''.join(character for character in unicodedata.normalize('NFKD', path.name.casefold())
                   if not unicodedata.combining(character))
    return not path.is_dir(), name, path.name


def _entry_icons():
    icons = ('📁', '📄')
    try:
        ''.join(icons).encode(getattr(sys.stdout, 'encoding', None) or 'utf-8')
    except (UnicodeEncodeError, LookupError):
        return '[DOSSIER]', '[EPUB]'
    return icons


def _selected_indices(raw, count):
    indices = set()
    for part in re.split(r'[\s,;]+', raw):
        if not part:
            continue
        match = re.fullmatch(r'([0-9]+)(?:-([0-9]+))?', part)
        if match is None:
            raise ValueError('Utilisez des numéros ou des plages, par exemple 1 2 4-8.')
        first = int(match[1])
        last = int(match[2]) if match[2] is not None else first
        if not 1 <= first <= last <= count:
            raise ValueError(f'Les numéros doivent être compris entre 1 et {count}, avec des plages croissantes.')
        indices.update(range(first - 1, last))
    if not indices:
        raise ValueError('Sélectionnez au moins un livre.')
    return sorted(indices)


def select_input_output(input_root, output_root):
    """Browse folders or select one/many EPUBs, with project-relative output."""
    input_root, output_root = Path(input_root).resolve(), Path(output_root).resolve()
    current = input_root
    folder_icon, file_icon = _entry_icons()
    while True:
        entries = sorted((path for path in current.iterdir()
                          if path.resolve().is_relative_to(input_root)
                          and (path.is_dir() or (path.is_file() and path.suffix.lower() == '.epub'))),
                         key=_entry_sort_key) if current.is_dir() else []
        relative = current.relative_to(input_root)
        label = 'input' if current == input_root else f'input / {relative}'
        print(f'\nDossiers et livres EPUB — {label} :')
        for index, path in enumerate(entries, 1):
            icon = folder_icon if path.is_dir() else file_icon
            print(f'{index}. {icon} {path.name}')
        if not entries:
            print('Aucun dossier ni fichier EPUB ici.')
        if current != input_root:
            print('0. Remonter au dossier parent')
        while True:
            raw = input('Numéro du dossier ou du livre (plusieurs EPUB : 1 2 4-8 ; Entrée = tous) : ').strip().strip('"')
            selected_files = None
            if raw in ('0', '..'):
                current = current.parent if current != input_root else input_root
                break
            if not raw:
                selected = current
            elif raw.isascii() and raw.isdigit():
                index = int(raw)
                if not 1 <= index <= len(entries):
                    print(f'Choisissez un numéro entre 1 et {len(entries)}, ou Entrée pour tout le dossier.')
                    continue
                selected = entries[index - 1].resolve()
            elif re.fullmatch(r'[0-9\s,;\-]+', raw):
                try:
                    indices = _selected_indices(raw, len(entries))
                except ValueError as exc:
                    print(exc)
                    continue
                selected_files = [entries[index].resolve() for index in indices]
                if any(not path.is_relative_to(input_root) or not path.is_file()
                       or path.suffix.lower() != '.epub' for path in selected_files):
                    print('Sélectionnez plusieurs fichiers EPUB, ou un seul dossier pour l’ouvrir.')
                    continue
                selected = current
            else:
                relative = Path(raw)
                if relative.is_absolute() or relative.drive or '..' in relative.parts:
                    print('Choisissez un dossier ou un EPUB situé à l’intérieur de input.')
                    continue
                selected = (current / relative).resolve()
            if not selected.is_relative_to(input_root):
                print('Choisissez un dossier ou un EPUB situé à l’intérieur de input.')
                continue
            if not (selected.is_dir() or (selected.is_file() and selected.suffix.lower() == '.epub')) and selected != input_root:
                print(f'Dossier ou fichier EPUB introuvable : {raw}')
                continue
            folder = selected.parent if selected.is_file() else selected
            output = (output_root / folder.relative_to(input_root)).resolve()
            if not output.is_relative_to(output_root):
                print('Le dossier de sortie doit rester à l’intérieur de output.')
                continue
            if output.exists() and not output.is_dir():
                print(f'Un fichier empêche la création du dossier de sortie : {output}')
                continue
            if raw and selected.is_dir() and selected_files is None:
                current = selected
                break
            if selected_files is not None:
                print(f'{len(selected_files)} livre(s) sélectionné(s) :')
                for path in selected_files:
                    print(f'  {path.name}')
            else:
                print(f'Entrée : {selected}')
            print(f'Sortie : {output}')
            return selected_files if selected_files is not None else selected, output


def parse_selection(selection_str, epubs):
    selection = selection_str.strip().lower()
    if not selection or selection == 'o':
        return epubs
    indices = set()
    for part in re.split(r'[\s,;]+', selection):
        if re.fullmatch(r'\d+-\d+', part):
            first, last = map(int, part.split('-'))
            indices.update(range(max(1, first), min(len(epubs), last) + 1))
        elif part.isdigit() and 1 <= int(part) <= len(epubs):
            indices.add(int(part))
    return [epubs[index - 1] for index in sorted(indices)]


def select_speed(default):
    while True:
        raw = input(f'Vitesse de lecture [{float(default)}] : ').strip()
        if not raw:
            return default
        try:
            speed = float(raw.replace(',', '.'))
        except ValueError:
            speed = float('nan')
        if math.isfinite(speed) and 0.5 <= speed <= 3:
            return speed
        print('Saisissez une vitesse entre 0.5 et 3 (exemple : 1.0 ou 1.1).')


def _select_number(label, choices, default, display=str, allowed=None):
    print(f'\n{label} :')
    for index, value in enumerate(choices, 1):
        unavailable = ' (indisponible pour ce codec)' if allowed is not None and value not in allowed else ''
        print(f'{index}. {display(value)}{unavailable}')
    default_index = choices.index(default) + 1
    while True:
        raw = input(f'{label} (numéro 1-{len(choices)}) [{default_index}] : ').strip()
        if not raw:
            return default
        if raw.isascii() and raw.isdigit() and 1 <= int(raw) <= len(choices):
            selected = choices[int(raw) - 1]
            if allowed is None or selected in allowed:
                return selected
            print('Fréquence incompatible avec ce codec. Choisissez 24 ou 48 kHz pour Opus.')
        else:
            print(f'Saisissez le numéro du choix, entre 1 et {len(choices)}.')


def select_batching(default):
    labels = {1: 'Sans batching (un passage à la fois)'}
    labels.update({size: f'Batching par {size} passages' for size in BATCH_SIZE_CHOICES})
    return _select_number('Batching Qwen', (1, *BATCH_SIZE_CHOICES), default, labels.__getitem__)


def select_segmentation(default):
    return _select_number('Découpage des segments', tuple(SEGMENTATION_MODES), default,
                          SEGMENTATION_MODES.__getitem__)


def select_llm_chapters(default):
    return _select_number('Améliorer le découpage des chapitres avec un LLM local', (False, True), default,
                          lambda value: 'Oui, analyser les chapitres' if value else 'Non, découpage habituel')


def select_audio_format(default):
    return _select_number('Codec audio', AUDIO_FORMATS, default,
                          lambda value: 'WAV (PCM original)' if value == 'wav' else value.upper())


def select_convert_numbers(default):
    return _select_number('Convertir les nombres en lettres', (False, True), default,
                          lambda value: 'Oui' if value else 'Non')


def select_normalization(default):
    return _select_number('Normaliser le volume final des chapitres', (False, True), default,
                          lambda value: 'Oui' if value else 'Non')


def select_bitrate(default):
    return _select_number('Débit', BITRATE_CHOICES, default, lambda value: f'{value} kbit/s')


def select_sample_rate(default, audio_format):
    if audio_format == 'wav':
        return 24000
    allowed = FORMAT_SAMPLE_RATES[audio_format]
    if default not in allowed:
        default = 48000
    if audio_format == 'opus':
        print('Opus : choix de la fréquence d’entrée de l’encodeur ; le fichier sera décodé à 48 kHz.')
    return _select_number('Fréquence', SAMPLE_RATE_CHOICES, default,
                          lambda value: f'{value / 1000:g} kHz'.replace('.', ','), allowed)
