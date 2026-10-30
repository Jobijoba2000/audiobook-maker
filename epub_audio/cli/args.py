"""Command-line arguments and validation."""
from pathlib import Path
import argparse
import math
from epub_audio.config.defaults import (DEFAULT_SPEED, DEFAULT_WORKERS, DEFAULT_BITRATE,
                                        DEFAULT_AUDIO_FORMAT, DEFAULT_SAMPLE_RATE,
                                        AUDIO_FORMATS, BITRATE_CHOICES, FORMAT_SAMPLE_RATES, SEGMENTATION_MODES, BATCH_SIZE_CHOICES)
from epub_audio.io.fs import INPUT_DIR, OUTPUT_DIR


def frequency(value):
    rates = {'24': 24000, '24000': 24000, '44.1': 44100, '44100': 44100, '48': 48000, '48000': 48000}
    try:
        return rates[value.strip().replace(',', '.')]
    except KeyError:
        raise argparse.ArgumentTypeError('Fréquences possibles : 24, 44.1 ou 48 kHz.') from None


def arguments(argv=None, default_directory=None):
    parser = argparse.ArgumentParser(description='Audiobook Maker : EPUB vers audio avec Faster-Qwen3-TTS 0.6B.')
    parser.add_argument('paths', nargs='*', type=Path, help='Fichiers EPUB ou dossiers (input du programme par défaut).')
    parser.add_argument('--output', '-o', type=Path,
                        help='Dossier de destination (output du programme par défaut).')
    parser.add_argument('--resume', action='store_true',
                        help='Reprendre les livres interrompus détectés dans output avec leurs réglages enregistrés.')
    parser.add_argument('--list-incomplete', action='store_true',
                        help='Lister les livres interrompus sans charger le modèle ni modifier les fichiers.')
    parser.add_argument('--batching', action=argparse.BooleanOptionalAction, default=None,
                        help='Batching activé par défaut ; --no-batching pour désactiver.')
    parser.add_argument('--continuous-batching', action=argparse.BooleanOptionalAction, default=None,
                        help='Faster uniquement, expérimental et désactivé par défaut : réutilisation continue des places GPU ; sinon groupes par longueur.')
    parser.add_argument('--shrinking-batching', action=argparse.BooleanOptionalAction, default=None,
                        help='Réduire les calculs GPU quand les passages terminent (dernier choix, sinon activé) ; --no-shrinking-batching pour comparer.')
    parser.add_argument('--batch-size', type=int, choices=BATCH_SIZE_CHOICES,
                        help='Active le batching avec 2, 3, 4, 5, 6, 7 ou 8 passages ; sinon taille mémorisée, par défaut 6.')
    parser.add_argument('--segmentation', choices=SEGMENTATION_MODES,
                        help='Découpage des segments : paragraph ou sentence ; dernier choix, sinon paragraph.')
    parser.add_argument('--llm-chapters', action=argparse.BooleanOptionalAction, default=None,
                        help='Améliorer les chapitres avec un LLM local ; --no-llm-chapters conserve le découpage habituel.')
    parser.add_argument('--llm-model', default='qwen3.5:4b', help='Modèle local Ollama pour les chapitres (qwen3.5:4b par défaut).')
    parser.add_argument('--llm-url', default='http://127.0.0.1:11434', help='Adresse du serveur Ollama local.')
    parser.add_argument('--chapters-only', action='store_true',
                        help='Exporter le plan des chapitres en JSON sans charger le TTS ni produire d’audio.')
    parser.add_argument('--convert-numbers', action=argparse.BooleanOptionalAction, default=None,
                        help='Convertir les nombres en lettres ; --no-convert-numbers pour conserver les chiffres (sinon choix mémorisé).')
    parser.add_argument('--voice', help='Identifiant de référence vocale locale ; voir --list-voices.')
    parser.add_argument('--language', default='fr-FR', help='Filtre des voix : fr-FR, fr, en ou all.')
    parser.add_argument('--list-voices', action='store_true', help='Afficher les voix sans télécharger de modèles.')
    parser.add_argument('--rate', type=float,
                        help='Vitesse de lecture : tempo FFmpeg pour Faster-Qwen3-TTS.')
    parser.add_argument('--format', '--codec', dest='audio_format', choices=AUDIO_FORMATS,
                        help='Format audio : dernier choix, sinon wav.')
    parser.add_argument('--bitrate', type=int, choices=BITRATE_CHOICES,
                        help='Débit en kbit/s : dernier choix, sinon 128 ; sans objet pour WAV.')
    parser.add_argument('--frequency', '--sample-rate', dest='sample_rate', type=frequency,
                        help='Fréquence : 24, 44.1 ou 48 kHz (24 ou 48 pour Opus ; WAV fixé à 24 kHz).')
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, default=None,
                        help='Normaliser le volume final de chaque chapitre ; --no-normalize pour désactiver (sinon choix mémorisé, non par défaut).')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help='1 à 20 workers (20 par défaut) : encodage final des chapitres ; la synthèse utilise un seul moteur GPU.')
    parser.add_argument('--offline', action='store_true', help='Interdire les téléchargements ; modèle déjà préparé requis.')
    parser.add_argument('--yes', '-y', action='store_true', help='Traiter tous les fichiers indiqués sans question.')
    args = parser.parse_args(argv)
    if args.batching is False and args.batch_size is not None:
        parser.error('--batch-size ne peut pas être combiné avec --no-batching.')
    args.paths_explicit = bool(args.paths)
    args.output_explicit = args.output is not None
    if args.output is None:
        args.output = OUTPUT_DIR
    args.format_explicit = args.audio_format is not None
    args.bitrate_explicit = args.bitrate is not None
    args.frequency_explicit = args.sample_rate is not None
    if args.audio_format is None:
        args.audio_format = DEFAULT_AUDIO_FORMAT
    if args.bitrate is None:
        args.bitrate = DEFAULT_BITRATE
    if args.sample_rate is None:
        args.sample_rate = DEFAULT_SAMPLE_RATE
    if args.format_explicit and args.audio_format == 'wav' and args.bitrate_explicit:
        parser.error('--bitrate ne s’applique pas au WAV PCM original.')
    if (args.format_explicit and args.frequency_explicit
            and args.sample_rate not in FORMAT_SAMPLE_RATES[args.audio_format]):
        parser.error('WAV PCM original : la fréquence doit rester à 24 kHz.' if args.audio_format == 'wav'
                     else 'Fréquence incompatible avec Opus : choisissez 24 ou 48 kHz.')
    args.rate_explicit = args.rate is not None
    if args.rate is None:
        args.rate = DEFAULT_SPEED
    if not math.isfinite(args.rate) or not 0.5 <= args.rate <= 3:
        parser.error('--rate doit être compris entre 0.5 et 3.')
    if not 1 <= args.workers <= 20:
        parser.error('--workers doit être compris entre 1 et 20.')
    if not args.paths:
        args.paths = [Path(default_directory) if default_directory is not None else INPUT_DIR]
    return args
