"""Persist synthesis settings and console question preferences."""
import json
import math
import os
from pathlib import Path
import tempfile
from epub_audio.io.fs import ROOT
from epub_audio.config.defaults import (DEFAULT_AUDIO_FORMAT, DEFAULT_BITRATE, DEFAULT_SAMPLE_RATE,
                                        AUDIO_FORMATS, BITRATE_CHOICES, SAMPLE_RATE_CHOICES, FORMAT_SAMPLE_RATES,
                                        SEGMENTATION_MODES, DEFAULT_SEGMENTATION, BATCH_SIZE_CHOICES,
                                        DEFAULT_CONVERT_NUMBERS, DEFAULT_NORMALIZE)

PREFERENCES_FILE = ROOT / '.cache' / 'preferences.json'
PROMPT_NAMES = ('voice', 'batching', 'shrinking_batching', 'segmentation', 'llm_chapters', 'convert_numbers', 'audio_format', 'bitrate', 'sample_rate', 'normalize', 'confirmation')


def load_preferences():
    try:
        data = json.loads(PREFERENCES_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    preferences = {}
    if type(data.get('normalize')) is bool:
        preferences['normalize'] = data['normalize']
    prompts = data.get('prompts')
    if isinstance(prompts, dict):
        preferences['prompts'] = {name: prompts[name] for name in PROMPT_NAMES if type(prompts.get(name)) is bool}
    if type(data.get('convert_numbers')) is bool:
        preferences['convert_numbers'] = data['convert_numbers']
    if type(data.get('llm_chapters')) is bool:
        preferences['llm_chapters'] = data['llm_chapters']
    if type(data.get('faster_small_batch_size')) is int and data['faster_small_batch_size'] in BATCH_SIZE_CHOICES:
        preferences['faster_small_batch_size'] = data['faster_small_batch_size']
    if type(data.get('faster_small_batching')) is bool:
        preferences['faster_small_batching'] = data['faster_small_batching']
    if type(data.get('shrinking_batching')) is bool:
        preferences['shrinking_batching'] = data['shrinking_batching']
    if isinstance(data.get('segmentation'), str) and data['segmentation'] in SEGMENTATION_MODES:
        preferences['segmentation'] = data['segmentation']
    # Migrate the last Faster reference from old per-engine preferences.
    voices = data.get('engine_voices')
    voice = voices.get('faster-qwen') if isinstance(voices, dict) else None
    voice = voice or data.get('voice')
    if isinstance(voice, str) and voice:
        preferences['voice'] = voice
    rate = data.get('rate')
    if (isinstance(rate, (int, float)) and not isinstance(rate, bool)
            and math.isfinite(rate) and 0.5 <= rate <= 3):
        preferences['rate'] = float(rate)
    codec = data.get('audio_format')
    if isinstance(codec, str) and codec in AUDIO_FORMATS:
        preferences['audio_format'] = codec
    bitrate = data.get('bitrate')
    if type(bitrate) is int and bitrate in BITRATE_CHOICES:
        preferences['bitrate'] = bitrate
    sample_rate = data.get('sample_rate')
    if (type(sample_rate) is int and sample_rate in SAMPLE_RATE_CHOICES
            and (codec not in AUDIO_FORMATS or sample_rate in FORMAT_SAMPLE_RATES[codec])):
        preferences['sample_rate'] = sample_rate
    return preferences


def save_preferences(voice, rate, bitrate=DEFAULT_BITRATE, audio_format=DEFAULT_AUDIO_FORMAT,
                     sample_rate=DEFAULT_SAMPLE_RATE,
                     segmentation=DEFAULT_SEGMENTATION, faster_small_batching=False, faster_small_batch_size=2,
                     prompts=None, convert_numbers=None, llm_chapters=None, normalize=None, shrinking_batching=None):
    existing = load_preferences()
    if shrinking_batching is None:
        shrinking_batching = existing.get('shrinking_batching', True)
    if type(shrinking_batching) is not bool:
        raise ValueError('Réduction progressive des lots invalide : utilisez true ou false.')
    if normalize is None:
        normalize = existing.get('normalize', DEFAULT_NORMALIZE)
    if type(normalize) is not bool:
        raise ValueError('Normalisation invalide : utilisez true ou false.')
    if prompts is None:
        prompts = existing.get('prompts', {})
    if convert_numbers is None:
        convert_numbers = existing.get('convert_numbers', DEFAULT_CONVERT_NUMBERS)
    if type(convert_numbers) is not bool:
        raise ValueError('Conversion des nombres invalide : utilisez true ou false.')
    if llm_chapters is None:
        llm_chapters = existing.get('llm_chapters', False)
    if type(llm_chapters) is not bool:
        raise ValueError('Découpage LLM invalide : utilisez true ou false.')
    if (not isinstance(prompts, dict)
            or any(name not in PROMPT_NAMES or type(value) is not bool for name, value in prompts.items())):
        raise ValueError('Affichage des questions invalide : utilisez true ou false pour chaque question.')
    if type(faster_small_batch_size) is not int or faster_small_batch_size not in BATCH_SIZE_CHOICES:
        raise ValueError('Taille du groupe de batching invalide : choisissez 2, 3, 4, 5, 6, 7 ou 8.')
    if type(faster_small_batching) is not bool:
        raise ValueError('Réglage de batching invalide.')
    if not isinstance(segmentation, str) or segmentation not in SEGMENTATION_MODES:
        raise ValueError('Découpage des segments inconnu : ' + str(segmentation))
    if audio_format == 'wav':
        bitrate, sample_rate = None, 24000
    PREFERENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=PREFERENCES_FILE.parent,
                                         prefix='.preferences-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            data = {'voice': voice, 'rate': rate, 'audio_format': audio_format,
                    'bitrate': bitrate, 'sample_rate': sample_rate,
                    'segmentation': segmentation, 'faster_small_batching': faster_small_batching,
                    'faster_small_batch_size': faster_small_batch_size, 'convert_numbers': convert_numbers,
                    'llm_chapters': llm_chapters, 'normalize': normalize, 'shrinking_batching': shrinking_batching}
            if prompts:
                data['prompts'] = prompts
            json.dump(data, stream,
                      ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, PREFERENCES_FILE)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
