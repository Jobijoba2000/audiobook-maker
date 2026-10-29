"""Encode each completed PCM chapter exactly once."""
from pathlib import Path
import json
import math
import os
import shutil
import subprocess

from epub_audio.config.defaults import (DEFAULT_SAMPLE_RATE, DEFAULT_BITRATE,
                                        DEFAULT_AUDIO_FORMAT, FORMAT_BITRATES, FORMAT_SAMPLE_RATES,
                                        NORMALIZE_LUFS, NORMALIZE_TRUE_PEAK, NORMALIZE_LRA)
from epub_audio.io.fs import ROOT


def tool_executable(name):
    local = ROOT / '.cache/ffmpeg/bin' / (name + '.exe')
    return str(local) if os.name == 'nt' and local.is_file() else shutil.which(name)


def require_tools(audio_format=None, *, normalize=False):
    if audio_format == 'wav' and not normalize:
        return
    for name in ('ffmpeg', 'ffprobe'):
        if not tool_executable(name):
            raise RuntimeError(f'{name} est introuvable. Lancez installer.bat pour installer FFmpeg localement.')
    if audio_format is not None:
        encoder = {'opus': 'libopus', 'mp3': 'libmp3lame', 'aac': 'aac', 'wav': 'pcm_s16le'}[audio_format]
        if not any(len(parts := line.split()) > 1 and parts[1] == encoder
                   for line in run_tool(['ffmpeg', '-hide_banner', '-encoders']).splitlines()):
            raise RuntimeError(f'Cet FFmpeg ne possède pas l’encodeur {encoder}.')


def run_tool(command, *, capture_stderr=False):
    if command[0] in ('ffmpeg', 'ffprobe'):
        command = [tool_executable(command[0]) or command[0], *command[1:]]
    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                            errors='replace', creationflags=flags)
    if result.returncode:
        raise RuntimeError(f'{Path(command[0]).name} a échoué : {result.stderr[-2000:]}')
    return result.stderr if capture_stderr else result.stdout


def chapter_loudness_filter(source):
    """Measure a complete chapter before applying one uniform loudness target."""
    target = f'loudnorm=I={NORMALIZE_LUFS:g}:TP={NORMALIZE_TRUE_PEAK:g}:LRA={NORMALIZE_LRA:g}'
    output = run_tool(['ffmpeg', '-hide_banner', '-nostats', '-nostdin', '-i', str(source),
                       '-map', '0:a:0', '-vn', '-af', target + ':print_format=json',
                       '-f', 'null', '-'], capture_stderr=True)
    try:
        stats = json.loads(output[output.rindex('{'):output.rindex('}') + 1])
        values = {key: float(stats[key]) for key in
                  ('input_i', 'input_tp', 'input_lra', 'input_thresh', 'target_offset')}
        # Silence has no measurable loudness. Keep it silent, with no gain.
        if values['input_i'] == values['input_tp'] == float('-inf'):
            return None
        # Clips shorter than the loudness meter's window have no integrated
        # measurement. Avoid amplifying them; only lower peaks if necessary.
        if values['input_i'] == float('-inf') and math.isfinite(values['input_tp']):
            return f'volume={min(0, NORMALIZE_TRUE_PEAK - values["input_tp"]):g}dB'
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('Mesure de volume non finie.')
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError('Impossible de mesurer le volume du chapitre.') from exc
    names = {'input_i': 'measured_I', 'input_tp': 'measured_TP', 'input_lra': 'measured_LRA',
             'input_thresh': 'measured_thresh', 'target_offset': 'offset'}
    measured = ':'.join(f'{names[key]}={value:g}' for key, value in values.items())
    # Prefer constant gain; FFmpeg limits dynamics only when the targets require it.
    return target + ':' + measured + ':linear=true'


def audio_info(path):
    data = json.loads(run_tool(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                               '-show_entries', 'stream=codec_name,sample_rate,channels,bit_rate:format=duration',
                               '-of', 'json', str(path)]))
    streams = data.get('streams', [])
    duration = float(data.get('format', {}).get('duration', 0))
    if not streams or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f'Fichier audio invalide : {path}')
    return dict(streams[0], duration=duration)


def encode_chapter(source, destination, *, bitrate=DEFAULT_BITRATE, title='', album='', track=1,
                   audio_format=DEFAULT_AUDIO_FORMAT, sample_rate=DEFAULT_SAMPLE_RATE, normalize=False):
    if type(normalize) is not bool:
        raise ValueError('Normalisation invalide : utilisez true ou false.')
    if audio_format == 'wav' and not normalize:
        raise ValueError('Le WAV doit être conservé directement, sans encodage FFmpeg.')
    if audio_format not in FORMAT_BITRATES:
        raise ValueError('Format audio inconnu : ' + str(audio_format))
    if audio_format != 'wav' and (type(bitrate) is not int or bitrate not in FORMAT_BITRATES[audio_format]):
        raise ValueError('Débit audio non pris en charge : ' + str(bitrate))
    if type(sample_rate) is not int or sample_rate not in FORMAT_SAMPLE_RATES[audio_format]:
        raise ValueError(f'Fréquence {sample_rate} Hz incompatible avec {audio_format.upper()}.')
    if audio_format == 'opus':
        encoding = ['-c:a', 'libopus', '-b:a', f'{bitrate}k', '-vbr', 'on',
                    '-application', 'audio', '-compression_level', '10']
    elif audio_format == 'aac':
        encoding = ['-c:a', 'aac', '-profile:a', 'aac_low', '-b:a', f'{bitrate}k', '-movflags', '+faststart']
    elif audio_format == 'mp3':
        encoding = ['-c:a', 'libmp3lame', '-abr', '0', '-b:a', f'{bitrate}k',
                    '-id3v2_version', '3', '-write_xing', '1']
    else:
        encoding = ['-c:a', 'pcm_s16le']
    if normalize:
        loudness = chapter_loudness_filter(source)
        if loudness is not None:
            encoding += ['-af', loudness]
    # loudnorm may internally resample: always restore the chosen export rate.
    encoding += ['-ac', '1', '-ar', str(sample_rate)]
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
               '-i', str(source), '-map', '0:a:0', '-vn', '-map_metadata', '-1',
               *encoding,
               '-metadata', f'title={title}', '-metadata', f'album={album}',
               '-metadata', f'track={track}', str(destination)]
    run_tool(command)
    info = audio_info(destination)
    expected_codec = 'pcm_s16le' if audio_format == 'wav' else audio_format
    if info['codec_name'] != expected_codec or info['channels'] != 1:
        raise RuntimeError('Le codec ou le nombre de canaux de l’export final est incorrect.')
    expected_rate = 48000 if audio_format == 'opus' else sample_rate
    if int(info['sample_rate']) != expected_rate:
        raise RuntimeError('Fréquence de sortie incorrecte.')
    if audio_format == 'mp3' and int(info.get('bit_rate', 0)) != bitrate * 1000:
        raise RuntimeError('L’export final ne contient pas le MP3 mono à débit constant attendu.')
    return info['duration']
