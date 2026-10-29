"""Install the pinned Windows/CUDA environment and local assets, without running TTS."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import venv
import wave
import zipfile

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def native_path(path):
    """Allow long pip/Hub temporary paths without changing Windows settings."""
    value = str(Path(path).resolve())
    if os.name != 'nt' or value.startswith('\\\\?\\'):
        return value
    return ('\\\\?\\UNC\\' + value[2:]) if value.startswith('\\\\') else ('\\\\?\\' + value)


def project_path(root, relative):
    relative = PurePosixPath(str(relative).replace('\\', '/'))
    path = (Path(root) / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'Chemin hors du projet : {relative}')
    return path


def run(command, root, *, env=None, capture=False):
    result = subprocess.run([str(arg) for arg in command], cwd=root, env=env,
                            capture_output=capture, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        detail = (result.stderr or result.stdout or '')[-2500:] if capture else ''
        raise RuntimeError(f'Échec de {Path(command[0]).name} (code {result.returncode}). {detail}')
    return result.stdout if capture else ''


def check_python(sources):
    wanted = sources['python']
    if sys.version_info[:2] != (wanted['major'], wanted['minor']) or struct.calcsize('P') * 8 != wanted['bits']:
        raise RuntimeError('Python 3.12 en version 64 bits est nécessaire. Installez-le depuis python.org.')
    if os.name != 'nt':
        raise RuntimeError('Cet installateur prépare la version Windows x64 avec carte NVIDIA.')


def seed_folders(root, sources):
    root = Path(root)
    sample = sources['sample']
    bundled = project_path(root, sample['source'])
    if not bundled.is_file() or sha256(bundled) != sample['sha256']:
        raise RuntimeError('EPUB de démonstration absent ou altéré dans assets/samples. Récupérez le projet complet.')
    target = project_path(root, sample['destination'])
    for path in (root / 'input', root / 'output', root / '.cache', target.parent):
        path.mkdir(parents=True, exist_ok=True)
    # An existing book is the user's file, even if its bytes differ from our sample.
    created = False
    try:
        with bundled.open('rb') as source, target.open('xb') as destination:
            created = True
            shutil.copyfileobj(source, destination)
    except FileExistsError:
        return False
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    return True


def valid_file(path, item):
    path = Path(path)
    return path.is_file() and path.stat().st_size == item['size'] and sha256(path) == item['sha256']


def model_directory(root):
    return project_path(root, read_json(Path(root) / 'qwen.json')['model_dir'])


def prepare_model(root, model, *, downloader=None):
    folder = model_directory(root)
    folder.mkdir(parents=True, exist_ok=True)
    for item in model['files']:
        target = project_path(folder, item['path'])
        if valid_file(target, item):
            print(f"  Vérifié : {item['path']}", flush=True)
            continue
        if downloader is None:
            from huggingface_hub import hf_hub_download
            downloader = hf_hub_download
        print(f"  Téléchargement : {item['path']}", flush=True)
        downloader(repo_id=model['repo'], revision=model['revision'], filename=item['path'],
                   local_dir=native_path(folder), token=False, force_download=target.exists())
        if not valid_file(target, item):
            raise RuntimeError(f"Poids incomplets ou empreinte incorrecte : {item['path']}. Relancez installer.bat.")
    stamp = {'repo': model['repo'], 'revision': model['revision'], 'files': model['files']}
    (Path(root) / '.cache/model-installation.json').write_text(json.dumps(stamp, indent=2), encoding='utf-8')


def download_archive(source, target):
    target = Path(target)
    if valid_file(target, source):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    last = time.monotonic()
    copied = 0
    request = urllib.request.Request(source['url'], headers={'User-Agent': 'Audiobook-Maker-Installer/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            copied += len(block)
            if time.monotonic() - last > 10:
                print(f"  FFmpeg : {copied / source['size']:.0%}", flush=True)
                last = time.monotonic()
    if not valid_file(temporary, source):
        raise RuntimeError('Archive FFmpeg incomplète ou empreinte incorrecte. Relancez installer.bat.')
    temporary.replace(target)


def prepare_ffmpeg(root, source):
    root = Path(root)
    folder = root / '.cache/ffmpeg'
    stamp = folder / 'installation.json'
    if stamp.is_file():
        try:
            previous = read_json(stamp)
        except (OSError, ValueError):
            previous = {}
        files = previous.get('files', [])
        if (previous.get('sha256') == source['sha256'] and len(files) == 2
                and all(valid_file(project_path(folder, item['path']), item) for item in files)):
            print('  FFmpeg local déjà vérifié.', flush=True)
            return
    archive = root / '.cache/downloads' / f"ffmpeg-{source['version']}.zip"
    download_archive(source, archive)
    wanted = {'ffmpeg.exe', 'ffprobe.exe'}
    installed = []
    with zipfile.ZipFile(archive) as package:
        for name in wanted:
            matches = [entry for entry in package.infolist()
                       if PurePosixPath(entry.filename).name == name and not entry.is_dir()]
            if len(matches) != 1:
                raise RuntimeError(f'Archive FFmpeg inattendue : {name} absent ou ambigu.')
            target = folder / 'bin' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.exe.part')
            with package.open(matches[0]) as data, temporary.open('wb') as output:
                shutil.copyfileobj(data, output)
            temporary.replace(target)
            installed.append({'path':f'bin/{name}', 'size':target.stat().st_size, 'sha256':sha256(target)})
        for entry in package.infolist():
            if PurePosixPath(entry.filename).name.lower() in ('license', 'license.txt', 'copying'):
                (folder / 'LICENSE.txt').write_bytes(package.read(entry))
                break
    stamp.write_text(json.dumps({**source, 'files':installed}, indent=2), encoding='utf-8')


GPU_CHECK = '''
import json, torch, torchaudio
from faster_qwen3_tts import FasterQwen3TTS
assert torch.cuda.is_available(), 'Carte NVIDIA ou pilote CUDA indisponible.'
assert torch.cuda.is_bf16_supported(), 'Le GPU doit prendre en charge bfloat16 (Ampere ou plus récent).'
assert torch.__version__.split('+')[0]=='2.6.0', 'Version PyTorch incorrecte.'
assert torchaudio.__version__.split('+')[0]=='2.6.0', 'Version torchaudio incorrecte.'
assert torch.version.cuda=='12.4', 'La version CUDA 12.4 de PyTorch est requise.'
x=torch.ones((2,2),device='cuda',dtype=torch.bfloat16)
assert (x@x).sum().item()==8
print(json.dumps({'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
                  'torchaudio':torchaudio.__version__, 'cuda':torch.version.cuda}))
'''


def verify(root, python, sources, model, env):
    root = Path(root)
    run([python, '-m', 'pip', 'check'], root, env=env)
    gpu = json.loads(run([python, '-c', GPU_CHECK], root, env=env, capture=True).splitlines()[-1])
    folder = model_directory(root)
    for item in model['files']:
        if not valid_file(project_path(folder, item['path']), item):
            raise RuntimeError(f"Poids manquants ou modifiés : {item['path']}. Relancez installer.bat.")
    config = read_json(folder / 'config.json')
    if config.get('tts_model_size') != '0b6' or config.get('tts_model_type') != 'base':
        raise RuntimeError('Le modèle téléchargé ne correspond pas au 0.6B Base.')
    voice = read_json(root / 'qwen.json')['voice']
    reference = project_path(root, f'voices/references_fr/{voice}.wav')
    if not reference.with_suffix('.txt').is_file():
        raise RuntimeError('La transcription de la voix de référence est absente.')
    with wave.open(str(reference)) as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 24000) or audio.getnframes() == 0:
            raise RuntimeError('La référence vocale fournie doit être mono PCM16 à 24 kHz.')
    binaries = root / '.cache/ffmpeg/bin'
    ffmpeg = binaries / 'ffmpeg.exe'
    ffprobe = binaries / 'ffprobe.exe'
    encoders = run([ffmpeg, '-hide_banner', '-encoders'], root, env=env, capture=True)
    names = {parts[1] for line in encoders.splitlines() if len(parts := line.split()) > 1}
    if not {'libmp3lame', 'aac', 'libopus', 'pcm_s16le'} <= names:
        raise RuntimeError('FFmpeg ne fournit pas tous les codecs nécessaires.')
    run([ffprobe, '-version'], root, env=env, capture=True)
    return {**gpu, 'model':model['repo'], 'revision':model['revision'],
            'model_files_verified':len(model['files']), 'ffmpeg':sources['ffmpeg']['version']}


@contextmanager
def install_lock(root):
    import msvcrt
    folder = Path(root) / '.cache'
    folder.mkdir(exist_ok=True)
    with (folder / 'installer.lock').open('a+b') as lock:
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Une autre installation est déjà en cours dans ce dossier.') from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Installer Faster-Qwen3-TTS 0.6B sous Windows x64.')
    parser.add_argument('--check', action='store_true', help='Vérifier l’installation existante, sans téléchargement ni génération TTS.')
    parser.add_argument('--no-pause', action='store_true', help='Ne pas attendre de touche à la fin du lanceur Windows.')
    args = parser.parse_args(argv)
    sources = read_json(ROOT / 'installation_sources.json')
    model = read_json(ROOT / 'model_sources.json')['model']
    check_python(sources)
    python = ROOT / '.venv/Scripts/python.exe'
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                       PIP_CACHE_DIR=native_path(ROOT / '.cache/pip'), HF_HOME=native_path(ROOT / '.cache/huggingface'),
                       HF_HUB_OFFLINE='0', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
    if Path(sys.executable).resolve() != python.resolve():
        if not python.is_file():
            if args.check:
                raise RuntimeError('Environnement .venv absent. Lancez installer.bat sans --check.')
            print('[1/6] Création de .venv avec Python 3.12 x64…', flush=True)
            temporary = ROOT / '.cache/pip-tmp'
            temporary.mkdir(parents=True, exist_ok=True)
            previous = {key:os.environ.get(key) for key in ('TEMP','TMP')}
            try:
                os.environ.update(TEMP=native_path(temporary), TMP=native_path(temporary))
                venv.EnvBuilder(with_pip=True).create(ROOT / '.venv')
            finally:
                for key,value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        return subprocess.call([str(python), str(ROOT / 'installer.py'), *(argv if argv is not None else sys.argv[1:])],
                               cwd=ROOT, env=environment)
    temporary = ROOT / '.cache/pip-tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    environment.update(TEMP=native_path(temporary), TMP=native_path(temporary))
    with install_lock(ROOT):
        if not args.check:
            seed_folders(ROOT, sources)
            print('[2/6] Installation des dépendances aux versions validées…', flush=True)
            pip = [python, '-m', 'pip', '--disable-pip-version-check', '--log', ROOT / '.cache/pip-install.log']
            run([*pip, 'install', '--upgrade', *(f'{name}=={version}' for name,version in sources['build_tools'].items())], ROOT, env=environment)
            cuda = sources['pytorch']
            run([*pip, 'install', f"torch=={cuda['torch']}", f"torchaudio=={cuda['torchaudio']}",
                 '--index-url', cuda['index_url']], ROOT, env=environment)
            run([*pip, 'install', '--no-build-isolation', '-r', ROOT / 'requirements.txt'], ROOT, env=environment)
            print('[3/6] Vérification de la carte NVIDIA et de CUDA…', flush=True)
            run([python, '-c', GPU_CHECK], ROOT, env=environment, capture=True)
            print('[4/6] Téléchargement et vérification des poids 0.6B et du tokenizer…', flush=True)
            # Set before importing the Hub; no token/account is required for this public model.
            os.environ.update(HF_HOME=environment['HF_HOME'], HF_HUB_OFFLINE='0', HF_HUB_DISABLE_TELEMETRY='1')
            prepare_model(ROOT, model)
            print('[5/6] Installation de FFmpeg et FFprobe locaux…', flush=True)
            prepare_ffmpeg(ROOT, sources['ffmpeg'])
        print('[6/6] Contrôles de l’installation, sans génération TTS…', flush=True)
        report = verify(ROOT, python, sources, model, environment)
        if not args.check:
            (ROOT / '.cache/installation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Installation vérifiée : {report['gpu']}, Faster 0.6B, CUDA {report['cuda']}.", flush=True)
        print('Lancez start_ebook.bat. Le premier chapitre de Vingt mille lieues est dans input/test.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInstallation interrompue. Vous pouvez relancer installer.bat.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'\nInstallation incomplète : {exc}\nJournal pip : {ROOT / ".cache/pip-install.log"}', file=sys.stderr)
        raise SystemExit(1)
