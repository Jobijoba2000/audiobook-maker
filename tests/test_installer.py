"""Fresh-install assets, pinned downloads, recovery and publication exclusions."""
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import installer
from epub_audio.io.epub import read_book
from epub_audio.adapters import ffmpeg

ROOT = Path(__file__).resolve().parents[1]


def item(name, data):
    return {'path':name, 'size':len(data), 'sha256':hashlib.sha256(data).hexdigest()}


class InstallerAssetsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / '.cache').mkdir()
        (self.root / 'qwen.json').write_text(json.dumps({'model_dir':'models/model'}), encoding='utf-8')
        self.enterContext(patch('sys.stdout', new=io.StringIO()))

    def sample(self):
        source = self.root / 'assets/sample.epub'
        source.parent.mkdir()
        source.write_bytes(b'Bundled public-domain EPUB')
        return {'sample':{'source':'assets/sample.epub', 'destination':'input/test/sample.epub',
                          'sha256':installer.sha256(source)}}

    def test_folders_and_sample_are_created_without_overwriting_a_user_book(self):
        sources = self.sample()
        self.assertTrue(installer.seed_folders(self.root,sources))
        target = self.root / sources['sample']['destination']
        self.assertTrue((self.root / 'output').is_dir())
        self.assertEqual(target.read_bytes(), (self.root/'assets/sample.epub').read_bytes())
        target.write_bytes(b'My own book')
        self.assertFalse(installer.seed_folders(self.root,sources))
        self.assertEqual(target.read_bytes(),b'My own book')

    def test_interrupted_copy_can_be_retried(self):
        sources = self.sample()
        target = self.root / sources['sample']['destination']
        with patch('installer.shutil.copyfileobj', side_effect=OSError('disk interrupted')):
            with self.assertRaises(OSError):
                installer.seed_folders(self.root,sources)
        self.assertFalse(target.exists())
        self.assertTrue(installer.seed_folders(self.root,sources))

    def test_altered_bundled_sample_is_rejected(self):
        sources = self.sample()
        (self.root / sources['sample']['source']).write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'EPUB'):
            installer.seed_folders(self.root,sources)
        self.assertFalse((self.root/'input').exists())

    def test_paths_cannot_escape_the_repository(self):
        for name in ('../outside', '/absolute', 'models/../../outside'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                installer.project_path(self.root,name)

    @unittest.skipUnless(os.name == 'nt','Windows long paths')
    def test_windows_native_path_is_idempotent_and_preserves_file_contents(self):
        path = self.root / 'espace été.txt'
        path.write_text('Exemple français.', encoding='utf-8')
        native = installer.native_path(path)
        self.assertTrue(native.startswith('\\\\?\\'))
        self.assertEqual(installer.native_path(native),native)
        self.assertEqual(Path(native).read_text(encoding='utf-8'),'Exemple français.')

    def test_model_download_is_pinned_validated_and_not_repeated(self):
        data = b'model bytes'
        spec = {'repo':'Qwen/test', 'revision':'a'*40, 'files':[item('model.safetensors',data)]}
        target = self.root / 'models/model/model.safetensors'
        def download(**kwargs):
            self.assertEqual(kwargs['revision'],'a'*40)
            self.assertFalse(kwargs['token'])
            self.assertEqual(kwargs['filename'],'model.safetensors')
            target.write_bytes(data)
        get = Mock(side_effect=download)
        installer.prepare_model(self.root,spec,downloader=get)
        installer.prepare_model(self.root,spec,downloader=get)
        get.assert_called_once()
        stamp = installer.read_json(self.root / '.cache/model-installation.json')
        self.assertEqual(stamp['revision'],'a'*40)

    def test_corrupt_model_is_repaired_even_when_its_size_matches(self):
        spec = {'repo':'Qwen/test', 'revision':'b'*40, 'files':[item('model.safetensors',b'good')]}
        target = self.root / 'models/model/model.safetensors'
        target.parent.mkdir(parents=True)
        target.write_bytes(b'bad!')
        def repair(**kwargs):
            self.assertTrue(kwargs['force_download'])
            target.write_bytes(b'good')
        installer.prepare_model(self.root,spec,downloader=repair)
        self.assertEqual(target.read_bytes(),b'good')

    def test_download_with_bad_hash_never_marks_model_installed(self):
        spec = {'repo':'Qwen/test', 'revision':'a'*40, 'files':[item('model.safetensors',b'good')]}
        def corrupt(**kwargs):
            (self.root/'models/model/model.safetensors').write_bytes(b'bad!')
        with self.assertRaisesRegex(RuntimeError,'empreinte'):
            installer.prepare_model(self.root,spec,downloader=corrupt)
        self.assertFalse((self.root/'.cache/model-installation.json').exists())

    def test_ffmpeg_extracts_only_tools_and_reuses_verified_binaries(self):
        archive = self.root / '.cache/downloads/ffmpeg-test.zip'
        archive.parent.mkdir()
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('package/bin/ffmpeg.exe',b'ffmpeg fixture')
            z.writestr('package/bin/ffprobe.exe',b'ffprobe fixture')
            z.writestr('package/LICENSE',b'GPL license fixture')
            z.writestr('../outside.txt',b'not extracted')
        source = {**item('',archive.read_bytes()),'version':'test','url':'https://invalid.example/ffmpeg.zip'}
        with patch('installer.urllib.request.urlopen',side_effect=AssertionError('unexpected network')):
            installer.prepare_ffmpeg(self.root,source)
            installer.prepare_ffmpeg(self.root,source)
        self.assertEqual((self.root/'.cache/ffmpeg/bin/ffprobe.exe').read_bytes(),b'ffprobe fixture')
        self.assertTrue((self.root/'.cache/ffmpeg/LICENSE.txt').exists())
        self.assertFalse((self.root/'outside.txt').exists())

    def test_local_ffmpeg_is_found_without_system_path(self):
        binary = self.root / '.cache/ffmpeg/bin/ffmpeg.exe'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'fixture')
        with patch('epub_audio.adapters.ffmpeg.ROOT',self.root), \
             patch('epub_audio.adapters.ffmpeg.shutil.which',return_value=None):
            if os.name == 'nt':
                self.assertEqual(ffmpeg.tool_executable('ffmpeg'),str(binary))
            self.assertIsNone(ffmpeg.tool_executable('ffprobe'))


class PublicationTests(unittest.TestCase):
    def test_bundled_epub_contains_one_complete_chapter_without_images(self):
        sources = installer.read_json(ROOT / 'installation_sources.json')
        sample = ROOT / sources['sample']['source']
        self.assertEqual(installer.sha256(sample),sources['sample']['sha256'])
        self.assertLess(sample.stat().st_size,100_000)
        with zipfile.ZipFile(sample) as package:
            self.assertEqual(package.read('mimetype'),b'application/epub+zip')
            self.assertFalse(any(n.lower().endswith(('.png','.jpg','.jpeg')) for n in package.namelist()))
        book = read_book(sample)
        self.assertEqual(len(book.chapters),1)
        paragraphs = book.chapters[0].paragraphs
        self.assertTrue(any(p.startswith('L’année 1866') for p in paragraphs))
        self.assertTrue(paragraphs[-1].endswith('de ce formidable cétacé.'))

    @unittest.skipUnless(shutil.which('git') and (ROOT / '.git').exists(), 'Requires a Git checkout')
    def test_git_excludes_personal_data_and_accepts_only_shared_assets(self):
        ignored = ['input/private.epub','output/private.wav','models/model.safetensors',
                   '.venv/pyvenv.cfg','.venv-check/file.py','.cache/preferences.json',
                   'voices/references_fr/private.wav','voices/references_fr/private.txt',
                   'voices/private.wav','.env','other/model.gguf']
        shared = ['installer.py','installation_sources.json','model_sources.json',
                  'assets/samples/vingt-mille-lieues-chapitre-1.epub',
                  'voices/references_fr/expr_chap_s01_0004-7e5a77eb.wav',
                  'voices/references_fr/expr_chap_s01_0001.txt','voices/references_fr/README.md']
        result = subprocess.run(['git','check-ignore','--no-index','--stdin'],cwd=ROOT,
                                input=('\n'.join(ignored+shared)+'\n').encode(),capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(set(result.stdout.decode().splitlines()),set(ignored))

    def test_voice_metadata_has_attribution_without_a_personal_path(self):
        data = installer.read_json(ROOT / 'voices/references_fr/expr_chap_s01_0004-7e5a77eb.json')
        self.assertEqual(data['license'],'CC-BY-4.0')
        self.assertFalse(Path(data['source']).is_absolute())
        self.assertNotIn('Users',data['source'])

    @unittest.skipUnless(os.name == 'nt','Windows bootstrap')
    def test_windows_launcher_help_works_outside_the_project(self):
        environment = dict(os.environ,AUDIOBOOK_PYTHON=os.sys.executable,PYTHONUTF8='1')
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                                     str(ROOT/'scripts/bootstrap.ps1'),'--help'],cwd=cwd,
                                    env=environment,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--check',result.stdout)
