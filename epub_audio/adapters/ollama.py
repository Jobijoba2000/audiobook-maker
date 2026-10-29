"""Local, structured chapter analysis, isolated from the TTS environment."""
from contextlib import AbstractContextManager
import json
import os
import shutil
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from epub_audio.io.fs import ROOT

DEFAULT_LLM_MODEL = 'qwen3.5:4b'
DEFAULT_LLM_URL = 'http://127.0.0.1:11434'


class ChapterLLMError(RuntimeError):
    pass


class OllamaChapters(AbstractContextManager):
    def __init__(self, model=DEFAULT_LLM_MODEL, url=DEFAULT_LLM_URL, *, timeout=240):
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1', '::1')
                or parsed.username or parsed.password or parsed.path not in ('', '/')
                or parsed.query or parsed.fragment):
            raise ValueError('Le LLM doit utiliser une adresse HTTP locale (localhost ou 127.0.0.1).')
        if not isinstance(model, str) or not model.strip() or 'cloud' in model.lower():
            raise ValueError('Choisissez un modèle LLM local.')
        self.model, self.url, self.timeout = model, url.rstrip('/'), timeout
        self._opener = build_opener(ProxyHandler({}))
        self._process = None
        self._log = None
        self._loaded = False
        self._ready = False

    def _request(self, endpoint, payload=None, *, timeout=None):
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request = Request(self.url + endpoint, data=data, headers={'Content-Type': 'application/json'})
        try:
            with self._opener.open(request, timeout=timeout or self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise ChapterLLMError(f'Ollama a refusé la demande (HTTP {exc.code}). '
                                  f'Vérifiez le modèle avec « ollama pull {self.model} ».') from exc
        except (URLError, OSError, ValueError) as exc:
            raise ChapterLLMError('LLM local indisponible ou réponse invalide. '
                                  'Démarrez Ollama et préparez le modèle : '
                                  f'ollama pull {self.model}') from exc
        if not isinstance(result, dict) or result.get('error'):
            raise ChapterLLMError('Réponse Ollama invalide ou génération impossible.')
        return result

    def _ensure_ready(self):
        if self._ready:
            return
        try:
            self._request('/api/tags', timeout=2)
        except ChapterLLMError:
            portable = ROOT / '.cache' / 'ollama' / 'ollama.exe'
            executable = str(portable) if portable.is_file() else shutil.which('ollama')
            if not executable:
                raise
            environment = os.environ.copy()
            environment['OLLAMA_HOST'] = self.url
            environment['OLLAMA_NO_CLOUD'] = '1'
            environment['OLLAMA_NUM_PARALLEL'] = '1'
            if portable.is_file():
                environment['OLLAMA_MODELS'] = str(ROOT / '.cache' / 'ollama-models')
            log_path = ROOT / '.cache' / 'chapter-llm.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = log_path.open('ab')
            self._process = subprocess.Popen(
                [executable, 'serve'], env=environment, stdin=subprocess.DEVNULL,
                stdout=self._log, stderr=self._log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            for _ in range(60):
                try:
                    self._request('/api/tags', timeout=1)
                    break
                except ChapterLLMError:
                    if self._process.poll() is not None:
                        raise ChapterLLMError('Le moteur Ollama local n’a pas démarré. Voir .cache/chapter-llm.log.')
                    time.sleep(0.5)
            else:
                raise ChapterLLMError('Le démarrage du LLM local a dépassé le délai prévu.')
        installed = self._request('/api/tags', timeout=5).get('models', [])
        selected = [item for item in installed if isinstance(item, dict)
                    and item.get('name') in (self.model, self.model + ':latest')]
        if not selected:
            raise ChapterLLMError(f'Modèle LLM absent : lancez « ollama pull {self.model} ».')
        details = self._request('/api/show', {'model': self.model}, timeout=10)
        if any(item.get('remote_model') or item.get('remote_host') for item in selected + [details]):
            raise ChapterLLMError('Le découpage nécessite un modèle installé en local, sans relais cloud.')
        self._ready = True

    def classify(self, system, content, schema):
        self._ensure_ready()
        # Even a failed/aborted generation can leave weights in video memory.
        self._loaded = True
        result = self._request('/api/chat', {
            'model': self.model, 'stream': False, 'think': False, 'format': schema,
            'keep_alive': '5m',
            'options': {'temperature': 0, 'seed': 42, 'num_ctx': 12288, 'num_predict': 4096},
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': content}],
        })
        if result.get('done') is not True or result.get('done_reason') == 'length':
            raise ChapterLLMError('L’analyse LLM est incomplète ; le découpage habituel est conservé.')
        try:
            return json.loads(result['message']['content'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ChapterLLMError('Le LLM n’a pas fourni un plan JSON valide.') from exc

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._loaded:
                self._request('/api/generate', {'model': self.model, 'keep_alive': 0}, timeout=60)
        except ChapterLLMError:
            raise ChapterLLMError('Impossible de libérer le LLM de la mémoire GPU. '
                                  'Arrêtez-le dans Ollama avant de lancer la synthèse.')
        finally:
            if self._process is not None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=10)
            if self._log is not None:
                self._log.close()
        return False
