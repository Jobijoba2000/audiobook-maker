"""Faster-Qwen3-TTS 0.6B Base, loaded once with a local voice reference."""
from pathlib import Path
import hashlib
from datetime import datetime
import uuid
import json
import math
import os
import tempfile
import threading
import time

from epub_audio.io.fs import ROOT
from epub_audio.config.defaults import ENGINE_NAME, DEFAULT_ENGINE, DEFAULT_MODEL_SIZE, BATCH_SIZE_CHOICES

CONFIG_FILE = ROOT / 'qwen.json'


def configuration():
    data = json.loads(CONFIG_FILE.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('Configuration Qwen invalide.')
    data['engine'] = DEFAULT_ENGINE
    data['model_size'] = DEFAULT_MODEL_SIZE
    for key in ('voice', 'model_dir'):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f'Configuration Qwen : {key} absent.')
    if type(data.get('seed')) is not int or not 0 <= data['seed'] < 2**32:
        raise ValueError('Graine Qwen invalide.')
    if type(data.get('sampling')) is not bool:
        raise ValueError('Mode de génération Qwen invalide.')
    for key, low, high in (('temperature', 0.1, 2), ('repetition_penalty', 1, 2)):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(f'Configuration Qwen : {key} invalide.')
    if type(data.get('max_new_tokens')) is not int or not 128 <= data['max_new_tokens'] <= 8192:
        raise ValueError('Limite de génération Qwen invalide.')
    maximum = data.setdefault('max_segment_characters', None)
    if maximum is not None and (type(maximum) is not int or not 100 <= maximum <= 1500):
        raise ValueError('Taille des passages Qwen invalide.')
    return data


def list_available_voices():
    from epub_audio.io.voices import reference_voices
    return reference_voices(ROOT / 'voices/references_fr')


def is_valid_voice_id(voice):
    return voice in [item['id'] for item in list_available_voices()]


class _ProgressNotificationError(Exception):
    """Keep a display failure separate from recoverable model RuntimeErrors."""


class QwenEngine:
    individual_progress = True
    # Un seul moteur GPU, partagé par une file de synthèse séquentielle.
    serial_gpu = True

    def __init__(self, voice=None, offline=True, message=print, speed=1.0, batching=False, batch_size=2, continuous_batching=False):
        self.settings = configuration()
        self.engine_id = DEFAULT_ENGINE
        self.model_size = DEFAULT_MODEL_SIZE
        if type(batching) is not bool:
            raise ValueError('Réglage de batching invalide.')
        if type(batch_size) is not int or batch_size not in BATCH_SIZE_CHOICES:
            raise ValueError('Taille du groupe de batching invalide : choisissez 2, 3, 4, 5, 6, 7 ou 8.')
        if type(continuous_batching) is not bool:
            raise ValueError('Réglage de batching continu invalide.')
        self.continuous_batching = batching and continuous_batching
        self.batching = batching
        self.batch_size = batch_size if batching else 1
        self.batch_backend = None
        self.voice = voice or self.settings['voice']
        self.message = message
        self.speed = speed
        self.model = None
        self.fast_model = None
        self.run_id = uuid.uuid4().hex
        self.prompt = None
        self.lock = threading.RLock()
        self.calls = 0

    @property
    def max_segment_characters(self):
        return self.settings['max_segment_characters']

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        value = float(value)
        if not math.isfinite(value) or not 0.5 <= value <= 3:
            raise ValueError('La vitesse doit être comprise entre 0.5 et 3.')
        self._speed = value

    def prepare(self):
        with self.lock:
            if self.model is not None and self.prompt is not None:
                return
            if not is_valid_voice_id(self.voice):
                raise ValueError('Référence Qwen introuvable : ' + self.voice)
            paths = [p for p in (ROOT / 'voices/references_fr').iterdir()
                     if p.stem == self.voice and p.suffix.lower() in ('.wav', '.flac')]
            if len(paths) != 1:
                raise ValueError('Référence Qwen ambiguë : ' + self.voice)
            reference = paths[0]
            transcript = reference.with_suffix('.txt').read_text(encoding='utf-8-sig').strip()
            if not transcript:
                raise ValueError('La transcription de la référence vocale est vide.')
            model_dir = (ROOT / self.settings['model_dir']).resolve()
            for filename in ('config.json', 'model.safetensors', 'speech_tokenizer/model.safetensors'):
                if not (model_dir / filename).is_file():
                    raise RuntimeError('Fichier Qwen manquant : ' + str(model_dir / filename))
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            import torch
            from transformers import set_seed
            if not torch.cuda.is_available():
                raise RuntimeError('Qwen nécessite ici la carte NVIDIA et PyTorch CUDA.')
            torch.set_num_threads(4)
            set_seed(self.settings['seed'])
            self.message(f'Chargement de {ENGINE_NAME} {self.model_size} et de la référence SIWIS…')
            from faster_qwen3_tts import FasterQwen3TTS
            attention = 'eager'
            fast = FasterQwen3TTS.from_pretrained(str(model_dir), device='cuda:0',
                dtype=torch.bfloat16, attn_implementation=attention,
                max_seq_len=3072, local_files_only=True)
            model = fast.model
            if model.model.tts_model_type != 'base' or model.model.tts_model_size != '0b6':
                raise RuntimeError('Le modèle chargé ne correspond pas à Qwen3-TTS 0.6B Base.')
            # Même valeur que le repli automatique de Transformers, définie une seule fois.
            talker_config = model.model.talker.generation_config
            if talker_config.pad_token_id is None:
                talker_config.pad_token_id = model.model.config.talker_config.codec_eos_token_id
            with torch.inference_mode():
                prompt = model.create_voice_clone_prompt(ref_audio=str(reference), ref_text=transcript,
                                                        x_vector_only_mode=False)
            fast.predictor_graph.do_sample = self.settings['sampling']
            fast.predictor_graph.temperature = self.settings['temperature']
            fast.predictor_graph.top_k = 50
            fast.predictor_graph.top_p = 1.0
            self.message('Préparation de l’accélération GPU (une fois au démarrage)…')
            fast.warmup(prefill_len=512)
            if self.batching:
                self.message(f'Préparation du batching par {self.batch_size} passages (une fois au démarrage)…')
                try:
                    from epub_audio.core.tts_batch import FasterBatchBackend
                    self.batch_backend = FasterBatchBackend(fast, self.settings, batch_size=self.batch_size,
                                                            adaptive_cache=not self.continuous_batching)
                    if not self.continuous_batching:
                        self.message('Cache GPU adaptatif prêt : 768 / 1024 / 1536 / 3072 positions.')
                except RuntimeError as exc:
                    raise RuntimeError('Impossible de préparer le batching. Relancez avec un groupe plus petit ou sans batching. '
                                       + str(exc)) from exc
            self.model, self.prompt, self.fast_model = model, prompt, fast
            report = dict(engine=self.engine_id, engine_version='0.3.2', run_id=self.run_id,
                          model=f'Qwen3-TTS-{self.model_size}-Base', model_size=self.model_size,
                          batching=self.batching, batch_size=self.batch_size, continuous_batching=self.continuous_batching,
                          adaptive_cache=self.batching and not self.continuous_batching,
                          precision='bfloat16', attention=attention,
                          voice=self.voice, reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                          reference_text=transcript, settings=self.settings, gpu=torch.cuda.get_device_name())
            cache = ROOT / '.cache'
            cache.mkdir(exist_ok=True)
            (cache / 'qwen-last-run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            if self.continuous_batching:
                self.message(f'Batching continu : jusqu’à {self.batch_size} passages simultanés.')
            self.message(f'{ENGINE_NAME} prêt. La même voix sera réutilisée pour tous les passages.')

    def initialize_worker(self):
        # OpenMP garde aussi un état par thread : initialiser le thread qui calcule.
        import torch
        torch.set_num_threads(4)

    def _generate(self, text):
        common = dict(text=text, language='French', voice_clone_prompt=self.prompt,
                      non_streaming_mode=True, do_sample=self.settings['sampling'],
                      temperature=self.settings['temperature'], top_k=50, top_p=1.0,
                      repetition_penalty=self.settings['repetition_penalty'],
                      max_new_tokens=self.settings['max_new_tokens'])
        self.fast_model.last_generation_timing = None
        waves, rate = self.fast_model.generate_voice_clone(
            **common, append_silence=False, xvec_only=False)
        return waves, rate, self.fast_model.last_generation_timing

    def synthesize(self, text, destination):
        import torch
        from transformers import set_seed
        destination = Path(destination)
        with self.lock:
            if self.model is None or self.prompt is None or self.fast_model is None:
                raise RuntimeError('Le moteur Qwen doit être préparé avant la synthèse.')
            if not text.strip():
                raise ValueError('Passage Qwen vide.')
            set_seed(self.settings['seed'])
            started_at = datetime.now().astimezone().isoformat()
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    waves, rate, timing = self._generate(text)
                if not isinstance(timing, dict) or timing.get('ended_with_eos') is not True:
                    raise RuntimeError(f'{ENGINE_NAME} a atteint sa limite avant de terminer le passage ; réessai en morceaux plus courts.')
                return self._save_wave(text, destination, waves[0], rate, timing, started_at, started)
            except RuntimeError:
                torch.cuda.empty_cache()
                raise

    def _save_wave(self, text, destination, wave, rate, timing, started_at, started,
                   batch_size=1, batch_id=None, generation_seconds=None):
        import numpy as np
        import soundfile as sf
        samples = np.asarray(wave, dtype=np.float32)
        if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
            raise RuntimeError('Qwen a produit un audio vide ou invalide.')
        if rate != 24000:
            raise RuntimeError('Le format renvoyé par le modèle Qwen est inattendu.')
        sf.write(str(destination), samples, rate, subtype='PCM_16')
        if self.speed != 1:
            from epub_audio.adapters.ffmpeg import run_tool
            with tempfile.TemporaryDirectory(prefix='.speed-', dir=destination.parent) as temp:
                adjusted = Path(temp) / 'adjusted.wav'
                run_tool(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(destination),
                          '-af', f'atempo={self.speed:g}', '-c:a', 'pcm_s16le', str(adjusted)])
                adjusted.replace(destination)
        self.calls += 1
        with (ROOT / '.cache/qwen-segments.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(dict(engine=self.engine_id, model_size=self.model_size, run_id=self.run_id,
                started_at=started_at, finished_at=datetime.now().astimezone().isoformat(),
                voice=self.voice, characters=len(text), text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                generation_seconds=round(generation_seconds if generation_seconds is not None else time.perf_counter() - started, 3),
                duration_seconds=len(samples) / rate / self.speed, finished=True,
                generated_tokens=timing.get('steps'), batch_size=batch_size, batch_id=batch_id,
                continuous_batching=timing.get('continuous_batching', False))) + '\n')
        return destination

    def synthesize_batch(self, texts, destinations, *, on_progress=None):
        """Keep independent PCM outputs; retry incomplete or failed groups individually."""
        from epub_audio.core.tts_retry import synthesize_with_fallback, retry_segment
        import torch
        from transformers import set_seed
        if not self.batching or not 1 <= len(texts) <= self.batch_size or len(texts) != len(destinations):
            raise ValueError(f'Le batching attend de 1 à {self.batch_size} passages avec leurs destinations.')
        if any(not text.strip() for text in texts):
            raise ValueError('Passage Qwen vide.')
        with self.lock:
            if self.batch_backend is None or self.model is None or self.prompt is None:
                raise RuntimeError('Le moteur Qwen doit être préparé avant la synthèse.')
            def notify(index, state):
                if on_progress is not None:
                    on_progress(index, state)

            def notify_generated(index):
                try:
                    notify(index, 'generated')
                except RuntimeError as exc:
                    raise _ProgressNotificationError() from exc

            if len(texts) == 1:
                result = synthesize_with_fallback(self, texts[0], destinations[0])
                notify(0, 'ready')
                return [result]
            set_seed(self.settings['seed'])
            started_at = datetime.now().astimezone().isoformat()
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    notifications = ({'on_generated': notify_generated}
                                     if on_progress is not None else {})
                    outputs = self.batch_backend.generate(texts, self.prompt, **notifications)
                if len(outputs) != len(texts):
                    raise RuntimeError('Le batching n’a pas renvoyé tous les passages.')
            except _ProgressNotificationError as exc:
                raise exc.__cause__ from None
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                for index in range(len(texts)):
                    notify(index, 'retry')
                results = []
                for index, (text, path) in enumerate(zip(texts, destinations)):
                    results.append(retry_segment(self, text, path, f'Groupe interrompu : {exc}.'))
                    notify(index, 'ready')
                return results
            batch_id = uuid.uuid4().hex
            seconds_per_sample = (time.perf_counter() - started) / len(texts)
            results = []
            for index, (text, path, (wave, rate, timing)) in enumerate(zip(texts, destinations, outputs)):
                try:
                    if wave is None or timing.get('ended_with_eos') is not True:
                        if timing.get('estimated_limit_reached'):
                            raise RuntimeError(
                                f'Limite estimée atteinte ({timing["generation_limit"]} étapes '
                                f'pour {len(text)} caractères).')
                        raise RuntimeError('Un passage du groupe ne s’est pas terminé.')
                    result = self._save_wave(text, Path(path), wave, rate, timing,
                        started_at, started, batch_size=len(texts), batch_id=batch_id,
                        generation_seconds=seconds_per_sample)
                except RuntimeError as exc:
                    notify(index, 'retry')
                    self._record_batch_failure(text, timing, batch_id, started_at, str(exc))
                    torch.cuda.empty_cache()
                    result = retry_segment(self, text, path, str(exc))
                results.append(result)
                notify(index, 'ready')
            return results


    def _record_batch_failure(self, text, timing, batch_id, started_at, reason):
        """Keep failed attempts separate from successfully generated segments."""
        record = dict(run_id=self.run_id, engine=self.engine_id, model_size=self.model_size,
                      batch_id=batch_id, started_at=started_at,
                      finished_at=datetime.now().astimezone().isoformat(), finished=False,
                      characters=len(text), text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      reason=reason, **timing)
        try:
            journal = ROOT / '.cache/qwen-batch-failures.jsonl'
            journal.parent.mkdir(exist_ok=True)
            with journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        except OSError as exc:
            self.message(f'Journal des réessais indisponible : {exc}')


    def synthesize_continuous(self, texts, destinations, *, on_ready=None, should_stop=None):
        """Reuse free GPU slots; retry only unfinished passages after draining the loop."""
        import torch
        from transformers import set_seed
        from concurrent.futures import CancelledError
        from epub_audio.core.tts_continuous import ContinuousBackend
        from epub_audio.core.tts_retry import synthesize_with_fallback, retry_segment
        if not self.batching or not 1 <= len(texts) <= self.batch_size * 8 or len(texts) != len(destinations):
            raise ValueError('Fenêtre de synthèse continue invalide.')
        if any(not text.strip() for text in texts):
            raise ValueError('Passage Qwen vide.')
        with self.lock:
            if self.batch_backend is None or self.model is None or self.prompt is None:
                raise RuntimeError('Le moteur Qwen doit être préparé avant la synthèse.')
            set_seed(self.settings['seed'])
            started_at = datetime.now().astimezone().isoformat()
            started = time.perf_counter()
            window_id = uuid.uuid4().hex
            saved = {}
            retry_reason = 'Passage non livré par le mode continu.'
            def check_stop():
                if should_stop is not None and should_stop():
                    raise CancelledError('Synthèse continue interrompue.')
            if len(texts) == 1:
                check_stop()
                path = synthesize_with_fallback(self, texts[0], destinations[0])
                if on_ready is not None:
                    on_ready(0, path)
                return [path]
            def deliver(index, audio, rate, timing):
                check_stop()
                if audio is None or timing.get('ended_with_eos') is not True:
                    return
                timing = dict(timing, continuous_batching=True)
                path = self._save_wave(texts[index], Path(destinations[index]), audio, rate,
                    timing, started_at, started, batch_size=self.batch_size, batch_id=window_id,
                    generation_seconds=timing.get('decode_s', time.perf_counter() - started))
                saved[index] = path
                if on_ready is not None:
                    on_ready(index, path)
            try:
                check_stop()
                with torch.inference_mode():
                    ContinuousBackend(self.batch_backend).generate(texts, self.prompt,
                        on_output=deliver, should_stop=should_stop)
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                retry_reason = f'Groupe continu interrompu : {exc}.'
            for index, text in enumerate(texts):
                if index not in saved:
                    check_stop()
                    saved[index] = retry_segment(self, text, destinations[index], retry_reason)
                    if on_ready is not None:
                        on_ready(index, saved[index])
            return [saved[index] for index in range(len(texts))]
