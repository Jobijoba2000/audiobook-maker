"""Batch-and-wait Faster inference, using one loaded Base model and ICL prompt."""

from epub_audio.config.defaults import BATCH_SIZE_CHOICES
from epub_audio.core.generation_budget import generation_step_budget


class FasterBatchBackend:
    def __init__(self, base, settings, batch_size=2, *, adaptive_cache=True, shrinking_batching=False):
        if type(batch_size) is not int or batch_size not in BATCH_SIZE_CHOICES:
            raise ValueError('Taille du groupe de batching invalide.')
        if type(adaptive_cache) is not bool:
            raise ValueError('Réglage du cache adaptatif invalide.')
        if type(shrinking_batching) is not bool or (shrinking_batching and not adaptive_cache):
            raise ValueError('La réduction des lots nécessite le cache adaptatif et le mode par lots.')
        self.batch_size = batch_size
        self.adaptive_cache = adaptive_cache
        from epub_audio._vendor.faster_qwen3_tts_batch.batched_predictor_graph import BatchedPredictorGraph
        from epub_audio._vendor.faster_qwen3_tts_batch.batched_talker_graph import BatchedTalkerGraph
        self.base = base
        self.settings = settings
        model = base.model.model
        talker = model.talker
        config = model.config.talker_config
        self.predictor = BatchedPredictorGraph(
            talker.code_predictor, talker.code_predictor.model.config, config.hidden_size,
            max_batch_size=batch_size, device=base.device, dtype=base.dtype,
            do_sample=settings['sampling'], temperature=settings['temperature'], top_k=50, top_p=1.0)
        if adaptive_cache:
            from epub_audio._vendor.faster_qwen3_tts_batch.adaptive_talker_graph import AdaptiveTalkerGraph
            graph_type = AdaptiveTalkerGraph
        else:
            graph_type = BatchedTalkerGraph
        self.talker = graph_type(talker.model, config,
            max_batch_size=batch_size, device=base.device, dtype=base.dtype, max_seq_len=3072)
        self.predictor.capture(num_warmup=3)
        self.talker.capture(prefill_len=512, num_warmup=3)
        self.codec_embeddings = None
        if adaptive_cache:
            from epub_audio._vendor.faster_qwen3_tts_batch.codec_embedding_graph import CodecEmbeddingGraph
            self.codec_embeddings = CodecEmbeddingGraph(talker, batch_size, base.device, base.dtype)
            self.codec_embeddings.capture()
        self.shrinking = None
        self.shrinking_error = None
        if shrinking_batching:
            import torch
            from epub_audio.core.tts_shrinking import ShrinkingBatch
            try:
                self.shrinking = ShrinkingBatch(self)
            except torch.cuda.OutOfMemoryError:
                self.shrinking_error = 'Mémoire GPU insuffisante pour les tailles dégressives ; lots fixes conservés.'
            if self.shrinking_error:
                import gc
                gc.collect()
            # Warmup of multiple shapes leaves large unused eager allocations.
            # Release them once so audio decoding has room beside graph pools.
            torch.cuda.empty_cache()

    def generate(self, texts, prompt, *, on_generated=None):
        import torch
        import numpy as np
        from epub_audio._vendor.faster_qwen3_tts_batch.batched_generate import batched_fast_generate
        if not 1 <= len(texts) <= self.batch_size:
            raise ValueError('Nombre de passages incompatible avec le groupe de batching.')
        prepared = [self.base._prepare_generation(
            text=text, language='French', voice_clone_prompt=prompt, xvec_only=False,
            non_streaming_mode=True, append_silence=False) for text in texts]
        model, talker, config, *_ = prepared[0]
        shrinking = getattr(self, 'shrinking', None)
        limits = [generation_step_budget(text, self.settings['max_new_tokens']) for text in texts]
        notifications = {'on_generated': on_generated} if on_generated is not None else {}
        if shrinking is not None:
            shrinking.reset(limits)
            def generated(index):
                shrinking.on_generated(index)
                if on_generated is not None:
                    on_generated(index)
            notifications = {'on_generated': generated}
        generated = batched_fast_generate(
            talker=talker, config=config,
            talker_input_embeds_list=[p[3] for p in prepared],
            attention_mask_list=[p[4] for p in prepared],
            trailing_text_hiddens_list=[p[5] for p in prepared],
            tts_pad_embed=prepared[0][6],
            predictor_graph=shrinking.predictor if shrinking is not None else self.predictor,
            talker_graph=shrinking.talker if shrinking is not None else self.talker,
            codec_embedding_graph=self.codec_embeddings,
            completion_check_interval=4,
            max_new_tokens=self.settings['max_new_tokens'],
            max_steps_per_sample=limits,
            temperature=self.settings['temperature'], top_k=50, top_p=1.0,
            do_sample=self.settings['sampling'], repetition_penalty=self.settings['repetition_penalty'],
            **notifications)
        if len(generated) != len(texts):
            raise RuntimeError('Le batching n’a pas renvoyé tous les passages.')
        outputs = []
        for (codes, timing), inputs in zip(generated, prepared):
            if shrinking is not None:
                timing = dict(timing, **shrinking.diagnostics())
            if codes is None or timing.get('ended_with_eos') is not True:
                outputs.append((None, self.base.sample_rate, timing))
                continue
            # Match the existing Faster ICL decoder: include the reference as
            # acoustic context, then remove its proportional PCM prefix.
            reference_codes = inputs[7]
            decode_codes = (torch.cat([reference_codes.to(codes.device), codes], dim=0)
                            if reference_codes is not None else codes)
            if shrinking is not None:
                # The waveform decoder needs large, varying workspaces. Cached
                # eager blocks can push an 8 GB Windows GPU into system memory.
                torch.cuda.empty_cache()
            audio_arrays, rate = model.speech_tokenizer.decode({'audio_codes': decode_codes.unsqueeze(0)})
            audio = audio_arrays[0]
            if hasattr(audio, 'cpu'):
                audio = audio.flatten().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).flatten()
            if reference_codes is not None:
                audio = audio[int(reference_codes.shape[0] / decode_codes.shape[0] * len(audio)):]
            outputs.append((audio, rate, timing))
        return outputs
