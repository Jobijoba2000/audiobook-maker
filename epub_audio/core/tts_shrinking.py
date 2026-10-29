"""Shrink captured GPU work as batch rows finish, preserving their KV history."""
from collections import Counter
import copy
import types

import torch

from epub_audio._vendor.faster_qwen3_tts_batch import batched_predictor_graph


def _width_view(parent, size, *, predictor=False):
    """Give each graph its own cache metadata, with shared tensor storage."""
    graph = copy.copy(parent)
    # Variants never run concurrently. Inputs, outputs and KV storage are
    # allocated outside capture; only disposable workspaces share this pool.
    graph._capture_pool = parent.graph.pool()
    graph.max_batch_size = size
    graph.static_cache = copy.copy(parent.static_cache)
    graph.static_cache.layers = [copy.copy(layer) for layer in parent.static_cache.layers]
    storage = ([(layer.keys, layer.values) for layer in parent.static_cache.layers]
               if predictor else parent._cache_storage)
    for layer, (keys, values) in zip(graph.static_cache.layers, storage):
        layer.keys, layer.values = keys[:size], values[:size]
    graph.input_buf = parent.input_buf[:size]
    if predictor:
        graph.output_tokens = parent.output_tokens[:size]
    else:
        graph.output_buf = parent.output_buf[:size]
        graph.rope_deltas = parent.rope_deltas[:size]
        graph.position_ids = parent.position_ids[:, :size]
        graph._attention_mask_2d = parent._full_attention_mask[:size]
        graph._variants = {}
        graph.capacity_counts = dict.fromkeys(parent.capacities, 0)
        graph._current_capacity = parent.max_seq_len
    return graph


class ShrinkingBatch:
    def __init__(self, backend):
        self.backend = backend
        self.maximum = backend.batch_size
        self.device = backend.base.device
        self.talkers = {self.maximum: backend.talker}
        self.predictors = {self.maximum: backend.predictor}
        self.maps = {size: torch.arange(size, device=self.device)
                     for size in range(1, self.maximum + 1)}
        sample_logits = batched_predictor_graph.batched_sample_logits
        for size in range(self.maximum - 1, 0, -1):
            talker = _width_view(backend.talker, size)
            talker.capture(prefill_len=512, num_warmup=1)
            self.talkers[size] = talker
            predictor = _width_view(backend.predictor, size, predictor=True)
            mapping = self.maps[size]

            def padded_sample(logits, *, _mapping=mapping, **kwargs):
                # Keep the original row positions and RNG consumption. Only
                # the expensive transformer work runs on the surviving rows.
                padded = torch.zeros((self.maximum, logits.shape[1]),
                                     device=logits.device, dtype=logits.dtype)
                padded.index_copy_(0, _mapping, logits)
                return sample_logits(padded, **kwargs).index_select(0, _mapping)

            # Bind this sampler only to the captured variant, without changing
            # module globals used by other engines or later eager forwards.
            scope = dict(predictor._full_loop.__func__.__globals__, batched_sample_logits=padded_sample)
            loop = types.FunctionType(predictor._full_loop.__func__.__code__, scope)
            predictor._full_loop = types.MethodType(loop, predictor)
            predictor.capture(num_warmup=1)
            self.predictors[size] = predictor
        self.predictor_output = torch.zeros_like(backend.predictor.output_tokens)
        self.talker_output = torch.zeros_like(backend.talker.output_buf)
        self.predictor = _Predictor(self)
        self.talker = _Talker(self)

    def reset(self, limits):
        if not 1 <= len(limits) <= self.maximum or any(type(n) is not int or n < 1 for n in limits):
            raise ValueError('Limites du lot dégressif invalides.')
        self.actual = len(limits)
        self.limits = list(limits)
        self.layout = list(range(self.maximum))
        self.finished = set(range(self.actual, self.maximum))
        self.step = 0
        self.prefill = 0
        self.selection = self.maps[self.maximum]
        self.trace = [dict(step=0, width=self.maximum, rows=self.layout.copy())]
        self.steps_by_width = Counter()

    def on_generated(self, index):
        if not 0 <= index < self.actual:
            raise ValueError('Indice de passage terminé invalide.')
        self.finished.add(index)

    def select(self, step):
        survivors = [i for i in self.layout
                     if i not in self.finished and step < self.limits[i]]
        # A budget can expire between host polls. Preserve the last nonempty
        # layout for those ignored dummy steps; never reactivate a finished row.
        if not survivors or survivors == self.layout:
            return
        size = min(n for n in self.talkers if n >= len(survivors))
        if size >= len(self.layout):
            return
        # Retain a few inactive rows as padding when no exact graph exists.
        # Production's GPU active mask still prevents them from contributing.
        padding = [i for i in self.layout if i not in survivors]
        survivors += padding[:size-len(survivors)]
        picks = torch.tensor([self.layout.index(i) for i in survivors], device=self.device)
        end = self.prefill + step
        for keys, values in self.backend.talker._cache_storage:
            keys[:size, :, :end].copy_(keys[:, :, :end].index_select(0, picks))
            values[:size, :, :end].copy_(values[:, :, :end].index_select(0, picks))
        mask = self.backend.talker._full_attention_mask
        mask[:size].copy_(mask.index_select(0, picks))
        rope = self.backend.talker.rope_deltas
        rope[:size].copy_(rope.index_select(0, picks))
        self.layout = survivors
        self.maps[size].copy_(torch.tensor(survivors, device=self.device))
        self.selection = self.maps[size]
        self.trace.append(dict(step=step, width=size, rows=survivors.copy()))

    def diagnostics(self):
        return dict(shrinking_batching=True, batch_width_steps=dict(self.steps_by_width),
                    batch_min_size=min(self.steps_by_width, default=self.maximum),
                    batch_width_transitions=self.trace)


class _Predictor:
    def __init__(self, owner):
        self.owner = owner

    def run(self, inputs):
        owner = self.owner
        owner.select(owner.step)
        size = len(owner.layout)
        owner.steps_by_width[size] += 1
        owner.step += 1
        if size == owner.maximum:
            return owner.predictors[size].run(inputs)
        result = owner.predictors[size].run(inputs.index_select(0, owner.selection))
        owner.predictor_output.zero_()
        owner.predictor_output.index_copy_(0, owner.selection, result)
        return owner.predictor_output


class _Talker:
    def __init__(self, owner):
        self.owner = owner
        self.max_batch_size = owner.maximum
        self.max_seq_len = owner.backend.talker.max_seq_len

    def prefill_kv_batched(self, *args, **kwargs):
        self.owner.prefill = self.owner.backend.talker.prefill_kv_batched(*args, **kwargs)
        return self.owner.prefill

    def set_generation_state(self, *args, **kwargs):
        return self.owner.backend.talker.set_generation_state(*args, **kwargs)

    def run(self, inputs, position):
        owner = self.owner
        size = len(owner.layout)
        if size == owner.maximum:
            return owner.talkers[size].run(inputs, position)
        result = owner.talkers[size].run(inputs.index_select(0, owner.selection), position)
        owner.talker_output.zero_()
        owner.talker_output.index_copy_(0, owner.selection, result)
        return owner.talker_output
