#!/usr/bin/env python3
"""
Batched CUDA graph capture for the talker's single-token decode step.

Supports:
- Fixed max_batch_size via left-padding alignment
- On-the-fly attention mask computation (no precomputed table needed)
- Mid-generation slot replacement for continuous batching

Key insight: left-padding all samples to the same max_prefill_len means
cache_position is a shared scalar — all samples decode at the same position.
Per-sample differences are handled by attention_mask and rope_deltas.
"""
import torch
from transformers import StaticCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


class BatchedTalkerGraph:
    """
    Batched talker decode with CUDA graph and continuous batching support.

    On-the-fly mask computation eliminates the precomputed mask table,
    enabling efficient slot replacement without expensive rebuilds.
    """

    def __init__(
        self,
        talker_model,
        talker_config,
        max_batch_size=4,
        device='cuda',
        dtype=torch.bfloat16,
        max_seq_len=512,
    ):
        self.device = device
        device_index = torch.device(device).index
        device_index = device_index if device_index is not None else torch.cuda.current_device()
        self.device_index = device_index

        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.hidden_size = talker_config.hidden_size
        self.num_layers = talker_config.num_hidden_layers

        # Reference to the inner model
        self.model = talker_model

        B = max_batch_size

        # StaticCache with batch_size=B
        self.static_cache = StaticCache(
            config=talker_config, max_cache_len=max_seq_len, batch_size=B
        )

        # Static I/O buffers — [B, 1, hidden_size]
        self.input_buf = torch.zeros(B, 1, self.hidden_size, dtype=dtype, device=device)
        self.output_buf = torch.zeros(B, 1, self.hidden_size, dtype=dtype, device=device)

        # Cache position — shared scalar (all samples at same decode position)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)

        # Per-sample rope deltas and position ids
        self.rope_deltas = torch.zeros(B, 1, dtype=torch.float32, device=device)
        self.position_ids = torch.zeros(3, B, 1, dtype=torch.float32, device=device)

        self.graph = None
        self.captured = False
        self.attn_mask = None

        # Attention mask: [B, max_seq_len] — 1 for valid, 0 for left-padding
        self._attention_mask_2d = torch.ones(B, max_seq_len, dtype=torch.long, device=device)

        # Mask creation function (causal or sliding-window)
        self._mask_fn = (
            create_causal_mask
            if talker_model.config.sliding_window is None
            else create_sliding_window_causal_mask
        )

        # Store config for cache layer initialization
        self._num_kv_heads = getattr(
            talker_config, 'num_key_value_heads', talker_config.num_attention_heads
        )
        self._head_dim = getattr(
            talker_config, 'head_dim',
            talker_config.hidden_size // talker_config.num_attention_heads,
        )

    def _init_cache_layers(self):
        """Force lazy initialization of StaticCache layers before graph capture."""
        B = self.max_batch_size
        dummy_k = torch.zeros(
            B, self._num_kv_heads, 1, self._head_dim,
            dtype=self.dtype, device=self.device,
        )
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy_k)

    def _compute_and_set_mask(self):
        """Compute attention mask for current cache_position and copy into static buffer."""
        mask = self._mask_fn(
            config=self.model.config,
            input_embeds=self.input_buf,
            attention_mask=self._attention_mask_2d,
            cache_position=self.cache_position,
            past_key_values=self.static_cache,
        )
        if mask is not None:
            self.attn_mask.copy_(mask)

    def _decode_step(self):
        """Single-token decode through the model's forward, batched."""
        out = self.model(
            inputs_embeds=self.input_buf,
            attention_mask=self.attn_mask,
            past_key_values=self.static_cache,
            cache_position=self.cache_position,
            position_ids=self.position_ids,
            use_cache=True,
        )
        self.output_buf.copy_(out.last_hidden_state)

    @torch.inference_mode()
    def capture(self, prefill_len=100, num_warmup=3):
        """Capture CUDA graph for batched single-token decode."""
        print(f"Warming up batched talker graph (B={self.max_batch_size}, {num_warmup} runs)...")

        self._init_cache_layers()

        # Initialize with no padding for capture
        self._attention_mask_2d.fill_(1)
        self.cache_position[0] = prefill_len

        # Compute initial mask to determine shape and allocate static buffer
        mask = self._mask_fn(
            config=self.model.config,
            input_embeds=self.input_buf,
            attention_mask=self._attention_mask_2d,
            cache_position=self.cache_position,
            past_key_values=self.static_cache,
        )
        self.attn_mask = mask.clone()

        for _ in range(num_warmup):
            self._decode_step()
        torch.cuda.synchronize()

        print("Capturing CUDA graph for batched talker decode...")

        with torch.cuda.device(self.device_index):
            self.graph = torch.cuda.CUDAGraph()

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._decode_step()
                torch.cuda.synchronize()

                with torch.cuda.graph(self.graph, pool=getattr(self, '_capture_pool', None)):
                    self._decode_step()

        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.captured = True
        print(f"Batched talker CUDA graph captured! (B={self.max_batch_size})")

    def reset(self):
        """Reset cache for new batch."""
        self.static_cache.reset()

    # ── KV cache population ──────────────────────────────────────────────

    def prefill_kv(self, past_key_values_list: list, prefill_lens: list[int]) -> int:
        """
        Copy left-padded KV caches from per-sample prefill into the batched StaticCache.

        Args:
            past_key_values_list: list of B DynamicCache objects from individual prefills.
            prefill_lens: list of B actual prefill lengths (before padding).

        Returns:
            max_prefill_len: the aligned prefill length (all samples padded to this).
        """
        self.static_cache.reset()

        B = len(past_key_values_list)
        max_prefill_len = max(prefill_lens)

        if max_prefill_len > self.max_seq_len:
            raise RuntimeError(
                f"Max prefill length {max_prefill_len} exceeds max_seq_len={self.max_seq_len}. "
                "Use shorter text or shorter reference audio."
            )

        cache_pos = torch.arange(max_prefill_len, device=self.device)

        for li in range(self.num_layers):
            padded_keys = []
            padded_vals = []
            for b in range(B):
                k, v = past_key_values_list[b][li]
                seq_len_b = k.shape[2]
                pad_len = max_prefill_len - seq_len_b

                if pad_len > 0:
                    pad_k = torch.zeros(
                        1, k.shape[1], pad_len, k.shape[3],
                        dtype=k.dtype, device=k.device,
                    )
                    pad_v = torch.zeros(
                        1, v.shape[1], pad_len, v.shape[3],
                        dtype=v.dtype, device=v.device,
                    )
                    k = torch.cat([pad_k, k], dim=2)
                    v = torch.cat([pad_v, v], dim=2)
                padded_keys.append(k)
                padded_vals.append(v)

            stacked_k = torch.cat(padded_keys, dim=0)
            stacked_v = torch.cat(padded_vals, dim=0)

            if B < self.max_batch_size:
                pad_b = self.max_batch_size - B
                extra_k = torch.zeros(
                    pad_b, stacked_k.shape[1], max_prefill_len, stacked_k.shape[3],
                    dtype=stacked_k.dtype, device=stacked_k.device,
                )
                extra_v = torch.zeros_like(extra_k)
                stacked_k = torch.cat([stacked_k, extra_k], dim=0)
                stacked_v = torch.cat([stacked_v, extra_v], dim=0)

            self.static_cache.update(stacked_k, stacked_v, li, {"cache_position": cache_pos})

        return max_prefill_len

    def prefill_kv_batched(self, past_key_values, max_prefill_len: int, B_actual: int) -> int:
        """
        Copy KV caches from a batched prefill directly into StaticCache.

        Args:
            past_key_values: DynamicCache from batched forward.
            max_prefill_len: sequence length in the cache (all samples equal).
            B_actual: number of real samples.

        Returns:
            max_prefill_len (unchanged, for API consistency).
        """
        self.static_cache.reset()

        if max_prefill_len > self.max_seq_len:
            raise RuntimeError(
                f"Max prefill length {max_prefill_len} exceeds max_seq_len={self.max_seq_len}."
            )

        cache_pos = torch.arange(max_prefill_len, device=self.device)

        for li in range(self.num_layers):
            k, v = past_key_values[li]

            if B_actual < self.max_batch_size:
                pad_b = self.max_batch_size - B_actual
                k = torch.cat([k, torch.zeros(
                    pad_b, *k.shape[1:], dtype=k.dtype, device=k.device,
                )], dim=0)
                v = torch.cat([v, torch.zeros(
                    pad_b, *v.shape[1:], dtype=v.dtype, device=v.device,
                )], dim=0)

            self.static_cache.update(k, v, li, {"cache_position": cache_pos})

        return max_prefill_len

    # ── Generation state ─────────────────────────────────────────────────

    def set_generation_state(
        self,
        prefill_lens: list[int],
        max_prefill_len: int,
        rope_deltas_list: list[torch.Tensor | None],
        padding_in_rope_deltas: bool = False,
    ):
        """Set padding-aware attention mask and per-sample rope deltas.

        Args:
            prefill_lens: list of B actual prefill lengths.
            max_prefill_len: the aligned (max) prefill length.
            rope_deltas_list: list of B rope_deltas tensors (or None).
            padding_in_rope_deltas: if True, rope_deltas already include the
                left-padding offset (from batched prefill with attention_mask).
        """
        B = len(prefill_lens)

        # Build attention mask: [max_batch_size, max_seq_len]
        self._attention_mask_2d.fill_(1)
        for b in range(B):
            pad_len = max_prefill_len - prefill_lens[b]
            if pad_len > 0:
                self._attention_mask_2d[b, :pad_len] = 0

        # Set per-sample rope_deltas
        self.rope_deltas.zero_()
        for b in range(B):
            if not padding_in_rope_deltas:
                pad_len = max_prefill_len - prefill_lens[b]
                self.rope_deltas[b, 0] = float(-pad_len)

            rd = rope_deltas_list[b]
            if rd is not None:
                if rd.dim() == 0:
                    self.rope_deltas[b, 0] += rd.to(dtype=torch.float32).item()
                elif rd.dim() == 1:
                    self.rope_deltas[b, 0] += rd[0].to(dtype=torch.float32).item()
                else:
                    self.rope_deltas[b, 0] += rd[0, 0].to(dtype=torch.float32).item()

    # ── Continuous batching: slot replacement ─────────────────────────────

    def replace_slot_kv(
        self,
        slot_idx: int,
        past_key_values,
        new_prefill_len: int,
        current_pos: int,
    ):
        """Replace KV cache for a single slot with a new sample's prefill result.

        Left-pads the new KV cache within the StaticCache to align with current_pos.

        Args:
            slot_idx: which batch slot to replace.
            past_key_values: DynamicCache from single-sample prefill.
                Each layer gives (K, V) of shape [1, kv_heads, new_prefill_len, head_dim].
            new_prefill_len: length of the new sample's prefill.
            current_pos: current decode position in the shared sequence.
        """
        start = current_pos - new_prefill_len
        for li in range(self.num_layers):
            k, v = past_key_values[li]
            layer = self.static_cache.layers[li]

            # Zero out the entire slot
            layer.keys[slot_idx].zero_()
            layer.values[slot_idx].zero_()

            # Write new KV at positions [start, current_pos)
            layer.keys[slot_idx, :, start:current_pos] = k[0]
            layer.values[slot_idx, :, start:current_pos] = v[0]

    def update_slot_state(
        self,
        slot_idx: int,
        new_prefill_len: int,
        current_pos: int,
        rope_delta: float,
    ):
        """Update attention mask and rope_delta for a replaced slot.

        Args:
            slot_idx: which batch slot was replaced.
            new_prefill_len: length of the new sample's prefill.
            current_pos: current decode position.
            rope_delta: rope_delta value from the new sample's prefill.
        """
        pad_len = current_pos - new_prefill_len
        self._attention_mask_2d[slot_idx].fill_(1)
        if pad_len > 0:
            self._attention_mask_2d[slot_idx, :pad_len] = 0
        self.rope_deltas[slot_idx, 0] = rope_delta

    def compact_cache(self, position: int, shift: int) -> int:
        """Reclaim common left padding while preserving absolute RoPE positions."""
        if not 0 < shift <= position < self.max_seq_len:
            raise ValueError('Décalage du cache continu invalide.')
        end = position - shift
        for layer in self.static_cache.layers:
            for buffer in (layer.keys, layer.values):
                buffer[:, :, :end].copy_(buffer[:, :, shift:position].clone())
                buffer[:, :, end:].zero_()
        self._attention_mask_2d[:, :end].copy_(self._attention_mask_2d[:, shift:position].clone())
        self._attention_mask_2d[:, end:].fill_(1)
        self.rope_deltas.add_(shift)
        return end

    # ── Decode step ──────────────────────────────────────────────────────

    @torch.inference_mode()
    def run(self, input_embeds: torch.Tensor, position: int) -> torch.Tensor:
        """
        Run one batched decode step.

        Args:
            input_embeds: [B, 1, hidden_size] — B must equal max_batch_size.
            position: current sequence position (shared across all samples).

        Returns:
            [B, 1, hidden_size] hidden states (static buffer — use immediately or clone).
        """
        self.input_buf.copy_(input_embeds)
        self.cache_position[0] = position
        self._compute_and_set_mask()
        # position_ids = cache_position + rope_deltas, expanded for 3 rope factor dims
        delta = self.rope_deltas + self.cache_position[0].to(self.rope_deltas.dtype)  # [B, 1]
        self.position_ids.copy_(delta.unsqueeze(0).expand(3, -1, -1))  # [3, B, 1]
        self.graph.replay()

        return self.output_buf  # [B, 1, H] — static buffer
