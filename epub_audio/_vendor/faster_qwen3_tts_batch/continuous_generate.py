#!/usr/bin/env python3
"""
Continuous batched generation with mid-generation slot replacement.

When a sample finishes (EOS), its slot can be immediately filled with the next
pending request — no need to wait for the entire batch to complete.

Key differences from batched_fast_generate:
- Per-slot generation_step tracking (for trailing_text_hidden injection)
- Mid-generation slot replacement with eager prefill + KV cache insertion
- Immediate result delivery via on_slot_done callback
"""
import time
from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .batched_predictor_graph import BatchedPredictorGraph
from .batched_sampling import batched_apply_repetition_penalty_mask, batched_sample_logits
from .batched_talker_graph import BatchedTalkerGraph


@dataclass
class SlotReplacement:
    """Everything needed to insert a new sample into a freed slot."""
    past_key_values: Any       # DynamicCache from single-sample talker prefill
    prefill_len: int           # sequence length of the prefill
    first_token: torch.Tensor  # sampled first token (scalar)
    past_hidden: torch.Tensor  # [1, 1, H] hidden state from prefill
    trailing_text_hiddens: torch.Tensor  # [1, text_len, H]
    text_len: int
    rope_delta: float          # rope_delta from model after prefill
    tag: Any                   # opaque tag for result routing (e.g., PendingRequest)


@torch.inference_mode()
def continuous_batched_generate(
    talker,
    talker_input_embeds_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    trailing_text_hiddens_list: list[torch.Tensor],
    request_tags: list[Any],
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: BatchedPredictorGraph,
    talker_graph: BatchedTalkerGraph,
    try_get_replacement: Optional[Callable[[], Optional[SlotReplacement]]] = None,
    on_slot_done: Optional[Callable[[Any, Optional[torch.Tensor], dict], None]] = None,
    minimum_cache_position: int = 0,
    should_stop: Optional[Callable[[], bool]] = None,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
) -> list[tuple[Any, Optional[torch.Tensor], dict]]:
    """
    Continuous batched autoregressive generation with CUDA-graphed predictor and talker.

    Args:
        talker: The Qwen3-TTS talker model.
        talker_input_embeds_list: list of B tensors, each [1, seq_len_b, H].
        attention_mask_list: list of B tensors, each [1, seq_len_b].
        trailing_text_hiddens_list: list of B tensors, each [1, text_len_b, H].
        request_tags: list of B opaque tags for result routing.
        tts_pad_embed: [1, 1, H] — shared pad embedding.
        config: model config with codec_eos_token_id, num_code_groups, vocab_size.
        predictor_graph: BatchedPredictorGraph (pre-captured).
        talker_graph: BatchedTalkerGraph (pre-captured).
        try_get_replacement: callback returning SlotReplacement or None.
        on_slot_done: callback(tag, codec_ids, timing) for immediate result delivery.

    Returns:
        list of (tag, codec_ids_or_None, timing_dict) tuples for all completed samples.
    """
    eos_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    vocab_size = config.vocab_size
    device = talker_input_embeds_list[0].device
    dtype = talker_input_embeds_list[0].dtype
    B_actual = len(talker_input_embeds_list)
    B_max = talker_graph.max_batch_size
    H = talker_input_embeds_list[0].shape[-1]

    # Suppress mask — shared across all samples
    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    suppress_start = max(0, vocab_size - 1024)
    for i in range(suppress_start, vocab_size):
        if i != eos_id:
            suppress_mask[i] = True

    # Extract model components
    predictor = talker.code_predictor
    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    predictor_codec_embeds = predictor.get_input_embeddings()

    # ========================
    # PREFILL (batched: left-pad + single forward)
    # ========================
    t_start = time.time()

    orig_seq_lens = [t.shape[1] for t in talker_input_embeds_list]
    max_seq_len = max(orig_seq_lens)

    batched_embeds = torch.zeros(B_actual, max_seq_len, H, dtype=dtype, device=device)
    batched_mask = torch.zeros(B_actual, max_seq_len, dtype=torch.long, device=device)

    for b in range(B_actual):
        seq_len = orig_seq_lens[b]
        pad_len = max_seq_len - seq_len
        batched_embeds[b, pad_len:] = talker_input_embeds_list[b][0]
        batched_mask[b, pad_len:] = attention_mask_list[b][0]

    out = talker.forward(
        inputs_embeds=batched_embeds,
        attention_mask=batched_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=None,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )

    # Collect rope_deltas from batched prefill
    rope_deltas_batched = getattr(talker, "rope_deltas", None)
    if rope_deltas_batched is not None:
        rope_deltas_batched = rope_deltas_batched.clone()
    rope_deltas_list = []
    for b in range(B_actual):
        if rope_deltas_batched is not None:
            rope_deltas_list.append(rope_deltas_batched[b:b+1])
        else:
            rope_deltas_list.append(None)

    # Batched first-token sampling
    first_logits = out.logits[:, -1, :]
    suppress_eos = min_new_tokens > 0
    first_tokens_batch = batched_sample_logits(
        first_logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        suppress_mask=suppress_mask,
        suppress_eos_mask=(
            torch.ones(B_actual, dtype=torch.bool, device=device)
            if suppress_eos else None
        ),
        eos_id=eos_id,
    )

    # Copy KV cache into StaticCache
    max_prefill_len = talker_graph.prefill_kv_batched(
        out.past_key_values, max_seq_len, B_actual
    )

    talker_graph.set_generation_state(
        orig_seq_lens, max_prefill_len, rope_deltas_list,
        padding_in_rope_deltas=True,
    )

    # Stack into batch tensors, pad to B_max
    tokens = torch.zeros(B_max, dtype=torch.long, device=device)
    tokens[:B_actual] = first_tokens_batch

    past_hiddens = torch.zeros(B_max, 1, H, dtype=dtype, device=device)
    past_hiddens[:B_actual] = out.past_hidden[:B_actual]
    del first_logits, out, batched_embeds, batched_mask

    # Per-slot trailing text hidden (with generous capacity for replacements)
    text_lens = [t.shape[1] for t in trailing_text_hiddens_list]
    max_text_len = max(text_lens) if text_lens else 0
    text_capacity = max(max_text_len, 256)

    text_lens_t = torch.zeros(B_max, dtype=torch.long, device=device)
    text_lens_t[:B_actual] = torch.tensor(text_lens, dtype=torch.long, device=device)

    trailing_text_padded = torch.zeros(
        B_max, text_capacity, H, dtype=dtype, device=device,
    )
    for b in range(B_actual):
        tl = text_lens[b]
        if tl > 0:
            trailing_text_padded[b, :tl] = trailing_text_hiddens_list[b][0]

    pad_embed = tts_pad_embed.expand(B_max, -1, -1)

    torch.cuda.synchronize()
    t_prefill = time.time() - t_start

    # ========================
    # DECODE LOOP (continuous)
    # ========================
    t_decode_start = time.time()

    active = torch.ones(B_max, dtype=torch.bool, device=device)
    active[B_actual:] = False

    # Per-slot state
    gen_steps = torch.zeros(B_max, dtype=torch.long, device=device)
    slot_steps = torch.zeros(B_max, dtype=torch.long, device=device)
    slot_tags: list[Any] = list(request_tags) + [None] * (B_max - B_actual)
    slot_start_times = [t_decode_start] * B_max

    # Per-slot codec accumulation
    all_codec_ids: list[list[torch.Tensor]] = [[] for _ in range(B_max)]

    # Per-sample token history for repetition penalty
    seen_tokens = torch.zeros(B_max, vocab_size, dtype=torch.bool, device=device)

    # Results collection
    all_results: list[tuple[Any, Optional[torch.Tensor], dict]] = []
    total_replaced = 0

    # Precompute range tensor for gather
    b_range = torch.arange(B_max, device=device)

    current_pos = max_prefill_len
    slot_starts = [max_prefill_len - length for length in orig_seq_lens] + [0] * (B_max - B_actual)
    while True:
        if should_stop is not None and should_stop():
            raise CancelledError('Synthèse continue interrompue.')
        # GPU-side EOS detection
        is_eos = tokens == eos_id
        newly_finished = active & (is_eos | (slot_steps >= max_new_tokens))

        # Process finished slots: collect results + attempt slot replacement
        finished_flags = newly_finished.tolist()
        if any(finished_flags):
            for b in range(B_max):
                if not finished_flags[b]:
                    continue

                # Collect result for this slot
                n_steps = len(all_codec_ids[b])
                slot_time = time.time() - slot_start_times[b]
                timing = {
                    'prefill_ms': t_prefill * 1000 / B_actual,
                    'decode_s': slot_time,
                    'steps': n_steps,
                    'ms_per_step': (slot_time / n_steps * 1000) if n_steps > 0 else 0,
                    'steps_per_s': (n_steps / slot_time) if slot_time > 0 else 0,
                    'batch_size': int(active.sum().item()),
                    'total_replaced': total_replaced,
                    'ended_with_eos': bool(is_eos[b].item()),
                }
                codec_tensor = torch.stack(all_codec_ids[b]) if all_codec_ids[b] else None
                result = (slot_tags[b], codec_tensor, timing)
                all_results.append(result)

                if on_slot_done is not None:
                    on_slot_done(*result)

                # Try slot replacement
                replacement = None
                if try_get_replacement is not None:
                    replacement = try_get_replacement()

                if replacement is not None:

                    # Insert new sample's KV cache into this slot
                    talker_graph.replace_slot_kv(
                        b, replacement.past_key_values,
                        replacement.prefill_len, current_pos,
                    )
                    talker_graph.update_slot_state(
                        b, replacement.prefill_len, current_pos,
                        replacement.rope_delta - (current_pos - replacement.prefill_len),
                    )

                    slot_starts[b] = current_pos - replacement.prefill_len
                    # Reset per-slot state for the new sample
                    tokens[b] = replacement.first_token
                    past_hiddens[b] = replacement.past_hidden[0]
                    gen_steps[b] = 0
                    slot_steps[b] = 0
                    slot_tags[b] = replacement.tag
                    slot_start_times[b] = time.time()
                    all_codec_ids[b] = []
                    seen_tokens[b].zero_()

                    # Update text hidden for this slot (expand capacity if needed)
                    tl = replacement.text_len
                    if tl > text_capacity:
                        new_cap = max(tl, text_capacity * 2)
                        new_padded = torch.zeros(B_max, new_cap, H, dtype=dtype, device=device)
                        new_padded[:, :text_capacity] = trailing_text_padded
                        trailing_text_padded = new_padded
                        text_capacity = new_cap

                    trailing_text_padded[b].zero_()
                    if tl > 0:
                        trailing_text_padded[b, :tl] = replacement.trailing_text_hiddens[0]
                    text_lens_t[b] = tl

                    # Keep slot active (replacement inserted)
                    total_replaced += 1
                    del replacement
                else:
                    # No replacement available — deactivate slot
                    active[b] = False

        active_flags = active.tolist()
        if not any(active_flags):
            break

        # --- CUDA-Graphed Code Predictor ---
        last_id_hidden = talker_codec_embed(tokens.unsqueeze(1))  # [B_max, 1, H]
        pred_input = torch.cat((past_hiddens, last_id_hidden), dim=1)  # [B_max, 2, H]
        codebook_token_ids = predictor_graph.run(pred_input)  # [B_max, 15]

        # Accumulate codec ids for active samples
        all_cb = torch.cat(
            [tokens.unsqueeze(1), codebook_token_ids], dim=1
        )  # [B_max, 16]
        for b in range(B_max):
            if active_flags[b]:
                all_codec_ids[b].append(all_cb[b].detach().clone())

        # Update token histories for repetition penalty
        columns = tokens.unsqueeze(1)
        seen_tokens.scatter_(1, columns, seen_tokens.gather(1, columns) | active.unsqueeze(1))

        # --- Build input embedding for talker ---
        codec_hiddens = [last_id_hidden]
        for i in range(num_code_groups - 1):
            codec_hiddens.append(
                predictor_codec_embeds[i](codebook_token_ids[:, i].unsqueeze(1))
            )
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)  # [B_max, 1, H]

        # Per-slot text hidden injection (each slot at its own gen_step)
        clamped_gs = gen_steps.clamp(max=text_capacity - 1)
        text_hidden = trailing_text_padded[b_range, clamped_gs].unsqueeze(1)  # [B_max, 1, H]
        needs_pad = gen_steps >= text_lens_t  # [B_max]
        text_hidden = torch.where(needs_pad[:, None, None], pad_embed, text_hidden)
        inputs_embeds = inputs_embeds + text_hidden

        # --- CUDA-Graphed Talker decode step ---
        if current_pos >= talker_graph.max_seq_len - 1:
            # Discard only padding shared by every active slot. Keep absolute
            # rotary positions unchanged while reclaiming physical cache space.
            shift = min(slot_starts[b] for b in range(B_max) if active_flags[b])
            shift = min(shift, current_pos - minimum_cache_position)
            if shift <= 0:
                break
            current_pos = talker_graph.compact_cache(current_pos, shift)
            slot_starts = [max(0, start - shift) for start in slot_starts]

        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)

        # --- Sampling ---
        logits = talker_codec_head(hidden_states[:, 0, :])  # [B_max, vocab]

        if repetition_penalty != 1.0:
            logits = batched_apply_repetition_penalty_mask(
                logits, seen_tokens, repetition_penalty, active
            )

        # Per-slot EOS suppression based on each slot's step count
        suppress_eos_mask = (slot_steps + 1 < min_new_tokens) & active
        tokens = batched_sample_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_eos_mask=suppress_eos_mask,
            eos_id=eos_id,
        )

        past_hiddens = hidden_states[:, :1, :].clone()
        gen_steps += active.to(torch.long)
        slot_steps += active.to(torch.long)
        current_pos += 1

    torch.cuda.synchronize()

    # Collect any remaining active slots (hit max_new_tokens without EOS)
    completed_tag_ids = {id(r[0]) for r in all_results}
    for b in range(B_max):
        if slot_tags[b] is not None and id(slot_tags[b]) not in completed_tag_ids:
            n_steps = len(all_codec_ids[b])
            slot_time = time.time() - slot_start_times[b]
            timing = {
                'prefill_ms': t_prefill * 1000 / B_actual,
                'decode_s': slot_time,
                'steps': n_steps,
                'ms_per_step': (slot_time / n_steps * 1000) if n_steps > 0 else 0,
                'steps_per_s': (n_steps / slot_time) if slot_time > 0 else 0,
                'batch_size': B_actual,
                'total_replaced': total_replaced,
                'ended_with_eos': False,
            }
            codec_tensor = torch.stack(all_codec_ids[b]) if all_codec_ids[b] else None
            result = (slot_tags[b], codec_tensor, timing)
            all_results.append(result)

            if on_slot_done is not None:
                on_slot_done(*result)

    return all_results
