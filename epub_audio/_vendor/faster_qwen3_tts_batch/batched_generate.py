#!/usr/bin/env python3
"""
Batched generation loop using CUDA graphs for both predictor and talker.

Key design:
- Batched prefill: left-pad inputs, single forward for all B samples
- Left-padded KV cache alignment to shared cache_position
- GPU row completion and codec lengths, with bounded periodic CPU polling
- Preallocated codec history and optional captured codec/text embeddings
"""
import time
from typing import Callable, Optional

import torch

from .batched_predictor_graph import BatchedPredictorGraph
from .batched_sampling import batched_apply_repetition_penalty_mask, batched_sample_logits
from .batched_talker_graph import BatchedTalkerGraph


@torch.inference_mode()
def batched_fast_generate(
    talker,
    talker_input_embeds_list: list[torch.Tensor],
    attention_mask_list: list[torch.Tensor],
    trailing_text_hiddens_list: list[torch.Tensor],
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: BatchedPredictorGraph,
    talker_graph: BatchedTalkerGraph,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    max_steps_per_sample: Optional[list[int]] = None,
    on_generated: Optional[Callable[[int], None]] = None,
    codec_embedding_graph=None,
    completion_check_interval: int = 1,
) -> list[tuple[Optional[torch.Tensor], dict]]:
    """
    Batched autoregressive generation with CUDA-graphed predictor and talker.

    Args:
        talker: The Qwen3-TTS talker model.
        talker_input_embeds_list: list of B tensors, each [1, seq_len_b, H].
        attention_mask_list: list of B tensors, each [1, seq_len_b].
        trailing_text_hiddens_list: list of B tensors, each [1, text_len_b, H].
        tts_pad_embed: [1, 1, H] — shared pad embedding.
        config: model config with codec_eos_token_id, num_code_groups, vocab_size.
        predictor_graph: BatchedPredictorGraph (pre-captured).
        talker_graph: BatchedTalkerGraph (pre-captured).
        max_steps_per_sample: optional row budgets; exhausted rows remain incomplete.
        on_generated: notify a real row when nonempty generation reaches EOS.
        codec_embedding_graph: optional pre-captured codec/text embedding sum.
        completion_check_interval: host poll interval; GPU EOS/budgets remain exact.

    Returns:
        list of B tuples (codec_ids_or_None, timing_dict).
    """
    eos_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    vocab_size = config.vocab_size
    device = talker_input_embeds_list[0].device
    dtype = talker_input_embeds_list[0].dtype
    B_actual = len(talker_input_embeds_list)
    B_max = talker_graph.max_batch_size
    H = talker_input_embeds_list[0].shape[-1]
    if type(completion_check_interval) is not int or not 1 <= completion_check_interval <= 8:
        raise ValueError('Intervalle de suivi de génération invalide.')

    if max_steps_per_sample is not None:
        if len(max_steps_per_sample) != B_actual or any(
            type(limit) is not int or not 1 <= limit <= max_new_tokens
            for limit in max_steps_per_sample
        ):
            raise ValueError('Limites individuelles de génération invalides.')
        row_limits = torch.full((B_max,), max_new_tokens, dtype=torch.long, device=device)
        row_limits[:B_actual] = torch.tensor(max_steps_per_sample, dtype=torch.long, device=device)
    else:
        row_limits = None
    budget_exhausted = torch.zeros(B_max, dtype=torch.bool, device=device)

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

    # Per-sample original sequence lengths (before left-padding)
    orig_seq_lens = [t.shape[1] for t in talker_input_embeds_list]
    max_seq_len = max(orig_seq_lens)

    # Left-pad input_embeds and attention_mask to max_seq_len
    batched_embeds = torch.zeros(B_actual, max_seq_len, H, dtype=dtype, device=device)
    batched_mask = torch.zeros(B_actual, max_seq_len, dtype=torch.long, device=device)

    for b in range(B_actual):
        seq_len = orig_seq_lens[b]
        pad_len = max_seq_len - seq_len
        batched_embeds[b, pad_len:] = talker_input_embeds_list[b][0]
        batched_mask[b, pad_len:] = attention_mask_list[b][0]

    # Single batched forward (trailing_text_hidden unused during prefill)
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

    gen_step = out.generation_step  # 0 after prefill

    # rope_deltas from model: [B_actual, 1], already includes left-padding offset
    rope_deltas_batched = getattr(talker, "rope_deltas", None)
    if rope_deltas_batched is not None:
        rope_deltas_batched = rope_deltas_batched.clone()
    rope_deltas_list = []
    for b in range(B_actual):
        if rope_deltas_batched is not None:
            rope_deltas_list.append(rope_deltas_batched[b:b+1])  # [1, 1]
        else:
            rope_deltas_list.append(None)

    # Batched first-token sampling from logits[:, -1, :]
    # With left-padding, the last position is always the real last token for all samples
    first_logits = out.logits[:, -1, :]  # [B_actual, vocab]
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
    )  # [B_actual]

    # Copy batched KV cache directly to StaticCache (already left-padded)
    max_prefill_len = talker_graph.prefill_kv_batched(
        out.past_key_values, max_seq_len, B_actual
    )

    # Set attention mask and rope deltas (padding already in rope_deltas)
    talker_graph.set_generation_state(
        orig_seq_lens, max_prefill_len, rope_deltas_list,
        padding_in_rope_deltas=True,
    )

    # Stack into batch tensors, pad to B_max
    tokens = torch.zeros(B_max, dtype=torch.long, device=device)
    tokens[:B_actual] = first_tokens_batch

    past_hiddens = torch.zeros(B_max, 1, H, dtype=dtype, device=device)
    past_hiddens[:B_actual] = out.past_hidden[:B_actual]

    # Prepare trailing_text_hiddens — pad to max_text_len
    text_lens = [t.shape[1] for t in trailing_text_hiddens_list]
    max_text_len = max(text_lens) if text_lens else 0
    text_lens_t = torch.tensor(
        text_lens + [0] * (B_max - B_actual), dtype=torch.long, device=device
    )

    trailing_text_padded = torch.zeros(
        B_max, max(max_text_len, 1), H,
        dtype=talker_input_embeds_list[0].dtype, device=device,
    )
    for b in range(B_actual):
        tl = text_lens[b]
        if tl > 0:
            trailing_text_padded[b, :tl] = trailing_text_hiddens_list[b][0]

    # Broadcast tts_pad_embed to [B_max, 1, H]
    pad_embed = tts_pad_embed.expand(B_max, -1, -1)  # [B_max, 1, H]

    torch.cuda.synchronize()
    t_prefill = time.time() - t_start

    # ========================
    # DECODE LOOP
    # ========================
    t_decode_start = time.time()

    active = torch.ones(B_max, dtype=torch.bool, device=device)
    active[B_actual:] = False  # unused slots are inactive
    finished_eos = torch.zeros(B_max, dtype=torch.bool, device=device)

    # One bounded allocation and one batch copy per step replace per-row clones.
    # Rows never reactivate, so their valid codes are always a contiguous prefix.
    codec_ids = torch.empty(max_new_tokens, B_actual, num_code_groups,
                            dtype=torch.long, device=device)
    codec_lengths = torch.zeros(B_actual, dtype=torch.long, device=device)

    # A fixed presence mask avoids CUDA unique() and growing per-row histories.
    # All updates and repetition checks stay on the GPU.
    seen_tokens = torch.zeros(B_max, vocab_size, dtype=torch.bool, device=device)

    reported_eos = [False] * B_actual

    for step_idx in range(max_new_tokens):
        # GPU-side EOS detection
        is_eos = tokens == eos_id
        newly_finished = active & is_eos
        finished_eos |= newly_finished
        active = active & ~is_eos

        # EOS takes precedence at the boundary. A timed-out row stays failed;
        # never fabricate EOS or accept its partial audio as complete.
        if row_limits is not None:
            newly_exhausted = active & (row_limits <= step_idx)
            budget_exhausted |= newly_exhausted
            active = active & ~newly_exhausted

        # EOS and budgets still deactivate rows at the exact GPU step. The host
        # only polls periodically; any extra work is ignored for inactive rows.
        if step_idx % completion_check_interval == 0 or step_idx == max_new_tokens - 1:
            if on_generated is not None:
                ready = finished_eos[:B_actual] & (codec_lengths > 0)
                status = torch.stack((active[:B_actual], ready)).tolist()
                active_flags, ready_flags = status
                for row, ready_flag in enumerate(ready_flags):
                    if ready_flag and not reported_eos[row]:
                        on_generated(row)
                        reported_eos[row] = True
            else:
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
        codec_lengths.add_(active[:B_actual])
        codec_ids[step_idx].copy_(all_cb[:B_actual])

        # Record active rows only, including tokens already seen previously.
        token_columns = tokens.unsqueeze(1)
        was_seen = seen_tokens.gather(1, token_columns)
        seen_tokens.scatter_(1, token_columns, was_seen | active.unsqueeze(1))

        # Add text hidden state (per-sample, based on shared gen_step)
        if gen_step < max_text_len:
            text_hidden = trailing_text_padded[:, gen_step, :].unsqueeze(1)  # [B_max, 1, H]
            needs_pad = gen_step >= text_lens_t  # [B_max] bool
            text_addend = torch.where(needs_pad[:, None, None], pad_embed, text_hidden)
        else:
            text_addend = pad_embed

        if codec_embedding_graph is not None:
            inputs_embeds = codec_embedding_graph.run(last_id_hidden, codebook_token_ids, text_addend)
        else:
            codec_hiddens = [last_id_hidden]
            for i in range(num_code_groups - 1):
                codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[:, i].unsqueeze(1)))
            inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True) + text_addend

        # --- CUDA-Graphed Talker decode step ---
        current_pos = max_prefill_len + step_idx
        if current_pos >= talker_graph.max_seq_len - 1:
            break

        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)
        # hidden_states: [B_max, 1, H] — static buffer

        # --- Sampling ---
        logits = talker_codec_head(hidden_states[:, 0, :])  # [B_max, vocab]

        # Per-sample repetition penalty
        if repetition_penalty != 1.0:
            logits = batched_apply_repetition_penalty_mask(
                logits, seen_tokens, repetition_penalty, active
            )

        # Per-sample EOS suppression
        step_count = step_idx + 1  # number of tokens generated so far
        suppress_eos_active = step_count < min_new_tokens
        suppress_eos_mask = (
            torch.ones(B_max, dtype=torch.bool, device=device)
            if suppress_eos_active
            else None
        )

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

        # The predictor consumes this view before the next talker replay writes
        # its static output again; CUDA stream ordering makes a clone unnecessary.
        past_hiddens = hidden_states[:, :1, :]
        gen_step += 1

    torch.cuda.synchronize()
    t_decode = time.time() - t_decode_start

    # ========================
    # COLLECT RESULTS
    # ========================
    results = []
    codec_lengths = codec_lengths.tolist()
    for b in range(B_actual):
        n_steps = codec_lengths[b]
        timing = {
            'prefill_ms': t_prefill * 1000 / B_actual,  # amortized per sample
            'decode_s': t_decode,
            'steps': n_steps,
            'ms_per_step': (t_decode / n_steps * 1000) if n_steps > 0 else 0,
            'steps_per_s': (n_steps / t_decode) if t_decode > 0 else 0,
            'batch_size': B_actual,
            'ended_with_eos': bool(finished_eos[b].item()),
            'generation_limit': max_steps_per_sample[b] if max_steps_per_sample is not None else max_new_tokens,
            'estimated_limit_reached': bool(budget_exhausted[b].item()),
        }
        if n_steps:
            codec_tensor = codec_ids[:n_steps, b].contiguous()  # [n_steps, 16]
            results.append((codec_tensor, timing))
        else:
            results.append((None, timing))
        # A global/cache boundary can fall between host polls. Deliver any
        # genuine EOS notification still pending, including these final rows.
        if on_generated is not None and n_steps and timing['ended_with_eos'] and not reported_eos[b]:
            on_generated(b)
            reported_eos[b] = True

    return results
