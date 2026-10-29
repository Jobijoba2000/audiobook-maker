"""Batched sampling helpers for dynamic batch TTS generation."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def batched_apply_repetition_penalty(
    logits: torch.Tensor,
    token_histories: list[torch.Tensor],
    repetition_penalty: float,
    active_mask: torch.Tensor | list[bool],
) -> torch.Tensor:
    """Apply per-sample repetition penalty.

    Args:
        logits: [B, vocab] logits tensor.
        token_histories: list of B 1-D tensors (variable-length per sample).
        repetition_penalty: HF-style repetition penalty (>1.0).
        active_mask: [B] bool tensor or cached CPU flags; only active rows change.
    """
    if repetition_penalty == 1.0:
        return logits
    # Callers may reuse CPU flags; tensor callers retain the original API
    # with one transfer instead of a separate scalar read for every row.
    active_flags = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
    for b in range(logits.shape[0]):
        if not active_flags[b]:
            continue
        history = token_histories[b]
        if history.numel() == 0:
            continue
        unique_toks = history.unique()
        tok_logits = logits[b, unique_toks]
        logits[b, unique_toks] = torch.where(
            tok_logits > 0,
            tok_logits / repetition_penalty,
            tok_logits * repetition_penalty,
        )
    return logits


def batched_apply_repetition_penalty_mask(
    logits: torch.Tensor,
    seen_tokens: torch.Tensor,
    repetition_penalty: float,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Same per-token penalty, with fixed GPU masks and no variable-size unique()."""
    if repetition_penalty == 1.0:
        return logits
    penalized = torch.where(logits > 0, logits / repetition_penalty,
                            logits * repetition_penalty)
    return torch.where(seen_tokens & active_mask[:, None], penalized, logits)


def batched_sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
    suppress_mask: Optional[torch.Tensor] = None,
    suppress_eos_mask: Optional[torch.Tensor] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Batched sampling from logits.

    Args:
        logits: [B, vocab] tensor.
        temperature: sampling temperature.
        top_k: top-k filtering.
        top_p: nucleus sampling threshold.
        do_sample: if False, use argmax.
        suppress_mask: [vocab] bool tensor — global suppression (same for all samples).
        suppress_eos_mask: [B] bool tensor — per-sample EOS suppression.
        eos_id: EOS token id (used with suppress_eos_mask).

    Returns:
        [B] tensor of sampled token ids.
    """
    logits = logits.clone()

    # Global suppression
    if suppress_mask is not None:
        logits.masked_fill_(suppress_mask.unsqueeze(0), float("-inf"))

    # Per-sample EOS suppression
    if suppress_eos_mask is not None and eos_id is not None:
        logits[:, eos_id].masked_fill_(suppress_eos_mask, float("-inf"))

    if not do_sample:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k, dim=-1)  # [B, k]
        threshold = topk_vals[:, -1:]  # [B, 1]
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 0] = False  # keep at least top-1
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)  # [B]
