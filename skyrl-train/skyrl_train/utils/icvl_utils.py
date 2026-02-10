"""
Utilities for ICVL (In-Context Value Learning) advantage estimation.

The critic is given other responses (and their rewards) from the same prompt group
as in-context examples, then predicts values for the current response. This leverages
the transformer's in-context learning ability: seeing (response, reward) pairs for the
same prompt gives the critic a much stronger signal for predicting the value of the
current response.
"""

from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig


def _encode_text(text: str, tokenizer: Any, pad_token_id: int, max_tokens: int = 20) -> List[int]:
    """Tokenize a short text string, returning a list of token ids."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = [pad_token_id]
    return ids[:max_tokens]


def format_icvl_batch_with_context(
    sequences: torch.Tensor,
    attention_masks: torch.Tensor,
    response_masks: torch.Tensor,
    rewards: torch.Tensor,
    index: np.ndarray,
    tokenizer: Any,
    config: DictConfig,
    pad_token_id: int,
    response_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build critic input with in-context (response, reward) examples from the same prompt.

    For each sample *i* in the batch we construct:

        [prompt_i]                                       (original prompt, left-pad preserved)
        Response 1: [resp_j1]
        [Reward 1: r_j1]      
        Response 2: [resp_j2]
        [Reward 2: r_j2] 
        ...
        [resp_i]                                         (current response — NO prefix)

    Args:
        sequences:       (B, seq_len)             full sequences [prompt | response].
        attention_masks: (B, seq_len)             1 for real tokens, 0 for padding.
        response_masks:  (B, response_length)     1 for real response tokens, 0 for pad.
        rewards:         (B, response_length)     per-token rewards.
        index:           (B,)                     prompt/group id (same id → same prompt).
        tokenizer:       HF tokenizer.
        config:          ``cfg.trainer.algorithm`` — reads ``config.icvl.*``.
        pad_token_id:    padding token id.
        response_length: padded response length (= ``response_masks.shape[1]``).

    Returns:
        formatted_sequences:        (B, new_seq_len)  left-padded to common length.
        formatted_attention_masks:  (B, new_seq_len)
    """
    device = sequences.device
    batch_size, seq_len = sequences.shape
    prompt_len = seq_len - response_length

    # ---- config ----
    icvl_cfg = getattr(config, "icvl", None) or {}
    reward_format = getattr(icvl_cfg, "reward_format", "raw")
    sort_context = getattr(icvl_cfg, "sort_context_by_reward", True)
    sort_order = getattr(icvl_cfg, "sort_order", "ascending")
    reward_precision = int(getattr(icvl_cfg, "reward_precision", 4))

    # ---- scalar reward per response ----
    scalar_rewards = (rewards.float() * response_masks.float()).sum(dim=-1)  # (B,)

    # ---- optional: normalize per group to [0, 1] ----
    if reward_format == "normalized":
        id2rows: defaultdict[str, List[int]] = defaultdict(list)
        for i in range(batch_size):
            id2rows[index[i]].append(i)
        normed = scalar_rewards.clone()
        for rows in id2rows.values():
            if len(rows) <= 1:
                normed[rows] = 0.5
                continue
            vals = scalar_rewards[rows]
            lo, hi = vals.min().item(), vals.max().item()
            if hi > lo:
                normed[rows] = (vals - lo) / (hi - lo)
            else:
                normed[rows] = 0.5
        scalar_rewards = normed

    # ---- group rows by prompt ----
    id2rows_all: defaultdict[str, List[int]] = defaultdict(list)
    for i in range(batch_size):
        id2rows_all[index[i]].append(i)

    # ---- per-row prompt boundaries (strip left-padding) ----
    # Find the first real token per row so we only include actual prompt tokens.
    prompt_attn_mask = attention_masks[:, :prompt_len]             # (B, prompt_len)
    # first_real[i] = index of first attn==1 token in the prompt region
    # If the entire prompt region is padding (shouldn't happen), default to prompt_len.
    has_prompt = prompt_attn_mask.any(dim=1)                        # (B,)
    prompt_start_idx = torch.where(
        has_prompt,
        prompt_attn_mask.int().argmax(dim=1),                     # first 1
        torch.full((batch_size,), prompt_len, device=device),
    )  # (B,)

    response_tokens = sequences[:, -response_length:]               # (B, response_length)

    all_tokens: List[List[int]] = []
    for i in range(batch_size):
        others = [j for j in id2rows_all[index[i]] if j != i]

        # Sort context examples by reward for better ICL pattern recognition
        if sort_context and len(others) > 1:
            others.sort(key=lambda j: scalar_rewards[j].item(), reverse=(sort_order == "descending"))

        # -- prompt (stripped of left-padding — only real tokens) --
        start = prompt_start_idx[i].item()
        toks: List[int] = sequences[i, start:prompt_len].tolist()

        # -- ICL context: (response, reward) pairs --
        for ctx_num, j in enumerate(others, start=1):
            toks.extend(_encode_text(f"\nResponse {ctx_num}: ", tokenizer, pad_token_id))
            toks.extend(response_tokens[j].tolist())
            r = scalar_rewards[j].item()
            toks.extend(_encode_text(
                f"\n[Reward {ctx_num}: {r:.{reward_precision}f}]", tokenizer, pad_token_id
            ))

        # -- current response (NO prefix so last response_length tokens align exactly) --
        toks.extend(response_tokens[i].tolist())
        all_tokens.append(toks)

    # ---- left-pad to common length ----
    # Every appended token is real (prompt padding was stripped), so attention mask
    # is simply 1 for token positions and 0 for left-pad.
    max_len = max(len(t) for t in all_tokens)

    out_seqs = torch.full((batch_size, max_len), pad_token_id, dtype=sequences.dtype, device=device)
    out_attn = torch.zeros(batch_size, max_len, dtype=attention_masks.dtype, device=device)

    for i in range(batch_size):
        L = len(all_tokens[i])
        out_seqs[i, max_len - L:] = torch.tensor(all_tokens[i], dtype=sequences.dtype, device=device)
        out_attn[i, max_len - L:] = 1

    return out_seqs, out_attn
