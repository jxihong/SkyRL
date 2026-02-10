"""
Utilities for Teacher-Distillation context formatting.

The teacher policy sees *privileged* in-context examples: complete (reward, trajectory)
pairs from the same prompt group, followed by a *success-conditioned* query.  The format
is designed to maximise the teacher's in-context learning ability:

    [prompt]
    [Reward: r_1]      ← sorted low → high (ascending)
    [trajectory_1]
    [Reward: r_2]
    [trajectory_2]
    ...
    [Reward: target_reward]      ← target reward (success conditioning)
    [current trajectory prefix]  ← the current trajectory prefix
"""

from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig


def _encode_text(text: str, tokenizer: Any, pad_token_id: int, max_tokens: int = 30) -> List[int]:
    """Tokenize a short text string, returning a list of token ids."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = [pad_token_id]
    return ids[:max_tokens]


def format_teacher_icl_context(
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
    """Build teacher input with in-context (reward, trajectory) examples.

    Args:
        sequences:       ``(B, seq_len)``         full sequences ``[prompt | response]``.
        attention_masks: ``(B, seq_len)``         1 for real tokens, 0 for padding.
        response_masks:  ``(B, response_length)`` 1 for real response tokens, 0 for pad.
        rewards:         ``(B, response_length)`` per-token rewards (scalar = sum).
        index:           ``(B,)``                 prompt/group id (same id → same prompt).
        tokenizer:       HF tokenizer.
        config:          reads ``config.teacher_distillation.*``.
        pad_token_id:    padding token id.
        response_length: padded response length (= ``response_masks.shape[1]``).

    Returns:
        formatted_sequences:       ``(B, new_seq_len)``  left-padded to common length.
        formatted_attention_masks: ``(B, new_seq_len)``
    """
    device = sequences.device
    batch_size, seq_len = sequences.shape
    prompt_len = seq_len - response_length

    td_cfg = getattr(config, "teacher_distillation", None) or {}
    reward_format = getattr(td_cfg, "reward_format", "normalized")
    sort_context = getattr(td_cfg, "sort_context_by_reward", True)
    sort_order = getattr(td_cfg, "sort_order", "ascending")
    reward_precision = int(getattr(td_cfg, "reward_precision", 2))
    target_reward = float(getattr(td_cfg, "target_reward", 1.0))

    # get trajectory-levle rewards
    scalar_rewards = (rewards.float() * response_masks.float()).sum(dim=-1)  # (B,)

    # normalize rewards
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

    # group trajectories by prompt
    id2rows_all: defaultdict[str, List[int]] = defaultdict(list)
    for i in range(batch_size):
        id2rows_all[index[i]].append(i)

    # strip left-padding from prompts
    prompt_attn_mask = attention_masks[:, :prompt_len]   # (B, prompt_len)
    has_prompt = prompt_attn_mask.any(dim=1)             # (B,)
    prompt_start_idx = torch.where(
        has_prompt,
        prompt_attn_mask.int().argmax(dim=1),
        torch.full((batch_size,), prompt_len, device=device),
    )  # (B,)

    response_tokens = sequences[:, -response_length:]    # (B, response_length)

    # strip right-padding from trajectories
    real_resp_lens = response_masks.sum(dim=-1).long()   # (B,)

    all_tokens: List[List[int]] = []
    for i in range(batch_size):
        others = [j for j in id2rows_all[index[i]] if j != i]

        # sort trajectories by reward
        if sort_context and len(others) > 1:
            others.sort(
                key=lambda j: scalar_rewards[j].item(),
                reverse=(sort_order == "descending"),
            )

        start = prompt_start_idx[i].item()
        toks: List[int] = sequences[i, start:prompt_len].tolist()

        for j in others:
            r = scalar_rewards[j].item()
            # Reward label BEFORE trajectory
            toks.extend(_encode_text(
                f"\n[Reward: {r:.{reward_precision}f}]\n",
                tokenizer,
                pad_token_id,
            ))
            # Trajectory tokens (stripped of right-padding)
            resp_len_j = real_resp_lens[j].item()
            toks.extend(response_tokens[j, :resp_len_j].tolist())

        toks.extend(_encode_text(
            f"\n[Reward: {target_reward:.{reward_precision}f}]\n",
            tokenizer,
            pad_token_id,
        ))
        # Current response: keep ALL response_length tokens (including padding)
        # so the last `response_length` positions are exactly the current response
        # and `num_actions` aligns with the model's forward pass.
        toks.extend(response_tokens[i].tolist())
        all_tokens.append(toks)

    # pad
    max_len = max(len(t) for t in all_tokens)

    out_seqs = torch.full(
        (batch_size, max_len), pad_token_id, dtype=sequences.dtype, device=device
    )
    out_attn = torch.zeros(batch_size, max_len, dtype=attention_masks.dtype, device=device)

    for i in range(batch_size):
        L = len(all_tokens[i])
        out_seqs[i, max_len - L :] = torch.tensor(
            all_tokens[i], dtype=sequences.dtype, device=device
        )
        out_attn[i, max_len - L :] = 1

    return out_seqs, out_attn
