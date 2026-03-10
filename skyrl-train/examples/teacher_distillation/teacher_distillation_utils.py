"""
Utilities for Teacher-Distillation context formatting.

The teacher policy sees *privileged* information built from the
student's own full rollout and achieved reward.

    [prompt]
    The following is what the student generated and its achieved reward:
    [student's full rollout]
    [Reward: student's achieved reward]
    Now, generate your own completion that fixes any mistakes in the student's.
    [current trajectory prefix]  ← the current trajectory prefix
"""

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
    _ = index  # prompt-group ids are unused in self-reference context mode

    td_cfg = getattr(config, "teacher_distillation", None) or {}
    reward_precision = int(getattr(td_cfg, "reward_precision", 2))
    max_reference_response_tokens = getattr(td_cfg, "max_reference_response_tokens", None)
    if max_reference_response_tokens is not None:
        max_reference_response_tokens = int(max_reference_response_tokens)
        if max_reference_response_tokens <= 0:
            max_reference_response_tokens = None

    preface_ids = _encode_text(
        "\nThe following is what the student generated and its achieved reward:\n",
        tokenizer,
        pad_token_id,
    )
    instruction_ids = _encode_text(
        "\nNow, generate your own completion that fixes any mistakes in the student's.\n",
        tokenizer,
        pad_token_id,
    )

    # get trajectory-levle rewards
    scalar_rewards = (rewards.float() * response_masks.float()).sum(dim=-1)  # (B,)

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
        start = prompt_start_idx[i].item()
        toks: List[int] = sequences[i, start:prompt_len].tolist()

        # Include the student's own full rollout and achieved reward as a reference.
        # This conditions teacher scoring toward improving this concrete rollout.
        r_ref = scalar_rewards[i].item()
        toks.extend(preface_ids)
        resp_len_i = real_resp_lens[i].item()
        ref_rollout_tokens = response_tokens[i, :resp_len_i].tolist()
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and ref_rollout_tokens and ref_rollout_tokens[-1] == eos_token_id:
            ref_rollout_tokens = ref_rollout_tokens[:-1]
        if max_reference_response_tokens is not None and len(ref_rollout_tokens) > max_reference_response_tokens:
            # Keep both the beginning and ending spans, and mark the omitted middle.
            head_len = max_reference_response_tokens // 2
            tail_len = max_reference_response_tokens - head_len
            omitted_ids = _encode_text("\n<OMITTED>\n", tokenizer, pad_token_id, max_tokens=16)
            ref_rollout_tokens = (
                ref_rollout_tokens[:head_len] + omitted_ids + ref_rollout_tokens[-tail_len:]
            )
        toks.extend(ref_rollout_tokens)
        toks.extend(_encode_text(
            f"\n[Reward: {r_ref:.{reward_precision}f}]\n",
            tokenizer,
            pad_token_id,
        ))
        toks.extend(instruction_ids)

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
