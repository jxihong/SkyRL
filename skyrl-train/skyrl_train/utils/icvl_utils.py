"""
Utilities for ICVL (In-Context Value Learning) advantage estimation.

The critic is given other responses (and their rewards) from the same prompt group
as in-context examples, then predicts values for the current response. This leverages
the transformer's in-context learning ability: seeing (response, reward) pairs for the
same prompt gives the critic a much stronger signal for predicting the value of the
current response.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig


def _encode_text(text: str, tokenizer: Any, pad_token_id: int, max_tokens: int = 20) -> List[int]:
    """Tokenize a short text string, returning a list of token ids."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = [pad_token_id]
    return ids[:max_tokens]


def _get_zip_template_tokens(tokenizer: Any) -> Dict[str, Any]:
    """Derive chat template structural tokens from the tokenizer.

    Uses ``apply_chat_template`` with minimal probe messages so the result
    is correct for any model family (Qwen, Llama, Gemma, …).

    Returns a dict with:
        assistant_header  – token ids that open an assistant turn
        eos_token_id      – single token id that closes an assistant turn
        user_header       – token ids that open a user turn (after a prior assistant turn)
        user_footer       – token ids that close a user turn
    """
    base = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=True, add_generation_prompt=False,
    )
    with_gen = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=True, add_generation_prompt=True,
    )
    assistant_header = list(with_gen[len(base):])

    one_asst = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        tokenize=True, add_generation_prompt=False,
    )
    two_turn = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
         {"role": "user", "content": "z"}],
        tokenize=True, add_generation_prompt=False,
    )

    # --- eos_token_id: first token after assistant content in a completed turn ---
    y_tokens = tokenizer.encode("y", add_special_tokens=False)
    after_asst_content = list(one_asst[len(base) + len(assistant_header) + len(y_tokens):])
    eos_token_id = after_asst_content[0] if after_asst_content else tokenizer.eos_token_id

    # --- user header / footer: structural tokens around the second user message ---
    second_user = list(two_turn[len(one_asst):])
    z_tokens = tokenizer.encode("z", add_special_tokens=False)
    z_start = None
    for idx in range(len(second_user) - len(z_tokens) + 1):
        if second_user[idx : idx + len(z_tokens)] == z_tokens:
            z_start = idx
            break
    if z_start is not None:
        user_header = second_user[:z_start]
        user_footer = second_user[z_start + len(z_tokens):]
    else:
        user_header = []
        user_footer = []

    return {
        "assistant_header": assistant_header,
        "eos_token_id": eos_token_id,
        "user_header": user_header,
        "user_footer": user_footer,
    }


def _build_zip_context_tokens(
    prompt_toks: List[int],
    others: List[int],
    response_tokens: torch.Tensor,
    response_masks: torch.Tensor,
    raw_scalar_rewards: torch.Tensor,
    target_response_toks: List[int],
    tokenizer: Any,
    tmpl: Dict[str, Any],
) -> List[int]:
    """Build token sequence matching the zip critic training format.

    The format mirrors ``train_in_context_critic.py``::

        {user_header}{prompt}{user_footer}
        {assistant_header}{resp_1}{eos}
        {user_header}Reward: {r}\\nLength: {l} tokens{user_footer}
        ...
        {assistant_header}{target_resp}

    ``tmpl`` is the dict returned by :func:`_get_zip_template_tokens`.
    """
    assistant_header = tmpl["assistant_header"]
    eos_token_id = tmpl["eos_token_id"]
    user_header = tmpl["user_header"]
    user_footer = tmpl["user_footer"]

    toks: List[int] = []

    # Prompt: strip trailing assistant header that the RL chat template appends,
    # since each response carries its own assistant header in the training format.
    hlen = len(assistant_header)
    if hlen and len(prompt_toks) >= hlen and prompt_toks[-hlen:] == assistant_header:
        toks.extend(prompt_toks[:-hlen])
    else:
        toks.extend(prompt_toks)

    for j in others:
        resp_len_j = int(response_masks[j].sum().item())
        resp_toks = response_tokens[j, :resp_len_j].tolist()

        has_eos = resp_toks and resp_toks[-1] == eos_token_id
        footer = [] if has_eos else [eos_token_id]
        content_plus_footer_len = len(resp_toks) + len(footer)

        # Assistant message (no trailing tokens after eos — matches training traj_block)
        toks.extend(assistant_header)
        toks.extend(resp_toks)
        toks.extend(footer)

        # User feedback message
        r = raw_scalar_rewards[j].item()
        feedback_str = f"Reward: {r}\nLength: {content_plus_footer_len} tokens"
        feedback_toks = tokenizer.encode(feedback_str, add_special_tokens=False)
        toks.extend(user_header)
        toks.extend(feedback_toks)
        toks.extend(user_footer)

    # Target response: re-add assistant header, then raw response tokens
    toks.extend(assistant_header)
    toks.extend(target_response_toks)
    return toks


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
    use_zip_format: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build critic input with in-context (response, reward) examples from the same prompt.

    When ``use_zip_format=False`` (default), uses the plain-text ICVL format:

        [prompt]
        Response 1: [resp_1]  
        Reward 1: [r_1]
        ...
        [target_resp]

    When ``use_zip_format=True``, uses the model's chat template (derived
    from the tokenizer) to match the format used by the ZIP training:

        {user turn: [prompt]}
        {assistant turn: [resp_1]}{eos}
        {user turn: Reward: [r_1]\\nLength: [l_1] tokens}
        ...
        {assistant turn: [target_resp]}

    In both modes the last ``response_length`` tokens are the target response,
    preserving value-head alignment.
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
    raw_scalar_rewards = (rewards.float() * response_masks.float()).sum(dim=-1)  # (B,)
    scalar_rewards = raw_scalar_rewards

    # ---- optional: normalize per group to [0, 1] (only for plain-text format) ----
    if reward_format == "normalized" and not use_zip_format:
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
    prompt_attn_mask = attention_masks[:, :prompt_len]              # (B, prompt_len)
    has_prompt = prompt_attn_mask.any(dim=1)                        # (B,)
    prompt_start_idx = torch.where(
        has_prompt,
        prompt_attn_mask.int().argmax(dim=1),
        torch.full((batch_size,), prompt_len, device=device),
    )  # (B,)

    response_tokens = sequences[:, -response_length:]               # (B, response_length)

    # Derive chat template tokens once for the whole batch
    zip_tmpl = _get_zip_template_tokens(tokenizer) if use_zip_format else None

    all_tokens: List[List[int]] = []
    for i in range(batch_size):
        others = [j for j in id2rows_all[index[i]] if j != i]

        if sort_context and len(others) > 1:
            others.sort(key=lambda j: scalar_rewards[j].item(), reverse=(sort_order == "descending"))

        start = prompt_start_idx[i].item()
        prompt_toks: List[int] = sequences[i, start:prompt_len].tolist()

        if use_zip_format:
            toks = _build_zip_context_tokens(
                prompt_toks=prompt_toks,
                others=others,
                response_tokens=response_tokens,
                response_masks=response_masks,
                raw_scalar_rewards=raw_scalar_rewards,
                target_response_toks=response_tokens[i].tolist(),
                tokenizer=tokenizer,
                tmpl=zip_tmpl,
            )
        else:
            toks = list(prompt_toks)
            for ctx_num, j in enumerate(others, start=1):
                resp_len_j = int(response_masks[j].sum().item())
                toks.extend(_encode_text(f"\nResponse {ctx_num}: ", tokenizer, pad_token_id))
                toks.extend(response_tokens[j, :resp_len_j].tolist())
                r = scalar_rewards[j].item()
                toks.extend(
                    _encode_text(f"\nReward {ctx_num}: {r:.{reward_precision}f}", tokenizer, pad_token_id)
                )
            toks.extend(response_tokens[i].tolist())

        all_tokens.append(toks)

    # ---- left-pad to common length ----
    max_len = max(len(t) for t in all_tokens)

    out_seqs = torch.full((batch_size, max_len), pad_token_id, dtype=sequences.dtype, device=device)
    out_attn = torch.zeros(batch_size, max_len, dtype=attention_masks.dtype, device=device)

    for i in range(batch_size):
        L = len(all_tokens[i])
        out_seqs[i, max_len - L:] = torch.tensor(all_tokens[i], dtype=sequences.dtype, device=device)
        out_attn[i, max_len - L:] = 1

    return out_seqs, out_attn
