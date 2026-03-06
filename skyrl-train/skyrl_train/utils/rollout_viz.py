import html
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


def _sample_indices(
    total_samples: int,
    max_samples_per_log: int,
    is_last_step: Optional[torch.Tensor] = None,
    uids: Optional[List[str]] = None,
) -> List[int]:
    if total_samples <= 0 or max_samples_per_log <= 0:
        return []
    if is_last_step is None:
        candidate_indices = list(range(total_samples))
    else:
        candidate_indices = [idx for idx in range(total_samples) if bool(is_last_step[idx].item())]

    if not uids:
        return candidate_indices[:max_samples_per_log]

    # Prefer one sample per problem UID so visualizations cover different problems.
    chosen_indices: List[int] = []
    seen_uids = set()
    for idx in candidate_indices:
        if idx >= len(uids):
            continue
        uid = uids[idx]
        if uid in seen_uids:
            continue
        seen_uids.add(uid)
        chosen_indices.append(idx)
        if len(chosen_indices) >= max_samples_per_log:
            return chosen_indices

    # If unique problems are fewer than requested, backfill with remaining candidates.
    if len(chosen_indices) < max_samples_per_log:
        chosen_set = set(chosen_indices)
        for idx in candidate_indices:
            if idx in chosen_set:
                continue
            chosen_indices.append(idx)
            if len(chosen_indices) >= max_samples_per_log:
                break
    return chosen_indices


def _expand_indices_for_selected_uids(
    *,
    selected_indices: List[int],
    total_samples: int,
    uids: Optional[List[str]] = None,
    is_last_step: Optional[torch.Tensor] = None,
) -> List[int]:
    if not selected_indices:
        return []

    # If UIDs are unavailable, fallback to the selected trajectory indices.
    if not uids:
        return selected_indices

    candidate_indices = (
        list(range(total_samples))
        if is_last_step is None
        else [idx for idx in range(total_samples) if bool(is_last_step[idx].item())]
    )

    selected_uid_order: List[str] = []
    selected_uid_set = set()
    for idx in selected_indices:
        if idx >= len(uids):
            continue
        uid = uids[idx]
        if uid in selected_uid_set:
            continue
        selected_uid_set.add(uid)
        selected_uid_order.append(uid)

    expanded_indices: List[int] = []
    for uid in selected_uid_order:
        for idx in candidate_indices:
            if idx >= len(uids):
                continue
            if uids[idx] == uid:
                expanded_indices.append(idx)

    return expanded_indices


def _normalize_value(value: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return max(-1.0, min(1.0, value / denom))


def _value_to_rgb(norm_value: float) -> str:
    # Diverging colormap: negative -> red, zero -> white, positive -> blue.
    if norm_value >= 0:
        intensity = int(255 - 120 * norm_value)
        r, g, b = intensity, intensity, 255
    else:
        intensity = int(255 - 120 * abs(norm_value))
        r, g, b = 255, intensity, intensity
    return f"rgb({r}, {g}, {b})"


def _decode_tokens(tokenizer, token_ids: Sequence[int]) -> List[str]:
    decoded = []
    for token_id in token_ids:
        decoded.append(
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    return decoded


def _compute_valid_length(response_ids: Sequence[int], response_mask_row: torch.Tensor) -> int:
    mask_length = int(response_mask_row.long().sum().item())
    return min(len(response_ids), max(mask_length, 0))


def _format_prompt(prompt: Optional[Any]) -> str:
    if prompt is None:
        return ""
    prompt_str = str(prompt)
    if len(prompt_str) > 4000:
        return f"{prompt_str[:4000]} ...[truncated]"
    return prompt_str


def build_rollout_value_advantage_wandb_payload(
    *,
    wandb_module: Any,
    tokenizer,
    response_ids: List[List[int]],
    values: Optional[torch.Tensor],
    advantages: Optional[torch.Tensor],
    response_mask: torch.Tensor,
    rewards: Optional[torch.Tensor],
    global_step: int,
    max_samples_per_log: int,
    max_tokens_per_sample: int,
    normalize_per_sample: bool = True,
    is_last_step: Optional[torch.Tensor] = None,
    uids: Optional[List[str]] = None,
    prompts: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    if values is None:
        return {}

    # Select problem-level seeds, then expand to all rollouts for those problems.
    selected_problem_indices = _sample_indices(
        total_samples=len(response_ids),
        max_samples_per_log=max_samples_per_log,
        is_last_step=is_last_step,
        uids=uids,
    )
    indices = _expand_indices_for_selected_uids(
        selected_indices=selected_problem_indices,
        total_samples=len(response_ids),
        uids=uids,
        is_last_step=is_last_step,
    )
    if not indices:
        return {}

    html_blocks: List[str] = []
    selected_values: List[np.ndarray] = []
    selected_token_ids: List[List[int]] = []
    selected_decoded_tokens: List[List[str]] = []
    selected_rewards: List[float] = []
    selected_uids: List[str] = []
    selected_prompts: List[str] = []

    for idx in indices:
        valid_len = _compute_valid_length(response_ids[idx], response_mask[idx])
        token_ids = response_ids[idx][:valid_len][:max_tokens_per_sample]

        if len(token_ids) == 0:
            continue

        value_row = values[idx][: len(token_ids)].detach().float().cpu().numpy()
        selected_values.append(value_row)
        selected_token_ids.append(token_ids)
        selected_decoded_tokens.append(_decode_tokens(tokenizer, token_ids))

        if rewards is not None:
            reward_row = rewards[idx][: len(token_ids)].detach().float().cpu()
            selected_rewards.append(float(reward_row.sum().item()))
        else:
            selected_rewards.append(0.0)

        selected_uids.append(uids[idx] if uids is not None and idx < len(uids) else "")
        selected_prompts.append(_format_prompt(prompts[idx]) if prompts is not None and idx < len(prompts) else "")

    if not selected_values:
        return {}

    global_norm = max(float(np.max(np.abs(np.concatenate(selected_values)))), 1e-8)

    for sample_i, (token_ids, decoded_tokens, value_row, reward_sum, uid, prompt_text) in enumerate(
        zip(
            selected_token_ids,
            selected_decoded_tokens,
            selected_values,
            selected_rewards,
            selected_uids,
            selected_prompts,
        )
    ):
        sample_norm = max(float(np.max(np.abs(value_row))), 1e-8)
        denom = sample_norm if normalize_per_sample else global_norm
        clipped_norm_values = [_normalize_value(float(v), denom) for v in value_row]

        rendered_value_tokens: List[str] = []
        for token, norm_value, value in zip(decoded_tokens, clipped_norm_values, value_row):
            bgcolor = _value_to_rgb(norm_value)
            token_html = html.escape(token).replace("\n", "\\n")
            rendered_value_tokens.append(
                (
                    f"<span title='value={float(value):.4f}' "
                    f"style='background-color:{bgcolor}; padding:1px 2px; border-radius:2px; margin-right:1px;'>"
                    f"{token_html}</span>"
                )
            )
        value_mean = float(np.mean(value_row))
        value_std = float(np.std(value_row))

        html_blocks.append(
            (
                "<div style='margin-bottom:12px;'>"
                f"<div><b>sample={sample_i}</b> uid={html.escape(uid)} reward_sum={reward_sum:.4f} "
                f"value_mean={value_mean:.4f} value_std={value_std:.4f} "
                f"</div>"
                "<div style='margin-top:6px;'><b>Prompt (no heatmap)</b></div>"
                f"<pre style='margin-top:2px; white-space:pre-wrap; background:#f7f7f7; padding:6px; border-radius:4px;'>{html.escape(prompt_text)}</pre>"
                "<div style='margin-top:4px;'><b>Critic Values Heatmap</b></div>"
                f"<div style='margin-top:2px; font-family:monospace; white-space:pre-wrap;'>{''.join(rendered_value_tokens)}</div>"
                "</div>"
            )
        )

    payload: Dict[str, Any] = {}

    html_doc = (
        "<html><body>"
        "<div><b>Rollout token heatmaps</b> (red=negative, blue=positive)</div>"
        "<div style='margin-bottom:8px;'>Each rollout shows critic value heatmap only.</div>"
        f"{''.join(html_blocks)}"
        "</body></html>"
    )
    payload["train/rollout_values_html"] = wandb_module.Html(html_doc)

    return payload
