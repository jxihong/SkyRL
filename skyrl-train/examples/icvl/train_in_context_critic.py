#!/usr/bin/env python3
"""
NOTE: Reference implementation of ZIP training (copied from external)

Fine-tune a model for Joint Reward-Cost Prediction (JointValueCost) with DDP support.
Includes "In-Context Critic" logic: prepends sibling responses (and optional reward/length
feedback) to the context to improve prediction of target responses.
Uses PRE-TOKENIZED data from the generation step to ensure <think> tags and 
special formatting are preserved exactly.

UPDATED: Uses a single packed sequence per prompt containing ALL trajectories 
for supervision in a single forward pass (supervising trajectories N..end).

Example usage:
    python train.py --model_id Qwen/Qwen3-1.7B \
                    --weights_path models/joint_critic --data_path data/data.parquet
"""
from __future__ import annotations
import argparse, math, os, time, wandb, ast, random
from typing import List, Dict, Any
import pyarrow.parquet as pq
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from visualization import visualize_predictions, log_prediction_distributions, log_joint_distribution_grid

# --- Qwen Specific Control Tokens ---
# Verify these match your specific model version if not using Qwen 2.5/3
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
USER_TOKEN_ID = 872
ASSISTANT_TOKEN_ID = 77091
SYSTEM_TOKEN_ID = 8948
NEWLINE_TOKEN_ID = 198

class JointDistributionDataset(Dataset):
    def __init__(self, table, tokenizer, max_length: int = 131_072, thinking_only: bool = False,
                 thinking_token_id: int = 151667, reward_values: List[float] | None = None,
                 label_column: str = "correct", ablation_type: str = "no_ans",
                 examples_per_prompt: int = 1, correctness_only: bool = False,
                 supervise_from_trajectory: int = 4, no_context: bool = False,
                 max_prompts: int | None = None, strip_context_thinking: bool = False,
                 inject_answer: bool = False):
        """
        Modified Dataset to implement In-Context Critic Hypothesis using PRE-TOKENIZED data.
        NOW PACKED: Creates one long sequence per prompt containing trajectories.
        Each packed example supervises trajectories `supervise_from_trajectory..end`
        in a single forward pass.

        If strip_context_thinking=True:
        - Context trajectories have thinking stripped (only answer after </think>)
        - Only the target trajectory (last one) keeps full thinking trace
        - Only the target trajectory is supervised

        If inject_answer=True (requires no_context=True):
        - Appends the ground truth answer to the prompt
        - Format: prompt + "\n\nThe correct answer is {answer}"
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inject_answer = inject_answer
        self.ablation_type = ablation_type
        self.correctness_only = correctness_only
        self.supervise_from_trajectory = int(supervise_from_trajectory)
        self.no_context = no_context
        self.strip_context_thinking = strip_context_thinking

        # Cache </think> tokens for stripping thinking from context
        if strip_context_thinking:
            self.end_think_tokens = tokenizer.encode("</think>", add_special_tokens=False)
            print(f"strip_context_thinking enabled. </think> token sequence: {self.end_think_tokens}")
        else:
            self.end_think_tokens = None
        
        # Load Data
        df = pq.read_table(table).to_pandas() if isinstance(table, str) else table
        
        # Validation
        required_cols = ["prompt_idx", "prompt", "output_token_ids"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Dataset missing required column: {c}")

        # Decide which supervision column to use
        choice = (label_column or "correct").lower()
        if choice == "auto":
            if "correct" in df.columns:
                self.reward_column = "correct"
            elif "value" in df.columns:
                self.reward_column = "value"
            else:
                raise ValueError("Data must contain 'correct' or 'value' column.")
        elif choice == "correct":
            if "correct" in df.columns:
                self.reward_column = "correct"
            else:
                raise ValueError("Requested label_column='correct' but column 'correct' is not present.")
        elif choice == "value":
            if "value" not in df.columns:
                raise ValueError("Requested label_column='value' but 'value' not found.")
            self.reward_column = "value"
        else:
            raise ValueError(f"Unknown label_column: {label_column}")

        # Normalize Reward Column
        if self.reward_column == "correct":
            df["correct"] = df["correct"].astype(float)

        # Filter thinking samples if requested
        if thinking_only:
            if "reasoning_enabled" in df.columns:
                df = df[df["reasoning_enabled"] == True].copy()
            else:
                print("Warning: --thinking_only requested but 'reasoning_enabled' column not found. Skipping filter.")

        # Filter to max_prompts if specified
        if max_prompts is not None:
            unique_prompts = df["prompt_idx"].unique()
            total_prompts = len(unique_prompts)
            if max_prompts < total_prompts:
                rng = np.random.default_rng(seed=42)
                rng.shuffle(unique_prompts)
                selected_prompts = unique_prompts[:max_prompts]
                df = df[df["prompt_idx"].isin(selected_prompts)].copy()
                print(f"Limiting to {max_prompts} prompts (from {total_prompts} total, shuffled with seed=42).")
            else:
                print(f"max_prompts={max_prompts} >= total prompts ({total_prompts}), using all data.")

        # --- Grouping Logic ---
        self.samples = []

        if self.no_context:
            # No-context mode: one sample per trajectory (no in-context conditioning)
            if self.inject_answer:
                if "answer" not in df.columns:
                    raise ValueError("inject_answer=True requires 'answer' column in data")
                print("No-context mode with answer injection: creating one sample per trajectory...")
            else:
                print("No-context mode: creating one sample per trajectory...")

            for _, row in df.iterrows():
                toks = row['output_token_ids']
                if hasattr(toks, 'tolist'): toks = toks.tolist()

                # Optionally inject answer into prompt
                prompt_text = row['prompt']
                if self.inject_answer:
                    answer = row.get('answer', '')
                    prompt_text = f"{prompt_text}\n\nThe correct answer is {answer}"

                self.samples.append({
                    "prompt": prompt_text,
                    "trajectory": {
                        "response_ids": toks,
                        "reward": float(row[self.reward_column])
                    }
                })
            print(f"Constructed {len(self.samples)} independent samples (no context, inject_answer={self.inject_answer}).")
        else:
            # Packed In-Context mode (original behavior)
            print("Grouping dataset by prompt for Packed In-Context generation...")

            # Group by prompt_idx
            grouped = df.groupby("prompt_idx")

            for prompt_idx, group in grouped:
                rows = group.to_dict('records')
                if not rows: continue

                # Common data elements
                prompt_text = rows[0]['prompt']

                # Collect all trajectories for this prompt
                trajectories = []
                for r in rows:
                    toks = r['output_token_ids']
                    if hasattr(toks, 'tolist'): toks = toks.tolist()

                    trajectories.append({
                        "response_ids": toks,
                        "reward": float(r[self.reward_column])
                    })

                # Create N examples per prompt by reshuffling trajectories.
                # Each example will supervise only the last trajectory (after suffix-fit selection).
                for _ in range(int(examples_per_prompt)):
                    trajs = list(trajectories)
                    random.shuffle(trajs)
                    self.samples.append({
                        "prompt": prompt_text,
                        "trajectories": trajs
                    })

            print(f"Constructed {len(self.samples)} packed samples ({examples_per_prompt} per prompt).")

        # --- Binning Setup ---
        if correctness_only:
            self.length_bins = [0, self.max_length + 1]
        else:
            self.length_bins = [0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        self.num_length_bins = len(self.length_bins) - 1
        
        if reward_values is not None:
            self.reward_values = reward_values
        else:
            if self.reward_column == "correct":
                self.reward_values = [0.0, 1.0]
            else:
                self.reward_values = [0.0, 0.1666667, 0.3333333, 0.5, 0.6666667, 0.8333333, 1.0]

        self.num_reward_states = len(self.reward_values)
        self.num_bins = self.num_length_bins * self.num_reward_states

        # Derive value bin edges
        if self.num_reward_states >= 2:
            edges = [0.0] * (self.num_reward_states + 1)
            for i in range(1, self.num_reward_states):
                edges[i] = 0.5 * (self.reward_values[i - 1] + self.reward_values[i])
            
            first_step = self.reward_values[1] - self.reward_values[0]
            last_step = self.reward_values[-1] - self.reward_values[-2]
            edges[0] = self.reward_values[0] - 0.5 * first_step
            edges[-1] = self.reward_values[-1] + 0.5 * last_step
        else:
            edges = [self.reward_values[0] - 0.5, self.reward_values[0] + 0.5]
        
        edges[0] = max(0.0, edges[0])
        edges[-1] = min(1.0, edges[-1])
        self.value_bin_edges = edges

    def __len__(self): return len(self.samples)

    def _get_bin_idx(self, tokens_to_completion, reward):
        length_bin = 0
        for i in range(len(self.length_bins) - 1):
            if tokens_to_completion >= self.length_bins[i] and tokens_to_completion < self.length_bins[i + 1]:
                length_bin = i
                break
        if tokens_to_completion >= self.length_bins[-1]:
            length_bin = self.num_length_bins - 1

        reward_state = 0
        for i in range(len(self.value_bin_edges) - 1):
            if reward >= self.value_bin_edges[i] and reward < self.value_bin_edges[i + 1]:
                reward_state = i
                break
        if reward >= self.value_bin_edges[-1]:
            reward_state = self.num_reward_states - 1

        return length_bin + reward_state * self.num_length_bins

    def _get_answer_start_idx(self, token_ids: List[int]) -> int:
        """
        Find where the answer starts (after </think> if present).
        Returns the index of the first token after </think>, or 0 if not found.
        """
        if self.end_think_tokens is None:
            return 0

        seq_len = len(self.end_think_tokens)
        if seq_len == 0:
            return 0

        # Search for </think> sequence in token_ids
        for i in range(len(token_ids) - seq_len + 1):
            if token_ids[i:i + seq_len] == self.end_think_tokens:
                return i + seq_len  # Return position after </think>

        return 0  # If not found, use full content (no thinking in this response)

    # --- Manual Token Construction Helpers ---
    @staticmethod
    def create_tokenized_message(role, tokenized_content):
        role_token_id = {
            "system": SYSTEM_TOKEN_ID,
            "user": USER_TOKEN_ID,
            "assistant": ASSISTANT_TOKEN_ID,
        }[role]
        
        content = list(tokenized_content)
        if content and content[-1] == IM_END_TOKEN_ID:
            content = content[:-1]
            
        return [IM_START_TOKEN_ID, role_token_id, NEWLINE_TOKEN_ID] + content + [IM_END_TOKEN_ID, NEWLINE_TOKEN_ID]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        full_input_ids = []
        full_label_positions = []
        full_bin_labels = []

        # 1. User: Prompt
        prompt_tokens = self.tokenizer.encode(sample['prompt'], add_special_tokens=False)
        full_input_ids.extend(self.create_tokenized_message("user", prompt_tokens))

        prefix_len = len(full_input_ids)

        # --- No-context mode: simple [prompt][trajectory] ---
        if self.no_context:
            traj = sample['trajectory']
            traj_content = list(traj['response_ids'])

            header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
            has_eos = (traj_content and traj_content[-1] == IM_END_TOKEN_ID)
            footer = [] if has_eos else [IM_END_TOKEN_ID]

            # Truncate if needed
            max_content_len = self.max_length - prefix_len - len(header) - len(footer)
            if max_content_len < 0:
                max_content_len = 0
            traj_content = traj_content[:max_content_len]

            # Recompute footer after truncation
            has_eos = (traj_content and traj_content[-1] == IM_END_TOKEN_ID)
            footer = [] if has_eos else [IM_END_TOKEN_ID]

            traj_block = header + traj_content + footer
            content_plus_footer_len = len(traj_content) + len(footer)

            # Add trajectory block
            current_start_idx = len(full_input_ids)
            full_input_ids.extend(traj_block)

            # Supervise all tokens in the trajectory
            reward_val = float(traj['reward'])
            reward_val = max(0.0, min(1.0, reward_val))

            content_start_rel_idx = len(header)
            block_len = len(traj_block)
            for i in range(content_plus_footer_len):
                abs_pos = current_start_idx + content_start_rel_idx + i
                tokens_to_completion = (current_start_idx + block_len) - abs_pos - 1
                bin_idx = self._get_bin_idx(tokens_to_completion, reward_val)
                full_label_positions.append(abs_pos)
                full_bin_labels.append(bin_idx)

            input_tensor = torch.tensor(full_input_ids, dtype=torch.long)
            return {
                "input_ids": input_tensor,
                "label_positions": full_label_positions,
                "bin_labels": full_bin_labels,
                "num_bins": self.num_bins
            }

        # --- Packed In-Context mode (original behavior) ---
        # 2. Trajectory packing: include a suffix that fits max_length and supervise
        # trajectories `supervise_from_trajectory..end` within the packed order.
        trajectories = sample['trajectories']
        if not trajectories:
            input_tensor = torch.tensor(full_input_ids, dtype=torch.long)
            return {
                "input_ids": input_tensor,
                "label_positions": full_label_positions,
                "bin_labels": full_bin_labels,
                "num_bins": self.num_bins
            }

        # Precompute blocks and feedback tokenization once.
        infos = []
        for traj in trajectories:
            traj_content = list(traj['response_ids'])

            header = [IM_START_TOKEN_ID, ASSISTANT_TOKEN_ID, NEWLINE_TOKEN_ID]
            has_eos = (traj_content and traj_content[-1] == IM_END_TOKEN_ID)
            footer = [] if has_eos else [IM_END_TOKEN_ID]
            traj_block = header + traj_content + footer

            content_plus_footer_len = len(traj_content) + len(footer)

            # For strip_context_thinking: also compute stripped (answer-only) version
            if self.strip_context_thinking:
                answer_start = self._get_answer_start_idx(traj_content)
                stripped_content = traj_content[answer_start:]
                stripped_has_eos = (stripped_content and stripped_content[-1] == IM_END_TOKEN_ID)
                stripped_footer = [] if stripped_has_eos else [IM_END_TOKEN_ID]
                stripped_block = header + stripped_content + stripped_footer
                stripped_block_len = len(stripped_block)
                stripped_content_plus_footer_len = len(stripped_content) + len(stripped_footer)
            else:
                stripped_content = traj_content
                stripped_footer = footer
                stripped_block = traj_block
                stripped_block_len = len(traj_block)
                stripped_content_plus_footer_len = content_plus_footer_len

            # Feedback always uses FULL trajectory length (as per user requirement)
            full_feedback_str = f"Reward: {traj['reward']}\nLength: {content_plus_footer_len} tokens"
            length_only_str = f"Length: {content_plus_footer_len} tokens"

            reward_tokens_full = self.tokenizer.encode(full_feedback_str, add_special_tokens=False)
            reward_tokens_len = self.tokenizer.encode(length_only_str, add_special_tokens=False)

            # create_tokenized_message adds 5 tokens around content
            reward_block_len_full = len(reward_tokens_full) + 5
            reward_block_len_len = len(reward_tokens_len) + 5

            infos.append({
                "traj": traj,
                "traj_content": traj_content,
                "header": header,
                "footer": footer,
                "traj_block": traj_block,
                "traj_block_len": len(traj_block),
                "content_plus_footer_len": content_plus_footer_len,
                # Stripped versions for context
                "stripped_content": stripped_content,
                "stripped_footer": stripped_footer,
                "stripped_block": stripped_block,
                "stripped_block_len": stripped_block_len,
                "stripped_content_plus_footer_len": stripped_content_plus_footer_len,
                # Feedback tokens
                "reward_tokens_full": reward_tokens_full,
                "reward_tokens_len": reward_tokens_len,
                "reward_block_len_full": reward_block_len_full,
                "reward_block_len_len": reward_block_len_len,
            })

        n = len(infos)

        # Suffix sums for lengths
        # For strip_context_thinking: context uses stripped lengths, target (last) uses full length
        assistant_suffix = [0] * (n + 1)
        full_fb_suffix = [0] * (n + 1)
        len_fb_suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            if self.strip_context_thinking:
                # Last trajectory uses full length, others use stripped
                if i == n - 1:
                    assistant_suffix[i] = assistant_suffix[i + 1] + infos[i]["traj_block_len"]
                else:
                    assistant_suffix[i] = assistant_suffix[i + 1] + infos[i]["stripped_block_len"]
            else:
                assistant_suffix[i] = assistant_suffix[i + 1] + infos[i]["traj_block_len"]

            full_fb_suffix[i] = full_fb_suffix[i + 1]
            len_fb_suffix[i] = len_fb_suffix[i + 1]
            if i != n - 1:
                full_fb_suffix[i] += infos[i]["reward_block_len_full"]
                len_fb_suffix[i] += infos[i]["reward_block_len_len"]

        # Pick earliest start index of a suffix that fits max_length (keeps supervised trajectory at end).
        start_idx = n - 1
        for s in range(n):
            base = prefix_len + assistant_suffix[s]
            if self.ablation_type == "full" or self.ablation_type == "no_ans":
                total = base + full_fb_suffix[s]
            elif self.ablation_type == "no_ans_no_rewards":
                total = base + len_fb_suffix[s]
            elif self.ablation_type == "no_ans_first_reward_only":
                if s == n - 1:
                    total = base
                else:
                    total = base + infos[s]["reward_block_len_full"] + len_fb_suffix[s + 1]
            else:
                total = base + full_fb_suffix[s]

            if total <= self.max_length:
                start_idx = s
                break

        included = infos[start_idx:]

        # If even the last trajectory alone cannot fit, truncate its content to fit.
        if start_idx == n - 1:
            last = included[0]
            budget = self.max_length - prefix_len
            min_block = len(last["header"]) + len(last["footer"])
            if budget < min_block:
                # No room for assistant content; return prefix only.
                input_tensor = torch.tensor(full_input_ids[:self.max_length], dtype=torch.long)
                return {
                    "input_ids": input_tensor,
                    "label_positions": [],
                    "bin_labels": [],
                    "num_bins": self.num_bins
                }

            max_content_len = budget - min_block
            if max_content_len < 0:
                max_content_len = 0

            trunc_content = list(last["traj_content"][:max_content_len])
            has_eos = (trunc_content and trunc_content[-1] == IM_END_TOKEN_ID)
            footer = [] if has_eos else [IM_END_TOKEN_ID]
            traj_block = last["header"] + trunc_content + footer

            included = [{
                **last,
                "traj_content": trunc_content,
                "footer": footer,
                "traj_block": traj_block,
                "traj_block_len": len(traj_block),
                "content_plus_footer_len": len(trunc_content) + len(footer),
            }]

        # Determine supervision start index
        if self.strip_context_thinking:
            # Only supervise the last trajectory (target)
            supervise_start_idx = len(included) - 1
        else:
            supervise_start_idx = max(0, self.supervise_from_trajectory - 1)
            if len(included) <= supervise_start_idx:
                supervise_start_idx = max(0, len(included) - 1)

        # Build sequence; supervise trajectories from supervise_start_idx..end in included
        for local_idx, info in enumerate(included):
            is_last = (local_idx == len(included) - 1)
            is_supervised = local_idx >= supervise_start_idx

            current_start_idx = len(full_input_ids)

            # For strip_context_thinking: context uses stripped block, target uses full block
            if self.strip_context_thinking and not is_last:
                traj_block = info["stripped_block"]
                block_len = info["stripped_block_len"]
                content_plus_footer_len = info["stripped_content_plus_footer_len"]
            else:
                traj_block = info["traj_block"]
                block_len = info["traj_block_len"]
                content_plus_footer_len = info["content_plus_footer_len"]

            if is_supervised:
                reward_val = float(info["traj"]["reward"])
                if reward_val < 0.0:
                    reward_val = 0.0
                elif reward_val > 1.0:
                    reward_val = 1.0

                content_start_rel_idx = len(info["header"])
                for i in range(content_plus_footer_len):
                    abs_pos = current_start_idx + content_start_rel_idx + i
                    tokens_to_completion = (current_start_idx + block_len) - abs_pos - 1
                    bin_idx = self._get_bin_idx(tokens_to_completion, reward_val)
                    full_label_positions.append(abs_pos)
                    full_bin_labels.append(bin_idx)

            full_input_ids.extend(traj_block)

            # Add reward/length feedback only when there is a next trajectory in-context.
            if local_idx != (len(included) - 1):
                if self.ablation_type == "full" or self.ablation_type == "no_ans":
                    reward_tokens = info["reward_tokens_full"]
                elif self.ablation_type == "no_ans_no_rewards":
                    reward_tokens = info["reward_tokens_len"]
                elif self.ablation_type == "no_ans_first_reward_only":
                    reward_tokens = info["reward_tokens_full"] if local_idx == 0 else info["reward_tokens_len"]
                else:
                    reward_tokens = info["reward_tokens_full"]

                full_input_ids.extend(self.create_tokenized_message("user", reward_tokens))

        input_tensor = torch.tensor(full_input_ids, dtype=torch.long)
        
        return {
            "input_ids": input_tensor,
            "label_positions": full_label_positions,
            "bin_labels": full_bin_labels,
            "num_bins": self.num_bins
        }

    @staticmethod
    def collate_fn(batch):
        max_len = max(s["input_ids"].size(0) for s in batch)
        pad_val = 0 
        
        return {
            "input_ids": torch.stack([F.pad(s["input_ids"], (0, max_len - s["input_ids"].size(0)), value=pad_val) for s in batch]),
            "label_positions": [s["label_positions"] for s in batch],
            "bin_labels": [s["bin_labels"] for s in batch],
            "num_bins": batch[0]["num_bins"]
        }

def compute_loss(model, batch, distribution_token_id, num_bins, num_length_bins):
    """
    Compute cross-entropy loss for joint distribution predictions.
    Also computes and reports marginal losses for Reward and Length (Cost).
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    
    # Forward pass: get hidden states
    outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states[-1]  # [B, S, E]
    
    flat_b, flat_s, flat_labels = [], [], []
    for i, (pos, labels) in enumerate(zip(batch["label_positions"], batch["bin_labels"])):
        flat_b.extend([i] * len(pos))
        flat_s.extend(pos)
        flat_labels.extend(labels)

    if not flat_labels:
        # Return 0 loss and 0 metrics if no labels
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"loss_reward": zero.detach(), "loss_length": zero.detach()}

    b_idx = torch.tensor(flat_b, device=device)
    s_idx = torch.tensor(flat_s, device=device)

    # Unwrap possible torch.compile and/or DDP wrappers for lm_head access
    tgt = model._orig_mod if hasattr(model, "_orig_mod") else model
    tgt = tgt.module if hasattr(tgt, "module") else tgt
    lm_head = tgt.lm_head if hasattr(tgt, "lm_head") else tgt.get_output_embeddings()
    
    w = lm_head.weight[distribution_token_id : distribution_token_id + num_bins]
    b = lm_head.bias[distribution_token_id : distribution_token_id + num_bins] if hasattr(lm_head, "bias") and lm_head.bias is not None else None
    
    # Project all positions to the small joint head to avoid materializing [N_selected, E]
    # (important when supervising many trajectories in one packed sequence).
    logits_all = F.linear(hidden_states, w, b)  # [B, S, num_bins]
    logits = logits_all[b_idx, s_idx]  # [N_selected, num_bins]
    
    # 1. Main Joint Loss (Unchanged)
    target_labels = torch.tensor(flat_labels, device=device)
    main_loss = F.cross_entropy(logits, target_labels)

    # 2. Marginalize for Reporting
    # Layout: [Reward 0 (L0..Lk), Reward 1 (L0..Lk), ...]
    num_reward_states = num_bins // num_length_bins
    
    # Reshape: [N, Num_Reward, Num_Length]
    logits_reshaped = logits.view(-1, num_reward_states, num_length_bins)
    
    # A. Reward Loss: Marginalize over Length (sum probs -> logsumexp over dim 2)
    # P(Reward) = Sum_over_Length(P(Reward, Length))
    logits_reward = torch.logsumexp(logits_reshaped, dim=2) # [N, Num_Reward]
    labels_reward = target_labels // num_length_bins
    loss_reward = F.cross_entropy(logits_reward, labels_reward)

    # B. Length (Cost) Loss: Marginalize over Reward (sum probs -> logsumexp over dim 1)
    # P(Length) = Sum_over_Reward(P(Reward, Length))
    logits_length = torch.logsumexp(logits_reshaped, dim=1) # [N, Num_Length]
    labels_length = target_labels % num_length_bins
    loss_length = F.cross_entropy(logits_length, labels_length)

    return main_loss, {"loss_reward": loss_reward, "loss_length": loss_length}

def print_trainable(model):
    print("\n=== Trainable parameters ===")
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            total += p.numel()
    print(f"TOTAL trainable elements: {total}\n")

def train(model, dataset, distribution_token_id, num_bins, weights_path,
          collate_fn, shuffle_data, seed, dtype, compile_mode, num_epochs, batch_size,
          gradient_accumulation_steps, learning_rate, min_learning_rate, warmup_ratio,
          weight_decay, beta_1, beta_2, grad_clip, wandb_project, dist_backend,
          max_steps=-1, visualization_freq=100):
    
    distributed = int(os.environ.get("RANK", -1)) != -1
    if distributed:
        dist.init_process_group("nccl")
        rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world, master = 1, True

    torch.manual_seed(seed)
    sampler = DistributedSampler(dataset, shuffle=shuffle_data) if distributed else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle_data and sampler is None),
                        sampler=sampler, collate_fn=collate_fn)

    if master and wandb_project:
        wandb.init(project=wandb_project, name=f"run_{int(time.time())}", config={
            "num_epochs": num_epochs, "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate, "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay, "compile_mode": compile_mode,
            "dist_backend": dist_backend, "distribution_token_id": distribution_token_id,
            "num_bins": num_bins, "num_reward_states": dataset.num_reward_states,
            "max_steps": max_steps,
            "num_samples": len(dataset),
            "joint_distribution_bins": f"{dataset.num_length_bins} length × {dataset.num_reward_states} reward",
            "reward_values": dataset.reward_values,
            "ablation_type": dataset.ablation_type,
            "supervise_from_trajectory": dataset.supervise_from_trajectory,
            "no_context": dataset.no_context,
            "strip_context_thinking": dataset.strip_context_thinking,
        })

    model = model.to(device)
    
    if distributed:
        model = DDP(model, device_ids=[device])
    
    if master: print_trainable(model)
    
    if compile_mode in {"default", "reduce-overhead", "max-autotune"}:
        model = torch.compile(model, mode=compile_mode)

    model.train()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], 
                      lr=learning_rate, betas=(beta_1, beta_2), weight_decay=0.0)
    scaler = torch.GradScaler(enabled=(dtype == "float16"))
    
    total_iters = (num_epochs * len(loader)) // gradient_accumulation_steps
    warmup_iters = int(warmup_ratio * total_iters)
    
    def lr_schedule(i):
        if i < warmup_iters:
            return learning_rate * i / warmup_iters
        progress = (i - warmup_iters) / (total_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (learning_rate - min_learning_rate)

    global_step = 0
    accum_loss = 0.0
    accum_reward_loss = 0.0
    accum_length_loss = 0.0
    
    for epoch in range(num_epochs):
        if distributed: sampler.set_epoch(epoch)
        
        for it, batch in enumerate(loader):
            update = (it + 1) % gradient_accumulation_steps == 0
            if distributed and isinstance(model, DDP):
                model.require_backward_grad_sync = update

            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=getattr(torch, dtype)):
                loss, metrics = compute_loss(model, batch, distribution_token_id, 
                                             batch["num_bins"], dataset.num_length_bins)
                loss_scaled = loss / gradient_accumulation_steps
            
            scaler.scale(loss_scaled).backward()
            
            # Aggregate stats for logging
            stats = torch.stack([
                loss.detach(), 
                metrics["loss_reward"].detach(), 
                metrics["loss_length"].detach()
            ])
            
            if distributed:
                dist.all_reduce(stats)
                stats /= world
            
            accum_loss += stats[0].item()
            accum_reward_loss += stats[1].item()
            accum_length_loss += stats[2].item()

            if not update: continue

            scaler.unscale_(optimizer)
            if grad_clip:
                params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                clip_grad_norm_(params, grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            lr = lr_schedule(global_step)
            for g in optimizer.param_groups: g["lr"] = lr
            
            if master and wandb_project:
                wandb.log({
                    "train/loss": accum_loss, 
                    "train/loss_reward": accum_reward_loss,
                    "train/loss_length": accum_length_loss,
                    "lr": lr, 
                    "step": global_step
                })
                accum_loss = 0.0
                accum_reward_loss = 0.0
                accum_length_loss = 0.0
            
            if master and global_step % visualization_freq == 0:
                print(f"\n{'='*80}")
                print(f"JOINT DISTRIBUTION VISUALIZATION - Step {global_step}")
                print(f"{'='*80}")
                
                position_to_probs = visualize_predictions(
                    model, batch, distribution_token_id, num_bins, 
                    dataset.length_bins, device
                )
                
                if not position_to_probs:
                    print("No valid predictions to visualize in this batch.")
                else:
                    for label, (pred_probs, gt_probs) in position_to_probs.items():
                        try:
                            import numpy as np
                            pred_grid = np.array(pred_probs).reshape(dataset.num_reward_states, dataset.num_length_bins)
                            gt_grid = np.array(gt_probs).reshape(dataset.num_reward_states, dataset.num_length_bins)
                            log_joint_distribution_grid(pred_grid, dataset.length_bins, dataset.num_length_bins, dataset.num_reward_states, dataset.reward_values, title_prefix=f"Predicted ({label})")
                            log_joint_distribution_grid(gt_grid, dataset.length_bins, dataset.num_length_bins, dataset.num_reward_states, dataset.reward_values, title_prefix=f"Ground Truth ({label})")
                        except Exception:
                            pass
                print(f"{'='*80}\n")
            
            if max_steps > 0 and global_step >= max_steps:
                break
        
        if max_steps > 0 and global_step >= max_steps:
            break

    if master:
        tgt = model.module if hasattr(model, "module") else model
        if hasattr(tgt, "save_pretrained"):
            tgt.save_pretrained(weights_path)
        else:
            torch.save(tgt.state_dict(), weights_path)
        if wandb_project: wandb.finish()
    
    if distributed: dist.barrier()

def main_worker(local_rank, world_size, cfg):
    os.environ.update(
        WORLD_SIZE=str(world_size), RANK=str(local_rank), LOCAL_RANK=str(local_rank),
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29500"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    # Ensure tokenizer has a pad token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device).gradient_checkpointing_enable()
    model.config.use_cache = False
    
    # Pass tokenizer to dataset
    dataset = JointDistributionDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        thinking_only=cfg.thinking_only,
        thinking_token_id=cfg.thinking_token_id,
        reward_values=(cfg.reward_values if cfg.reward_values is not None else None),
        label_column=cfg.label_column,
        ablation_type=cfg.ablation_type,
        examples_per_prompt=cfg.examples_per_prompt,
        correctness_only=cfg.correctness_only,
        supervise_from_trajectory=cfg.supervise_from_trajectory,
        no_context=cfg.no_context,
        max_prompts=cfg.max_prompts,
        strip_context_thinking=cfg.strip_context_thinking,
        inject_answer=cfg.inject_answer,
    )
    num_bins = dataset.num_bins
    
    if local_rank == 0 and not os.path.exists(cfg.weights_path):
        tokenizer.save_pretrained(cfg.weights_path)
    
    train(model, dataset, cfg.distribution_token_id, num_bins,
          cfg.weights_path, JointDistributionDataset.collate_fn, True, 42, "bfloat16", cfg.compile_mode,
          cfg.num_epochs, cfg.batch_size, cfg.gradient_accumulation_steps, cfg.learning_rate,
          0.0, cfg.warmup_ratio, cfg.weight_decay, 0.9, 0.95, 1.0, cfg.wandb_project,
          cfg.dist_backend, cfg.max_steps, cfg.visualization_freq)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-1.7B")
    p.add_argument("--weights_path", default="models/joint_critic")
    p.add_argument("--data_path", default="data/zip_training_adaptivemath_data_qwen17b_thinking.parquet")
    
    # Token ID for distribution prediction
    p.add_argument("--distribution_token_id", type=int, default=151669, 
                   help="Starting token ID for joint distribution prediction P(reward, tokens_remaining)")
    p.add_argument("--thinking_only", action="store_true", 
                   help="Only train on samples that contain thinking token")
    p.add_argument("--thinking_token_id", type=int, default=151667,
                   help="Token ID for detecting thinking/reasoning samples")
    p.add_argument("--reward_values", type=float, nargs="+", default=None,
                   help="Midpoints for reward bins. If omitted: [0,1] for correctness; 7 bins for value.")
    p.add_argument("--label-column", choices=["auto","correct","value"], default="correct",
                   help="Which column to use for reward supervision. 'auto' prefers 'correct' if present.")
    
    # Ablation
    # Removed unused ablations, kept strict list as requested
    p.add_argument("--ablation_type", 
                   choices=["full", "no_ans", "no_ans_no_rewards", "no_ans_first_reward_only"], 
                   default="no_ans", 
                   help="Controls in-context feedback configuration. Note: the ground-truth answer is never added to the prompt; 'full' is kept as an alias of 'no_ans'.")

    # Dataset packing / supervision
    p.add_argument("--examples_per_prompt", type=int, default=1,
                   help="Number of packed examples per prompt (reshuffle trajectories each time).")
    p.add_argument("--supervise_from_trajectory", type=int, default=4,
                   help="1-indexed trajectory number to start supervision (e.g., 4 => supervise trajectories 4..end; first 3 are context-only).")

    # Output space
    p.add_argument("--correctness_only", action="store_true",
                   help="If set, predict correctness only (1 length bin).")

    # No-context baseline mode
    p.add_argument("--no_context", action="store_true",
                   help="If set, train without in-context conditioning. Each trajectory becomes an independent example.")

    # Inject answer into prompt (requires --no_context)
    p.add_argument("--inject_answer", action="store_true",
                   help="If set (with --no_context), inject ground truth answer into the prompt. "
                        "Format: prompt + '\\n\\nThe correct answer is {answer}'")

    # Strip thinking from context mode (for reasoning models)
    p.add_argument("--strip_context_thinking", action="store_true",
                   help="If set, strip thinking tokens from context trajectories (keep only answer after </think>). "
                        "Only the target trajectory (last one) keeps full thinking trace and is supervised.")

    # Data subset selection
    p.add_argument("--max_prompts", type=int, default=None,
                   help="If set, limit training to this many unique prompts (all rollouts per prompt are kept).")

    # Training parameters
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=131_072)
    p.add_argument("--max_steps", type=int, default=-1, help="If >0, stop training after this many optimizer steps")
    p.add_argument("--visualization_freq", type=int, default=100, help="Frequency (in steps) to display prediction visualizations")
    
    # Model configuration
    p.add_argument("--compile_mode", default="none", choices=["none", "default", "reduce-overhead", "max-autotune"])
    p.add_argument("--wandb_project", default="new_joint_critic")
    p.add_argument("--dist-backend", choices=["ddp"], default="ddp", help="Distributed training backend")
    
    return p.parse_args()

def main():
    cfg = parse_args()
    ngpus = torch.cuda.device_count() or 1
    mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg))

if __name__ == "__main__":
    main()
