"""
Teacher-Distillation: A teacher policy trained with DRO on privileged informatio, then distilled to a student policy.

Training loop:
  1. Student generates N rollouts per prompt.
  2. Environment scores each rollout.
  3. Teacher sees privileged information and computes log-probs on the student's rollout tokens.
  4. Teacher is trained via DRO to maximize environment rewards with regularization to student.
  5. Student token reward signal = (teacher_log_probs - student_log_probs).
  6. Student is trained with PPO on teacher-derived advantages.

Usage:
    uv run --isolated --extra vllm -m examples.teacher_distillation.main_teacher_distillation \\
        trainer.critic.model.path="Qwen/Qwen3-1.7B" ...

See run_teacher_distillation_deepscaler.sh for a full example.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from loguru import logger
from omegaconf import DictConfig

import hydra

from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.generators.base import GeneratorInterface
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.utils import initialize_ray, Timer, validate_cfg
from skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    compute_grpo_outcome_advantage,
    masked_mean,
    reduce_loss,
    register_policy_loss,
)


@register_policy_loss("dro")
def dro_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    """
    DRO (Direct Reward Optimization) loss for teacher training.

    Args:
        log_probs:     Teacher's current log-probs (has gradient).
        old_log_probs: Student's log-probs (sampling distribution, no gradient).
        advantages:    GRPO-normalized environment rewards.
        config:        Algorithm config — reads ``config.teacher_distillation.dro_beta``.
        loss_mask:     Token-level mask.
    """
    td_cfg = config.teacher_distillation
    beta = td_cfg.dro_beta

    # Quadratic penalty: keeps teacher close to student distribution
    quadratic_term = (log_probs - old_log_probs) ** 2

    # DRO objective: maximize expected reward subject to quadratic constraint
    dro_objective = log_probs * advantages - 0.5 * beta * quadratic_term

    # Loss = negative objective (we minimize)
    loss = -dro_objective

    loss = reduce_loss(loss, loss_mask, config.loss_reduction, config.max_seq_len)
    return loss, 0.0  # no clip ratio for DRO


def compute_student_distillation_gae(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lambd: float,
) -> torch.Tensor:
    """
    GAE-style smoothing using distillation rewards.
    """
    with torch.no_grad():
        lastgaelam = torch.zeros(token_level_rewards.shape[0], device=token_level_rewards.device)
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            reward_t = token_level_rewards[:, t] * response_mask[:, t]
            lastgaelam = reward_t + gamma * lambd * lastgaelam
            lastgaelam = lastgaelam * response_mask[:, t]
            advantages_reversed.append(lastgaelam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        return advantages * response_mask


class SharedTeacherStudentWorker:
    """
    A single shared policy model/optimizer that supports two update modes:
      - student: standard PPO policy update
      - teacher: DRO update on teacher-conditioned inputs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        td_cfg = getattr(self.cfg.trainer.algorithm, "teacher_distillation", None) or {}
        teacher_loss_type = getattr(td_cfg, "teacher_loss_type", "dro")
        self.teacher_policy_loss_fn = PolicyLossRegistry.get(teacher_loss_type)
        self.teacher_loss_alpha = float(getattr(td_cfg, "teacher_loss_alpha", 1.0))
        self.training_mode = "student"

    def set_training_mode(self, mode: str):
        if mode not in ("student", "teacher"):
            raise ValueError(f"Unknown training_mode: {mode}")
        self.training_mode = mode

    def forward_backward(self, experience, accumulation_steps) -> Dict[str, float]:
        if self.training_mode != "teacher":
            return super().forward_backward(experience, accumulation_steps)

        # Teacher-mode update on the same policy parameters.
        self.model.train()
        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        old_action_log_probs = experience.action_log_probs  # detached student log-probs on rollout inputs
        advantages = experience.advantages  # teacher GRPO advantages
        num_actions = experience.num_actions
        attention_mask = experience.attention_mask
        loss_mask = experience.loss_mask

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            action_log_probs, output = self.model(
                sequences,
                num_actions,
                attention_mask=attention_mask,
                temperature=self.cfg.generator.sampling_params.temperature,
                return_output=True,
                compute_entropy=True,
                entropy_requires_grad=False,
            )
            teacher_loss, _ = self.teacher_policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
            )
            weighted_teacher_loss = self.teacher_loss_alpha * teacher_loss

        with torch.no_grad():
            entropy_BS = output["entropy"]
            entropy_BS = entropy_BS[:, -num_actions - 1 : -1]
            entropy = masked_mean(entropy_BS, loss_mask)

        loss = weighted_teacher_loss / accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)

        return {
            "teacher_loss": teacher_loss.item(),
            "teacher_loss_alpha": self.teacher_loss_alpha,
            "teacher_weighted_loss": weighted_teacher_loss.item(),
        }


def format_teacher_context(
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
    """
    Format teacher input with ICL context:

        [prompt]
        The following is what the student generated and its achieved reward:
        [student's full rollout]
        [Reward: student's achieved reward]
        Now, generate your own completion that fixes any mistakes in the student's.
        [current trajectory prefix]
    """
    from .teacher_distillation_utils import format_teacher_icl_context

    return format_teacher_icl_context(
        sequences=sequences,
        attention_masks=attention_masks,
        response_masks=response_masks,
        rewards=rewards,
        index=index,
        tokenizer=tokenizer,
        config=config,
        pad_token_id=pad_token_id,
        response_length=response_length,
    )


class TeacherDistillationTrainer(RayPPOTrainer):
    """
    Instead of the value-function critic, we use a trainable teacher policy that sees
    privileged information (on-policy rollouts + rewards).
    """

    @torch.no_grad()
    def fwd_logprobs_values_reward(self, training_input: TrainingInputBatch) -> TrainingInputBatch:
        """
        Compute log-probs from student and teacher models.

        Adds to ``training_input``:
          - ``action_log_probs``: student policy log-probs
          - ``base_action_log_probs``: student policy log-probs snapshot (used as PPO base policy)
          - ``values``: None (no value function)
          - ``teacher_action_log_probs``: teacher log-probs with privileged context
        """
        data_fwd_pass = training_input.select(
            keys=["sequences", "attention_mask"], metadata_keys=["response_length"]
        )

        def collect_results(actor_infos, results, key):
            ret_outputs: TrainingOutputBatch = concatenate_outputs_after_mesh_dispatch(actor_infos, results)
            return ret_outputs[key]

        teacher_log_probs = None
        action_log_probs = None
        teacher_chunk_batch_size = self._get_teacher_chunk_size()

        # Teacher forward is always distinguished by privileged context formatting.
        formatted_sequences, formatted_attention_masks = format_teacher_context(
            sequences=training_input["sequences"],
            attention_masks=training_input["attention_mask"],
            response_masks=training_input["response_mask"],
            rewards=training_input["rewards"],
            index=np.array(training_input.metadata["uids"]),
            tokenizer=self.tokenizer,
            config=self.cfg.trainer.algorithm,
            pad_token_id=self.tokenizer.pad_token_id,
            response_length=training_input.metadata["response_length"],
        )
        logger.info(
            "Teacher context formatted: student_seq_len={}, teacher_seq_len={}",
            training_input["sequences"].shape[1],
            formatted_sequences.shape[1],
        )
        teacher_fwd_pass = TrainingInputBatch({
            "sequences": formatted_sequences,
            "attention_mask": formatted_attention_masks,
        })
        teacher_fwd_pass.metadata = {"response_length": training_input.metadata["response_length"]}

        # Store formatted sequences for teacher training later (move to CPU to save GPU mem)
        training_input.metadata["teacher_formatted_sequences"] = formatted_sequences.cpu()
        training_input.metadata["teacher_formatted_attention_mask"] = formatted_attention_masks.cpu()
        training_input.metadata["teacher_context_applied"] = True

        # forward the shared student policy on rollout inputs and teacher-conditioned inputs
        if self.colocate_all:
            self.policy_model.backload_to_gpu(backload_optimizer=False, backload_model=True)

        action_log_probs_refs = self.policy_model.async_run_ray_method("mesh", "forward", data=data_fwd_pass)

        all_rank_action_log_probs: List[TrainingOutputBatch] = ray.get(action_log_probs_refs)
        action_log_probs = collect_results(self.policy_model.actor_infos, all_rank_action_log_probs, key="output")

        if teacher_chunk_batch_size is not None and teacher_chunk_batch_size < len(teacher_fwd_pass):
            teacher_outputs = []
            num_chunks = (len(teacher_fwd_pass) + teacher_chunk_batch_size - 1) // teacher_chunk_batch_size
            logger.info(
                "Teacher forward chunking enabled: batch_size={}, chunk_size={}, num_chunks={}",
                len(teacher_fwd_pass),
                teacher_chunk_batch_size,
                num_chunks,
            )
            for teacher_chunk in teacher_fwd_pass.chunk(teacher_chunk_batch_size):
                teacher_refs = self.policy_model.async_run_ray_method("mesh", "forward", data=teacher_chunk)
                all_rank_teacher = ray.get(teacher_refs)
                teacher_outputs.append(
                    collect_results(self.policy_model.actor_infos, all_rank_teacher, key="output")
                )
                if not self.colocate_all:
                    empty_cache_refs = self.policy_model.async_run_ray_method("pass_through", "empty_cache")
                    ray.get(empty_cache_refs)
            teacher_log_probs = torch.cat(teacher_outputs, dim=0) if teacher_outputs else None
        else:
            teacher_refs = self.policy_model.async_run_ray_method("mesh", "forward", data=teacher_fwd_pass)
            all_rank_teacher = ray.get(teacher_refs)
            teacher_log_probs = collect_results(self.policy_model.actor_infos, all_rank_teacher, key="output")

        if self.colocate_all:
            self.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)

        if not self.colocate_all:
            empty_cache_refs = self.policy_model.async_run_ray_method("pass_through", "empty_cache")
            ray.get(empty_cache_refs)

        sequences_all = training_input["sequences"]
        action_log_probs = action_log_probs[: len(sequences_all)]
        teacher_log_probs = teacher_log_probs[: len(sequences_all)] if teacher_log_probs is not None else None

        # Ref policy is replaced by student policy snapshot on the rollout inputs.
        training_input["base_action_log_probs"] = action_log_probs
        training_input["action_log_probs"] = action_log_probs
        training_input["values"] = None  # No value function in teacher-distillation

        # Store teacher log-probs as a proper tensor field
        if teacher_log_probs is not None:
            training_input["teacher_action_log_probs"] = teacher_log_probs

        return training_input

    def _get_teacher_chunk_size(self) -> Optional[int]:
        """Resolve a mesh-safe teacher chunk size from config."""
        td_cfg = getattr(self.cfg.trainer.algorithm, "teacher_distillation", None) or {}
        teacher_chunk_batch_size = getattr(td_cfg, "teacher_chunk_batch_size", None)
        if teacher_chunk_batch_size is None:
            return None

        teacher_chunk_batch_size = int(teacher_chunk_batch_size)
        if teacher_chunk_batch_size <= 0:
            return None

        dp_size = self.policy_model.actor_infos[0].rank.dp_size
        # Mesh dispatch requires each data shard batch to be divisible by dp_size.
        # Align chunk size so each per-call teacher step satisfies this contract.
        if teacher_chunk_batch_size < dp_size:
            teacher_chunk_batch_size = dp_size
        teacher_chunk_batch_size = (teacher_chunk_batch_size // dp_size) * dp_size
        return teacher_chunk_batch_size

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """
        Compute advantages for both teacher and student.
        """
        token_level_rewards = data["rewards"]
        response_mask = data["response_mask"]
        uids = data.metadata["uids"]

        # compute teacher advantages from environment rewards (always GRPO for teacher training)
        teacher_advantages, teacher_returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=np.array(uids),
            grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
        )

        # compute student advantages from teacher-student distillation signal
        td_cfg = getattr(self.cfg.trainer.algorithm, "teacher_distillation", None) or {}
        teacher_log_probs = data.get("teacher_action_log_probs", None)
        student_log_probs = data.get("action_log_probs", None)

        if teacher_log_probs is None:
            # no teacher: student uses env rewards (same as teacher advantages)
            student_advantages = teacher_advantages
        else:
            if student_log_probs is None:
                raise ValueError(
                    "Teacher distillation student advantages require `action_log_probs` (student policy), "
                    "but student log-probs are missing."
                )
            distill_token_rewards = (teacher_log_probs - student_log_probs) * response_mask
            # Optional GAE smoothing over (teacher-student); lambda=0 is exact immediate token signal.
            gamma = getattr(td_cfg, "student_gae_gamma", getattr(self.cfg.trainer.algorithm, "gamma", 1.0))
            lambd = getattr(td_cfg, "student_gae_lambda", 0.0)
            student_advantages = compute_student_distillation_gae(
                token_level_rewards=distill_token_rewards,
                response_mask=response_mask,
                gamma=gamma,
                lambd=lambd,
            )

        data["advantages"] = student_advantages
        data["returns"] = student_advantages  # no value-function returns

        # Store teacher advantages for teacher training
        data["teacher_advantages"] = teacher_advantages
        data["teacher_returns"] = teacher_returns

        pad_size = data.metadata.get("pad_size", 0)
        num_samples = len(token_level_rewards)

        return_sums = token_level_rewards.sum(dim=-1)[: num_samples - pad_size]
        avg_rewards: float = return_sums.mean().item()
        avg_response_length = data.metadata["avg_response_length"]

        # Move to CPU
        data = data.to("cpu")

        # Also move teacher formatted sequences to CPU
        if "teacher_formatted_sequences" in data.metadata:
            if isinstance(data.metadata["teacher_formatted_sequences"], torch.Tensor):
                data.metadata["teacher_formatted_sequences"] = data.metadata["teacher_formatted_sequences"].cpu()
            if isinstance(data.metadata["teacher_formatted_attention_mask"], torch.Tensor):
                data.metadata["teacher_formatted_attention_mask"] = data.metadata[
                    "teacher_formatted_attention_mask"
                ].cpu()

        valid_advantages = torch.masked_select(
            data["advantages"][: num_samples - pad_size],
            data["response_mask"][: num_samples - pad_size].bool(),
        )
        avg_advantages: float = valid_advantages.mean().item()
        avg_advantages_abs: float = valid_advantages.abs().mean().item()

        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}
        data.metadata["metrics"].update({
            "avg_final_rewards": avg_rewards,
            "avg_response_length": avg_response_length,
            "avg_advantages": avg_advantages,
            "avg_advantages_abs": avg_advantages_abs,
        })

        logger.info(f"avg_final_rewards: {avg_rewards}, avg_response_length: {avg_response_length}")
        self.all_metrics.update({
            "loss/avg_final_rewards": avg_rewards,
            "loss/avg_raw_advantages": avg_advantages,
            "loss/avg_raw_advantages_abs": avg_advantages_abs,
        })

        # Log teacher-specific metrics
        valid_teacher_adv = torch.masked_select(
            data["teacher_advantages"][: num_samples - pad_size],
            data["response_mask"][: num_samples - pad_size].bool(),
        )
        self.all_metrics.update({
            "teacher/avg_advantages": valid_teacher_adv.mean().item(),
            "teacher/avg_advantages_abs": valid_teacher_adv.abs().mean().item(),
        })
        if teacher_log_probs is not None:
            teacher_lp_cpu = data["teacher_action_log_probs"]
            student_lp_cpu = data.get("action_log_probs", None)
            if teacher_lp_cpu is not None and student_lp_cpu is not None:
                # Original DRO loss (token-level):
                beta = float(getattr(self.cfg.trainer.algorithm.teacher_distillation, "dro_beta", 1.0))
                valid_mask = data["response_mask"][: num_samples - pad_size]
                dro_sq_err = (
                    data["teacher_advantages"][: num_samples - pad_size]
                    - beta * (teacher_lp_cpu[: num_samples - pad_size] - student_lp_cpu[: num_samples - pad_size])
                ) ** 2
                dro_loss = masked_mean(dro_sq_err, valid_mask).item()
                self.all_metrics.update({
                    "teacher/dro_loss": dro_loss,
                })

        return data

    def train_teacher_and_student(self, data: TrainingInputBatch):
        """Train teacher and student objectives sequentially on one shared policy model."""
        data.metadata["global_step"] = self.global_step

        teacher_data = self._build_teacher_training_batch(data)
        student_data = self._build_student_training_batch(data)

        if self.colocate_all:
            self.policy_model.backload_to_gpu()

        with Timer("teacher_train", self.all_timings):
            ray.get(self.policy_model.async_run_ray_method("pass_through", "set_training_mode", "teacher"))
            teacher_chunk_size = self._get_teacher_chunk_size()
            if teacher_chunk_size is not None and teacher_chunk_size < len(teacher_data):
                teacher_status_chunks: List[Dict[str, float]] = []
                for teacher_chunk in teacher_data.chunk(teacher_chunk_size):
                    chunk_statuses = ray.get(self.policy_model.async_run_ray_method("mesh", "ppo_train", teacher_chunk))
                    teacher_status_chunks.append(chunk_statuses[0].metadata["train_status"])
                teacher_status = {
                    k: float(np.mean([status[k] for status in teacher_status_chunks]))
                    for k in teacher_status_chunks[0]
                }
            else:
                teacher_statuses = ray.get(self.policy_model.async_run_ray_method("mesh", "ppo_train", teacher_data))
                teacher_status = teacher_statuses[0].metadata["train_status"]

        with Timer("policy_train", self.all_timings):
            ray.get(self.policy_model.async_run_ray_method("pass_through", "set_training_mode", "student"))
            policy_statuses = ray.get(self.policy_model.async_run_ray_method("mesh", "ppo_train", student_data))

        empty_cache_refs = []
        for k, v in teacher_status.items():
            self.all_metrics.update({f"teacher/{k}": v})

        # policy/ = student policy only (from ppo_train(student_data) with mode "student")
        policy_status = policy_statuses[0].metadata["train_status"]
        for k, v in policy_status.items():
            self.all_metrics.update({f"policy/{k}": v})
        empty_cache_refs += self.policy_model.async_run_ray_method("pass_through", "empty_cache")
        ray.get(empty_cache_refs)

        return policy_status

    def train_critic_and_policy(self, data: TrainingInputBatch):
        """
        Compatibility override for RayPPOTrainer's call site.
        This trainer does not use a critic; delegate to train_teacher_and_student.
        """
        return self.train_teacher_and_student(data)

    def _build_teacher_training_batch(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """
        Build a TrainingInputBatch for teacher training with:
        - Context-formatted sequences (privileged info)
        - Student's log-probs (for DRO quadratic penalty)
        - environment rewards
        """
        teacher_sequences = data.metadata.get("teacher_formatted_sequences")
        teacher_attention_mask = data.metadata.get("teacher_formatted_attention_mask")
        teacher_context_applied = data.metadata.get("teacher_context_applied", False)

        if teacher_sequences is None or teacher_attention_mask is None or not teacher_context_applied:
            raise RuntimeError(
                "Teacher training requires context-formatted teacher inputs, but context metadata was not found."
            )

        teacher_batch = TrainingInputBatch({
            "sequences": teacher_sequences,
            "attention_mask": teacher_attention_mask,
            # KL term compares two passes from same shared policy:
            # teacher pass uses context-formatted inputs; this ref pass uses normal student inputs.
            "action_log_probs": data["action_log_probs"],
            "base_action_log_probs": None,                       # teacher has no ref model
            "values": None,                                      # no value function
            "returns": data["teacher_returns"],                  # GRPO returns
            "advantages": data["teacher_advantages"],            # GRPO advantages
            "response_mask": data["response_mask"],
            "loss_mask": data["loss_mask"],
            "rollout_logprobs": None,
        })
        teacher_batch.metadata = {
            "response_length": data.metadata["response_length"],
            "global_step": data.metadata.get("global_step", 0),
        }
        return teacher_batch

    def _build_student_training_batch(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """
        Build a TrainingInputBatch for student policy training with:
        - Original sequences (no privileged context)
        - Teacher-distillation advantages (default: teacher_log_probs - student_log_probs)
        """
        student_batch = TrainingInputBatch({
            "sequences": data["sequences"],
            "attention_mask": data["attention_mask"],
            "action_log_probs": data["action_log_probs"],
            "base_action_log_probs": data["base_action_log_probs"],
            "values": None,
            "returns": data["returns"],
            "advantages": data["advantages"],
            "response_mask": data["response_mask"],
            "loss_mask": data["loss_mask"],
            "rollout_logprobs": data.get("rollout_logprobs", None),
        })
        student_batch.metadata = {
            "response_length": data.metadata["response_length"],
            "global_step": data.metadata.get("global_step", 0),
        }
        if "is_last_step" in data:
            student_batch["is_last_step"] = data["is_last_step"]
        return student_batch


class TeacherDistillationExp(BasePPOExp):
    """Experiment class that wires up the teacher worker and custom trainer."""

    def get_trainer(self, *args, **kwargs):
        return TeacherDistillationTrainer(*args, **kwargs)

    def _setup_trainer(self):
        """Override to run teacher+student updates on one shared policy worker."""
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        # Import strategy-specific workers for policy and ref
        if self.cfg.trainer.strategy == "deepspeed":
            from skyrl_train.workers.deepspeed.deepspeed_worker import DeepSpeedPolicyWorkerBase, RefWorker
            PolicyWorkerBaseImpl = DeepSpeedPolicyWorkerBase
        elif self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase, RefWorker
            PolicyWorkerBaseImpl = FSDPPolicyWorkerBase
        elif self.cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase, RefWorker
            PolicyWorkerBaseImpl = MegatronPolicyWorkerBase
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")

        class SharedPolicyWorkerBase(SharedTeacherStudentWorker, PolicyWorkerBaseImpl):
            pass
        SharedPolicyWorker = ray.remote(num_gpus=1)(SharedPolicyWorkerBase)

        # Disable separate teacher/critic model; teacher objective runs on policy_model in teacher mode.
        self.cfg.trainer.critic.model.path = None
        # Reference policy is replaced by student policy snapshot in this setup.
        self.cfg.trainer.algorithm.use_kl_loss = False

        tracker = self.get_tracker()
        tokenizer = self.tokenizer

        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(
                self.cfg, self.colocate_pg, tokenizer
            )
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)
        generator: GeneratorInterface = self.get_generator(self.cfg, tokenizer, inference_engine_client)

        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )

        # Build models: use one shared policy worker for both teacher and student phases.
        trainer.build_models(SharedPolicyWorker, SharedPolicyWorker, RefWorker)
        return trainer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = TeacherDistillationExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
