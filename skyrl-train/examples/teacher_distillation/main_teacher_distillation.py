"""
Teacher-Distillation: A teacher policy trained with DRO on privileged information
(on-policy rollouts + rewards), distilled to a student policy.

Training loop:
  1. Student generates N rollouts per prompt.
  2. Environment scores each rollout.
  3. Teacher sees privileged information and computes log-probs on the student's rollout tokens.
  4. Teacher is trained via DRO to maximize environment rewards with regularization to student.
  5. Student advantages = teacher_log_probs.
  6. Student is trained on these advantages.

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
import torch.distributed
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
from skyrl_train.workers.worker import PolicyWorkerBase


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


class TeacherWorkerBase(PolicyWorkerBase):
    """
    Teacher worker: another policy model but trained with DRO loss.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override the loss function with the teacher's DRO loss
        td_cfg = getattr(self.cfg.trainer.algorithm, "teacher_distillation", None)
        teacher_loss_type = getattr(td_cfg, "teacher_loss_type", "dro") if td_cfg else "dro"
        self.policy_loss_fn = PolicyLossRegistry.get(teacher_loss_type)

    def _normalize_mini_batch_size(self):
        """Use critic mini-batch size since teacher occupies the critic model slot."""
        if not hasattr(self, "mesh_rank") or self.mesh_rank is None:
            raise RuntimeError("mesh_rank must be initialized before calling _normalize_mini_batch_size()")
        dp_size = self.mesh_rank.dp_size
        # occupies critic model slot, so use critic mini-batch size
        self.policy_mini_batch_size_per_gpu = (
            self.cfg.trainer.critic_mini_batch_size * self.cfg.generator.n_samples_per_prompt // dp_size
        )

    def forward_backward(self, experience, accumulation_steps) -> Dict[str, float]:
        """Teacher forward-backward with DRO loss."""
        self.model.train()
        experience.to_device(torch.cuda.current_device())

        sequences = experience.sequences
        old_action_log_probs = experience.action_log_probs  # student's log-probs
        advantages = experience.advantages  # environment rewards
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
            teacher_loss, _ = self.policy_loss_fn(
                action_log_probs,
                old_action_log_probs,
                advantages,
                config=self.cfg.trainer.algorithm,
                loss_mask=loss_mask,
            )

        # Entropy for logging only (no gradient)
        with torch.no_grad():
            entropy_BS = output["entropy"]
            entropy_BS = entropy_BS[:, -num_actions - 1 : -1]
            entropy = masked_mean(entropy_BS, loss_mask)

        loss = teacher_loss / accumulation_steps
        self.strategy.backward(loss, self.model, self.optimizer)

        return {
            # Teacher-specific metrics
            "teacher_loss": teacher_loss.item(),
            "teacher_entropy": entropy.item(),
            # Compatibility keys for ppo_train progress bar
            "policy_loss": teacher_loss.item(),
            "policy_entropy": entropy.item(),
            "response_length": num_actions,
        }


class FSDPTeacherWorkerBase(TeacherWorkerBase):
    """FSDP-specific teacher worker. Uses critic config for optimizer/FSDP settings."""

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        from transformers import AutoConfig

        from skyrl_train.distributed.fsdp_strategy import FSDPStrategy
        from skyrl_train.distributed.fsdp_utils import get_init_weight_context_manager
        from skyrl_train.model_wrapper import HFModelWrapper
        from skyrl_train.utils.trainer_utils import get_rope_scaling_config, get_rope_theta_config

        assert self.cfg.trainer.strategy in ("fsdp", "fsdp2")

        # Use *critic* config for optimizer, FSDP, and model settings
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.trainer.critic.fsdp_config,
            optimizer_config=self.cfg.trainer.critic.optimizer_config,
            fsdp_strategy=self.cfg.trainer.strategy,
            seed=self.cfg.trainer.seed,
            micro_train_batch_size_per_gpu=self.cfg.trainer.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy
        self._is_lora = self.cfg.trainer.critic.model.lora.rank > 0

        self._normalize_mini_batch_size()

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        with init_context():
            # Policy architecture (HFModelWrapper) — no value head
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.trainer.flash_attn,
                bf16=False,  # fp32 for training initialization
                lora_rank=self.cfg.trainer.critic.model.lora.rank,
                lora_alpha=self.cfg.trainer.critic.model.lora.alpha,
                lora_dropout=self.cfg.trainer.critic.model.lora.dropout,
                target_modules=self.cfg.trainer.critic.model.lora.target_modules,
                exclude_modules=self.cfg.trainer.critic.model.lora.exclude_modules,
                sequence_parallel_size=self.cfg.trainer.critic.sequence_parallel_size,
                use_sample_packing=self.cfg.trainer.use_sample_packing,
                rope_scaling=get_rope_scaling_config(self.cfg.trainer),
                rope_theta=get_rope_theta_config(self.cfg.trainer),
            )
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

            if self.cfg.trainer.gradient_checkpointing:
                wrapped_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": self.cfg.trainer.gradient_checkpointing_use_reentrant
                    }
                )

        self.model, self.optimizer, self.scheduler = strategy.prepare((wrapped_model, None, None))
        assert (
            self.optimizer is not None and self.scheduler is not None
        ), "FSDP preparation should create optimizer and scheduler for teacher"

    def forward(self, data: TrainingInputBatch) -> TrainingOutputBatch:
        from skyrl_train.distributed.fsdp_utils import fsdp_version

        output = super().forward(data)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


# Ray remote actor
TeacherWorker = ray.remote(num_gpus=1)(FSDPTeacherWorkerBase)


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
        [Reward: r_1]  ← lowest reward first (ascending)
        [trajectory_1]
        [Reward: r_2]
        [trajectory_2]
        ...
        [Reward: 1.0]  ← success conditioning
        [current trajectory prefix]
    """
    from skyrl_train.examples.teacher_distillation.teacher_distillation_utils import format_teacher_icl_context

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
        Compute log-probs from policy, ref, and teacher models.

        Adds to ``training_input``:
          - ``action_log_probs``: student policy log-probs
          - ``base_action_log_probs``: ref model log-probs (if ref model exists)
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
        base_log_probs = None
        action_log_probs = None

        # add privileged information to the teacher's context
        teacher_fwd_pass = data_fwd_pass
        if self.critic_model is not None:
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
            teacher_fwd_pass = TrainingInputBatch({
                "sequences": formatted_sequences,
                "attention_mask": formatted_attention_masks,
            })
            teacher_fwd_pass.metadata = {"response_length": training_input.metadata["response_length"]}

            # Store formatted sequences for teacher training later (move to CPU to save GPU mem)
            training_input.metadata["teacher_formatted_sequences"] = formatted_sequences.cpu()
            training_input.metadata["teacher_formatted_attention_mask"] = formatted_attention_masks.cpu()

        # forward the teacher model
        if self.colocate_all and self.critic_model is not None:
            self.critic_model.backload_to_gpu(backload_optimizer=False, backload_model=True)

        if self.critic_model is not None:
            teacher_refs = self.critic_model.async_run_ray_method("mesh", "forward", data=teacher_fwd_pass)
            if self.colocate_all:
                all_rank_teacher = ray.get(teacher_refs)
                teacher_log_probs = collect_results(self.critic_model.actor_infos, all_rank_teacher, key="output")
                self.critic_model.offload_to_cpu(offload_optimizer=False, offload_model=True)

        # forward the ref model (same as base)
        if self.ref_model is not None:
            if self.cfg.trainer.placement.colocate_policy_ref or self.colocate_all:
                self.ref_model.backload_to_gpu()
            base_action_log_probs_refs = self.ref_model.async_run_ray_method("mesh", "forward", data=data_fwd_pass)

        if self.ref_model is not None:
            if self.cfg.trainer.placement.colocate_policy_ref or self.colocate_all:
                all_rank_base_log_probs: List[TrainingOutputBatch] = ray.get(base_action_log_probs_refs)
                base_log_probs = collect_results(self.ref_model.actor_infos, all_rank_base_log_probs, key="output")
                self.ref_model.offload_to_cpu()
                ray.get(self.ref_model.async_run_ray_method("pass_through", "empty_cache"))
        else:
            base_log_probs = None

        # forward the student model (same as base)
        if self.colocate_all:
            self.policy_model.backload_to_gpu(backload_optimizer=False, backload_model=True)

        action_log_probs_refs = self.policy_model.async_run_ray_method("mesh", "forward", data=data_fwd_pass)
        if self.colocate_all:
            all_rank_action_log_probs: List[TrainingOutputBatch] = ray.get(action_log_probs_refs)
            action_log_probs = collect_results(
                self.policy_model.actor_infos, all_rank_action_log_probs, key="output"
            )
            self.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)

        if not self.colocate_all:
            if not self.cfg.trainer.placement.colocate_policy_ref:
                if self.critic_model is not None:
                    all_rank_teacher = ray.get(teacher_refs)
                    teacher_log_probs = collect_results(
                        self.critic_model.actor_infos, all_rank_teacher, key="output"
                    )
                if self.ref_model is not None:
                    all_rank_base_log_probs = ray.get(base_action_log_probs_refs)
                    base_log_probs = collect_results(
                        self.ref_model.actor_infos, all_rank_base_log_probs, key="output"
                    )
                else:
                    base_log_probs = None
            elif self.critic_model is not None:
                all_rank_teacher = ray.get(teacher_refs)
                teacher_log_probs = collect_results(
                    self.critic_model.actor_infos, all_rank_teacher, key="output"
                )

            all_rank_action_log_probs = ray.get(action_log_probs_refs)
            action_log_probs = collect_results(
                self.policy_model.actor_infos, all_rank_action_log_probs, key="output"
            )

        if not self.colocate_all:
            empty_cache_refs = self.policy_model.async_run_ray_method("pass_through", "empty_cache")
            if self.ref_model is not None:
                empty_cache_refs.extend(self.ref_model.async_run_ray_method("pass_through", "empty_cache"))
            if self.critic_model is not None:
                empty_cache_refs.extend(self.critic_model.async_run_ray_method("pass_through", "empty_cache"))
            ray.get(empty_cache_refs)

        sequences_all = training_input["sequences"]
        base_log_probs = base_log_probs[: len(sequences_all)] if base_log_probs is not None else None
        action_log_probs = action_log_probs[: len(sequences_all)]
        teacher_log_probs = teacher_log_probs[: len(sequences_all)] if teacher_log_probs is not None else None

        training_input["base_action_log_probs"] = base_log_probs
        training_input["action_log_probs"] = action_log_probs
        training_input["values"] = None  # No value function in teacher-distillation

        # Store teacher log-probs as a proper tensor field
        if teacher_log_probs is not None:
            training_input["teacher_action_log_probs"] = teacher_log_probs

        return training_input

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """
        Compute advantages for both teacher and student.
        """
        token_level_rewards = data["rewards"]
        response_mask = data["response_mask"]
        uids = data.metadata["uids"]

        # compute teacher advantages from environment rewards
        teacher_advantages, teacher_returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=np.array(uids),
            grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
        )

        # compute student advantages from teacher log-probs
        teacher_log_probs = data.get("teacher_action_log_probs", None)
        if teacher_log_probs is not None:
            student_advantages = teacher_log_probs * response_mask
        else:
            # if no teacher, use environment rewards directly
            student_advantages = teacher_advantages

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
            student_lp_cpu = data["action_log_probs"]
            if teacher_lp_cpu is not None and student_lp_cpu is not None:
                kl_signal = (teacher_lp_cpu - student_lp_cpu) * data["response_mask"]
                valid_kl = torch.masked_select(
                    kl_signal[: num_samples - pad_size],
                    data["response_mask"][: num_samples - pad_size].bool(),
                )
                self.all_metrics.update({
                    "teacher/avg_reverse_kl": valid_kl.mean().item(),
                })

        return data

    def train_critic_and_policy(self, data: TrainingInputBatch):
        """
        Train the teacher with DRO on environment rewards, then train the student
        policy with PPO on teacher-distillation advantages.
        """
        data.metadata["global_step"] = self.global_step

        teacher_data = self._build_teacher_training_batch(data)
        student_data = self._build_student_training_batch(data)

        if self.colocate_all:
            if self.critic_model is not None:
                with Timer("teacher_train", self.all_timings):
                    self.critic_model.backload_to_gpu()
                    teacher_statuses = ray.get(
                        self.critic_model.async_run_ray_method("mesh", "ppo_train", teacher_data)
                    )
                    self.critic_model.offload_to_cpu()
            with Timer("policy_train", self.all_timings):
                self.policy_model.backload_to_gpu()
                policy_statuses = ray.get(
                    self.policy_model.async_run_ray_method("mesh", "ppo_train", student_data)
                )
        else:
            # Overlapped: teacher and student in parallel
            if self.critic_model is not None:
                with Timer("policy_teacher_overlap_train", self.all_timings):
                    policy_refs = self.policy_model.async_run_ray_method("mesh", "ppo_train", student_data)
                    teacher_refs = self.critic_model.async_run_ray_method("mesh", "ppo_train", teacher_data)
                    policy_statuses = ray.get(policy_refs)
                    teacher_statuses = ray.get(teacher_refs)
            else:
                with Timer("policy_train", self.all_timings):
                    policy_statuses = ray.get(
                        self.policy_model.async_run_ray_method("mesh", "ppo_train", student_data)
                    )

        empty_cache_refs = []
        if self.critic_model is not None:
            teacher_status = teacher_statuses[0].metadata["train_status"]
            for k, v in teacher_status.items():
                self.all_metrics.update({f"teacher/{k}": v})
            empty_cache_refs += self.critic_model.async_run_ray_method("pass_through", "empty_cache")

        policy_status = policy_statuses[0].metadata["train_status"]
        for k, v in policy_status.items():
            self.all_metrics.update({f"policy/{k}": v})
        empty_cache_refs += self.policy_model.async_run_ray_method("pass_through", "empty_cache")
        ray.get(empty_cache_refs)

        return policy_status

    def _build_teacher_training_batch(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """
        Build a TrainingInputBatch for teacher training with:
        - Context-formatted sequences (privileged info)
        - Student's log-probs (for DRO quadratic penalty)
        - environment rewards
        """
        teacher_sequences = data.metadata.get("teacher_formatted_sequences")
        teacher_attention_mask = data.metadata.get("teacher_formatted_attention_mask")

        if teacher_sequences is None:
            # Fallback: use original sequences (no context)
            logger.warning("No teacher-formatted sequences found; using original sequences for teacher training.")
            teacher_sequences = data["sequences"]
            teacher_attention_mask = data["attention_mask"]

        teacher_batch = TrainingInputBatch({
            "sequences": teacher_sequences,
            "attention_mask": teacher_attention_mask,
            "action_log_probs": data["action_log_probs"],        # student's log-probs (for DRO penalty)
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
        - Teacher-distillation advantages (reverse KL)
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
        """Override to use TeacherWorker in place of CriticWorker."""
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        # Import strategy-specific workers for policy and ref
        if self.cfg.trainer.strategy == "deepspeed":
            from skyrl_train.workers.deepspeed.deepspeed_worker import PolicyWorker, RefWorker
        elif self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, RefWorker
        elif self.cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, RefWorker
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")

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

        # Build models: use TeacherWorker in place of CriticWorker
        trainer.build_models(PolicyWorker, TeacherWorker, RefWorker)
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
