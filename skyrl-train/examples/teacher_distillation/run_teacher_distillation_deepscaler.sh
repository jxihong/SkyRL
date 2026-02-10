set -x

# Teacher-Distillation PPO training on the DeepScaleR math dataset.
#
# Prerequisites:
#   uv run examples/teacher_distillation/deepscaler_dataset.py --output_dir $HOME/data/deepscaler
#   export WANDB_API_KEY=<your_key_here>
#
# Usage:
#   bash examples/teacher_distillation/run_teacher_distillation_deepscaler.sh
#
# You can override defaults via env vars, e.g.:
#   NUM_GPUS=8 MODEL_NAME=Qwen/Qwen2.5-7B-Instruct bash examples/teacher_distillation/run_teacher_distillation_deepscaler.sh

# --- Configurable env vars with defaults ---
: "${DATA_DIR:="$HOME/data/deepscaler"}"
: "${MODEL_NAME:="Qwen/Qwen3-1.7B"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"

# --- Training hyperparameters ---
: "${TRAIN_BATCH_SIZE:=1024}"
: "${POLICY_MINI_BATCH_SIZE:=256}"
: "${CRITIC_MINI_BATCH_SIZE:=256}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_RESPONSE_LENGTH:=2048}"
: "${POLICY_LR:=1e-6}"
: "${TEACHER_LR:=1e-6}"
: "${EPOCHS:=20}"
: "${KL_LOSS_COEF:=0.001}"

# --- Teacher-distillation-specific parameters ---
# teacher_loss_type: "dro" for Distributionally Robust Optimization
: "${TEACHER_LOSS_TYPE:=dro}"
# dro_beta: quadratic regularization coefficient (higher = teacher stays closer to student)
: "${DRO_BETA:=0.1}"
# reward_format: "normalized" normalizes rewards to [0, 1] per group; "raw" keeps them as-is
: "${TD_REWARD_FORMAT:=normalized}"
# sort_context_by_reward: sort context trajectories by reward for better ICL pattern learning
: "${TD_SORT_CONTEXT:=true}"
# sort_order: "ascending" (worst to best) or "descending" (best to worst)
: "${TD_SORT_ORDER:=ascending}"
# reward_precision: number of decimal places for reward formatting in context
: "${TD_REWARD_PRECISION:=2}"

uv run --isolated --extra $INFERENCE_BACKEND -m examples.teacher_distillation.main_teacher_distillation \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.teacher_distillation.teacher_loss_type="$TEACHER_LOSS_TYPE" \
  trainer.algorithm.teacher_distillation.dro_beta=$DRO_BETA \
  trainer.algorithm.teacher_distillation.reward_format="$TD_REWARD_FORMAT" \
  trainer.algorithm.teacher_distillation.sort_context_by_reward=$TD_SORT_CONTEXT \
  trainer.algorithm.teacher_distillation.sort_order="$TD_SORT_ORDER" \
  trainer.algorithm.teacher_distillation.reward_precision=$TD_REWARD_PRECISION \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.critic.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=$EPOCHS \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.critic_mini_batch_size=$CRITIC_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$POLICY_LR \
  trainer.critic.optimizer_config.lr=$TEACHER_LR \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="teacher_distillation_deepscaler" \
  trainer.run_name="td_deepscaler_$(basename $MODEL_NAME)" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/td_deepscaler_ckpt" \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@
