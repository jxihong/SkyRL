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
: "${NUM_GPUS:=8}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${RAY_TMPDIR:="$HOME/.cache/ray"}"
mkdir -p "$RAY_TMPDIR"
export RAY_TMPDIR

# --- Training hyperparameters ---
: "${TRAIN_BATCH_SIZE:=1024}"
: "${POLICY_MINI_BATCH_SIZE:=256}"
: "${CRITIC_MINI_BATCH_SIZE:=256}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_RESPONSE_LENGTH:=8196}"
: "${POLICY_LR:=1e-6}"
: "${TEACHER_LR:=1e-6}"
: "${EPOCHS:=10}"

# Set to "true" to enable thinking, "false" to disable. Empty = model default (usually true).
: "${ENABLE_THINKING:=}"
if [ -n "$ENABLE_THINKING" ]; then
  : "${BATCHED:=false}"
else
  : "${BATCHED:=true}"
fi

# --- Teacher-distillation-specific parameters ---
# teacher_loss_type: "dro" for Distributionally Robust Optimization
: "${TEACHER_LOSS_TYPE:=dro}"
# dro_beta: quadratic regularization coefficient (higher = teacher stays closer to student)
: "${DRO_BETA:=0.1}"
# teacher loss alpha: scales teacher update magnitude relative to student PPO update
: "${TEACHER_LOSS_ALPHA:=1.0}"
# reward_precision: number of decimal places for reward formatting in context
: "${TD_REWARD_PRECISION:=2}"
# Student distillation signal is always (teacher_log_probs - student_log_probs).
# GAE smoothing is controlled by STUDENT_GAE_LAMBDA/GAMMA.
# With STUDENT_GAE_LAMBDA=0, this is exactly immediate on-policy distillation.
: "${STUDENT_GAE_LAMBDA:=0.0}"
: "${STUDENT_GAE_GAMMA:=1.0}"
# Teacher runs with much longer context than student; chunk both teacher forward and teacher train to avoid OOM.
: "${TEACHER_CHUNK_BATCH_SIZE:=$(($NUM_GPUS*2))}"
# Cap reference rollout tokens injected into teacher context to avoid very long teacher forwards.
# Set empty/null to keep full rollout.
: "${TD_MAX_REFERENCE_RESPONSE_TOKENS:=2048}"
# Dynamic sampling filter drops prompt groups with reward std == 0
# (all sampled responses for a prompt are all-correct or all-incorrect).
# This filtering happens before optimization, so both teacher and student skip them.
: "${DYNAMIC_SAMPLING_TYPE:=filter}"
: "${DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES:=10}"

uv run --isolated --extra $INFERENCE_BACKEND -m examples.teacher_distillation.main_teacher_distillation \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.teacher_distillation.teacher_loss_type="$TEACHER_LOSS_TYPE" \
  trainer.algorithm.teacher_distillation.dro_beta=$DRO_BETA \
  trainer.algorithm.teacher_distillation.teacher_loss_alpha=$TEACHER_LOSS_ALPHA \
  trainer.algorithm.teacher_distillation.student_gae_lambda=$STUDENT_GAE_LAMBDA \
  trainer.algorithm.teacher_distillation.student_gae_gamma=$STUDENT_GAE_GAMMA \
  trainer.algorithm.teacher_distillation.teacher_chunk_batch_size=$TEACHER_CHUNK_BATCH_SIZE \
  trainer.algorithm.teacher_distillation.reward_precision=$TD_REWARD_PRECISION \
  trainer.algorithm.teacher_distillation.max_reference_response_tokens=$TD_MAX_REFERENCE_RESPONSE_TOKENS \
  trainer.algorithm.dynamic_sampling.type=$DYNAMIC_SAMPLING_TYPE \
  trainer.algorithm.dynamic_sampling.max_sample_batches=$DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES \
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
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.export_path="$HOME/exports/td_deepscaler" \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=$POLICY_LR \
  trainer.critic.optimizer_config.lr=$TEACHER_LR \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=$BATCHED \
  environment.env_class=aime \
  +environment.skyrl_gym.aime.strict_box_verify=${STRICT_BOX_VERIFY:-true} \
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
  ${ENABLE_THINKING:++generator.chat_template_kwargs={enable_thinking:$ENABLE_THINKING}} \
  "${EXTRA_OVERRIDES[@]}" \
  "$@"
