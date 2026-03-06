set -x

# ICVL (In-Context Value Learning) PPO training on the DeepScaleR math dataset.
#
# ICVL gives the critic additional (response, reward) context from the same prompt group,
# allowing it to leverage in-context learning to better estimate values for the current response.
# Training otherwise proceeds as standard PPO with GAE advantage estimation.
#
# Prerequisites:
#   uv run examples/icvl/deepscaler_dataset.py --output_dir $HOME/data/deepscaler
#   export WANDB_API_KEY=<your_key_here>
#
# Usage:
#   bash examples/icvl/run_icvl_deepscaler.sh
#
# Choose which algorithm: ESTIMATOR=ppo, ESTIMATOR=grpo, or ESTIMATOR=icvl (default).
#   ESTIMATOR=ppo  bash examples/icvl/run_icvl_deepscaler.sh
#   ESTIMATOR=grpo bash examples/icvl/run_icvl_deepscaler.sh
#   ESTIMATOR=icvl bash examples/icvl/run_icvl_deepscaler.sh
# Other overrides, e.g.:
#   ENABLE_THINKING=false MODEL_NAME=Qwen/Qwen2.5-7B-Instruct bash examples/icvl/run_icvl_deepscaler.sh

# --- Configurable env vars with defaults ---
: "${DATA_DIR:="$HOME/data/deepscaler"}"
: "${MODEL_NAME:="Qwen/Qwen3-1.7B"}"
: "${NUM_GPUS:=8}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${ROLLOUT_VIZ:=true}"
: "${ROLLOUT_VIZ_BEFORE_TRAIN:=true}"
: "${ROLLOUT_VIZ_INTERVAL:=5}"
: "${ROLLOUT_VIZ_SAMPLES:=4}"
: "${ROLLOUT_VIZ_NORMALIZE_PER_SAMPLE:=false}"

# --- Training hyperparameters ---
: "${TRAIN_BATCH_SIZE:=1024}"
: "${POLICY_MINI_BATCH_SIZE:=256}"
: "${CRITIC_MINI_BATCH_SIZE:=256}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_RESPONSE_LENGTH:=8192}"
: "${LR:=1e-6}"
: "${EPOCHS:=5}"
: "${KL_LOSS_COEF:=0.001}"
# GAE (Generalized Advantage Estimation): lambda in [0,1]; default 1.0 (no discount on advantages)
: "${GAE_LAMBDA:=0.0}"
# --- Dynamic sampling (DAPO-style filter) ---
# filter: drop prompt-groups with std==0 rewards and resample to fill batch
: "${DYNAMIC_SAMPLING_TYPE:=filter}"
: "${DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES:=10}"

# --- ICVL-specific parameters ---
: "${ICVL_REWARD_FORMAT:=raw}"
# sort_context_by_reward: sort context trajectories by reward for better ICL pattern learning
: "${ICVL_SORT_CONTEXT:=true}"
# sort_order: "ascending" (worst to best) or "descending" (best to worst)
: "${ICVL_SORT_ORDER:=ascending}"
# reward_precision: number of decimal places for reward formatting in context
: "${ICVL_REWARD_PRECISION:=2}"
# micro batch size for critic training (smaller than policy to handle longer ICVL sequences)
: "${MICRO_CRITIC_TRAIN_BS:=2}"

# --- Value head type: regression (default), cross_entropy (bins), sigmoid (binary BCE), or zip (joint distribution) ---
# USE_CE_VALUE_HEAD=true: critic predicts reward (cross-entropy). Uses VALUE_MIN, VALUE_MAX, VALUE_NUM_BINS.
# USE_SIGMOID_VALUE_HEAD=true: critic predicts binary reward (BCE) with classes VALUE_MIN, VALUE_MAX.
: "${USE_CE_VALUE_HEAD:=false}"
: "${USE_SIGMOID_VALUE_HEAD:=false}"
: "${VALUE_MIN:=-1.0}"
: "${VALUE_MAX:=1.0}"
: "${VALUE_NUM_BINS:=51}"
# USE_ZIP_VALUE_HEAD=true: critic uses ZIP to predict joint distribution of reward and length.
# can load from pretrained critic using ZIP_CRITIC_PATH.
: "${USE_ZIP_VALUE_HEAD:=true}"
: "${ZIP_CRITIC_PATH:=/home/rohin/icl_value/models/joint_distribution_critic_no_ans_supervise_from_8_with_32}"
: "${ZIP_DISTRIBUTION_TOKEN_ID:=151669}"
: "${ZIP_REWARD_VALUES:=[0.0,1.0]}"
: "${ZIP_NUM_LENGTH_BINS:=8}"
: "${ZIP_SCALE_ZERO_ONE_REWARDS:=true}"

# Set to "true" to enable thinking, "false" to disable. Empty = model default (usually true).
: "${ENABLE_THINKING:=}"
if [ -n "$ENABLE_THINKING" ]; then
  : "${BATCHED:=false}"
else
  : "${BATCHED:=true}"
fi

# Shared wandb group so runs can be grouped for comparison across invocations.
: "${EXPERIMENT_GROUP:=icvl_deepscaler_$(basename $MODEL_NAME)}"
# Which algorithm to run: "ppo" (GAE baseline), "grpo", or "icvl". Only one runs per invocation.
: "${ESTIMATOR:=icvl}"

_run_common() {
  uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
    data.train_data="['$DATA_DIR/train.parquet']" \
    data.val_data="['$DATA_DIR/validation.parquet','$DATA_DIR/validation_aime2025.parquet']" \
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
    trainer.micro_critic_train_batch_size_per_gpu=$MICRO_CRITIC_TRAIN_BS \
    trainer.ckpt_interval=5 \
    trainer.hf_save_interval=5 \
    trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
    generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
    trainer.policy.optimizer_config.lr=$LR \
    trainer.algorithm.use_kl_loss=true \
    trainer.algorithm.kl_loss_coef=$KL_LOSS_COEF \
    trainer.algorithm.lambd=$GAE_LAMBDA \
    trainer.algorithm.dynamic_sampling.type=$DYNAMIC_SAMPLING_TYPE \
    trainer.algorithm.dynamic_sampling.max_sample_batches=$DYNAMIC_SAMPLING_MAX_SAMPLE_BATCHES \
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
    trainer.rollout_visualization.enabled=$ROLLOUT_VIZ \
    trainer.rollout_visualization.rollout_viz_before_train=$ROLLOUT_VIZ_BEFORE_TRAIN \
    trainer.rollout_visualization.log_interval=$ROLLOUT_VIZ_INTERVAL \
    trainer.rollout_visualization.max_samples_per_log=$ROLLOUT_VIZ_SAMPLES \
    trainer.rollout_visualization.max_tokens_per_sample=$MAX_RESPONSE_LENGTH \
    trainer.rollout_visualization.normalize_per_sample=$ROLLOUT_VIZ_NORMALIZE_PER_SAMPLE \
    trainer.project_name="icvl_deepscaler" \
    trainer.run_group="$EXPERIMENT_GROUP" \
    trainer.resume_mode=null \
    trainer.eval_batch_size=1024 \
    trainer.eval_before_train=true \
    trainer.eval_interval=5 \
    ${ENABLE_THINKING:++generator.chat_template_kwargs={enable_thinking:$ENABLE_THINKING}} \
    ${USE_CE_VALUE_HEAD:+trainer.algorithm.value_head_type=cross_entropy} \
    ${USE_CE_VALUE_HEAD:+trainer.algorithm.value_min=$VALUE_MIN} \
    ${USE_CE_VALUE_HEAD:+trainer.algorithm.value_max=$VALUE_MAX} \
    ${USE_CE_VALUE_HEAD:+trainer.algorithm.value_num_bins=$VALUE_NUM_BINS} \
    ${USE_SIGMOID_VALUE_HEAD:+trainer.algorithm.value_head_type=sigmoid} \
    ${USE_SIGMOID_VALUE_HEAD:+trainer.algorithm.value_min=$VALUE_MIN} \
    ${USE_SIGMOID_VALUE_HEAD:+trainer.algorithm.value_max=$VALUE_MAX} \
    ${USE_ZIP_VALUE_HEAD:+trainer.algorithm.value_head_type=zip} \
    ${USE_ZIP_VALUE_HEAD:+trainer.algorithm.zip_distribution_token_id=$ZIP_DISTRIBUTION_TOKEN_ID} \
    ${USE_ZIP_VALUE_HEAD:+"trainer.algorithm.zip_reward_values=$ZIP_REWARD_VALUES"} \
    ${USE_ZIP_VALUE_HEAD:+trainer.algorithm.zip_num_length_bins=$ZIP_NUM_LENGTH_BINS} \
    ${USE_ZIP_VALUE_HEAD:+trainer.algorithm.zip_scale_zero_one_rewards=$ZIP_SCALE_ZERO_ONE_REWARDS} \
    ${ZIP_CRITIC_PATH:+trainer.critic.model.path="$ZIP_CRITIC_PATH"} \
    "$@"
}

case "$ESTIMATOR" in
  ppo)
    echo "========== Running PPO (GAE) baseline =========="
    _run_common \
      trainer.algorithm.advantage_estimator="gae" \
      trainer.run_name="ppo_deepscaler_$(basename $MODEL_NAME)" \
      trainer.ckpt_path="$HOME/ckpts/ppo_deepscaler_ckpt" \
      "$@"
    ;;
  grpo)
    echo "========== Running GRPO baseline =========="
    _run_common \
      trainer.algorithm.advantage_estimator="grpo" \
      trainer.run_name="grpo_deepscaler_$(basename $MODEL_NAME)" \
      trainer.ckpt_path="$HOME/ckpts/grpo_deepscaler_ckpt" \
      "$@"
    ;;
  icvl)
    echo "========== Running ICVL =========="
    _run_common \
      trainer.algorithm.advantage_estimator="icvl" \
      trainer.algorithm.icvl.reward_format="$ICVL_REWARD_FORMAT" \
      trainer.algorithm.icvl.sort_context_by_reward=$ICVL_SORT_CONTEXT \
      trainer.algorithm.icvl.sort_order="$ICVL_SORT_ORDER" \
      trainer.algorithm.icvl.reward_precision=$ICVL_REWARD_PRECISION \
      trainer.run_name="icvl_deepscaler_$(basename $MODEL_NAME)" \
      trainer.ckpt_path="$HOME/ckpts/icvl_deepscaler_ckpt" \
      "$@"
    ;;
  *)
    echo "Invalid ESTIMATOR='$ESTIMATOR'. Use 'ppo', 'grpo', or 'icvl'." >&2
    exit 1
    ;;
esac
