# This script will run an MNIST training workload and save measurements as CSV and metadata as JSON.
#
# Author: Daniel Snider <danielsnider12@gmail.com>
#
# Usage:
# bash ./experiments/model_arch_mnist/measure_batchnorm_checkpoints.sh 2>&1 | tee -a ./experiments/batchnorm_checkpoints/logs/console_output.log

set -e # exit on error

export WANDB_ENTITY=danielsnider
export WANDB_PROJECT="mnist_batch_norm"
export WANDB_NOTES=""
# export WANDB_MODE="offline"


LOGGING_DIR='./experiments/batchnorm_checkpoints-2-batch_stats/logs'
rm -rf $LOGGING_DIR
mkdir -p $LOGGING_DIR

BATCH_NORMS='affine-activation-batchnorm affine-batchnorm-activation off'
ACTIVATIONS='relu'
MODEL_WIDTHS='100'
MODEL_DEPTHS='3'
DROPOUT_RATES='0'
BATCH_SIZES='1024'
OPTIMIZER='adam'

EVAL_FREQUENCY_OVERRIDE='1 step'
TARGET_VALUE='0.873'
NUM_TRIALS='3'

HYPERPARAM_CONFIG='adam_mnist_tuning_search_space.json'
cat <<EOF > adam_mnist_tuning_search_space.json
{
  "learning_rate": {"feasible_points": [1e-3]},
  "one_minus_beta_1": {"feasible_points": [0.99]},
  "epsilon": {"feasible_points": [1e-5]}
}
EOF

MAX_ALLOWED_RUNTIME_SEC='900000'
EARLY_STOPPING_CONFIG='mnist_early_stopping_config.json'
cat <<EOF > mnist_early_stopping_config.json
{
  "metric_name": "loss",
  "min_delta": 0,
  "patience": 99,
  "min_steps": 58,
  "max_steps": 90000,
  "mode": "min",
  "baseline": null
}
EOF


# Count number of experiments so that we can track progress
activation_count=$( awk -F" " '{print NF-1}' <<<"${ACTIVATIONS}" )
activation_count=$(echo $(( activation_count + 1 )))
width_count=$( awk -F" " '{print NF-1}' <<<"${MODEL_WIDTHS}" )
width_count=$(echo $(( width_count + 1 )))
depth_count=$( awk -F" " '{print NF-1}' <<<"${MODEL_DEPTHS}" )
depth_count=$(echo $(( depth_count + 1 )))
BATCH_NORMS_count=$( awk -F" " '{print NF-1}' <<<"${BATCH_NORMS}" )
BATCH_NORMS_count=$(echo $(( BATCH_NORMS_count + 1 )))
total=$(echo $((activation_count * width_count * depth_count * BATCH_NORMS_count)))
iteration=1


run_cmd () {
  echo -e "\n[INFO $(date +"%d-%I:%M%p")] Starting iteration $iteration of $total."
  iteration=$(echo $(( iteration + 1 )))
  EXPERIMENT_DIR="$LOGGING_DIR/batchnorm_$BATCH_NORM-activation_$ACTIVATION-width_$MODEL_WIDTH-depth_$MODEL_DEPTH-dropout_$DROPOUT_RATE-batch_$BATCH_SIZE/"
  mkdir -p $EXPERIMENT_DIR
  export WANDB_NAME=$BATCH_NORM
  set -x
  CUDA_VISIBLE_DEVICES=0 python3 submission_runner.py \
    --framework=jax \
    --workload=configurable_mnist_jax \
    --submission_path=baselines/mnist/configurable_mnist_jax/submission.py \
    --tuning_search_space="$HYPERPARAM_CONFIG" \
    --num_tuning_trials="$NUM_TRIALS" \
    --logging_dir="$EXPERIMENT_DIR" \
    --eval_frequency_override="$EVAL_FREQUENCY_OVERRIDE" \
    --early_stopping_config="$EARLY_STOPPING_CONFIG" \
    --save_checkpoints \
    --enable_wandb \
    --extra_metadata="mnist_config.target_value=$TARGET_VALUE" \
    --extra_metadata="mnist_config.max_allowed_runtime_sec=$MAX_ALLOWED_RUNTIME_SEC" \
    --extra_metadata="mnist_config.activation_fn=$ACTIVATION" \
    --extra_metadata="mnist_config.model_width=$MODEL_WIDTH" \
    --extra_metadata="mnist_config.model_depth=$MODEL_DEPTH" \
    --extra_metadata="mnist_config.dropout_rate=$DROPOUT_RATE" \
    --extra_metadata="mnist_config.batch_size=$BATCH_SIZE" \
    --extra_metadata="mnist_config.optimizer=$OPTIMIZER" \
    --extra_metadata="mnist_config.batch_norm=$BATCH_NORM"
  set +x
}

for ACTIVATION in $ACTIVATIONS
do
  for MODEL_WIDTH in $MODEL_WIDTHS
  do
    for MODEL_DEPTH in $MODEL_DEPTHS
    do
      for DROPOUT_RATE in $DROPOUT_RATES
      do
        for BATCH_SIZE in $BATCH_SIZES
        do
          for BATCH_NORM in $BATCH_NORMS
          do
            run_cmd
          done
        done
      done
    done
  done
done


# Combine all tuning trials into one CSV
# python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

# echo "[INFO $(date +"%d-%I:%M%p")] Generated files:"
# find $LOG_DIR

# # Check status of each experiment (requires zsh)
# for FILE in ./experiments/model_arch_mnist/logs/**/*.json
# do
#   STATUS=$(cat $FILE | jq -r '.status')
#   echo "$STATUS $FILE"
# done

echo "[INFO $(date +"%d-%I:%M%p")] Finished."

exit 0

for FILE in ./experiments/batchnorm_checkpoints/logs/*/**/trial_*/*.json
do
    global_step=$(cat $FILE | jq -r '.global_step')
    accuracy=$(cat $FILE | jq -r '.accuracy')
    loss=$(cat $FILE | jq -r '.loss')
    echo "$global_step $accuracy $loss $FILE"
done