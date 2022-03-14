# This script will run an MNIST training workload and save measurements as CSV and metadata as JSON.
#
# Author: Daniel Snider <danielsnider12@gmail.com>
#
# Usage:
# bash ./experiments/model_arch_mnist/measure_example.sh 2>&1 | tee -a ./experiments/model_arch_mnist/logs/console_output.log

set -e # exit on error

LOGGING_DIR='./experiments/model_arch_mnist/logs'
rm -rf $LOGGING_DIR
mkdir -p $LOGGING_DIR

ACTIVATIONS='relu sigmoid hard_tahn gelu'
MODEL_WIDTHS='16 32 128 256'
MODEL_DEPTHS='1 2 4 8'
DROPOUT_RATES='0'
BATCH_SIZES='1024'
OPTIMIZER='adam'

EVAL_FREQUENCY_OVERRIDE='10 step'
TARGET_VALUE='0.6'
MAX_ALLOWED_RUNTIME_SEC='300'
NUM_TRIALS='1'

HYPERPARAM_CONFIG='adam_mnist_tuning_search_space.json'
cat <<EOF > adam_mnist_tuning_search_space.json
{
  "learning_rate": {"min": 1e-4, "max": 1e-2, "scaling": "log"},
  "one_minus_beta_1": {"min": 0.9, "max": 0.999, "scaling": "log"},
  "epsilon": {"feasible_points": [1e-8, 1e-5, 1e-3]}
}
EOF

EARLY_STOPPING_CONFIG='mnist_early_stopping_config.json'
cat <<EOF > mnist_early_stopping_config.json
{
  "metric_name": "loss",
  "min_delta": 0,
  "patience": 5,
  "min_steps": 0,
  "max_steps": 290,
  "mode": "min",
  "baseline": null
}
EOF


# Count number of experiments so that we can track progress
activation_count=$( awk -F" " '{print NF-1}' <<<"${ACTIVATIONS}" )
activation_count=$(echo $(( activation_count + 1 )))
width_count=$( awk -F" " '{print NF-1}' <<<"${ACTIVATIONS}" )
width_count=$(echo $(( width_count + 1 )))
depth_count=$( awk -F" " '{print NF-1}' <<<"${ACTIVATIONS}" )
depth_count=$(echo $(( depth_count + 1 )))
total=$(echo $((activation_count * width_count * depth_count)))
iteration=1


run_cmd () {
  echo -e "\n[INFO $(date +"%d-%I:%M%p")] Starting iteration $iteration of $total."
  iteration=$(echo $(( iteration + 1 )))
  EXPERIMENT_DIR="$LOGGING_DIR/activation_$ACTIVATION-width_$MODEL_WIDTH-depth_$MODEL_DEPTH-dropout_$DROPOUT_RATE-batch_$BATCH_SIZE/"
  mkdir -p $EXPERIMENT_DIR
  set -x
  python3 algorithmic_efficiency/submission_runner.py \
    --framework=jax \
    --workload=configurable_mnist_jax \
    --submission_path=baselines/mnist/configurable_mnist_jax/submission.py \
    --tuning_search_space="$HYPERPARAM_CONFIG" \
    --num_tuning_trials="$NUM_TRIALS" \
    --logging_dir="$EXPERIMENT_DIR" \
    --eval_frequency_override="$EVAL_FREQUENCY_OVERRIDE" \
    --early_stopping_config="$EARLY_STOPPING_CONFIG" \
    --extra_metadata="mnist_config.target_value=$TARGET_VALUE" \
    --extra_metadata="mnist_config.max_allowed_runtime_sec=$MAX_ALLOWED_RUNTIME_SEC" \
    --extra_metadata="mnist_config.activation_fn=$ACTIVATION" \
    --extra_metadata="mnist_config.model_width=$MODEL_WIDTH" \
    --extra_metadata="mnist_config.model_depth=$MODEL_DEPTH" \
    --extra_metadata="mnist_config.dropout_rate=$DROPOUT_RATE" \
    --extra_metadata="mnist_config.batch_size=$BATCH_SIZE" \
    --extra_metadata="mnist_config.optimizer=$OPTIMIZER"
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
          run_cmd
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