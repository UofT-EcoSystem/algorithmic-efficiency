# This script will run an MNIST training workload and save measurements as CSV and metadata as JSON.
#
# Author: Daniel Snider <danielsnider12@gmail.com>
#
# Usage:
# mkdir -p ./experiments/mnist_batch_size/logs; bash ./experiments/mnist_batch_size/measure_batch.sh 2>&1 | tee -a ./experiments/mnist_batch_size/logs/console_output.log

# set -e # exit on error

LOG_DIR=./experiments/mnist_batch_size/logs
mkdir -p $LOG_DIR/

# Full data collection
NUM_TRIALS='1'
TARGET_VALUE='0.7'

# Quick data collection
NUM_TRIALS='1'
TARGET_VALUE='0.3'

echo -e "\n[INFO $(date +"%d-%I:%M%p")] Starting."
while IFS=, read -r architecture batch_size trial_id step_to_threshold learning_rate train_cross_entropy_error train_classification_error val_cross_entropy_error val_classification_error best_config_path; do
    echo "INPUT CONFIG: arch $architecture, batch $batch_size, lr $learning_rate";
    LOG_DIR_SUB_EXPERIMENT=$LOG_DIR/$architecture/$batch_size
    set -x
    mkdir -p $LOG_DIR_SUB_EXPERIMENT
   python3 algorithmic_efficiency/submission_runner.py \
       --framework=jax \
       --workload=mnist_jax \
       --submission_path=baselines/mnist/mnist_jax/submission.py \
       --tuning_search_space=baselines/mnist/tuning_search_space.json \
       --num_tuning_trials=$NUM_TRIALS \
       --log_dir=$LOG_DIR_SUB_EXPERIMENT \
       --extra_metadata="batch_science.architecture=$architecture" \
       --extra_metadata="batch_science.batch_size=$batch_size" \
       --extra_metadata="batch_science.trial_id=$trial_id" \
       --extra_metadata="batch_science.step_to_threshold=$step_to_threshold" \
       --extra_metadata="batch_science.learning_rate=$learning_rate" \
       --extra_metadata="batch_science.train_cross_entropy_error=$train_cross_entropy_error" \
       --extra_metadata="batch_science.train_classification_error=$train_classification_error" \
       --extra_metadata="batch_science.val_cross_entropy_error=$val_cross_entropy_error" \
       --extra_metadata="batch_science.val_classification_error=$val_classification_error" \
       --extra_metadata="batch_science.best_config_path=$best_config_path" \
       --target_value=$TARGET_VALUE \
       --architecture=$architecture \
       --learning_rate=$learning_rate
    #    --batch_size=$batch_size \
    set +x
    break
done < ./best_parameters.csv

# Combine all tuning trials into one CSV
# python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

echo "[INFO $(date +"%d-%I:%M%p")] Generated files:"
find $LOG_DIR

# Check status of each experiment (requires zsh)
for FILE in ./experiments/mnist_batch_size/logs/**/*.json
do
    STATUS=$(cat $FILE | jq -r '.status')
    echo "$STATUS $FILE"
done

echo "[INFO $(date +"%d-%I:%M%p")] Finished."