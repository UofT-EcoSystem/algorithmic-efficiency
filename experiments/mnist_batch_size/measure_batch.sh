# This script will run an MNIST training workload and save measurements as CSV and metadata as JSON.
#
# Author: Daniel Snider <danielsnider12@gmail.com>
#
# Usage:
## bash ./experiments/mnist_batch_size/measure_batch.sh.sh 2>&1 | tee -a $LOG_DIR/console_output.log

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
while IFS=, read -r architecture batch_size trial_id step_to_threshold learning_rate train.cross_entropy_error train.classification_error val.cross_entropy_error val.classification_error best_config_path; do
    echo "arch $architecture, batch $batch_size, lr $learning_rate";
    LOG_DIR_SUB_EXPERIMENT=$LOG_DIR/$architecture/$batch_size
    set -x
    mkdir -p $LOG_DIR_SUB_EXPERIMENT
#    python3 algorithmic_efficiency/submission_runner.py \
#        --framework=jax \
#        --workload=mnist_jax \
#        --submission_path=baselines/mnist/mnist_jax/submission.py \
#        --tuning_search_space=baselines/mnist/tuning_search_space.json \
#        --num_tuning_trials=$NUM_TRIALS \
#        --log_dir=$LOG_DIR_SUB_EXPERIMENT
#        --architecture=$architecture
#        --batch_size=$batch_size
#        --learning_rate=$learning_rate
#        --extra_metadata="architecture=batch_science.$architecture"
#        --extra_metadata="batch_size=batch_science.$batch_size"
#        --extra_metadata="trial_id=batch_science.$trial_id"
#        --extra_metadata="step_to_threshold=batch_science.$step_to_threshold"
#        --extra_metadata="learning_rate=batch_science.$learning_rate"
#        --extra_metadata="train.cross_entropy_error=batch_science.$train.cross_entropy_error"
#        --extra_metadata="train.classification_error=batch_science.$train.classification_error"
#        --extra_metadata="val.cross_entropy_error=batch_science.$val.cross_entropy_error"
#        --extra_metadata="val.classification_error=batch_science.$val.classification_error"
#        --extra_metadata="best_config_path=batch_science.$best_config_path"
#        --target_value=$TARGET_VALUE
    set +x
done < /home/dans/csc2541_colab/best_parameters.csv

# Combine all tuning trials into one CSV
# python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

echo "[INFO $(date +"%d-%I:%M%p")] Generated files:"
find $LOG_DIR

# Check status of each experiment
for FILE in experiments/simple_example_mnist_loss/logs/*/*.json;
do
    STATUS=$(cat experiments/simple_example_mnist_loss/logs/mnist_jax/metadata.json | jq -r '.status')
    echo "$STATUS $FILE"
done;

echo "[INFO $(date +"%d-%I:%M%p")] Finished."