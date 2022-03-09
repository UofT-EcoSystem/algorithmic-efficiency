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
while IFS=, read -r ARCHITECTURE BATCH_SIZE TRIAL_ID STEP_TO_THRESHOLD LEARNING_RATE REMAINDER; do
    echo "arch $ARCHITECTURE, batch $BATCH_SIZE, lr $LEARNING_RATE";
    LOG_DIR_SUB_EXPERIMENT=$LOG_DIR/$ARCHITECTURE/$BATCH_SIZE
    set -x
    mkdir -p $LOG_DIR_SUB_EXPERIMENT
#    python3 algorithmic_efficiency/submission_runner.py \
#        --framework=jax \
#        --workload=mnist_jax \
#        --submission_path=baselines/mnist/mnist_jax/submission.py \
#        --tuning_search_space=baselines/mnist/tuning_search_space.json \
#        --num_tuning_trials=$NUM_TRIALS \
#        --log_dir=$LOG_DIR_SUB_EXPERIMENT
#        --architecture=$ARCHITECTURE
#        --batch_size=$BATCH_SIZE
#        --learning_rate=$LEARNING_RATE
#        --target_value=$TARGET_VALUE
    set +x
done < /home/dans/csc2541_colab/best_parameters.csv

# Combine all tuning trials into one CSV
# python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

echo "[INFO $(date +"%d-%I:%M%p")] Generated files:"
find $LOG_DIR
echo "[INFO $(date +"%d-%I:%M%p")] Finished."
