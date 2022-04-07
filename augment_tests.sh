# This script will run an MNIST training workload and save measurements as CSV and metadata as JSON.
#
# Author: Daniel Snider <danielsnider12@gmail.com>
#
# Usage:
# mkdir -p ./experiments/mnist_batch_size/logs; bash ./experiments/mnist_batch_size/measure_batch.sh

# set -e # exit on error




LOG_DIR=./experiments/mnist/logs
CKPT_DIR=./experiments/mnist/chpts
mkdir -p $LOG_DIR/

# Full data collection
# NUM_TRIALS='20'
# TARGET_VALUE='0.7'

# Quick data collection
NUM_TRIALS='1'
TARGET_VALUE='0.3'

for apply_augment in 0 1;
do
    if [ $apply_augment -eq 0 ]
    then
        WORKLOAD=mnist_jax
        SUBMISSION_PATH=baselines/mnist/mnist_jax/submission.py
        AUG=no_aug
    else
        WORKLOAD=mnist_jax_augmentation
        SUBMISSION_PATH=baselines/mnist/mnist_jax/augmentation/submission.py
        AUG=with_aug
    fi

    for num_data in {10..100..10};
    do 
        echo "INPUT CONFIG: pct data pts: $num_data, augments applied: $apply_augment";
        LOG_DIR_SUB_EXPERIMENT=$LOG_DIR/$AUG/$num_data
        CKPT_DIR_SUB_EXPERIMENT=$CKPT_DIR/$AUG/$num_data

            set -x
            mkdir -p $LOG_DIR_SUB_EXPERIMENT
            mkdir -p $CKPT_DIR_SUB_EXPERIMENT
            python submission_runner.py \
                --framework=jax \
                --workload=$WORKLOAD \
                --submission_path=$SUBMISSION_PATH \
                --tuning_search_space=baselines/mnist/tuning_search_space.json \
                --cp_dir=$CKPT_DIR_SUB_EXPERIMENT \
                --num_tuning_trials=$NUM_TRIALS \
                --percent_data_selection=$num_data \
                --logging_dir=$LOG_DIR_SUB_EXPERIMENT

                set +x
        done #< ./best_parameters.csv
    done
done

# # Combine all tuning trials into one CSV
# # python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

# echo "[INFO $(date +"%d-%I:%M%p")] Generated files:"
# find $LOG_DIR

# echo "[INFO $(date +"%d-%I:%M%p")] Finished."
# exit 0

# # # Check status of each experiment (requires zsh)
# for FILE in ./experiments/mnist_batch_size/logs*/**/trial_*/*.json
# do
#     STATUS=$(cat $FILE | jq -r '.status')
#     global_step=$(cat $FILE | jq -r '.global_step')
#     accuracy=$(cat $FILE | jq -r '.accuracy')
#     step_to_threshold=$(cat $FILE | jq -r '."extra.batch_science.step_to_threshold"')
#     echo "$STATUS theirs $step_to_threshold ours $global_step $accuracy $FILE"
# done

# # only one field
# for FILE in ./experiments/mnist_batch_size/logs*/**/trial_*/*.json
# do
#     early_stop=$(cat $FILE | jq -r '.early_stop')
#     echo "$early_stop $FILE"
# done

# # rebuild best parameters csv
# echo "architecture,batch_size,trial_id,step_to_threshold,learning_rate,train.cross_entropy_error,train.classification_error,val.cross_entropy_error,val.classification_error,best_config_path" > out.csv
# for FILE in ./experiments/mnist_batch_size/logs*/**/trial_**/*.json
# do
#     JSON=$(cat $FILE)
#     global_step=$(echo $JSON | jq -r '.global_step')
#     architecture=$(echo $JSON | jq -r .'"extra.batch_science.architecture"')
#     batch_size=$(echo $JSON | jq -r .'"batch_size"')
#     trial_id=$(echo $JSON | jq -r .'"extra.batch_science.trial_id"')
#     step_to_threshold=$(echo $JSON | jq -r .'"global_step"')
#     # learning_rate=$(echo $JSON | jq -r .'"extra.batch_science.learning_rate"')
#     learning_rate=$(echo $JSON | jq -r .'"learning_rate"')
#     train_cross_entropy_error=$(echo $JSON | jq -r .'"extra.batch_science.train_cross_entropy_error"')
#     train_classification_error=$(echo $JSON | jq -r .'"extra.batch_science.train_classification_error"')
#     val_cross_entropy_error=$(echo $JSON | jq -r .'"extra.batch_science.val_cross_entropy_error"')
#     val_classification_error=$(echo $JSON | jq -r .'"extra.batch_science.val_classification_error"')
#     best_config_path=$(echo $JSON | jq -r .'"extra.batch_science.best_config_path"')

#     echo "$architecture,$batch_size,$trial_id,$step_to_threshold,$learning_rate,$train_cross_entropy_error,$train_classification_error,$val_cross_entropy_error,$val_classification_error,$best_config_path" | tee -a out.csv
# done

# python3 experiments/mnist_batch_size/plot_batch.py out.csv plot_mnist_batch-20trials.png