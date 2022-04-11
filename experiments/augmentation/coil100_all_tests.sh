#! /bin/bash

# Usage:
# Must be located in main dir of algorithmic-efficiency project
# sh experiments/augmentation/aug_test_cifar10.sh path/to/save/dir

SAVE_DIR=$1
TARGET_VALUE=1.0
NUM_TRIALS=5


if [ -d $SAVE_DIR ]; then
  echo "save_dir: ${SAVE_DIR} already exists"
  exit 1
fi

mkdir -p $SAVE_DIR

LOG_DIR="$SAVE_DIR/tex_aug"
mkdir -p $LOG_DIR
AUG_CONFIG=experiments/augmentation/coil_texture_augments.json
cp $AUG_CONFIG $LOG_DIR/
python submission_runner.py \
  --framework=jax \
  --workload=coil100_jax \
  --data_dir=algorithmic_efficiency/workloads/coil100/coil100_train_test_split.pkl \
  --submission_path=baselines/coil100/coil100_jax/submission.py \
  --tuning_search_space=baselines/coil100/tuning_search_space.json \
  --num_tuning_trials=$NUM_TRIALS \
  --logging_dir=$LOG_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/coil100/early_stop_config.json \
  --cp_dir=$LOG_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --extra_metadata="coil100.target_value=$TARGET_VALUE" \
  --augments=$AUG_CONFIG \

python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

LOG_DIR="$SAVE_DIR/geo_aug"
mkdir -p $LOG_DIR
AUG_CONFIG=experiments/augmentation/coil_geo_augments.json
cp $AUG_CONFIG $LOG_DIR/
python submission_runner.py \
  --framework=jax \
  --workload=coil100_jax \
  --data_dir=algorithmic_efficiency/workloads/coil100/coil100_train_test_split.pkl \
  --submission_path=baselines/coil100/coil100_jax/submission.py \
  --tuning_search_space=baselines/coil100/tuning_search_space.json \
  --num_tuning_trials=$NUM_TRIALS \
  --logging_dir=$LOG_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/coil100/early_stop_config.json \
  --cp_dir=$LOG_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --extra_metadata="coil100.target_value=$TARGET_VALUE" \
  --extra_metadata='coil100.aug_fliplr=0.5' \
  --extra_metadata='cifar10.aug_width_shift_max=13' \
  --extra_metadata='cifar10.aug_height_shift_max=13' \
  --augments=$AUG_CONFIG \

python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

LOG_DIR="$SAVE_DIR/all_aug"
mkdir -p $LOG_DIR
AUG_CONFIG=experiments/augmentation/coil_all_augments.json
cp $AUG_CONFIG $LOG_DIR/
python submission_runner.py \
  --framework=jax \
  --workload=coil100_jax \
  --data_dir=algorithmic_efficiency/workloads/coil100/coil100_train_test_split.pkl \
  --submission_path=baselines/coil100/coil100_jax/submission.py \
  --tuning_search_space=baselines/coil100/tuning_search_space.json \
  --num_tuning_trials=$NUM_TRIALS \
  --logging_dir=$LOG_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/coil100/early_stop_config.json \
  --cp_dir=$LOG_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --extra_metadata="coil100.target_value=$TARGET_VALUE" \
  --extra_metadata='coil100.aug_fliplr=0.5' \
  --extra_metadata='cifar10.aug_width_shift_max=13' \
  --extra_metadata='cifar10.aug_height_shift_max=13' \
  --augments=$AUG_CONFIG \

python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

LOG_DIR="$SAVE_DIR/no_aug"
mkdir -p $LOG_DIR
python submission_runner.py \
  --framework=jax \
  --workload=coil100_jax \
  --data_dir=algorithmic_efficiency/workloads/coil100/coil100_train_test_split.pkl \
  --submission_path=baselines/coil100/coil100_jax/submission.py \
  --tuning_search_space=baselines/coil100/tuning_search_space.json \
  --num_tuning_trials=$NUM_TRIALS \
  --logging_dir=$LOG_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/coil100/early_stop_config.json \
  --cp_dir=$LOG_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --extra_metadata="coil100.target_value=$TARGET_VALUE" \

python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

LOG_DIR="$SAVE_DIR/no_aug"




