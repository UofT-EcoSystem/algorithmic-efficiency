#! /bin/bash

# Usage:
# Must be located in main dir of algorithmic-efficiency project
# sh experiments/augmentation/cifar10_baseline.sh path/to/save/dir

SAVE_DIR=$1
TARGET_VALUE=1.0

if [ -d $SAVE_DIR ]; then
  echo "save_dir: ${SAVE_DIR} already exists"
  exit 1
fi

mkdir -p $SAVE_DIR

LOG_DIR="$SAVE_DIR/tex_aug"
mkdir -p $LOG_DIR
AUG_CONFIG=experiments/augmentation/cifar_texture_augments.json
python submission_runner.py \
  --framework=jax \
  --workload=cifar10_jax \
  --submission_path=baselines/cifar10/cifar10_jax/submission.py \
  --tuning_search_space=baselines/cifar10/tuning_search_space.json \
  --num_tuning_trials=5 \
  --logging_dir=$LOG_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/cifar10/early_stop_config.json \
  --cp_dir=$LOG_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --cp_max=101 \
  --extra_metadata="cifar10.target_value=$TARGET_VALUE" \
  --augments=$AUG_CONFIG \

python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

LOG_DIR="$SAVE_DIR/all_aug"
mkdir -p $LOG_DIR
AUG_CONFIG=experiments/augmentation/cifar_all_augments.json
python submission_runner.py \
  --framework=jax \
  --workload=cifar10_jax \
  --submission_path=baselines/cifar10/cifar10_jax/submission.py \
  --tuning_search_space=baselines/cifar10/tuning_search_space.json \
  --num_tuning_trials=5 \
  --logging_dir=$LOG_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/cifar10/early_stop_config.json \
  --cp_dir=$LOG_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --cp_max=101 \
  --extra_metadata="cifar10.target_value=$TARGET_VALUE" \
  --augments=$AUG_CONFIG \

python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

LOG_DIR="$SAVE_DIR/geo_aug"
mkdir -p $LOG_DIR
AUG_CONFIG=experiments/augmentation/cifar_augments.json
python submission_runner.py \
  --framework=jax \
  --workload=cifar10_jax \
  --submission_path=baselines/cifar10/cifar10_jax/submission.py \
  --tuning_search_space=baselines/cifar10/tuning_search_space.json \
  --num_tuning_trials=5 \
  --logging_dir=$LOG_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/cifar10/early_stop_config.json \
  --cp_dir=$LOG_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --cp_max=101 \
  --extra_metadata="cifar10.target_value=$TARGET_VALUE" \
  --augments=$AUG_CONFIG \

python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"

LOG_DIR="$SAVE_DIR/no_aug"
mkdir -p $LOG_DIR
python submission_runner.py \
  --framework=jax \
  --workload=cifar10_jax \
  --submission_path=baselines/cifar10/cifar10_jax/submission.py \
  --tuning_search_space=baselines/cifar10/tuning_search_space.json \
  --num_tuning_trials=5 \
  --logging_dir=$LOG_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/cifar10/early_stop_config.json \
  --cp_dir=$LOG_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --cp_max=101 \
  --extra_metadata="cifar10.target_value=$TARGET_VALUE" \

python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOG_DIR')"