#! /bin/bash

# Usage:
# Must be located in main dir of algorithmic-efficiency project
# sh experiments/augmentation/aug_test_cifar10.sh path/to/save/dir

SAVE_DIR=$1
TARGET_VALUE=1.0
NUM_TRIALS=2
AUG_CONFIG=experiments/augmentation/cifar_augments.json

if [ -d $SAVE_DIR ]; then
  echo "save_dir: ${SAVE_DIR} already exists"
  exit 1
fi

mkdir -p $SAVE_DIR

python submission_runner.py \
  --framework=jax \
  --workload=cifar10_jax \
  --submission_path=baselines/cifar10/cifar10_jax/submission.py \
  --tuning_search_space=baselines/cifar10/tuning_search_space.json \
  --num_tuning_trials=$NUM_TRIALS \
  --logging_dir=$SAVE_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/cifar10/early_stop_config.json \
  --cp_dir=$SAVE_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --extra_metadata="cifar10.target_value=$TARGET_VALUE" \
  --augments=$AUG_CONFIG \


