#! /bin/bash

# Usage:
# Must be located in main dir of algorithmic-efficiency project
# sh experiments/augmentation/cifar10_custom.sh path/to/save/dir

SAVE_DIR=$1
TARGET_VALUE=1.0

if [ -d $SAVE_DIR ]; then
  echo "save_dir: ${SAVE_DIR} already exists"
  exit 1
fi

mkdir -p $SAVE_DIR

python submission_runner.py \
  --framework=jax \
  --workload=cifar10_jax_custom \
  --submission_path=baselines/cifar10/cifar10_jax/submission.py \
  --tuning_search_space=baselines/cifar10/tuning_search_space.json \
  --num_tuning_trials=5 \
  --logging_dir=$SAVE_DIR \
  --eval_frequency_override="1 epoch" \
  --early_stopping_config=baselines/cifar10/early_stop_config.json \
  --cp_dir=$SAVE_DIR \
  --console_verbosity=1 \
  --cp_step="epoch" \
  --cp_freq=1 \
  --extra_metadata="cifar10.target_value=$TARGET_VALUE" \
  --extra_metadata='cifar10_custom.aug_fliplr=0.5' \
  --extra_metadata='cifar10_custom.aug_width_shift_max=2' \
  --extra_metadata='cifar10_custom.aug_height_shift_max=2' \
  --data_dir=baselines/cifar10/augmented_datasets/cifar10_aug1.pkl \


