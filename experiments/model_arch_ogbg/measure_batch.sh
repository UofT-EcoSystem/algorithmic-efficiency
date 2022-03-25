# This script will run a graph DNN training workload and save measurements as CSV and metadata as JSON.
#
# Author: Daniel Snider <danielsnider12@gmail.com>
#
# Usage:
# kill %; bash ./experiments/model_arch_ogbg/measure_batch.sh 2>&1 | tee -a ./experiments/model_arch_ogbg/logs/console_output.log

set -e # exit on error

# ECO-12
LOGGING_DIR='./experiments/model_arch_ogbg/logs'
# rm -rf $LOGGING_DIR
mkdir -p $LOGGING_DIR

ACTIVATIONS='relu'
LATENT_DIMS='128 256'
HIDDEN_DIMS='128 256'
NUM_MESSAGE_PASSING_STEPS='3 5'
DROPOUT_RATES='0.1'
BATCH_SIZES='128 256 512 1024 2048 4096 8192'
OPTIMIZER='adam'

EVAL_FREQUENCY_OVERRIDE='100 step'
TARGET_VALUE='0.1'
NUM_TRIALS='3'

HYPERPARAM_CONFIG='adam_ogbg_tuning_search_space.json'
cat <<EOF > adam_ogbg_tuning_search_space.json
{
  "learning_rate": {"feasible_points": [1e-3]}
}
EOF
# "learning_rate": {"min": 1e-4, "max": 1e-2, "scaling": "log"}, # ADAM MNIST 

MAX_ALLOWED_RUNTIME_SEC='3600'
EARLY_STOPPING_CONFIG='ogbg_early_stopping_config.json'
cat <<EOF > ogbg_early_stopping_config.json
{
  "metric_name": "mean_average_precision",
  "min_delta": 0,
  "patience": 5,
  "min_steps": 100,
  "max_steps": 2000,
  "mode": "max",
  "baseline": null
}
EOF


# Count number of experiments so that we can track progress
activation_count=$( awk -F" " '{print NF-1}' <<<"${ACTIVATIONS}" )
activation_count=$(echo $(( activation_count + 1 )))
LATENT_DIMS_count=$( awk -F" " '{print NF-1}' <<<"${LATENT_DIMS}" )
LATENT_DIMS_count=$(echo $(( LATENT_DIMS_count + 1 )))
HIDDEN_DIMS_count=$( awk -F" " '{print NF-1}' <<<"${HIDDEN_DIMS}" )
HIDDEN_DIMS_count=$(echo $(( HIDDEN_DIMS_count + 1 )))
BATCH_SIZES_count=$( awk -F" " '{print NF-1}' <<<"${BATCH_SIZES}" )
BATCH_SIZES_count=$(echo $(( BATCH_SIZES_count + 1 )))
NUM_MESSAGE_PASSING_STEPS_count=$( awk -F" " '{print NF-1}' <<<"${NUM_MESSAGE_PASSING_STEPS}" )
NUM_MESSAGE_PASSING_STEPS_count=$(echo $(( NUM_MESSAGE_PASSING_STEPS_count + 1 )))
total=$(echo $((activation_count * LATENT_DIMS_count * HIDDEN_DIMS_count * NUM_MESSAGE_PASSING_STEPS_count * BATCH_SIZES_count)))
iteration=1


run_cmd () {
  set -x
  python3 submission_runner.py --framework=jax --workload=configurable_ogb_jax --submission_path=baselines/configurable_ogbg/ogbg_jax/submission.py \
    --tuning_search_space="$HYPERPARAM_CONFIG" \
    --num_tuning_trials="$NUM_TRIALS" \
    --logging_dir="$EXPERIMENT_DIR" \
    --eval_frequency_override="$EVAL_FREQUENCY_OVERRIDE" \
    --early_stopping_config="$EARLY_STOPPING_CONFIG" \
    --extra_metadata="ogbg_config.max_allowed_runtime_sec=$MAX_ALLOWED_RUNTIME_SEC" \
    --extra_metadata="ogbg_config.target_value=$TARGET_VALUE" \
    --extra_metadata="ogbg_config.activation_fn=$ACTIVATION" \
    --extra_metadata="ogbg_config.dropout_rate=$DROPOUT_RATE" \
    --extra_metadata="ogbg_config.latent_dim=$LATENT_DIM" \
    --extra_metadata="ogbg_config.hidden_dims=$HIDDEN_DIM" \
    --extra_metadata="ogbg_config.num_message_passing_steps=$NUM_MESSAGE_PASSING_STEP" \
    --extra_metadata="ogbg_config.batch_size=$BATCH_SIZE" \
    --extra_metadata="ogbg_config.optimizer=$OPTIMIZER"
  set +x
}

for ACTIVATION in $ACTIVATIONS
do
  for LATENT_DIM in $LATENT_DIMS
  do
    for HIDDEN_DIM in $HIDDEN_DIMS
    do
      for NUM_MESSAGE_PASSING_STEP in $NUM_MESSAGE_PASSING_STEPS
      do
        for DROPOUT_RATE in $DROPOUT_RATES
        do
          for BATCH_SIZE in $BATCH_SIZES
          do
            echo -e "\n[INFO $(date +"%d-%I:%M%p")] Starting iteration $iteration of $total."
            iteration=$(echo $(( iteration + 1 )))
            EXPERIMENT_DIR="$LOGGING_DIR/activation_$ACTIVATION-latent_$LATENT_DIM-hidden_$HIDDEN_DIM-num_message_$NUM_MESSAGE_PASSING_STEP-dropout_$DROPOUT_RATE-batch_$BATCH_SIZE/"
            # if [ "$iteration" -lt "99973" ]; then
            #   echo 'Skipping...'
            #   continue
            # fi
            mkdir -p $EXPERIMENT_DIR
            run_cmd
          done
        done
      done
    done
  done
done


# Combine all tuning trials into one CSV
# python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOGGING_DIR')"

echo "[INFO $(date +"%d-%I:%M%p")] Generated files:"
find $LOGGING_DIR

# # Check status of each experiment (requires zsh)
# for FILE in ./experiments/model_arch_ogbg/logs/**/*.json
# do
#   STATUS=$(cat $FILE | jq -r '.status')
#   echo "$STATUS $FILE"
# done

echo "[INFO $(date +"%d-%I:%M%p")] Finished."

exit 0

# # Check status of each experiment (requires zsh)
for FILE in ./experiments/model_arch_ogbg/logs*/**/trial_*/*.json
do
    STATUS=$(cat $FILE | jq -r '.status')
    mean_average_precision=$(cat $FILE | jq -r '.mean_average_precision')
    global_step=$(cat $FILE | jq -r '.global_step')
    goal_reached=$(cat $FILE | jq -r '.goal_reached')
    is_time_remaining=$(cat $FILE | jq -r '.is_time_remaining')
    early_stop=$(cat $FILE | jq -r '.early_stop')
    training_complete=$(cat $FILE | jq -r '.training_complete')

    echo -e "$FILE \n  $STATUS mean_average_precision=$mean_average_precision global_step=$global_step goal_reached=$goal_reached is_time_remaining=$is_time_remaining early_stop=$early_stop training_complete=$training_complete \n"
done


# rebuild best parameters csv
echo "architecture,batch_size,step_to_threshold" > gnn_out.csv
for FILE in ./experiments/model_arch_ogbg/logs*/**/trial_**/*.json
do
    JSON=$(cat $FILE)
    # architecture=$(echo $JSON | jq -r .'"extra.batch_science.architecture"')
    batch_size=$(echo $JSON | jq -r .'"batch_size"')
    step_to_threshold=$(echo $JSON | jq -r .'"global_step"')
    latent_dim=$(echo $JSON | jq -r .'"extra.ogbg_config.latent_dim"')
    hidden_dims=$(echo $JSON | jq -r .'"extra.ogbg_config.hidden_dims"')
    num_message_passing_steps=$(echo $JSON | jq -r .'"extra.ogbg_config.num_message_passing_steps"')
    ARCH="latent$latent_dim-hidden$hidden_dims-pass$num_message_passing_steps"
    echo "$ARCH,$batch_size,$step_to_threshold" | tee -a gnn_out.csv
done


python3 experiments/mnist_batch_size/plot_batch.py gnn_out.csv plot_gnn_batch.png