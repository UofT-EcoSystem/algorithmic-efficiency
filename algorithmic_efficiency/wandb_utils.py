import datetime
import sys
import json
import subprocess
import wandb


def setup(config=None, name=''):
  run = wandb.init(config=config, reinit=True)
  config = wandb.config
  if name:
    run_name = wandb.run.name + '_' + str(name)
    wandb.run.name = run_name
    wandb.run.save()
  return run

def log(metrics):
  wandb.log(metrics)

def write_tuning_search_space(config):
  params = {
    "learning_rate": {"feasible_points": [config["learning_rate"]]}
  }
  filename = 'ogbg_tuning_search_space.json'
  with open(filename, 'w') as file:
    json.dump(params, file)
  return filename

def write_early_stopping(config):
  params = {
    "metric_name": "mean_average_precision",
    "min_delta": 0,
    "patience": 5,
    "min_steps": 100,
    "max_steps": 6000,
    # "min_steps": 1,
    # "max_steps": 2,
    "mode": "max",
    "baseline": None
  }
  filename = 'ogbg_early_stopping_config.json'
  with open(filename, 'w') as file:
    json.dump(params, file)
  return filename

def main():
  config = {
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'batch_size': 2048,
    'activation': 'relu',
    'latent_dim': 256,
    'hidden_dim': 256,
    'num_message_passing_step': 5,
    'dropout_rate': 0.1,
    'num_trials': 1,
    'logging_dir': '/FS1/dans/mlc-gnn-logs',
    'eval_frequency_override': '100 step',
    # 'eval_frequency_override': '1 step',
    'max_allowed_runtime_sec': 7200,
    'target_value': 0.24,
    'layer_norm': True,
  }
  if len(sys.argv) > 1:
    overrides_file = sys.argv[1]
    with open(overrides_file, 'r') as f:
      overrides = json.load(f)
    config.update(overrides)

  time_now  = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
  experiment_dir = f'{config["logging_dir"]}/{time_now}-layer_norm_{config["layer_norm"]}-learning_rate{config["learning_rate"]}-activation_{config["activation"]}-latent_{config["latent_dim"]}-hidden_{config["hidden_dim"]}-num_message_{config["num_message_passing_step"]}-dropout_{config["dropout_rate"]}-batch_{config["batch_size"]}/'

  tuning_search_space = write_tuning_search_space(config)
  early_stopping_config = write_early_stopping(config)

  cmd = [
    'python3',
    'submission_runner.py',
    '--framework=jax',
    '--workload=configurable_ogb_jax',
    '--submission_path=baselines/configurable_ogbg/ogbg_jax/submission.py',
    '--enable_wandb',
    '--save_checkpoints',
    f'--tuning_search_space={tuning_search_space}',
    f'--early_stopping_config={early_stopping_config}',
    f'--num_tuning_trials={config["num_trials"]}',
    f'--logging_dir={experiment_dir}',
    f'--eval_frequency_override={config["eval_frequency_override"]}',
    f'--extra_metadata="ogbg_config.max_allowed_runtime_sec={config["max_allowed_runtime_sec"]}',
    f'--extra_metadata="ogbg_config.target_value={config["target_value"]}"',
    f'--extra_metadata="ogbg_config.activation_fn={config["activation"]}"',
    f'--extra_metadata="ogbg_config.dropout_rate={config["dropout_rate"]}"',
    f'--extra_metadata="ogbg_config.latent_dim={config["latent_dim"]}"',
    f'--extra_metadata="ogbg_config.hidden_dims={config["hidden_dim"]}"',
    f'--extra_metadata="ogbg_config.num_message_passing_steps={config["num_message_passing_step"]}"',
    f'--extra_metadata="ogbg_config.batch_size={config["batch_size"]}"',
    f'--extra_metadata="ogbg_config.layer_norm={config["layer_norm"]}"',
  ]
  print(" ".join(cmd))
  subprocess.run(cmd)

if __name__ == '__main__':
  main()
