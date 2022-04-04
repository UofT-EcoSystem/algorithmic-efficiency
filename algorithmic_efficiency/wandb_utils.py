import json
import subprocess
import wandb


def setup(config):
  wandb.init(config=config)
  config = wandb.config

def log(metrics):
  wandb.log(metrics)

def write_tuning_search_space(config):
  params = {
    "learning_rate": {"feasible_points": [config.learning_rate]}
  }
  filename = 'ogbg_tuning_search_space.json'
  with open(filename, 'r') as file:
    json.dump(params, file)
  return filename

def write_early_stopping(config):
  params = {
    "metric_name": "mean_average_precision",
    "min_delta": 0,
    "patience": 5,
    "min_steps": 100,
    "max_steps": 2000,
    "mode": "max",
    "baseline": None
  }
  filename = 'ogbg_early_stopping_config.json'
  with open(filename, 'r') as file:
    json.dump(params, file)
  return filename

def main():
  defaults = {
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'batch_size': 2048,
    'activation': 'relu',
    'latent_dim': 256,
    'hidden_dim': 256,
    'num_message_passing_step': 5,
    'dropout_rate': 0.1,
    'num_trials': '',
    'logging_dir': '',
    'eval_frequency_override': '',
    'early_stopping_config': '',
    'max_allowed_runtime_sec': '',
    'target_value': '',
  }
  wandb.init(config=defaults)
  config = wandb.config

  experiment_dir = f"{config.logging_dir}/activation_{config.activation}-latent_{config.latent_dim}-hidden_{config.hidden_dim}-num_message_{config.num_message_passing_step}-dropout_{config.dropout_rate}-batch_{config.batch_size}/"

  tuning_search_space = write_tuning_search_space(config)
  early_stopping_config = write_early_stopping(config)

  cmd = [
    'python3 submission_runner.py --framework=jax --workload=configurable_ogb_jax --submission_path=baselines/configurable_ogbg/ogbg_jax/submission.py',
    f'--tuning_search_space="{tuning_search_space}"',
    f'--early_stopping_config="{early_stopping_config}"',
    f'--num_tuning_trials="{config.num_trials}"',
    f'--logging_dir="{experiment_dir}"',
    f'--eval_frequency_override="{config.eval_frequency_override}"',
    f'--extra_metadata="ogbg_config.max_allowed_runtime_sec={config.max_allowed_runtime_sec}"',
    f'--extra_metadata="ogbg_config.target_value={config.target_value}"',
    f'--extra_metadata="ogbg_config.activation_fn={config.activation}"',
    f'--extra_metadata="ogbg_config.dropout_rate={config.dropout_rate}"',
    f'--extra_metadata="ogbg_config.latent_dim={config.latent_dim}"',
    f'--extra_metadata="ogbg_config.hidden_dims={config.hidden_dim}"',
    f'--extra_metadata="ogbg_config.num_message_passing_steps={config.num_message_passing_step}"',
    f'--extra_metadata="ogbg_config.batch_size={config.batch_size}"',
  ]
  print(cmd)
  # subprocess.run(cmd)

if __name__ == '__main__':
  main()
