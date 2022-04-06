import signal
import wandb

def signal_handler(signum, frame):
  # https://docs.wandb.ai/guides/track/advanced/resuming#preemptible-sweeps
  wandb.mark_preempting()

def setup(config=None, name=''):
  run = wandb.init(config=config, reinit=True)
  config = wandb.config
  signal.signal(signal.SIGTERM, signal_handler)
  if name:
    run_name = wandb.run.name + '_' + str(name)
    wandb.run.name = run_name
    wandb.run.save()
  return run

def log(metrics):
  wandb.log(metrics)

