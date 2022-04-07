import signal
import wandb

from absl import flags
from algorithmic_efficiency.logging_utils import _get_extra_metadata_as_dict

FLAGS = flags.FLAGS


def signal_handler(signum, frame):
  # https://docs.wandb.ai/guides/track/advanced/resuming#preemptible-sweeps
  wandb.mark_preempting()

def setup(config=None):
  run = wandb.init(config=config, reinit=True)
  config = wandb.config
  signal.signal(signal.SIGTERM, signal_handler)

  extra_metadata = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
  name = extra_metadata.get('wandb.name', None)
  if name:
    wandb.run.name = str(name)
    wandb.run.save()

  return run

def log(metrics):
  wandb.log(metrics)

