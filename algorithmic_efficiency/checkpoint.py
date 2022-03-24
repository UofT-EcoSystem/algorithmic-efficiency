import dill
from flax.training import checkpoints
import os
from absl import logging
from algorithmic_efficiency import spec
from typing import Optional


def save_checkpoint(
          params,
          model_state,
          workload: spec.Workload,
          output_dir: str,
          step: int,
          trial_idx: int,
          prefix: str = 'checkpoint_', keep: int = float('inf'), overwrite: bool = True, keep_every_n_steps: Optional[int] = None):
  '''Save a checkpoint of the model.

  Attempts to be pre-emption safe by writing to temporary before
  a final rename and cleanup of past files.

  Docs: https://flax.readthedocs.io/en/latest/flax.training.html

  Args:
    params: The model parameters. Usually a dictionary.
    model_state: The model state. Usually a dictionary.
    workload: The workload class.
    output_dir: str: directory to save to.
    step: int or float: training step number or other metric number.
    prefix: str: checkpoint file name prefix.
    keep: number of past checkpoint files to keep.
    overwrite: overwrite existing checkpoint files if a checkpoint
    at the current or a later step already exits (default: False).
    keep_every_n_steps: if defined, keep every checkpoints every n steps (in
    addition to keeping the last 'keep' checkpoints).
  Returns:
    Filename of saved checkpoint.
  '''
  if not output_dir:
    return

  # Create output folder
  save_path = os.path.join(output_dir, f'trial_{trial_idx}')
  os.makedirs(save_path, exist_ok=True)

  # Save model
  model = getattr(workload, '_model', None)
  model_path = os.path.join(save_path, 'model.pkl')
  if model and not os.path.isfile(model_path):
    # model doesn't change so only write once
    with open(model_path, 'wb') as f:
      dill.dump(workload._model, f)

  # Safely transform params
  param_dict = None
  try:
    param_dict = dict(params)
  except Exception:
    to_dict = getattr(params, "state_dict", None)
    if callable(to_dict):
      param_dict = to_dict()
    else:
      logging.warn('Could not convert model params to a dict. Checkpoint not saved')
      return

  # Save checkpoint
  checkpoint = {'params': param_dict, 'model_state': model_state}
  checkpoint_path = checkpoints.save_checkpoint(
    ckpt_dir=save_path,
    target=checkpoint,
    step=step,
    prefix=prefix,
    keep=keep,
    overwrite=overwrite,
    keep_every_n_steps=keep_every_n_steps
    )

  return checkpoint_path


def load_checkpoint(ckpt_dir, prefix='', step=None, target=None):
  '''Load a checkpoint from a checkpoint file. The checkpoints are
  dictionaries with the model parameters and model state.

  Args:
    ckpt_dir: The path to the directory containing the checkpoints
    prefix: Prefix used for the checkpoint file. Used to differentiate
      different tuning runs.
    step: Which checkpoint file to load. By default loads the most recent
    target: matching object to rebuild via deserialized state-dict. If None,
      the deserialized state-dict is returned as-is.

  Returns:
    checkpoint (dict): A dictionary containing 'params' and 'model_state'
  '''

  checkpoint = checkpoints.restore_checkpoint(
    ckpt_dir=ckpt_dir,
    target=target,
    step=step,
    prefix=prefix,
  )
  if checkpoint is None:
    logging.warn(f'Could not find file containing prefix {prefix}, in directory'
                 + f' {ckpt_dir}')

  return checkpoint
