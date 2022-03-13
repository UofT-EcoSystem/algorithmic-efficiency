import json

from absl import logging
import numpy as np


class EarlyStopping:
  """Stop training early if not improving."""

  def __init__(self, config):
    if config and type(config) == dict:
      self.enabled = True
    elif config and type(config) == str:
      with open(config, 'r') as file:
        config = json.load(file)
        self.enabled = True
    else:
      config = {}
      self.enabled = False

    self.metric_name = config.get('metric_name', None)
    self.min_delta = config.get('min_delta', 0)
    self.patience = config.get('patience', 0)
    self.min_steps = config.get('min_steps', 0)
    self.max_steps = config.get('max_steps', None)
    self.mode = config.get('mode', 'min')
    self.baseline_score = config.get('baseline', None)
    self.no_change_count = 0

    try:
      assert (self.mode in ['min', 'max'])
    except:
      logging.error(
          'Failed to parse early_stopping config. Please check "mode" setting.')
      raise
    if self.mode == 'min':
      self.compare_fn = lambda a, b: np.less(a, b - self.min_delta)
      self.best_score = np.Inf
    elif self.mode == 'max':
      self.compare_fn = lambda a, b: np.greater(a, b + self.min_delta)
      self.best_score = -np.Inf
    if self.baseline_score:
      self.best_score = self.baseline_score

  def early_stop_check(self, metrics: dict, step_count: int):
    """Returns True if it is time to stop."""
    if not self.enabled:
      return False
    if self.max_steps and step_count > self.max_steps:
      return True

    current_score = metrics[self.metric_name]

    if self.compare_fn(current_score, self.best_score):
      self.best_score = current_score
      self.no_change_count = 0
      return False
    else:
      if self.no_change_count >= self.patience:
        if step_count >= self.min_steps:
          return True
        else:
          return False
      else:
        self.no_change_count += 1
        return False
