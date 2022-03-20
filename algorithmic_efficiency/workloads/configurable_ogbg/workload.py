from absl import flags

from algorithmic_efficiency import spec
from algorithmic_efficiency.logging_utils import _get_extra_metadata_as_dict

FLAGS = flags.FLAGS


class OGBG(spec.Workload):

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['mean_average_precision'] > self.target_value

  @property
  def target_value(self):
    extra_metadata = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
    target_value = float(extra_metadata.get('extra.ogbg_config.target_value', 0.24))
    # From Flax example
    # https://tensorboard.dev/experiment/AAJqfvgSRJaA1MBkc0jMWQ/#scalars.
    return target_value

  @property
  def loss_type(self):
    return spec.LossType.SIGMOID_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 350343

  @property
  def num_eval_train_examples(self):
    return 350343

  @property
  def num_validation_examples(self):
    return 43793

  @property
  def train_mean(self):
    raise NotImplementedError

  @property
  def train_stddev(self):
    raise NotImplementedError

  @property
  def max_allowed_runtime_sec(self):
    return 10000  # 3h non-default

  @property
  def eval_period_time_sec(self):
    return 60 # non-default
