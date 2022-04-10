from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.logging_utils import _get_extra_metadata_as_dict
from absl import flags
FLAGS = flags.FLAGS


class CIFAR10(spec.Workload):

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['accuracy'] > self.target_value

  @property
  def target_value(self):
    if FLAGS.extra_metadata is not None:
      meta = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
      return float(meta.get('extra.cifar10.target_value', 0.65))
    else:
      return 0.65

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    p = float(float(FLAGS.percent_data_selection) / 100)
    return int(p*50000)

  @property
  def num_eval_train_examples(self):
    p = float(float(FLAGS.percent_data_selection) / 100)
    return int(p*50000)

  @property
  def num_validation_examples(self):
    return 10000

  @property
  def train_mean(self):
    return [0.49139968 * 255.0, 0.48215827 * 255.0, 0.44653124 * 255.0]

  @property
  def train_stddev(self):
    return [0.24703233 * 255.0, 0.24348505 * 255.0, 0.26158768 * 255.0]

  @property
  def max_allowed_runtime_sec(self):
    return 3600

  @property
  def eval_period_time_sec(self):
    return 30

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    raise NotImplementedError

  def eval_model(self,
                 params: spec.ParameterContainer,
                 model_state: spec.ModelAuxiliaryState,
                 rng: spec.RandomState,
                 data_dir: str):
    """Run a full evaluation of the model."""
    data_rng, model_rng = prng.split(rng, 2)
    eval_batch_size = 2000
    self._eval_ds = self.build_input_queue(
        data_rng, 'test', data_dir, batch_size=eval_batch_size)

    total_metrics = {
        'accuracy': 0.,
        'loss': 0.,
    }
    n_data = 0
    for (images, labels, _) in self._eval_ds:
      images, labels = self.preprocess_for_eval(images, labels, None, None)
      logits, _ = self.model_fn(
          params,
          images,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          update_batch_norm=False)
      # TODO(znado): add additional eval metrics?
      batch_metrics = self._eval_metric(logits, labels)
      total_metrics = {
          k: v + batch_metrics[k] for k, v in total_metrics.items()
      }
      n_data += batch_metrics['n_data']
    return {k: float(v / n_data) for k, v in total_metrics.items()}
