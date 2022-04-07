"""MNIST workload implemented in Jax."""

from typing import Tuple

from jax import lax
from absl import flags
from flax import linen as nn
import jax
import jax.numpy as jnp
from algorithmic_efficiency import random_utils as prng
import tensorflow as tf
import tensorflow_datasets as tfds

from algorithmic_efficiency import spec
from algorithmic_efficiency.logging_utils import _get_extra_metadata_as_dict

FLAGS = flags.FLAGS
class _Model(nn.Module):
  def setup(self):
    extra_metadata = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
    self._target_value = float(
        extra_metadata.get('extra.mnist_config.target_value', None))
    self._max_allowed_runtime_sec = int(
        extra_metadata.get('extra.mnist_config.max_allowed_runtime_sec', None))
    activitation_fn_map = {
        'relu': jax.nn.relu,
        'sigmoid': jax.nn.sigmoid,
        'hard_tanh': jax.nn.hard_tanh,
        'gelu': jax.nn.gelu
    }
    self.activation_fn = activitation_fn_map[extra_metadata.get(
        'extra.mnist_config.activation_fn', 'relu')]
    self.model_width = int(extra_metadata.get('extra.mnist_config.model_width', 128))
    self.model_depth = int(extra_metadata.get('extra.mnist_config.model_depth', 1))
    self.dropout_rate = float(
        extra_metadata.get('extra.mnist_config.dropout_rate', 0))
    self.batch_size = int(
        extra_metadata.get('extra.mnist_config.batch_size', None))
    self.optimizer = extra_metadata.get('extra.mnist_config.optimizer', None)
    self.batch_norm = extra_metadata.get('extra.mnist_config.batch_norm', 'off')

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool):

    input_size = 28 * 28
    num_classes = 10
    x = x.reshape((x.shape[0], input_size))  # Flatten.
    for _ in range(self.model_depth):
      if self.batch_norm == 'off':
        x = nn.Dense(features=self.model_width, use_bias=True)(x)
        x = self.activation_fn(x)
      elif self.batch_norm == 'affine-activation-batchnorm':
        x = nn.Dense(features=self.model_width, use_bias=True)(x)
        x = self.activation_fn(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.99)(x)
      elif self.batch_norm == 'affine-batchnorm-activation':
        x = nn.Dense(features=self.model_width, use_bias=True)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.99)(x)
        x = self.activation_fn(x)
    x = nn.Dense(features=num_classes, use_bias=True)(x)
    x = nn.log_softmax(x)
    return x

class MnistWorkload(spec.Workload):

  def __init__(self):
    self._eval_ds = None
    self._param_shapes = None
    extra_metadata = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
    self._target_value = float(
        extra_metadata.get('extra.mnist_config.target_value', None))
    self._max_allowed_runtime_sec = int(
        extra_metadata.get('extra.mnist_config.max_allowed_runtime_sec', None))
    activitation_fn_map = {
        'relu': jax.nn.relu,
        'sigmoid': jax.nn.sigmoid,
        'hard_tanh': jax.nn.hard_tanh,
        'gelu': jax.nn.gelu
    }
    self.dropout_rate = float(
        extra_metadata.get('extra.mnist_config.dropout_rate', 0))
    self.batch_size = int(
        extra_metadata.get('extra.mnist_config.batch_size', None))
    self.optimizer = extra_metadata.get('extra.mnist_config.optimizer', None)
    self.batch_norm = extra_metadata.get('extra.mnist_config.batch_norm', 'off')

    self._model = _Model()

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['accuracy'] > self.target_value

  @property
  def target_value(self):
    if self._target_value:
      return self._target_value
    return 0.9

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 60000

  @property
  def num_eval_train_examples(self):
    return 60000

  @property
  def num_validation_examples(self):
    return 10000

  @property
  def train_mean(self):
    return 0.1307

  @property
  def train_stddev(self):
    return 0.3081

  @property
  def max_allowed_runtime_sec(self):
    if self._max_allowed_runtime_sec:
      return self._max_allowed_runtime_sec
    return 900

  @property
  def eval_period_time_sec(self):
    return 60

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  def model_params_types(self):
    pass

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def eval_model(self, params: spec.ParameterContainer,
                 model_state: spec.ModelAuxiliaryState, rng: spec.RandomState,
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

  def _normalize(self, image):
    return (tf.cast(image, tf.float32) - self.train_mean) / self.train_stddev

  def _build_dataset(self, data_rng: jax.random.PRNGKey, split: str,
                     data_dir: str, batch_size):
    ds = tfds.load('mnist', split=split)
    ds = ds.cache()
    ds = ds.map(lambda x: (self._normalize(x['image']), x['label'], None))
    if split == 'train':
      ds = ds.shuffle(1024, seed=data_rng[0])
      ds = ds.repeat()
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)

  def build_input_queue(self, data_rng: jax.random.PRNGKey, split: str,
                        data_dir: str, batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))

  def preprocess_for_train(self, selected_raw_input_batch: spec.Tensor,
                           selected_label_batch: spec.Tensor,
                           train_mean: spec.Tensor, train_stddev: spec.Tensor,
                           rng: spec.RandomState) -> spec.Tensor:
    del rng
    return selected_raw_input_batch, selected_label_batch

  def preprocess_for_eval(self, raw_input_batch: spec.Tensor,
                          raw_label_batch: spec.Tensor, train_mean: spec.Tensor,
                          train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return raw_input_batch, raw_label_batch

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
    variables = self._model.init(rng, init_val, train=True)
    model_state, initial_params = variables.pop('params')
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      initial_params)
    return initial_params, model_state

  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(self, logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    if loss_type == spec.LossType.SOFTMAX_CROSS_ENTROPY:
      return jax.nn.softmax(logits_batch, axis=-1)
    if loss_type == spec.LossType.SIGMOID_CROSS_ENTROPY:
      return jax.nn.sigmoid(logits_batch)
    if loss_type == spec.LossType.MEAN_SQUARED_ERROR:
      return logits_batch

  def sync_batch_stats(self, model_state):
    """Sync the batch statistics across replicas."""
    # An axis_name is passed to pmap which can then be used by pmean.
    # In this case each device has its own version of the batch statistics and
    # we average them.
    if self.batch_norm == 'off':
      return model_state
    avg_fn = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    new_model_state = model_state.copy(
        {'batch_stats': avg_fn(model_state['batch_stats'])})
    return new_model_state


  def model_fn(
      self, params: spec.ParameterContainer, input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState, mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del rng
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    if train:
      logits_batch, new_model_state = self._model.apply({'params': params, **model_state},
                                      input_batch,
                                      mutable=['batch_stats'],
                                      train=train)
      return logits_batch, new_model_state
    else:
      logits_batch = self._model.apply({'params': params, **model_state},
                                      input_batch,
                                      mutable=False,
                                      train=train)
      return logits_batch, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self, label_batch: spec.Tensor,
              logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    one_hot_targets = jax.nn.one_hot(label_batch, 10)
    return -jnp.sum(one_hot_targets * nn.log_softmax(logits_batch), axis=-1)

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    # not accuracy, but nr. of correct predictions
    accuracy = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
    loss = jnp.sum(self.loss_fn(labels, logits))
    n_data = len(logits)
    return {'accuracy': accuracy, 'loss': loss, 'n_data': n_data}
