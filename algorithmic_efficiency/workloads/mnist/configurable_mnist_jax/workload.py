"""MNIST workload implemented in Jax."""

from typing import Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from algorithmic_efficiency import spec

from absl import flags

FLAGS = flags.FLAGS


class _Model(nn.Module):

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool):
    del train
    num_hidden = 128
    num_layer = 1
    activation_fn = 'relu'
    input_size = 28 * 28
    num_classes = 10
    dropout = 0
    activitation_fn_map = {
      'relu': jax.nn.relu,
      'sigmoid': jax.nn.sigmoid,
      'hard_tanh': jax.nn.hard_tanh,
      'gelu': jax.nn.gelu
    }
    activation_fn = activitation_fn_map[activation_fn]

    x = x.reshape((x.shape[0], input_size))  # Flatten.
    for _ in range(num_layer - 1):
      x = nn.Dense(features=num_hidden, use_bias=True)(x)
      x = activation_fn(x)
    x = nn.Dense(features=num_classes, use_bias=True)(x)
    x = nn.log_softmax(x)
    return x


class MnistWorkload(spec.Workload):

  def __init__(self):
    self._eval_ds = None
    self._param_shapes = None
    num_hidden = 128
    num_layer = 1
    activation_fn = 'relu'
    input_size = 28 * 28
    num_classes = 10
    dropout = 0
    self._model = _Model()

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result['accuracy'] > self.target_value

  @property
  def target_value(self):
    if 'target_value' in FLAGS and FLAGS.target_value:
      return FLAGS.target_value
    return 0.9

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def num_train_examples(self):
    return 60000

  @property
  def num_eval_examples(self):
    return 10000

  @property
  def train_mean(self):
    return 0.1307

  @property
  def train_stddev(self):
    return 0.3081

  @property
  def max_allowed_runtime_sec(self):
    return 900

  @property
  def eval_period_time_sec(self):
    return 60

  def _eval_metric(self, logits, labels):
    """Return the mean accuracy and loss as a dict."""
    raise NotImplementedError

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
    if 'batch_size' in FLAGS and FLAGS.batch_size:
      eval_batch_size = FLAGS.batch_size
    self._eval_ds = self.build_input_queue(
        data_rng, 'test', data_dir, batch_size=eval_batch_size)

    total_metrics = {
        'accuracy': 0.,
        'loss': 0.,
    }
    n_data = 0
    dropout_rate = 0.4
    params2 = params.unfreeze()
    for key in params.keys():
      params2[key]["kernel"] = params[key]["kernel"] * (1.0 - dropout_rate)
    params = flax.core.frozen_dict.freeze(params2)
    for (images, labels) in self._eval_ds:
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
    ds = ds.map(lambda x: (self._normalize(x['image']), x['label']))
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
    initial_params = self._model.init(rng, init_val, train=True)['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      initial_params)
    return initial_params, None

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

  def model_fn(
      self, params: spec.ParameterContainer, input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState, mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    logits_batch = self._model.apply({'params': params},
                                     input_batch,
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
