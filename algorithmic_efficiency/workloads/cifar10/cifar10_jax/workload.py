"""MNIST workload implemented in Jax."""

from typing import Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.cifar10.workload import CIFAR10
from absl import flags
from algorithmic_efficiency.augmentation import ImageAugmenter
from absl import logging

FLAGS = flags.FLAGS

class VGGblock(nn.Module):
  'A VGG Block'

  num_filters: int

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=self.num_filters, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=self.num_filters, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    return x

class _Model(nn.Module):

  num_classes: 10

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool):
    del train
    x = VGGblock(num_filters=32)(x)
    x = VGGblock(num_filters=64)(x)
    x = VGGblock(num_filters=128)(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=128)(x)
    x = nn.relu(x)
    x = nn.Dense(features=self.num_classes)(x)
    x = nn.log_softmax(x)
    return x

class CIFAR10Workload(CIFAR10):

  def __init__(self):
    self._eval_ds = None
    self._param_shapes = None
    self._model = _Model(num_classes=10)

  def _normalize(self, image):
    return tf.cast(image, tf.float32) / 255.0

  def _build_dataset(self,
                     data_rng: jax.random.PRNGKey,
                     split: str,
                     data_dir: str,
                     batch_size):

    split_pct = split
    if FLAGS.percent_data_selection < 100 and split == 'train':
      split_pct = split + '[{pct}%:]'.format(pct=FLAGS.percent_data_selection)
      
    ds = tfds.load('cifar10', split=split_pct, shuffle_files=True)
    ds = ds.cache()
    ds = ds.map(lambda x: (self._normalize(x['image']), x['label'], None))
    if split == 'train':
      ds = ds.shuffle(1024, seed=data_rng[0])
      ds = ds.repeat()

    # Must drop remainder so that batch size is not None for augmentations
    ds = ds.batch(batch_size, drop_remainder=True) 

    if FLAGS.augments is not None:
      logging.info('Augmenting data with: %s' % FLAGS.augments)
      data_rng, aug_rng = jax.random.split(data_rng)
      aug = ImageAugmenter(FLAGS.augments, rng=aug_rng)
      ds = ds.map(
        lambda im_batch, l_batch, m_batch:
        (aug.apply_augmentations(im_batch), l_batch, m_batch)
      ) # Apply augmentations to whole batch
    
    return tfds.as_numpy(ds)

  def build_input_queue(self,
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str,
                        batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))

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

  def preprocess_for_train(self,
                           selected_raw_input_batch: spec.Tensor,
                           selected_label_batch: spec.Tensor,
                           train_mean: spec.Tensor,
                           train_stddev: spec.Tensor,
                           rng: spec.RandomState) -> spec.Tensor:
    del rng
    return selected_raw_input_batch, selected_label_batch

  def preprocess_for_eval(self,
                          raw_input_batch: spec.Tensor,
                          raw_label_batch: spec.Tensor,
                          train_mean: spec.Tensor,
                          train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return raw_input_batch, raw_label_batch

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    init_val = jnp.ones((1, 32, 32, 3), jnp.float32)
    initial_params = self._model.init(rng, init_val, train=True)['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      initial_params)
    return initial_params, None

  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    if loss_type == spec.LossType.SOFTMAX_CROSS_ENTROPY:
      return jax.nn.softmax(logits_batch, axis=-1)
    if loss_type == spec.LossType.SIGMOID_CROSS_ENTROPY:
      return jax.nn.sigmoid(logits_batch)
    if loss_type == spec.LossType.MEAN_SQUARED_ERROR:
      return logits_batch

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    logits_batch = self._model.apply({'params': params},
                                     augmented_and_preprocessed_input_batch,
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

  def get_model_class(self):
    '''Return a target class representing the workload model. The object
    whose state is being updated during training.
    '''
    # Makes sense for this to be an abstract class in spec.py but I didn't 
    # want to enforce this method on other workloads and break things.
    # As an alternative, could enforce a self._model property to all workloads
    return self._model