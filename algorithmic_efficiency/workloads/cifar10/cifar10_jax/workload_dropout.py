"""MNIST workload implemented in Jax."""

from typing import Tuple

from flax import linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.cifar10.cifar10_jax.workload import CIFAR10Workload, VGGblock
from absl import flags
from algorithmic_efficiency.augmentation import ImageAugmenter
from absl import logging

FLAGS = flags.FLAGS

class _Model(nn.Module):

  num_classes: 10

  @nn.compact
  def __call__(self, x: spec.Tensor, train: bool):
    x = VGGblock(num_filters=32)(x)
    x = nn.Dropout(rate=0.2)(x, deterministic = not train)
    x = VGGblock(num_filters=64)(x)
    x = nn.Dropout(rate=0.2)(x, deterministic = not train)
    x = VGGblock(num_filters=128)(x)
    x = nn.Dropout(rate=0.2)(x, deterministic = not train)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=128)(x)
    x = nn.relu(x)
    x = nn.Dropout(rate=0.2)(x, deterministic = not train)
    x = nn.Dense(features=self.num_classes)(x)
    x = nn.log_softmax(x)
    return x

class Dropout_CIFAR10Workload(CIFAR10Workload):
  def __init__(self):
    self._eval_ds = None
    self._param_shapes = None
    self._model = _Model(num_classes=10)

  def _normalize(self, image):
    return (tf.cast(image, tf.float32) - self.train_mean) / self.train_stddev

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    init_val = jnp.ones((1, 32, 32, 3), jnp.float32)
    param_rng, dropout_rng = jax.random.split(rng)
    initial_params = self._model.init({'params': param_rng, 'dropout': dropout_rng}, init_val, train=True)['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      initial_params)
    return initial_params, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del update_batch_norm
    train = mode == spec.ForwardPassMode.TRAIN
    if train:
      logits_batch = self._model.apply({'params': params},
                                      augmented_and_preprocessed_input_batch,
                                      train=train,
                                      rngs={'dropout': rng})
    else:
      del rng
      logits_batch = self._model.apply({'params': params},
                                      augmented_and_preprocessed_input_batch,
                                      train=train)
    return logits_batch, None