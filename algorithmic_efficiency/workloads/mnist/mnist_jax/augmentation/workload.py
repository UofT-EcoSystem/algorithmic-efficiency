"""MNIST workload implemented in Jax. 
    Adapted to allow for data augmentation"""

from typing import Tuple

from flax import linen as nn
from PIL import Image
import json
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from workloads.mnist.mnist_jax.workload import MnistWorkload
from absl import flags

class MnistAugmentation(MnistWorkload):

  def __init__(self):
    super().__init__()
    FLAGS = flags.FLAGS

    if FLAGS.augments:
      with open(FLAGS.augments, 'r') as augments_file:
        self.augment_params = json.load(augments_file)
    else:
      self.augment_params = {}

  def apply_augmentations(self, x):

    images, labels = x[0], x[1]
    for aug in self.augment_params:
      augment_fn = getattr(self,'_augment_%s' % (aug))
      augment_map = lambda img: augment_fn(img, self.augment_params[aug])

      #images = jnp.apply_along_axis(augment_map, -1, images)
      for i,im in enumerate(images):
        images[i] = augment_map(im)

    return images,labels

  def _augment_rotate(self, image, magnitude):
    """Rotation by *magnitude* radians"""
    image = tfa.image.rotate(image, tf.constant(magnitude))
    return image

  def _augment_blur(self, image, span):
    """2D mean filtering"""
    tfa.image.mean_filter2d(image, filter_shape=span)
    return image

  def _augment_flip(self, image, flip):
    """Horizontal flip"""
    im = Image.fromarray(image[:,:,0])
    return jnp.array(im.transpose(Image.FLIP_LEFT_RIGHT)).reshape(image.shape)
    
  def _augment_translate(self, image, shift):
    """Translations"""
    pass

  

