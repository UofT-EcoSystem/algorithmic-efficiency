import jax
import jax.numpy as jnp
import tensorflow_addons as tfa
import tensorflow as tf
from typing import Dict, Tuple, Any, List
import json
from absl import logging
from sklearn.model_selection import train_test_split

class ImageAugmenter():

  def __init__(self, augments_file, rng: jax.random.PRNGKey):
    self.augments_file=augments_file
    self.rng = rng
    if self.augments_file:
      try:
        with open(self.augments_file, 'r') as file:
          self.augment_kwargs = json.load(file)
      except:
        logging.warn('Could not read augmentations file \'%s\'; Proceeding without augmentations')
        self.augment_kwargs = {}

    self.aug_fns = {}
    for aug in self.augment_kwargs.keys():
      try:
        self.aug_fns[aug] = getattr(self, '_%s' %(aug))
      except:
        logging.warn('Could not find desired augmentation function: %s' %(aug))
  
  def apply_augmentations(self, x):
    for (aug, aug_fn) in self.aug_fns.items():
      self.rng, aug_rng = jax.random.split(self.rng)
      x = aug_fn(x, rng=aug_rng, **self.augment_kwargs[aug])
    return x

  def _shift(self, x, rng, width_shift_max: int, height_shift_max: int, fill_mode='nearest'):
    '''Args must be shift ranges measured in pixels. Will randomly shift
    images between -max and max.
    '''
    w = jnp.arange(-width_shift_max, width_shift_max+1)
    h = jnp.arange(-height_shift_max, height_shift_max+1)
    w_rng, h_rng = jax.random.split(rng)
    w_shift = jax.random.choice(w_rng, w, shape=[jnp.shape(x)[0]])
    h_shift = jax.random.choice(h_rng, h, shape=[jnp.shape(x)[0]])
    shift = jnp.stack((w_shift, h_shift), axis=-1)
    return tfa.image.translate(images=x, translations=tf.cast(shift, tf.float32), fill_mode=fill_mode)

  def _fliplr(self, x, rng):
    '''Randomly flips images with a probability of 0.5
    '''
    seed = jax.random.randint(rng, shape=[2], minval=-1000, maxval=1000)
    return tf.image.stateless_random_flip_left_right(x, seed)

  def _gaussian_noise(self, x, rng):
    '''Adds guassian noise to the images
    '''
    noise = 0.1*jax.random.normal(rng, shape=x.shape)
    return x + noise

  def _invert(self, x, rng):
    '''Randomly inverts images by flipping the intensities. Assumes the image
    intensities are normalized with zero mean.
    '''
    m = jax.random.choice(rng, jnp.array([-1, 1]), shape=[jnp.shape(x)[0]])
    m = tf.reshape(tf.cast(m, tf.float32), shape=[-1, 1, 1, 1])
    return tf.math.multiply(m, x)

  
