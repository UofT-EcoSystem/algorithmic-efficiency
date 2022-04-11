from algorithmic_efficiency.workloads.cifar10.cifar10_jax.workload import CIFAR10Workload
import jax
import tensorflow_datasets as tfds
import dill
import json
import tensorflow as tf
import jax.numpy as jnp
import tensorflow_addons as tfa
from absl import flags
from absl import logging

class Custom_CIFAR10Workload(CIFAR10Workload):
  def _build_dataset(self,
                     data_rng: jax.random.PRNGKey,
                     split: str,
                     data_dir: str,
                     batch_size):
    '''Load from a pickled dataset as opposed to a dataset builder.
    data_dir should be the path to the pickled dataset. The dataset should
    be a dict with keys 'train' and 'test'. Each split should also be a dict
    with keys 'image' and 'label'
    '''
    with open(data_dir, 'rb') as file:
      data = dill.load(file)
    data = data[split]
    ds = tf.data.Dataset.from_tensor_slices((self._normalize(data['image']), data['label'], None))
    ds = ds.cache()
    if split == 'train':
      ds = ds.shuffle(1024, seed=data_rng[0])
      ds = ds.repeat()
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)


class CIFAR10Augmentation(CIFAR10Workload):
  '''For online augmentations on CIFAR10 using subsets of the training set'''
  def __init__(self):
    super().__init__()
    FLAGS = flags.FLAGS

    if FLAGS.augments:
      with open(FLAGS.augments, 'r') as augments_file:
        self.augment_params = json.load(augments_file)
    else:
      self.augment_params = {}
    self.aug_fns = {}

    for aug in self.augment_params.keys():
      try:
        self.aug_fns[aug] = getattr(self, '_%s' %(aug))
      except:
        logging.warn('Could not find desired augmentation function: %s' %(aug))

  def _normalize(self, image):
    return (tf.cast(image, tf.float32) - self.train_mean) / self.train_stddev

  def _build_dataset(self,
                     data_rng: jax.random.PRNGKey,
                     split: str,
                     data_dir: str,
                     batch_size):

    self.rng = data_rng
    FLAGS = flags.FLAGS
    split_pct = split
    if FLAGS.percent_data_selection < 100 and split == 'train':
      split_pct = split + '[{pct}%:]'.format(pct=FLAGS.percent_data_selection)

    ds = tfds.load('cifar10', split=split_pct)
    ds = ds.cache()
    ds = ds.map(lambda x: (self._normalize(x['image']), x['label'], None))
    if split == 'train':
      ds = ds.shuffle(1024, seed=data_rng[0])
      ds = ds.repeat()
    ds = ds.batch(batch_size)

    return tfds.as_numpy(ds)

  def apply_augmentations(self, x):
    for (aug, aug_fn) in self.aug_fns.items():
      self.rng, aug_rng = jax.random.split(self.rng)
      x = aug_fn(x, rng=aug_rng, **self.augment_params[aug])
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
    return tfa.image.translate(images=x, translations=tf.cast(shift, tf.float32), fill_mode=fill_mode).numpy()

  def _fliplr(self, x, rng):
    '''Randomly flips images with a probability of 0.5
    '''
    seed = jax.random.randint(rng, shape=[2], minval=-1000, maxval=1000)
    return tf.image.stateless_random_flip_left_right(x, seed).numpy()
  

