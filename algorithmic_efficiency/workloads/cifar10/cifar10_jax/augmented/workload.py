from algorithmic_efficiency.workloads.cifar10.cifar10_jax.workload import CIFAR10Workload
import jax
import tensorflow_datasets as tfds
import dill
import tensorflow as tf

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
