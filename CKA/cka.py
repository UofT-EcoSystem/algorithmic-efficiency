"""
Sourced from https://github.com/RobertTLange/code-and-blog
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import jax.random as jax_rng

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
import tensorflow_datasets as tfds

from flax.training import checkpoints

from algorithmic_efficiency import checkpoint
from algorithmic_efficiency.workloads.mnist.mnist_jax.augmentation.workload import MnistAugmentation
from algorithmic_efficiency.workloads.cifar10.cifar10_jax.workload import _ModelActivations

def CKA(X, Y, kernel="linear", sigma_frac=0.4):
    """Centered Kernel Alignment."""
    if kernel == "linear":
        K, L = linear_kernel(X, Y)
    elif kernel == "rbf":
        K, L = rbf_kernel(X, Y, sigma_frac)
    return HSIC(K, L) / jnp.sqrt(HSIC(K, K) * HSIC(L, L))


#@jax.jit
def linear_kernel(X, Y):
    K = X @ X.T
    L = Y @ Y.T
    return K, L


#@jax.jit
def rbf_kernel(X, Y, sigma_frac=0.4):
    """Compute radial basis function kernels."""
    # Define helper for euclidean distance
    def euclidean_dist_matrix(X, Y):
        """Compute matrix of pairwise, squared Euclidean distances."""
        norms_1 = (X ** 2).sum(axis=1)
        norms_2 = (Y ** 2).sum(axis=1)
        return jnp.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * jnp.dot(X, Y.T))

    # Define σ as a fraction of the median distance between examples
    dist_X = euclidean_dist_matrix(X, X)
    dist_Y = euclidean_dist_matrix(Y, Y)
    sigma_x = sigma_frac * jnp.percentile(dist_X, 0.5)
    sigma_y = sigma_frac * jnp.percentile(dist_Y, 0.5)
    K = jnp.exp(-dist_X / (2 * sigma_x ** 2))
    L = jnp.exp(-dist_Y / (2 * sigma_y ** 2))
    return K, L


#@jax.jit
def HSIC(K, L):
    """Hilbert-Schmidt Independence Criterion."""

    m = K.shape[0]
    H = jnp.eye(m) - 1 / m * jnp.ones((m, m))
    numerator = jnp.trace(K @ H @ L @ H)
    return numerator / (m - 1) ** 2


def get_cka_matrix(activations_1, activations_2, kernel="linear"):
    """Loop over layer combinations & construct CKA matrix."""
    num_layers_1 = len(activations_1)
    num_layers_2 = len(activations_2)
    cka_matrix = np.zeros((num_layers_1, num_layers_2))
    symmetric = num_layers_1 == num_layers_2

    for i in range(num_layers_1):
        if symmetric:
            for j in range(i, num_layers_2):
                X, Y = activations_1[i], activations_2[j]
                cka_temp = CKA(X, Y, kernel)
                cka_matrix[num_layers_1 - i - 1, j] = cka_temp
                cka_matrix[i, num_layers_1 - j - 1] = cka_temp
        else:
            for j in range(num_layers_2):
                X, Y = activations_1[i], activations_2[j]
                cka_temp = CKA(X, Y)
                cka_matrix[num_layers_1 - i - 1, j] = cka_temp
    return cka_matrix

def build_dataset(data_rng: jax.random.PRNGKey,
                     split: str,
                     data_dir: str,
                     batch_size,
                     dataset):
    def _normalize(image):
        return tf.cast(image, tf.float32) / 255.0

    ds = tfds.load(dataset, split=split)
    ds = ds.cache()
    ds = ds.map(lambda x: (_normalize(x['image']), x['label'], None))
    if split == 'train':
      ds = ds.shuffle(1024, seed=data_rng[0])
      ds = ds.repeat()
    ds = ds.batch(batch_size)

    return tfds.as_numpy(ds)

def main(fname, kernel="linear"):
    log = load_log(os.path.join("../checkpoints/", fname))
    # Get CIFAR-10 dataloader
    train_loader, test_loader = get_dataloaders(2056)
    # Get batch data & compute final activations
    for batch_idx, (data, target) in enumerate(test_loader):
        batch_images = jnp.array(data)
        break


    cnn_vars = load_model(log.meta.model_ckpt, model_type="jax")
    model = All_CNN_C_Features(**log.meta.config_dict["model_config"])
    activations_final = model.apply(cnn_vars, batch_images, train=False)

    # Reload the trained final checkpoint
    all_cka_matrices = []
    ckpt_list = [log.meta.init_ckpt]
    ckpt_list = ckpt_list + log.meta.every_k_ckpt_list

    # Loop over all checkpoint and construct CKA of ckpt vs final ckpt
    for ckpt_path in ckpt_list:
        cnn_vars = load_model(ckpt_path, model_type="jax")
        activations_ckpt = model.apply(cnn_vars, batch_images, train=False)
        cka_matrix = get_cka_matrix(activations_ckpt, activations_final, kernel)
        all_cka_matrices.append(cka_matrix)
        print(ckpt_path)

    stacked_rsm = np.stack(all_cka_matrices, axis=0)
    np.save(f"rsm_{fname}_{kernel}.npy", stacked_rsm)

    reload = np.load(f"rsm_{fname}_{kernel}.npy")
    print(reload.shape)


def get_checkpoint(load_path, prefix, step=None):
    '''Load a checkpoint from a checkpoint file. The checkpoints are
    dictionaries with the model parameters and model state.

    Args:
        load_path: The path to the directory containing the checkpoints
        prefix: Prefix used for the checkpoint file. Used to differentiate
            different tuning runs
        step: Which checkpoint file to load. By default loads the most recent
        model: An instance of the model class. If provided, checkpoint 
            parameters/state will be loaded into the model

    Returns:
        checkpoint (dict): A dictionary containing 'params' and 'model_state'
    '''

    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=load_path,
        target=None,
        step=step,
        prefix=prefix,
    )
    if ckpt is None:
        logging.warn('Could not find file containing prefix %s, \
            in directory %s' % (prefix, load_path))

    return ckpt

def plot_cka_matrix(cka_matrix, title="All-CNN-C",
                    xlabel="Layer ID", ylabel="Layer ID",
                    every_nth_tick=1, ax=None, fig=None):
    """" Helper Function for Plotting CKA Matrices. """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    im = ax.imshow(cka_matrix, vmin=0, vmax=1)
    
    yticklabels = list(reversed(["L" + str(i+1) for i 
                                 in range(cka_matrix.shape[0])]))
    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)

    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth_tick != 0:
            label.set_visible(False)

    xticklabels = ["L" + str(i+1) for i in range(cka_matrix.shape[1])]
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=14)

    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth_tick != 0:
            label.set_visible(False)
    ax.set_title(title, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=75, ha="right", rotation_mode="anchor")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("CKA Score", rotation=270, labelpad=30, fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    fig.tight_layout()
    plt.savefig("CKA")


if __name__ == "__main__":

    import pickle
    from algorithmic_efficiency import random_utils as prng
    comparison_checkpoint = './experiments/augmentation/saved/cifar10_baseline/checkpoints/tune1_epoch_15'
    checkpoints_dir = './experiments/augmentation/saved/cifar10_baseline/checkpoints'

    ckpt = get_checkpoint(comparison_checkpoint, "tune")
    #This is for CIFAR10, see workloads/cifar10/cifar10_jax/workload.py to make one for other models

    model = _ModelActivations(num_classes=10) 

    input_queue = iter(build_dataset(jax_rng.PRNGKey(0), 'train', data_dir='~/', batch_size=32, dataset='cifar10'))
    batch_images = np.array(next(input_queue))

    #activations_final = model.apply(batch_images, train=False)
    activations_final = model.apply({'params':ckpt['params']},
                                     batch_images[0],
                                     train=False)
    ckpt_list = ['tune2_epoch_13'] # List of all the checkpoints you want to compare against the comparison here. 
    for single_ckpt in ckpt_list:
        ckpt = get_checkpoint(os.path.join(checkpoints_dir, single_ckpt), "tune")
        activations_ckpt = model.apply({'params':ckpt['params']},
                                 batch_images[0],
                                 train=False)

        cka_matrix = get_cka_matrix(activations_final, activations_ckpt, "linear")
        print(cka_matrix)
        # Just plot them for now
        plot_cka_matrix(cka_matrix, title="CNN",
                    xlabel="Layer ID", ylabel="Layer ID",
                    every_nth_tick=1, ax=None, fig=None)
