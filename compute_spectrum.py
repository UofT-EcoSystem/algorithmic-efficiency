import sys
module_dirs = ["/home/adamo/Documents/spectral-density/jax","/home/adamo/Documents/algorithmic-efficiency"]
for module_dir in module_dirs:
    if module_dir not in sys.path:
        sys.path.append(module_dir)
# print(sys.path)

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import linen as nn
from flax.training import train_state
import numpy as np
import tensorflow_datasets as tfds
from typing import Any, Callable, Sequence, Optional
import dill
import os
import json
import importlib
import time
import matplotlib.pyplot as plt
import argparse

import submission_runner
from submission_runner import FLAGS, _import_workload, WORKLOADS
from algorithmic_efficiency.checkpoint import load_checkpoint
from algorithmic_efficiency.spec import ForwardPassMode

import density as density_lib
import hessian_computation
import lanczos


def main(args):

    assert os.path.isfile(args.metadata_fp)
    with open(args.metadata_fp,"r") as json_file:
        metadata_d = json.load(json_file)
    set_up_flags(metadata_d)
    wl = set_up_workload(metadata_d)
    batches_list = get_batches_list(wl)
    # TODO: scan for multiple checkpoints
    for step in args.steps:
        print(f">>> step = {step}")
        params, model_state = load_params_model_state(args.checkpoint_dp,step)
        tridiags, vecses, density, grids = compute_eigvals_density(args,wl,batches_list,params,model_state)
        plot_density(grids,density)


def set_up_flags(metadata_d):
    workload = metadata_d["workload"]
    flags_list = [""]
    flags_list.append(f"--workload={workload}")
    flags_list.append(f"--logging_dir=logging")
    flags_list.append(f"--framework=jax")
    flags_extra_list = []
    for k,v in metadata_d.items():
        if k.startswith("extra.mnist_config."):
            flags_k = k[len("extra."):]
            flags_extra_list.append(f"{flags_k}={v}")
    # flags_extra_str = ",".join(flags_extra_list)
    for flag in flags_extra_list:
        flags_list.append(f"--extra_metadata={flag}")
    print(flags_list)
    FLAGS(flags_list)


def set_up_workload(metadata_d):
    workload = metadata_d["workload"]
    workload_metadata = WORKLOADS[workload]
    workload_path = workload_metadata["workload_path"]
    workload_class_name = workload_metadata["workload_class_name"]
    wl = _import_workload(workload_path=workload_path,workload_class_name=workload_class_name)
    # print(wl)
    return wl


def load_params_model_state(checkpoint_dp,step):
    assert os.path.isdir(checkpoint_dp)
    checkpoint = load_checkpoint(checkpoint_dp,prefix="checkpoint_",step=step)
    params = checkpoint["params"]
    model_state = checkpoint["model_state"]
    return params, model_state


def get_batches_list(wl):
    dummy_key = jax.random.PRNGKey(0)
    batch_size = wl.batch_size
    input_queue = wl.build_input_queue(dummy_key,"train","",batch_size,repeat=False)
    # TODO: what is the point of preprocess_for_train? not used for MNIST
    batches_list = [(b[0],b[1]) for b in input_queue]
    return batches_list


def compute_eigvals_density(args,wl,batches_list,params,model_state):

    dummy_key = jax.random.PRNGKey(0)
    # batch_size = wl.batch_size
    # num_points = wl.num_eval_train_examples
    
    num_batches = len(batches_list)
    def batches_fn():
        for b in batches_list:
            yield b

    # TODO: I think we should actually use a real key (i.e. for Dropout...)
    # TODO: what about update_batch_norm?
    def loss(params,batch):
        input_batch, label_batch = batch
        # print(input_batch.shape)
        # print(label_batch.shape)
        logits_batch, _ = wl.model_fn(
            params,
            input_batch,
            model_state,
            ForwardPassMode.TRAIN,
            dummy_key,
            False
        )
        loss_score = wl.loss_fn(
            label_batch, logits_batch
        )
        return jnp.mean(loss_score)

    # test it out
    test_batch = batches_list[0]
    print(lax.stop_gradient(loss(params,test_batch)))

    order = args.order
    num_samples = args.num_samples
    hvp, unravel, num_params = hessian_computation.get_hvp_fn(loss, params, batches_fn)
    hvp_cl = lambda v: hvp(params, v) / num_batches # Match the API required by lanczos_alg

    print("num_params: {}".format(num_params))
    start = time.time()
    hvp_cl(jnp.ones(num_params)).block_until_ready() # first call of a jitted function compiles it
    end = time.time()
    print("hvp compile time: {}".format(end-start))
    start = time.time()
    hvp_cl(2*jnp.ones(num_params)).block_until_ready() # second+ call will be much faster
    end = time.time()
    print("hvp compute time: {}".format(end-start))

    rng = jax.random.PRNGKey(420420)
    rngs = jax.random.split(rng,num=num_samples+1)
    rng = rngs[0]
    start = time.time()
    tridiags, vecses = [], []
    for i in range(num_samples):
        tridiag, vecs = lanczos.lanczos_alg(hvp_cl, num_params, order, rngs[i+1])
        tridiags.append(tridiag)
        vecses.append(vecs)
    end = time.time()
    print("Lanczos time: {}".format(end-start)) # this should be ~ num_samples * order * hvp compute time
    density, grids = density_lib.tridiag_to_density(tridiags, grid_len=10000, sigma_squared=1e-5)
    
    return tridiags, vecses, density, grids


def compute_metrics(tridiags,vecses,density,grids):

    pass


def plot_density(grids, density, label=None):
    plt.semilogy(grids, density, label=label)
    plt.ylim(1e-10, 1e2)
    plt.ylabel("Density")
    plt.xlabel("Eigenvalue")
    if not (label is None):
        plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_fp",type=str, default="/home/adamo/Downloads/mnist-batchnorm-checkpoints-with-pickles/logs/batchnorm_off-activation_relu-width_256-depth_1-dropout_0-batch_1024/configurable_mnist_jax/trial_1/metadata.json")
    parser.add_argument("--checkpoint_dp",type=str,default="/home/adamo/Downloads/mnist-batchnorm-checkpoints-with-pickles/logs/batchnorm_off-activation_relu-width_256-depth_1-dropout_0-batch_1024/configurable_mnist_jax/trial_1/checkpoints")
    parser.add_argument("--steps",nargs="+",type=int,default=[0,100])
    parser.add_argument("--order",type=int,default=90)
    parser.add_argument("--num_samples",type=int,default=10)
    args = parser.parse_args()
    print(args)
    main(args)