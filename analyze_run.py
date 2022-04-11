import sys
import os
sys.path.append(os.path.join(os.getcwd(),"spectral-density"))
# module_dirs = ["/home/adamo/Documents/spectral-density/jax","/home/adamo/Documents/algorithmic-efficiency"]
# for module_dir in module_dirs:
#     if module_dir not in sys.path:
#         sys.path.append(module_dir)
# print(sys.path)

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.config import config
from jax.tree_util import tree_map

import json
import time
import matplotlib.pyplot as plt
import argparse
import glob
from tqdm import tqdm
from pprint import pprint
import wandb
import numpy as np

import submission_runner
from submission_runner import FLAGS, _import_workload, WORKLOADS
from algorithmic_efficiency.checkpoint import load_checkpoint
from algorithmic_efficiency.spec import ForwardPassMode
from algorithmic_efficiency.workloads.mnist.configurable_mnist_jax.workload import MnistWorkload

import density as density_lib
import hessian_computation
import lanczos
import metrics
from simple_test import compute_metrics, compute_spectrum

run_name_to_exp = {
    "act_gelu_100": "activation/batchnorm_off-activation_gelu-width_100-depth_3-dropout_0-batch_1024",
    "act_gelu_200": "activation/batchnorm_off-activation_gelu-width_200-depth_3-dropout_0-batch_1024",
    "act_htanh_100": "activation/batchnorm_off-activation_hard_tanh-width_100-depth_3-dropout_0-batch_1024",
    "act_htanh_200": "activation/batchnorm_off-activation_hard_tanh-width_200-depth_3-dropout_0-batch_1024",
    "act_relu_100": "activation/batchnorm_off-activation_relu-width_100-depth_3-dropout_0-batch_1024",
    "act_relu_200": "activation/batchnorm_off-activation_relu-width_200-depth_3-dropout_0-batch_1024",
    "act_sigmoid_100": "activation/batchnorm_off-activation_sigmoid-width_100-depth_3-dropout_0-batch_1024",
    "act_sigmoid_200": "activation/batchnorm_off-activation_sigmoid-width_200-depth_3-dropout_0-batch_1024",
    "bn_on_3": "batchnorm/batchnorm_affine-activation-batchnorm-activation_relu-width_100-depth_3-dropout_0-batch_1024",
    "bn_on_5": "batchnorm/batchnorm_affine-activation-batchnorm-activation_relu-width_100-depth_5-dropout_0-batch_1024",
    "bn_on_7": "batchnorm/batchnorm_affine-activation-batchnorm-activation_relu-width_100-depth_7-dropout_0-batch_1024",
    "bn_off_3": "batchnorm/batchnorm_off-activation_relu-width_100-depth_3-dropout_0-batch_1024",
    "bn_off_5": "batchnorm/batchnorm_off-activation_relu-width_100-depth_5-dropout_0-batch_1024",
    "bn_off_7": "batchnorm/batchnorm_off-activation_relu-width_100-depth_7-dropout_0-batch_1024",
}

def main(args):

    exp_dp = os.path.join(args.base_dp,run_name_to_exp[args.run_name])
    assert os.path.isdir(exp_dp), exp_dp
    checkpoint_dp = os.path.join(exp_dp,"configurable_mnist_jax",f"trial_{args.trial}","checkpoints")
    assert os.path.isdir(checkpoint_dp), checkpoint_dp
    metadata_fp = os.path.join(exp_dp,"configurable_mnist_jax",f"trial_{args.trial}","metadata.json")
    assert os.path.isfile(metadata_fp), metadata_fp
    args.checkpoint_dp = checkpoint_dp
    args.metadata_fp = metadata_fp
    # set up wandb
    run = wandb.init(
        project=args.project_name,
        name=f"{args.run_name}_trial_{args.trial}",
        config=vars(args),
        group=args.run_name,
        mode=args.wandb_mode,
        dir=args.wandb_meta_dp
    )
    # check jax stuff
    if args.precision == "double":
        config.update("jax_enable_x64", True)
    print(jax.devices())
    print(jnp.ones(3).device_buffer.device())
    # load
    with open(args.metadata_fp,"r") as json_file:
        metadata_d = json.load(json_file)
    set_up_flags(metadata_d)
    wl = set_up_workload(metadata_d)
    batches_list = get_batches_list(args,wl)
    checkpoint_names = glob.glob(os.path.join(args.checkpoint_dp,"checkpoint_*"))
    all_steps = sorted([int(os.path.basename(name)[len("checkpoint_"):]) for name in checkpoint_names])
    if args.steps is None:
        steps = all_steps
    else:
        steps = [step for step in args.steps if step in all_steps]
    print(f">>> total num steps = {len(steps)}")
    for step in steps:
        print(f">>> step = {step}")
        params, model_state = load_params_model_state(args,step)
        print("model_state",not(model_state is None))
        metric_d = {}
        for mvp_type in args.mvp_types:
            print(f">> mvp_type = {mvp_type}")
            mvp_spec_d = analyze(args,wl,batches_list,params,model_state,mvp_type)
            # plot_density(grids,density)
            mvp_metric_d = compute_metrics(mvp_spec_d)
            pprint(mvp_metric_d)
            for k,v in mvp_metric_d.items():
                metric_d[f"{mvp_type}_{k}"] = v
        metric_d["chkpt_step"] = step
        wandb.log(metric_d,commit=True)
    run.finish()

def set_up_flags(metadata_d):
    workload = metadata_d["workload"]
    flags_list = [""]
    flags_list.append(f"--workload={workload}")
    flags_list.append(f"--logging_dir=logging")
    flags_list.append(f"--framework=jax")
    flags_extra_list = []
    for k,v in metadata_d.items():
        if k.startswith("extra.mnist_config.") or k.startswith("extra.ogbg_config."):
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


def load_params_model_state(args,step):
    assert os.path.isdir(args.checkpoint_dp)
    checkpoint = load_checkpoint(args.checkpoint_dp,prefix="checkpoint_",step=step)
    params = checkpoint["params"]
    model_state = checkpoint["model_state"]
    if args.precision == "double":
        params = tree_map(lambda p: p.astype(jnp.float64),params)
        if not (model_state is None):
            model_state = tree_map(lambda p: p.astype(jnp.float64),model_state)
    return params, model_state


def get_batches_list(args,wl):
    dummy_key = jax.random.PRNGKey(0)
    if args.batch_size == -1:
        batch_size = wl.batch_size
    else:
        batch_size = args.batch_size
    input_queue = wl.build_input_queue(dummy_key,"train",args.data_dp,batch_size,repeat=False)
    # TODO: what is the point of preprocess_for_train? not used for MNIST
    if args.num_batches == -1:
        batches_list = [(b[0],b[1],b[2]) for b in input_queue]
    else:
        batches_list = []
        for i in range(args.num_batches):
            b = next(input_queue)
            batches_list.append((b[0],b[1],b[2]))
    return batches_list


def analyze(args,wl,batches_list,params,model_state,mvp_type):

    dummy_key = jax.random.PRNGKey(0)
    
    num_batches = len(batches_list)
    print("num_batches", num_batches)
    def batches_fn():
        for b in batches_list:
            yield b

    assert jax.local_device_count() == 1

    # TODO: I think we should actually use a real key (i.e. for Dropout...)
    # TODO: what about update_batch_norm?
    if isinstance(wl,MnistWorkload):

        def loss(params,batch):
            input_batch, label_batch, _ = batch
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
            mean_loss = jnp.mean(loss_score)
            return mean_loss

        def model(params,batch):
            input_batch, label_batch, _ = batch
            logits_batch, _ = wl.model_fn(
                params,
                input_batch,
                model_state,
                ForwardPassMode.TRAIN,
                dummy_key,
                False
            )
            return logits_batch

        def model_loss(logits_batch,batch):
            _, label_batch, _ = batch
            loss_score = wl.loss_fn(
                label_batch, logits_batch
            )
            mean_loss = jnp.mean(loss_score)
            return mean_loss

    else:

        # un-pmap
        params = jax.tree_map(lambda x: x[0:1], params)

        def _loss(params,input_batch,label_batch,mask_batch):
            logits_batch, _ = wl.model_fn(
                params,
                input_batch,
                model_state,
                ForwardPassMode.TRAIN,
                dummy_key,
                False
            )
            per_example_losses = wl.loss_fn(label_batch, logits_batch, mask_batch)
            mean_loss = (
                jnp.sum(jnp.where(mask_batch, per_example_losses, 0)) /
                jnp.sum(mask_batch)
            )
            return mean_loss
        pmap_loss = jax.pmap(_loss,axis_name='batch',in_axes=(0,0,0,0))

        def loss(params,batch):
            input_batch, label_batch, mask_batch = batch
            pmap_mean_loss = pmap_loss(params,input_batch,label_batch,mask_batch)
            mean_loss = jnp.mean(pmap_mean_loss)
            # print(mean_loss.shape)
            return mean_loss

    # test it out
    test_batch = batches_list[0]
    print(lax.stop_gradient(loss(params,test_batch)))

    if mvp_type == "hvp":

        hvp, unravel, num_params = hessian_computation.get_hvp_fn(loss, params, batches_fn)
        mvp_cl = lambda v: hvp(params, v) / num_batches

    elif mvp_type == "ggnvp":

        ggnvp, unravel, num_params = hessian_computation.get_ggnvp_fn(model,model_loss,params,batches_fn)
        mvp_cl = lambda v: ggnvp(params,v)
    
    elif mvp_type == "jjvp":

        jjvp, unravel, num_params = hessian_computation.get_jjvp_fn(loss, params, batches_fn)
        mvp_cl = lambda v: jjvp(params, v) / num_batches

    jac = hessian_computation.get_jac(loss, params, batches_fn)
    print(jac.shape, jac.sum())

    spec_d = compute_spectrum(mvp_cl,num_params,jac,args.order,args.num_samples)
    return spec_d

    # order = args.order
    # num_samples = args.num_samples
    # hvp, unravel, num_params = hessian_computation.get_hvp_fn(loss, params, batches_fn)
    # hvp_cl = lambda v: hvp(params, v) / num_batches # Match the API required by lanczos_alg

    # print("num_params: {}".format(num_params))
    # start = time.time()
    # hvp_cl(jnp.ones(num_params)).block_until_ready() # first call of a jitted function compiles it
    # end = time.time()
    # print("hvp compile time: {}".format(end-start))
    # start = time.time()
    # hvp_cl(2*jnp.ones(num_params)).block_until_ready() # second+ call will be much faster
    # end = time.time()
    # print("hvp compute time: {}".format(end-start))
    # print(f"estimated time = {(end-start)*order*num_samples}")

    # rng = jax.random.PRNGKey(420420)
    # rngs = jax.random.split(rng,num=num_samples+1)
    # rng = rngs[0]
    # start = time.time()
    # tridiags, lcz_vecs = [], []
    # for i in tqdm(range(num_samples)):
    #     tridiag, vec = lanczos.lanczos_alg(hvp_cl, num_params, order, rngs[i+1])
    #     tridiags.append(tridiag)
    #     lcz_vecs.append(vec)
    # end = time.time()
    # print("Lanczos time: {}".format(end-start)) # this should be ~ num_samples * order * hvp compute time
    # eig_vals, _, eig_vecs = density_lib.tridiag_to_eigv(tridiags, get_eig_vecs=True)
    # density, grids = density_lib.tridiag_to_density(tridiags, grid_len=args.grid_len, sigma_squared=args.sigma_squared)
    # out_d = {
    #     "eig_vals": eig_vals,
    #     "eig_vecs": eig_vecs,
    #     "lcz_vecs": jnp.stack(lcz_vecs,axis=0),
    #     "density": density,
    #     "grids": grids
    # }
    # return out_d


def plot_density(grids, density, label=None):
    plt.semilogy(grids, density, label=label)
    # plt.ylim(1e-10, 1e2)
    plt.ylabel("Density")
    plt.xlabel("Eigenvalue")
    if not (label is None):
        plt.legend()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--metadata_fp",type=str,default="/home/adamo/Downloads/mnist_checkpoints_2/batchnorm/metadata.json")
    # parser.add_argument("--checkpoint_dp",type=str,default="/home/adamo/Downloads/mnist_checkpoints_2/batchnorm/checkpoints")
    # parser.add_argument("--metadata_fp",type=str, default="/home/adamo/Downloads/mnist-batchnorm-checkpoints-with-pickles/logs/batchnorm_off-activation_relu-width_256-depth_1-dropout_0-batch_1024/configurable_mnist_jax/trial_1/metadata.json")
    # parser.add_argument("--checkpoint_dp",type=str,default="/home/adamo/Downloads/mnist-batchnorm-checkpoints-with-pickles/logs/batchnorm_off-activation_relu-width_256-depth_1-dropout_0-batch_1024/configurable_mnist_jax/trial_1/checkpoints")
    # parser.add_argument("--run_dp",type=str,default="/home/adamo/Downloads/mnist_checkpoints_2/batchnorm/")
    parser.add_argument("--run_name",type=str,required=True)
    parser.add_argument("--base_dp",type=str,default="/home/adamo/Downloads/mnist_checkpoints_2")
    parser.add_argument("--trial",type=int,default=1)
    parser.add_argument("--steps",nargs="+",type=int,default=None)
    parser.add_argument("--order",type=int,default=90)
    parser.add_argument("--num_samples",type=int,default=10)
    parser.add_argument("--data_dp",type=str,default="/home/adamo/Documents/tfds_datasets")
    parser.add_argument("--batch_size",type=int,default=-1)
    parser.add_argument("--num_batches",type=int,default=-1)
    parser.add_argument("--precision",type=str,default="double",choices=["single","double"])
    parser.add_argument("--grid_len",type=int,default=10000)
    parser.add_argument("--sigma_squared",type=float,default=1e-5)
    parser.add_argument("--mvp_types",type=str,nargs="+",default=["hvp"])
    parser.add_argument("--project_name",type=str,default="CSC2541-2022")
    parser.add_argument("--wandb_mode",type=str,default="offline",choices=["online","offline","disabled"])
    parser.add_argument("--wandb_meta_dp",type=str,default=os.getcwd())
    args = parser.parse_args()
    print(args)
    main(args)