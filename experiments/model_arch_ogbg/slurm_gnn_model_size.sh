#!/bin/bash
#
# Usage:
# sbatch experiments/model_arch_ogbg/slurm_gnn_model_size.sh
#
# OR launch many jobs in parallel
#
# wandb sweep /ssd003/home/dans/algorithmic-efficiency/experiments/model_arch_ogbg/sweep_gnn_model_size.yaml
# # copy sweep ID
# cd ~/slurm_logs-2
# for run in {1..161}; do
#   sbatch /ssd003/home/dans/algorithmic-efficiency/experiments/model_arch_ogbg/slurm_gnn_model_size.sh
# done
#
# build 1 large JSON file
# jq -s 'flatten' /h/dans/algorithmic-efficiency/logs/gnn_model_size_first_slurm/*/**/trial_**/*.json > slurm_gnn_size_merged.json

#SBATCH --nodes 1
#SBATCH --partition t4v2,rtx6000
#SBATCH --time=4:00:00
#SBATCH -c 4
#SBATCH --mem=8GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=gnn_model_size
#SBATCH --output=gnn_model_size_job_%j.out
#SBATCH --signal=B:TERM@60 # tells the controller
                           # to send SIGTERM to the job 60 secs
                           # before its time ends to give it a
                           # chance for better cleanup.


module load jax0.2.24-cuda11.0-python3.8_jupyter
source ~/venv2/bin/activate
cd ~/algorithmic-efficiency
wandb login ab8a559717268962dc374078301ec56912d60370
exec wandb agent --count 1 danielsnider/mlc_held_out_gnn/fzvw4nhj # copy sweep ID here