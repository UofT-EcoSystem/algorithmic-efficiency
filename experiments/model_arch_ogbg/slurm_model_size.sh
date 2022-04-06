#SBATCH --nodes 2
#SBATCH --partition t4v2,rtx6000
#SBATCH --time=2:30:00
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
# wandb sweep experiments/model_arch_ogbg/sweep_gnn_model_size.yaml
exec wandb agent --count 1 danielsnider/mlc_held_out_gnn/tivftlsr