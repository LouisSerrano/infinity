#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=sdf
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate infinity_env 

data_to_encode=sdf
task=scarce
epochs=5000
batch_size=64
latent_dim=256
lr_inr=5e-5

python3 training/inr.py --config-name=siren.yaml "data.data_to_encode=$data_to_encode" "data.task=$task" "optim.epochs=$epochs" "optim.batch_size=$batch_size" "optim.lr_inr=$lr_inr" "inr.latent_dim=$latent_dim" "optim.meta_lr_code=0"


