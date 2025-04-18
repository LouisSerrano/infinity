#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=geo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate infinity_env 

data_to_encode=all_physics_fields
task=full
batch_size=64
epochs=10000
depth=5
hidden_dim=256
latent_dim=256
max_frequencies=32
base_frequency=1.25
num_frequencies=64
meta_lr_code=5e-5
include_input=True

python3 training/inr_fields.py "data.data_to_encode=$data_to_encode" "data.task=$task" "optim.epochs=$epochs" "optim.batch_size=$batch_size" "optim.meta_lr_code=$meta_lr_code" "inr.hidden_dim=$hidden_dim" "inr.latent_dim=$latent_dim" "inr.depth=$depth" "inr.max_frequencies=$max_frequencies" "inr.base_frequency=$base_frequency" "inr.num_frequencies=$num_frequencies" "inr.include_input=$include_input"


