#!/bin/bash
#SBATCH --partition=funky
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


task=full
batch_size=64
epochs=500
depth=4
hidden_dim=256

all_physics_fields=young-pond-3
sdf=sunny-deluge-10 #sunny-deluge-10.pt
n=None

python3 training/regression.py "data.task=$task" "optim.epochs=$epochs" "optim.batch_size=$batch_size" "model.width=$hidden_dim" "model.depth=$depth" "inr.run_dict={all_physics_fields: $all_physics_fields, n: $n, sdf: $sdf}"
