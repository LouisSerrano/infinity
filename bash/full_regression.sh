#!/bin/bash
#SBATCH --partition=funky
#SBATCH --job-name=geo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate infinity_env


data_to_encode=n
task=full
batch_size=64
epochs=500
depth=4
hidden_dim=256

vx=bright-totem-286
vy=devoted-puddle-287
p=serene-vortex-284
nu=wandering-bee-288
sdf=earnest-paper-289
n=astral-leaf-330

python3 training/code_regression.py "data.task=$task" "optim.epochs=$epochs" "optim.batch_size=$batch_size" "model.width=$hidden_dim" "model.depth=$depth" "inr.run_dict={vx: $vx, vy: $vy, p: $p,  nu: $nu, sdf: $sdf, n: $n}" 


