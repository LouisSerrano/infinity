#!/bin/bash
#SBATCH --partition=funky
#SBATCH --job-name=reg
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
task=scarce
batch_size=16
epochs=500
depth=4
hidden_dim=256


vx=crisp-disco-451
vy=leafy-lion-452
p=woven-mountain-453	
nu=pleasant-gorge-454
sdf=zany-snowball-1185
n=legendary-smoke-417
fields=rosy-firefly-1181
#fields=frosty-dust-1256
#jumping-elevator-453
#frosty-river-456

python3 training/code_regression_without_normal.py "data.task=$task" "optim.epochs=$epochs" "optim.batch_size=$batch_size" "model.width=$hidden_dim" "model.depth=$depth" "inr.run_dict={vx: $vx, vy: $vy, p: $p,  nu: $nu, sdf: $sdf, n: $n, fields: $fields}" 'data.seed=1060' 'optim.lr=1e-3' 'optim.weight_decay=1e-6' 'model.ddpm_steps=1000'


