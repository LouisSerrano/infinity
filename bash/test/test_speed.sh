#!/bin/bash
#SBATCH --partition=electronic
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

run_name=serene-shadow-483

python3 test/test_inference_speed.py "data.task=$task" "optim.epochs=$epochs" "optim.batch_size=$batch_size" "model.width=$hidden_dim" "model.depth=$depth" "model.run_name=$run_name" 'data.seed=1060' 'optim.lr=1e-3' 'optim.weight_decay=1e-6'


