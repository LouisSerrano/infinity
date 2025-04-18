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


task=full
#ddpm_run=autumn-butterfly-1179
#ddpm_run=serene-universe-1230
ddpm_run=serene-thunder-1255

glamorous-leaf-1241

#true-brook-1235

python3 training/test_ddpm.py "data.task=$task" "model.run_name=$ddpm_run" 'optim.batch_size=4' 'optim.batch_size_val=4'


