#!/bin/bash
#SBATCH --partition=hard
#SBATCH -w lizzy
#SBATCH --job-name=geo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate wisp

python3 experiments/graph_inr.py "optim.batch_size=64" "optim.batch_size_val=64" "optim.lr_inr=1e-4" 'inr.w0=30' "data.data_to_encode=sdf" "inr.scale_factor=1" 'inr.modulate_scale=False' 'optim.epochs=5000' 'optim.meta_lr_code=1e-4' 'optim.weight_decay_code=0' "inr.num_nodes=64" "inr.latent_dim=32" 'inr.frequency_embedding=nerf' 'inr.include_input=False' 'inr.scale=5' 'inr.max_frequencies=8' 'inr.num_frequencies=32' 'inr.base_frequency=2' 'inr.include_sdf=False'


