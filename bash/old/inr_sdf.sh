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

#python3 training/inr_regression.py 'data.dataset_name=shallow-water' 'data.data_to_encode=vorticity' 'optim.batch_size=16' 'optim.batch_size_val=4' 'inr.w0=20' 'inr.hidden_dim=256' 'inr.latent_dim=256' 'inr.depth=6' 'data.sub_tr=0.25' 'data.sub_te=1' 'inr.model_type=siren' 'mlp.inference_model=resnet' 'optim.epochs=10000' 'optim.lr_inr=3e-6' 


#python3 training/inr_regression.py 'data.dataset_name=shallow-water' 'data.data_to_encode=vorticity' 'optim.batch_size=16' 'optim.batch_size_val=16' 'inr.hidden_dim=256' 'inr.latent_dim=256' 'data.sub_tr=2' 'data.sub_te=2' 'inr.model_type=bacon' 'mlp.inference_model=resnet' 'optim.epochs=10000' 'optim.lr_inr=1e-3' 'inr.base_freq_multiplier=1' 'inr.input_scales=[1/8, 1/8, 1/4, 1/2]' 'inr.output_layers=[3]' 'inr.filter_type=gabor' 'inr.modulate_scale=True'


python3 experiments/inr_sdf.py "optim.batch_size=64" "optim.batch_size_val=64" "inr.latent_dim=256" "inr.hidden_dim=128" "inr.depth=5" "optim.lr_inr=5e-5" 'inr.w0=5' "data.data_to_encode=sdf" "inr.scale_factor=1" "inr.model_type=pos_embedder" 'inr.modulate_scale=False' 'optim.epochs=5000' 'optim.meta_lr_code=0' 'optim.weight_decay_code=0' 


