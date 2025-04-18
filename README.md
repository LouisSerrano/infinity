First create a new conda environment:
* conda create -n infinity_env python=3.9

Then install the infinity libray in develop mode:
* pip install -e .

Then create a slurm_run directory at the root of the repo:
* mkdir slurm_run

To launch an INR run, you can use the minimalist bash scripts: 
* fourier: sbatch bash/inr.sh
* siren: sbatch bash/inr_fourier.sh

To launch a regression between the codes, you can use the following command:
* sbatch bash/regression.sh

The regression has not been testes yet. 

The config for each channel used until now:

* vx: 
   * bright-totem-286"
   * fourier_features
   * embedding=nerf
   * max_frequencies=10
   * num_frequencies=32
   * base_frequency=2

* vy: 
   * "devoted-puddle-287"
   * fourier_features
   * embedding=nerf
   * max_frequencies=10
   * num_frequencies=32
   * base_frequency=2

* p: 
   * "serene-vortex-284"
   * fourier_features
   * embedding=nerf
   * max_frequencies=10
   * num_frequencies=32
   * base_frequency=2

* nu: 
   * "wandering-bee-288"
   * fourier_features
   * embedding=nerf
   * max_frequencies=10
   * num_frequencies=32
   * base_frequency=2

* sdf:
   * "earnest-paper-289"
   * siren
   * hidden_dim=128
   * depth=5
   * w0=5
   * meta_lr_code=0

* n:
   * "astral-leaf-330"
   * fourier_features
   * hidden_dim=128
   * depth=5
   * embedder=nerf
   * num_frequencies=128
   * max_frequencies=32 
   * base_frequency=1.25
 
* SDF small latent space:
  * misty-morning-762 for a very very small one
  * or lunar-field-749 for a small one with same results as before
* Joint learning of fields:
   * mild-sound-698
	
