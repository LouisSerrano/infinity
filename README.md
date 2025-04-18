# INFINITY: Code of the INFINITY Paper  
📄 [OpenReview Link](https://openreview.net/forum?id=B3n6VOBTjx#all)

## 📚 Citation

To cite our work:

```bibtex
@inproceedings{
  serrano2023infinity,
  title={{INFINITY}: Neural Field Modeling for Reynolds-Averaged Navier-Stokes Equations},
  author={Louis Serrano and L{\'e}on Migus and Yuan Yin and Jocelyn Ahmed Mazari and Jean-No{\"e}l Vittaut and Patrick Gallinari},
  booktitle={1st Workshop on the Synergy of Scientific and Machine Learning Modeling @ ICML2023},
  year={2023},
  url={https://openreview.net/forum?id=B3n6VOBTjx}
}
```

---

## 🔧 1. Code Installation and Setup

### 🧪 Create and install environment

```bash
conda create -n infinity-env python=3.9.0
conda activate infinity-env
pip install -e .
```

### 📁 Create a `slurm_run` directory

```bash
mkdir slurm_run
```

### 🪄 Setup your `wandb` config

Add the following to your `~/.bashrc`:

```bash
export WANDB_API_TOKEN=your_key
export WANDB_DIR=your_dir
export WANDB_CACHE_DIR=your_cache_dir
export WANDB_CONFIG_DIR="${WANDB_DIR}config/wandb"
export MINICONDA_PATH=your_anaconda_path
```

---

## 📦 2. Dataset Download

Download and unzip the dataset:

```bash
conda activate infinity-env
python download_data.py --data_dir my_data_dir

cd my_data_dir
unzip Dataset.zip
rm Dataset.zip
```

> 💡 **Note:** Dataset preprocessing is slow (10–15 minutes on load). For debugging, we recommend using the `scarce` regime instead of `full`.

---

## 🏋️‍♂️ 3. Training

### 🛠 Notes

- The simplest setup is to train:
  - an INR to auto-decode the **physical fields**.
  - an INR to auto-decode the **SDF** (surface distance field).
- Autodecoding the **normals** is optional.
- For the ICML workshop paper, separate INRs were used per modality. This is often unnecessary and computationally expensive.
- You can use a shared INR for physics fields, but frequency interference may reduce airfoil profile quality.
- Training an INR for 10,000 epochs can take up to **36 hours** on an RTX 24GB GPU.

### 🚀 Run training

#### Physical Fields (Full Regime)

```bash
sbatch bash/full/inr_physics_fields.sh
```

#### SDF

```bash
sbatch bash/full/inr_sdf.sh
```

> ℹ️ The latent dimension must be the same for both SDF and field encodings.

### 🔁 Regression Script

Using the two previous `run_names`:

```bash
sbatch bash/full/regression.sh
```

You can change the `task` parameter to adjust the dataset configuration.

---

## 🧪 4. Testing

- Volume and surface metrics are computed at the end of training automatically.
- To compute **drag and lift metrics**, run:

```bash
sbatch bash/test/test.sh
```
