import os
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader
from time import time

import json
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from functools import partial
import torch.nn.functional as F
import pdb

from infinity.utils.load_inr import load_inr_model
from infinity.utils.load_modulations import load_modulations
from infinity.utils.reorganize import reorganize
from infinity.data.dataset import GeometryDatasetFull, set_seed
from infinity.mlp import ResNet
from infinity.utils.metrics import (
    Airfoil_mean,
    Airfoil_test,
    Compute_coefficients,
    rel_err,
)
from infinity.utils import metrics_NACA
from tqdm import tqdm
import pyvista as pv
import scipy as sc
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from infinity.graph_metalearning import outer_step
import seaborn as sns


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def Airfoil_test(internal, airfoil, outs, coef_norm, bool_surf):
    # Produce multiple copies of a simulation for different predictions.
    # stocker les internals, airfoils, calculer le wss, calculer le drag, le lift, plot pressure coef, plot skin friction coef, plot drag/drag, plot lift/lift
    # calcul spearsman coef, boundary layer
    internals = []
    airfoils = []
    for out in outs:
        intern = internal.copy()
        aerofoil = airfoil.copy()

        point_mesh = intern.points[bool_surf, :2]
        point_surf = aerofoil.points[:, :2]
        # print('out', out)
        out = (out * (coef_norm[3] + 1e-8) + coef_norm[2]).numpy()
        out[bool_surf.numpy(), :2] = np.zeros_like(out[bool_surf.numpy(), :2])
        out[bool_surf.numpy(), 3] = np.zeros_like(out[bool_surf.numpy(), 3])
        intern.point_data["U"][:, :2] = out[:, :2]
        intern.point_data["p"] = out[:, 2]
        intern.point_data["nut"] = out[:, 3]

        surf_p = intern.point_data["p"][bool_surf]
        surf_p = reorganize(point_mesh, point_surf, surf_p)
        aerofoil.point_data["p"] = surf_p

        intern = intern.ptc(pass_point_data=True)
        aerofoil = aerofoil.ptc(pass_point_data=True)

        internals.append(intern)
        airfoils.append(aerofoil)

    return internals, airfoils


@hydra.main(config_path="config/", config_name="regression.yaml")
def main(cfg: DictConfig) -> None:
    #
    # data
    data_dir = cfg.data.dir
    task = cfg.data.task
    data_to_encode = cfg.data.data_to_encode
    score = cfg.data.score
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    seed = cfg.data.seed

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    epochs = cfg.optim.epochs
    lr = cfg.optim.lr
    weight_decay = cfg.optim.weight_decay

    # model
    load_run_name = cfg.model.run_name

    # model
    model_type = cfg.model.model_type
    depth = cfg.model.depth
    width = cfg.model.width
    activation = cfg.model.activation

    # wandb
    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = cfg.wandb.id
    run_name = cfg.wandb.name
    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"), f"wandb/{cfg.wandb.dir}")
        if cfg.wandb.dir is not None
        else None
    )

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=None,
    )

    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    run_name = wandb.run.name

    print("id", run.id)
    print("dir", run.dir)

    LOAD_DIR = Path(os.getenv("WANDB_DIR")) / "airfrans" / task / "inr"
    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / "airfrans" / task / "model"
    MODULATIONS_DIR = Path(os.getenv("WANDB_DIR")) / "airfrans" / task / "modulations"

    load_model_input = torch.load(RESULTS_DIR / f"{load_run_name}.pt")
    input_cfg = load_model_input["cfg"]

    # inr
    run_name_vx = input_cfg.inr.run_dict.vx  # "bright-totem-286"
    run_name_vy = input_cfg.inr.run_dict.vy  # "devoted-puddle-287"
    run_name_p = input_cfg.inr.run_dict.p  # "serene-vortex-284"
    run_name_nu = input_cfg.inr.run_dict.nu  # "wandering-bee-288"
    run_name_sdf = input_cfg.inr.run_dict.sdf  # "earnest-paper-289"
    run_name_n = input_cfg.inr.run_dict.n  # "astral-leaf-330"

    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    os.makedirs(str(MODULATIONS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

    set_seed(seed)

    run.tags = ("regression",) + (task,) + (model_type,)

    # train
    with open(Path(data_dir) / "Dataset/manifest.json", "r") as f:
        manifest = json.load(f)

    trainset = manifest[task + "_train"][:10]
    testset = (
        manifest[task + "_test"][:100]
        if task != "scarce"
        else manifest["full_test"][:10]
    )

    ntrain = len(trainset)
    ntest = len(testset)
    tmp = torch.load(LOAD_DIR / "vx" / f"{run_name_vx}.pt")
    latent_dim = tmp["cfg"].inr.latent_dim

    # default sample is none
    trainset = GeometryDatasetFull(
        trainset,
        key=data_to_encode,
        latent_dim=latent_dim,
        norm=True,
        sample=None,
        n_boot=4000,
    )
    testset = GeometryDatasetFull(
        testset,
        key=data_to_encode,
        latent_dim=latent_dim,
        sample="mesh",
        n_boot=150000,
        coef_norm=trainset.coef_norm,
    )
    coef_norm = trainset.coef_norm
    device = torch.device("cuda")

    # load inr and modulations

    inr_vx, alpha_vx = load_inr_model(
        LOAD_DIR / "vx", run_name_vx, input_dim=3, output_dim=1
    )
    inr_vy, alpha_vy = load_inr_model(
        LOAD_DIR / "vy", run_name_vy, input_dim=3, output_dim=1
    )
    inr_p, alpha_p = load_inr_model(
        LOAD_DIR / "p", run_name_p, input_dim=3, output_dim=1
    )
    inr_nu, alpha_nu = load_inr_model(
        LOAD_DIR / "nu", run_name_nu, input_dim=3, output_dim=1
    )
    inr_sdf, alpha_sdf = load_inr_model(
        LOAD_DIR / "sdf", run_name_sdf, input_dim=2, output_dim=1
    )
    inr_n, alpha_n = load_inr_model(
        LOAD_DIR / "n", run_name_n, input_dim=2, output_dim=2
    )

    mod_vx = load_modulations(
        trainset, testset, inr_vx, MODULATIONS_DIR, run_name_vx, "vx", alpha=alpha_vx
    )
    mod_vy = load_modulations(
        trainset, testset, inr_vy, MODULATIONS_DIR, run_name_vy, "vy", alpha=alpha_vy
    )
    mod_p = load_modulations(
        trainset, testset, inr_p, MODULATIONS_DIR, run_name_p, "p", alpha=alpha_p
    )
    mod_nu = load_modulations(
        trainset, testset, inr_nu, MODULATIONS_DIR, run_name_nu, "nu", alpha=alpha_nu
    )
    mod_sdf = load_modulations(
        trainset,
        testset,
        inr_sdf,
        MODULATIONS_DIR,
        run_name_sdf,
        "sdf",
        alpha=alpha_sdf,
        input_dim=2,
    )
    mod_n = load_modulations(
        trainset,
        testset,
        inr_n,
        MODULATIONS_DIR,
        run_name_n,
        "n",
        alpha=alpha_n,
        input_dim=2,
    )

    mu_vx = mod_vx["z_train"].mean(0)
    sigma_vx = mod_vx["z_train"].std(0)
    mu_vy = mod_vy["z_train"].mean(0)
    sigma_vy = mod_vy["z_train"].std(0)
    mu_p = mod_p["z_train"].mean(0)
    sigma_p = mod_p["z_train"].std(0)
    mu_nu = mod_nu["z_train"].mean(0)
    sigma_nu = mod_nu["z_train"].std(0)
    mu_sdf = mod_sdf["z_train"].mean(0)
    sigma_sdf = mod_sdf["z_train"].std(0)
    mu_n = mod_n["z_train"].mean(0)
    sigma_n = mod_n["z_train"].std(0)

    print("mu_sdf train", mu_sdf, sigma_sdf)
    print("mu_sdf test", mod_sdf["z_test"].mean(0), mod_sdf["z_test"].std(0))

    trainset.out_modulations["vx"] = (mod_vx["z_train"] - mu_vx) / sigma_vx
    trainset.out_modulations["vy"] = (mod_vy["z_train"] - mu_vy) / sigma_vy
    trainset.out_modulations["p"] = (mod_p["z_train"] - mu_p) / sigma_p
    trainset.out_modulations["nu"] = (mod_nu["z_train"] - mu_nu) / sigma_nu
    trainset.in_modulations["sdf"] = (mod_sdf["z_train"] - mu_sdf) / sigma_sdf
    trainset.in_modulations["n"] = (mod_n["z_train"] - mu_n) / sigma_n

    testset.out_modulations["vx"] = (mod_vx["z_test"] - mu_vx) / sigma_vx
    testset.out_modulations["vy"] = (mod_vy["z_test"] - mu_vy) / sigma_vy
    testset.out_modulations["p"] = (mod_p["z_test"] - mu_p) / sigma_p
    testset.out_modulations["nu"] = (mod_nu["z_test"] - mu_nu) / sigma_nu
    testset.in_modulations["sdf"] = (mod_sdf["z_test"] - mu_sdf) / sigma_sdf
    testset.in_modulations["n"] = (mod_n["z_test"] - mu_n) / sigma_n

    print("trainset", trainset.out_modulations)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    # test
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    model = ResNet(
        input_dim=2 * latent_dim + 2,
        hidden_dim=width,
        output_dim=4 * latent_dim,
        depth=depth,
        dropout=0.0,
        activation=activation,
    ).cuda()

    model.load_state_dict(load_model_input["model"])
    model.eval()

    vx_params = sum(p.numel() for p in inr_vx.parameters())
    vy_params = sum(p.numel() for p in inr_vy.parameters())
    p_params = sum(p.numel() for p in inr_p.parameters())
    nu_params = sum(p.numel() for p in inr_nu.parameters())
    sdf_params = sum(p.numel() for p in inr_sdf.parameters())
    n_params = sum(p.numel() for p in inr_n.parameters())
    model_params = sum(p.numel() for p in model.parameters())
    total_params = (
        vx_params
        + vy_params
        + p_params
        + nu_params
        + sdf_params
        + n_params
        + model_params
    )

    print(
        f"Total : {total_params}, vx {vx_params}, vy: {vy_params}, p: {p_params}, nu: {nu_params}, sdf: {sdf_params}, n: {n_params}, model : {model_params}"
    )

    best_loss = np.inf

    mu_vx = mu_vx.cuda()
    sigma_vx = sigma_vx.cuda()
    mu_vy = mu_vy.cuda()
    sigma_vy = sigma_vy.cuda()
    mu_p = mu_p.cuda()
    sigma_p = sigma_p.cuda()
    mu_nu = mu_nu.cuda()
    sigma_nu = sigma_nu.cuda()
    mu_sdf = mu_sdf.cuda()
    sigma_sdf = sigma_sdf.cuda()
    mu_n = mu_n.cuda()
    sigma_n = sigma_n.cuda()

    inner_steps = 3

    code_train_mse = 0
    code_test_mse = 0
    vx_train_mse = 0
    vx_test_mse = 0
    vy_train_mse = 0
    vy_test_mse = 0
    p_train_mse = 0
    p_test_mse = 0
    p_surf_train_mse = 0
    p_surf_test_mse = 0
    nu_train_mse = 0
    nu_test_mse = 0

    step_show = True

    testset_pred = []

    model.eval()
    # encode sdf

    num_tries = 100

    t_in = time()

    time_list = []

    for graph, j in test_loader:
        graph = graph.cuda()
        n_samples = 1  # len(graph)

        t_step_in = time()

        graph.tmp_pos = graph.pos
        graph.tmp_batch = graph.batch

        graph = graph.cuda()
        n_samples = len(graph)

        graph.images = graph.sdf
        graph.batch = graph.tmp_batch
        graph.pos = graph.tmp_pos
        graph.modulations = torch.zeros((len(graph), latent_dim)).cuda()

        outputs = outer_step(
            inr_sdf,
            graph,
            inner_steps,
            alpha_sdf,
            is_train=False,
            return_reconstructions=True,
            gradient_checkpointing=False,
            use_rel_loss=False,
        )
        z_sdf = outputs["modulations"]
        z_sdf = (z_sdf - mu_sdf) / (sigma_sdf)

        # encode n
        graph.images = torch.cat([graph.nx, graph.ny], axis=-1)
        mask = graph.surface
        graph.pos = graph.tmp_pos[mask]
        graph.batch = graph.tmp_batch[mask]
        graph.images = graph.images[mask]
        graph.modulations = torch.zeros((len(graph), latent_dim)).cuda()

        outputs = outer_step(
            inr_n,
            graph,
            inner_steps,
            alpha_n,
            is_train=False,
            return_reconstructions=True,
            gradient_checkpointing=False,
            use_rel_loss=False,
        )

        z_n = outputs["modulations"]
        z_n = (z_n - mu_sdf) / (sigma_sdf)

        with torch.no_grad():
            graph.batch = graph.tmp_batch
            ipt = torch.cat(
                [z_sdf, z_n, graph.inlet_x.unsqueeze(-1), graph.inlet_y.unsqueeze(-1)],
                axis=-1,
            )

            pred = model(ipt)

            graph.new_pos = torch.cat([graph.tmp_pos, graph.sdf], axis=-1)
            z_pred = pred.reshape(-1, 4, latent_dim)
            z_vx_pred = z_pred[:, 0, :] * sigma_vx + mu_vx
            z_vy_pred = z_pred[:, 1, :] * sigma_vy + mu_vy
            z_p_pred = z_pred[:, 2, :] * sigma_p + mu_p
            z_nu_pred = z_pred[:, 3, :] * sigma_nu + mu_nu

            # num_points = n_samples * 4000
            mask = ...
            mask_surf = graph.surface
            vx_pred = inr_vx.modulated_forward(
                graph.new_pos[mask], z_vx_pred[graph.batch[mask]]
            )
            vy_pred = inr_vy.modulated_forward(
                graph.new_pos[mask], z_vy_pred[graph.batch[mask]]
            )
            p_pred = inr_p.modulated_forward(
                graph.new_pos[mask], z_p_pred[graph.batch[mask]]
            )
            nu_pred = inr_nu.modulated_forward(
                graph.new_pos[mask], z_nu_pred[graph.batch[mask]]
            )
        t_step_out = time()
        time_list.append(t_step_out - t_step_in)

    t_out = time()

    print("one shot", (t_out - t_in) / num_tries)
    print("list", time_list)
    print("mean", np.array(time_list).mean(), np.array(time_list).std())

    return (t_out - t_in) / num_tries


if __name__ == "__main__":
    main()
