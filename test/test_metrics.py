import os
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader

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

    trainset = manifest[task + "_train"]
    testset = manifest[task + "_test"] if task != "scarce" else manifest["full_test"]

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
        sample=None,
        n_boot=4000,
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

    for substep, (graph, idx) in enumerate(test_loader):
        model.eval()
        graph = graph.cuda()
        n_samples = 1  # len(graph)

        ipt = torch.cat(
            [
                graph.z_sdf,
                graph.z_n,
                graph.inlet_x.unsqueeze(-1),
                graph.inlet_y.unsqueeze(-1),
            ],
            axis=-1,
        )
        pred = model(ipt)
        loss = ((pred - graph.z) ** 2).mean()
        code_test_mse += loss.item() * n_samples

        graph.pos = torch.cat([graph.pos, graph.sdf], axis=-1)
        z_pred = pred.reshape(-1, 4, latent_dim)
        z_vx_pred = z_pred[:, 0, :] * sigma_vx.cuda() + mu_vx.cuda()
        z_vy_pred = z_pred[:, 1, :] * sigma_vy.cuda() + mu_vy.cuda()
        z_p_pred = z_pred[:, 2, :] * sigma_p.cuda() + mu_p.cuda()
        z_nu_pred = z_pred[:, 3, :] * sigma_nu.cuda() + mu_nu.cuda()

        # num_points = n_samples * 4000
        mask = ...
        mask_surf = graph.surface

        with torch.no_grad():
            vx_pred = inr_vx.modulated_forward(
                graph.pos[mask], z_vx_pred[graph.batch[mask]]
            )
            vy_pred = inr_vy.modulated_forward(
                graph.pos[mask], z_vy_pred[graph.batch[mask]]
            )
            p_pred = inr_p.modulated_forward(
                graph.pos[mask], z_p_pred[graph.batch[mask]]
            )
            nu_pred = inr_nu.modulated_forward(
                graph.pos[mask], z_nu_pred[graph.batch[mask]]
            )

        vx_test_mse += ((vx_pred - graph.vx[mask]) ** 2).mean().item() * n_samples
        vy_test_mse += ((vy_pred - graph.vy[mask]) ** 2).mean().item() * n_samples
        p_test_mse += ((p_pred - graph.p[mask]) ** 2).mean().item() * n_samples
        p_surf_test_mse += (
            (p_pred[mask_surf] - graph.p[mask][mask_surf]) ** 2
        ).mean().item() * n_samples
        nu_test_mse += ((nu_pred - graph.nu[mask]) ** 2).mean().item() * n_samples

        # print('graph.pos', graph.pos.shape, graph.inlet_x.shape, graph.inlet_y.shape, graph.sdf.shape, graph.nx.shape, graph.ny.shape)
        graph_pred = Data(
            x=torch.cat(
                [
                    graph.pos,
                    graph.inlet_x.unsqueeze(0).repeat(graph.pos.shape[0], 1),
                    graph.inlet_y.unsqueeze(0).repeat(graph.pos.shape[0], 1),
                    graph.sdf,
                    graph.nx,
                    graph.ny,
                ],
                axis=-1,
            )
        )
        graph_pred.y = torch.cat([vx_pred, vy_pred, p_pred, nu_pred], axis=-1)
        graph_pred.surface = graph.surface

        testset_pred.append(graph_pred.cpu())

    code_test_loss = code_test_mse / ntest
    vx_test_mse = vx_test_mse / ntest
    vy_test_mse = vy_test_mse / ntest
    p_test_mse = p_test_mse / ntest
    p_surf_test_mse = p_surf_test_mse / ntest
    nu_test_mse = nu_test_mse / ntest

    print(
        f"Test code: {code_test_loss}, vx: {vx_test_mse}, vy: {vy_test_mse}, p: {p_test_mse}, nu: {nu_test_mse}, p surf: {p_surf_test_mse}"
    )

    wandb.log(
        {
            "code_test_loss": code_test_loss,
            "vx_test_mse": vx_test_mse,
            "vy_test_mse": vy_test_mse,
            "p_test_mse": p_test_mse,
            "p_surf_test_mse": p_surf_test_mse,
            "nu_test_mse": nu_test_mse,
        },
    )

    # next steps is to obtain the physics results directly

    # Compute scores and all metrics for a
    sns.set()
    path_in = "/data/serrano/airfrans/Dataset/"
    with open(path_in + "manifest.json", "r") as f:
        manifest = json.load(f)

    task = cfg.data.task
    test_dataset_list = manifest[task + "_test"]
    # idx = random.sample(range(len(test_dataset)), k = n_test)
    # idx.sort()

    # test_dataset_vtk = Dataset(test_dataset, sample = None, coef_norm = coef_norm)
    # test_loader = DataLoader(test_dataset_vtk, shuffle = False)

    criterion = "MSE"
    if criterion == "MSE":
        criterion = nn.MSELoss(reduction="none")
    elif criterion == "MAE":
        criterion = nn.L1Loss(reduction="none")

    scores_vol = []
    scores_surf = []
    scores_force = []
    scores_p = []
    scores_wss = []
    internals = []
    airfoils = []
    true_internals = []
    true_airfoils = []
    times = []
    true_coefs = []
    pred_coefs = []

    avg_loss_per_var = np.zeros((1, 4))
    avg_loss = np.zeros(1)
    avg_loss_surf_var = np.zeros((1, 4))
    avg_loss_vol_var = np.zeros((1, 4))
    avg_loss_surf = np.zeros(1)
    avg_loss_vol = np.zeros(1)
    avg_rel_err_force = np.zeros((1, 2))
    avg_loss_p = np.zeros((1))
    avg_loss_wss = np.zeros((1, 2))
    internal = []
    airfoil = []
    pred_coef = []

    for j, data in enumerate(tqdm(test_loader)):
        Uinf, angle = float(test_dataset_list[j].split("_")[2]), float(
            test_dataset_list[j].split("_")[3]
        )
        print(f"uinf: {Uinf}, angle: {angle}")
        data[0].y = torch.cat([data[0].vx, data[0].vy, data[0].p, data[0].nu], axis=-1)

        # to replace >>>>> outs, tim = Infer_test(device, model, hparams, data, coef_norm = coef_norm)
        outs = [testset_pred[j]]
        # times.append(tim)
        intern = pv.read(
            "/data/serrano/airfrans/Dataset/"
            + test_dataset_list[j]
            + "/"
            + test_dataset_list[j]
            + "_internal.vtu"
        )
        aerofoil = pv.read(
            "/data/serrano/airfrans/Dataset/"
            + test_dataset_list[j]
            + "/"
            + test_dataset_list[j]
            + "_aerofoil.vtp"
        )
        tc, true_intern, true_airfoil = Compute_coefficients(
            [intern], [aerofoil], data[0].surface, Uinf, angle, keep_vtk=True
        )
        tc, true_intern, true_airfoil = tc[0], true_intern[0], true_airfoil[0]
        intern, aerofoil = Airfoil_test(
            intern, aerofoil, [out.y for out in outs], coef_norm, data[0].surface
        )
        pc, intern, aerofoil = Compute_coefficients(
            intern, aerofoil, data[0].surface, Uinf, angle, keep_vtk=True
        )

        true_coefs.append(tc)
        pred_coef.append(pc)

        internal.append(intern)
        airfoil.append(aerofoil)
        true_internals.append(true_intern)
        true_airfoils.append(true_airfoil)

        for n, out in enumerate(outs):
            # print('out', out.y.shape)
            # print('target', data[0].y.shape)
            loss_per_var = criterion(out.y, data[0].y).mean(dim=0)
            loss = loss_per_var.mean()
            loss_surf_var = criterion(
                out.y[data[0].surface, :], data[0].y[data[0].surface, :]
            ).mean(dim=0)
            loss_vol_var = criterion(
                out.y[~data[0].surface, :], data[0].y[~data[0].surface, :]
            ).mean(dim=0)
            loss_surf = loss_surf_var.mean()
            loss_vol = loss_vol_var.mean()

            avg_loss_per_var[n] += loss_per_var.cpu().numpy()
            avg_loss[n] += loss.cpu().numpy()
            avg_loss_surf_var[n] += loss_surf_var.cpu().numpy()
            avg_loss_vol_var[n] += loss_vol_var.cpu().numpy()
            avg_loss_surf[n] += loss_surf.cpu().numpy()
            avg_loss_vol[n] += loss_vol.cpu().numpy()
            avg_rel_err_force[n] += rel_err(tc, pc[n])
            avg_loss_wss[n] += rel_err(
                true_airfoil.point_data["wallShearStress"],
                aerofoil[n].point_data["wallShearStress"],
            ).mean(axis=0)
            avg_loss_p[n] += rel_err(
                true_airfoil.point_data["p"], aerofoil[n].point_data["p"]
            ).mean(axis=0)

    internals.append(internal)
    airfoils.append(airfoil)
    pred_coefs.append(pred_coef)

    score_var = np.array(avg_loss_per_var) / ntest
    score = np.array(avg_loss) / ntest
    score_surf_var = np.array(avg_loss_surf_var) / ntest
    score_vol_var = np.array(avg_loss_vol_var) / ntest
    score_surf = np.array(avg_loss_surf) / ntest
    score_vol = np.array(avg_loss_vol) / ntest
    score_force = np.array(avg_rel_err_force) / ntest
    score_p = np.array(avg_loss_p) / ntest
    score_wss = np.array(avg_loss_wss) / ntest

    score = score_surf + score_vol
    scores_vol.append(score_vol_var)
    scores_surf.append(score_surf_var)
    scores_force.append(score_force)
    scores_p.append(score_p)
    scores_wss.append(score_wss)

    scores_vol = np.array(scores_vol)
    scores_surf = np.array(scores_surf)
    scores_force = np.array(scores_force)
    scores_p = np.array(scores_p)
    scores_wss = np.array(scores_wss)
    times = np.array(times)
    true_coefs = np.array(true_coefs)
    pred_coefs = np.array(pred_coefs)
    pred_coefs_mean = pred_coefs.mean(axis=0)
    pred_coefs_std = pred_coefs.std(axis=0)

    spear_coefs = []
    for j in range(pred_coefs.shape[0]):
        spear_coef = []
        for k in range(pred_coefs.shape[2]):
            spear_drag = sc.stats.spearmanr(true_coefs[:, 0], pred_coefs[j, :, k, 0])[0]
            spear_lift = sc.stats.spearmanr(true_coefs[:, 1], pred_coefs[j, :, k, 1])[0]
            spear_coef.append([spear_drag, spear_lift])
        spear_coefs.append(spear_coef)
    spear_coefs = np.array(spear_coefs)

    print(
        {
            #'mean_time': times.mean(axis = 0),
            #'std_time': times.std(axis = 0),
            "mean_score_vol": scores_vol.mean(axis=0),
            "std_score_vol": scores_vol.std(axis=0),
            "mean_score_surf": scores_surf.mean(axis=0),
            "std_score_surf": scores_surf.std(axis=0),
            "mean_rel_p": scores_p.mean(axis=0),
            "std_rel_p": scores_p.std(axis=0),
            "mean_rel_wss": scores_wss.mean(axis=0),
            "std_rel_wss": scores_wss.std(axis=0),
            "mean_score_force": scores_force.mean(axis=0),
            "std_score_force": scores_force.std(axis=0),
            "spearman_coef_mean": spear_coefs.mean(axis=0),
            "spearman_coef_std": spear_coefs.std(axis=0),
        }
    )

    with open(f"{RESULTS_DIR}/score_{load_run_name}.json", "w") as f:
        json.dump(
            {
                #'mean_time': times.mean(axis = 0),
                #'std_time': times.std(axis = 0),
                "mean_score_vol": scores_vol.mean(axis=0),
                "std_score_vol": scores_vol.std(axis=0),
                "mean_score_surf": scores_surf.mean(axis=0),
                "std_score_surf": scores_surf.std(axis=0),
                "mean_rel_p": scores_p.mean(axis=0),
                "std_rel_p": scores_p.std(axis=0),
                "mean_rel_wss": scores_wss.mean(axis=0),
                "std_rel_wss": scores_wss.std(axis=0),
                "mean_score_force": scores_force.mean(axis=0),
                "std_score_force": scores_force.std(axis=0),
                "spearman_coef_mean": spear_coefs.mean(axis=0),
                "spearman_coef_std": spear_coefs.std(axis=0),
            },
            f,
            indent=4,
            cls=NumpyEncoder,
        )

    return scores_force, spear_coefs


if __name__ == "__main__":
    main()
