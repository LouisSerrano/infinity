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
from infinity.data.dataset import GeometryDatasetFull, set_seed
from infinity.mlp import ResNet


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

    # inr
    run_name_vx = cfg.inr.run_dict.vx  # "bright-totem-286"
    run_name_vy = cfg.inr.run_dict.vy  # "devoted-puddle-287"
    run_name_p = cfg.inr.run_dict.p  # "serene-vortex-284"
    run_name_nu = cfg.inr.run_dict.nu  # "wandering-bee-288"
    run_name_sdf = cfg.inr.run_dict.sdf  # "earnest-paper-289"
    run_name_n = cfg.inr.run_dict.n  # "astral-leaf-330"

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

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # test
    test_loader = DataLoader(testset, batch_size=batch_size_val, shuffle=True)

    model = ResNet(
        input_dim=2 * latent_dim + 2,
        hidden_dim=width,
        output_dim=4 * latent_dim,
        depth=depth,
        dropout=0.0,
        activation=activation,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=100,
        threshold=0.01,
        threshold_mode="rel",
        cooldown=0,
        min_lr=5e-5,
        eps=1e-08,
        verbose=True,
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

    for step in range(epochs):
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

        for substep, (graph, idx) in enumerate(train_loader):
            model.train()
            graph = graph.cuda()
            n_samples = len(graph)

            # print(graph.z_sdf.shape, graph.z_n.shape, graph.inlet_x.shape, graph.inlet_y.shape)

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            code_train_mse += loss.item() * n_samples

            if step_show:
                num_points = n_samples * 4000
                mask = torch.randperm(graph.pos.shape[0])[:num_points]
                mask_surf = graph.surface[mask]

                graph.pos = torch.cat([graph.pos, graph.sdf], axis=-1)
                z_pred = pred.reshape(-1, 4, latent_dim)
                z_vx_pred = z_pred[:, 0, :] * sigma_vx + mu_vx
                z_vy_pred = z_pred[:, 1, :] * sigma_vy + mu_vy
                z_p_pred = z_pred[:, 2, :] * sigma_p + mu_p
                z_nu_pred = z_pred[:, 3, :] * sigma_nu + mu_nu

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

                vx_train_mse += (
                    (vx_pred - graph.vx[mask]) ** 2
                ).mean().item() * n_samples
                vy_train_mse += (
                    (vy_pred - graph.vy[mask]) ** 2
                ).mean().item() * n_samples
                p_train_mse += ((p_pred - graph.p[mask]) ** 2).mean().item() * n_samples
                p_surf_train_mse += (
                    (p_pred[mask_surf] - graph.p[mask][mask_surf]) ** 2
                ).mean().item() * n_samples
                nu_train_mse += (
                    (nu_pred - graph.nu[mask]) ** 2
                ).mean().item() * n_samples

        code_train_loss = code_train_mse / ntrain
        scheduler.step(code_train_loss)
        if step % 20 == 0:
            print("train code", step, code_train_loss)
        if step_show:
            vx_train_mse = vx_train_mse / ntrain
            vy_train_mse = vy_train_mse / ntrain
            p_train_mse = p_train_mse / ntrain
            p_surf_train_mse = p_surf_train_mse / ntrain
            nu_train_mse = nu_train_mse / ntrain
            print(
                f"train {step} vx: {vx_train_mse}, vy: {vy_train_mse}, p: {p_train_mse}, nu: {nu_train_mse}, p surf: {p_surf_train_mse}"
            )

        if step_show:
            for substep, (graph, idx) in enumerate(test_loader):
                model.eval()
                graph = graph.cuda()
                n_samples = len(graph)

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
                z_vx_pred = z_pred[:, 0, :] * sigma_vx + mu_vx
                z_vy_pred = z_pred[:, 1, :] * sigma_vy + mu_vy
                z_p_pred = z_pred[:, 2, :] * sigma_p + mu_p
                z_nu_pred = z_pred[:, 3, :] * sigma_nu + mu_nu

                num_points = n_samples * 4000
                mask = torch.randperm(graph.pos.shape[0])[:num_points]
                mask_surf = graph.surface[mask]

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

                vx_test_mse += (
                    (vx_pred - graph.vx[mask]) ** 2
                ).mean().item() * n_samples
                vy_test_mse += (
                    (vy_pred - graph.vy[mask]) ** 2
                ).mean().item() * n_samples
                p_test_mse += ((p_pred - graph.p[mask]) ** 2).mean().item() * n_samples
                p_surf_test_mse += (
                    (p_pred[mask_surf] - graph.p[mask][mask_surf]) ** 2
                ).mean().item() * n_samples
                nu_test_mse += (
                    (nu_pred - graph.nu[mask]) ** 2
                ).mean().item() * n_samples

            code_test_loss = code_test_mse / ntest
            vx_test_mse = vx_test_mse / ntest
            vy_test_mse = vy_test_mse / ntest
            p_test_mse = p_test_mse / ntest
            p_surf_test_mse = p_surf_test_mse / ntest
            nu_test_mse = nu_test_mse / ntest

            print(
                f"Test {step} code: {code_test_loss}, vx: {vx_test_mse}, vy: {vy_test_mse}, p: {p_test_mse}, nu: {nu_test_mse}, p surf: {p_surf_test_mse}"
            )

        if step_show:
            wandb.log(
                {
                    "code_train_loss": code_train_loss,
                    "code_test_loss": code_test_loss,
                    "vx_train_mse": vx_train_mse,
                    "vx_test_mse": vx_test_mse,
                    "vy_train_mse": vy_train_mse,
                    "vy_test_mse": vy_test_mse,
                    "p_train_mse": p_train_mse,
                    "p_test_mse": p_test_mse,
                    "p_surf_train_mse": p_surf_train_mse,
                    "p_surf_test_mse": p_surf_test_mse,
                    "nu_train_mse": nu_train_mse,
                    "nu_test_mse": nu_test_mse,
                },
            )

        else:
            wandb.log(
                {
                    "code_train_loss": code_train_loss,
                },
                step=step,
                commit=not step_show,
            )

        if code_train_loss < best_loss:
            best_loss = code_train_loss

            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "model": model.state_dict(),
                    "optimizer_model": optimizer.state_dict(),
                    "loss": code_test_loss,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )

    # load the best model during training
    load_dict = torch.load(f"{RESULTS_DIR}/{run_name}.pt")
    model.load_state_dict(load_dict["model"])
    model.eval()

    # test with full points (no masking)
    test_loader_full = DataLoader(testset, batch_size=2, shuffle=True)

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

    for substep, (graph, idx) in enumerate(test_loader_full):
        model.eval()
        graph = graph.cuda()
        n_samples = len(graph)

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
        z_vx_pred = z_pred[:, 0, :] * sigma_vx + mu_vx
        z_vy_pred = z_pred[:, 1, :] * sigma_vy + mu_vy
        z_p_pred = z_pred[:, 2, :] * sigma_p + mu_p
        z_nu_pred = z_pred[:, 3, :] * sigma_nu + mu_nu

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
            (p_pred[mask_surf] - graph.p[mask_surf]) ** 2
        ).mean().item() * n_samples
        nu_test_mse += ((nu_pred - graph.nu[mask]) ** 2).mean().item() * n_samples

    code_test_loss = code_test_mse / ntest
    vx_test_mse = vx_test_mse / ntest
    vy_test_mse = vy_test_mse / ntest
    p_test_mse = p_test_mse / ntest
    p_surf_test_mse = p_surf_test_mse / ntest
    nu_test_mse = nu_test_mse / ntest

    print(
        f"Test code: {code_test_loss}, vx: {vx_test_mse}, vy: {vy_test_mse}, p: {p_test_mse}, nu: {nu_test_mse}, p surf: {p_surf_test_mse}"
    )

    # next steps is to obtain the physics results directly

    return code_test_loss


if __name__ == "__main__":
    main()
