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
    include_normal = cfg.data.include_normal

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    epochs = cfg.optim.epochs
    lr = cfg.optim.lr
    weight_decay = cfg.optim.weight_decay

    # inr
    run_name_sdf = cfg.inr.run_dict.sdf
    run_name_n = cfg.inr.run_dict.n
    run_name_fields = cfg.inr.run_dict.all_physics_fields

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

    tmp = torch.load(
        LOAD_DIR / "all_physics_fields" / f"{run_name_fields}.pt", weights_only=False
    )
    latent_dim = tmp[
        "cfg"
    ].inr.latent_dim  # WARNING we suppose every INR has the same latent dim.

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

    inr_fields, alpha_fields = load_inr_model(
        LOAD_DIR / "all_physics_fields", run_name_fields, input_dim=3, output_dim=4
    )

    inr_sdf, alpha_sdf = load_inr_model(
        LOAD_DIR / "sdf", run_name_sdf, input_dim=2, output_dim=1
    )

    if include_normal:
        inr_n, alpha_n = load_inr_model(
            LOAD_DIR / "n", run_name_n, input_dim=2, output_dim=2
        )

    mod_fields = load_modulations(
        trainset,
        testset,
        inr_fields,
        MODULATIONS_DIR,
        run_name_fields,
        "all_physics_fields",
        alpha=alpha_fields,
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
    if include_normal:
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

    gamma = 1

    mu_fields = mod_fields["z_train"].mean(0)
    sigma_fields = mod_fields["z_train"].std(0) * gamma

    mu_sdf = mod_sdf["z_train"].mean(0)
    sigma_sdf = mod_sdf["z_train"].std(0)

    if include_normal:
        mu_n = mod_n["z_train"].mean(0)
        sigma_n = mod_n["z_train"].std(0)

    trainset.out_modulations["fields"] = (
        mod_fields["z_train"] - mu_fields
    ) / sigma_fields
    trainset.in_modulations["sdf"] = (mod_sdf["z_train"] - mu_sdf) / sigma_sdf

    if include_normal:
        trainset.in_modulations["n"] = (mod_n["z_train"] - mu_n) / sigma_n

    testset.out_modulations["fields"] = (
        mod_fields["z_test"] - mu_fields
    ) / sigma_fields
    testset.in_modulations["sdf"] = (mod_sdf["z_test"] - mu_sdf) / sigma_sdf

    if include_normal:
        testset.in_modulations["n"] = (mod_n["z_test"] - mu_n) / sigma_n

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # test
    test_loader = DataLoader(testset, batch_size=batch_size_val, shuffle=True)

    model = ResNet(
        input_dim=2 * latent_dim + 2 if include_normal else latent_dim + 2,
        hidden_dim=width,
        output_dim=latent_dim,
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

    mu_fields = mu_fields.cuda()
    sigma_fields = sigma_fields.cuda()

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

        step_show = step % 100 == 0

        for substep, (graph, idx) in enumerate(train_loader):
            model.train()
            graph = graph.cuda()
            n_samples = len(graph)

            if include_normal:
                inpt = torch.cat(
                    [
                        graph.z_sdf,
                        graph.z_n,
                        graph.inlet_x.unsqueeze(-1),
                        graph.inlet_y.unsqueeze(-1),
                    ],
                    axis=-1,
                )
            else:
                inpt = torch.cat(
                    [
                        graph.z_sdf,
                        graph.inlet_x.unsqueeze(-1),
                        graph.inlet_y.unsqueeze(-1),
                    ],
                    axis=-1,
                )
            z_pred = model(inpt)
            loss = ((z_pred - graph.z_fields) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            code_train_mse += loss.item() * n_samples

            if step_show:
                num_points = n_samples * 4000  # 4000 points per sample
                mask = torch.randperm(graph.pos.shape[0])[:num_points]
                batch = graph.batch[mask]
                mask_surf = graph.surface[mask]

                graph = graph.cuda()
                images = torch.cat([graph.vx, graph.vy, graph.p, graph.nu], axis=-1)

                pos = torch.cat([graph.pos, graph.sdf], axis=-1)
                with torch.no_grad():
                    pred = inr_fields.modulated_forward(
                        pos[mask], z_pred[batch] * sigma_fields + mu_fields
                    )
                    pred_surf = inr_fields.modulated_forward(
                        pos[graph.surface],
                        z_pred[graph.batch[graph.surface]] * sigma_fields + mu_fields,
                    )

                mse = ((pred - images[mask]) ** 2).mean(0)

                vx_train_mse += mse[0] * n_samples
                vy_train_mse += mse[1] * n_samples
                p_train_mse += mse[2] * n_samples
                nu_train_mse += mse[3] * n_samples
                p_surf_train_mse += (
                    (pred_surf[..., 2] - images[graph.surface, 2]) ** 2
                ).mean() * n_samples

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

                if include_normal:
                    inpt = torch.cat(
                        [
                            graph.z_sdf,
                            graph.z_n,
                            graph.inlet_x.unsqueeze(-1),
                            graph.inlet_y.unsqueeze(-1),
                        ],
                        axis=-1,
                    )

                else:
                    inpt = torch.cat(
                        [
                            graph.z_sdf,
                            graph.inlet_x.unsqueeze(-1),
                            graph.inlet_y.unsqueeze(-1),
                        ],
                        axis=-1,
                    )

                z_pred = model(inpt)
                loss = ((z_pred - graph.z_fields) ** 2).mean()
                code_test_mse += loss.item() * n_samples

                images = torch.cat([graph.vx, graph.vy, graph.p, graph.nu], axis=-1)

                num_points = n_samples * 4000
                mask = torch.randperm(graph.pos.shape[0])[:num_points]
                batch = graph.batch[mask]
                mask_surf = graph.surface[mask]

                pos = torch.cat([graph.pos, graph.sdf], axis=-1)
                with torch.no_grad():
                    pred = inr_fields.modulated_forward(
                        pos[mask], z_pred[batch] * sigma_fields + mu_fields
                    )
                    pred_surf = inr_fields.modulated_forward(
                        pos[graph.surface],
                        z_pred[graph.batch[graph.surface]] * sigma_fields + mu_fields,
                    )

                mse = ((pred - images[mask]) ** 2).mean(0)

                vx_test_mse += mse[0] * n_samples
                vy_test_mse += mse[1] * n_samples
                p_test_mse += mse[2] * n_samples
                nu_test_mse += mse[3] * n_samples
                p_surf_test_mse += (
                    (pred_surf[..., 2] - images[graph.surface, 2]) ** 2
                ).mean() * n_samples

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

    # test to have the final results at the end of training

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

        if include_normal:
            inpt = torch.cat(
                [
                    graph.z_sdf,
                    graph.z_n,
                    graph.inlet_x.unsqueeze(-1),
                    graph.inlet_y.unsqueeze(-1),
                ],
                axis=-1,
            )
        else:
            inpt = torch.cat(
                [
                    graph.z_sdf,
                    graph.inlet_x.unsqueeze(-1),
                    graph.inlet_y.unsqueeze(-1),
                ],
                axis=-1,
            )

        with torch.no_grad():
            z_pred = model(inpt)
            loss = ((z_pred - graph.z_fields) ** 2).mean()
            code_test_mse += loss.item() * n_samples

        mask = ...
        mask_surf = graph.surface

        images = torch.cat([graph.vx, graph.vy, graph.p, graph.nu], axis=-1)

        pos = torch.cat([graph.pos, graph.sdf], axis=-1)
        with torch.no_grad():
            pred = inr_fields.modulated_forward(
                pos[mask], z_pred[graph.batch] * sigma_fields + mu_fields
            )
            pred_surf = inr_fields.modulated_forward(
                pos[graph.surface],
                z_pred[graph.batch[graph.surface]] * sigma_fields + mu_fields,
            )

        mse = ((pred - images[mask]) ** 2).mean(0)

        vx_test_mse += mse[0] * n_samples
        vy_test_mse += mse[1] * n_samples
        p_test_mse += mse[2] * n_samples
        nu_test_mse += mse[3] * n_samples
        p_surf_test_mse += (
            (pred_surf[..., 2] - images[graph.surface, 2]) ** 2
        ).mean() * n_samples

    code_test_loss = code_test_mse / ntest
    vx_test_mse = vx_test_mse / ntest
    vy_test_mse = vy_test_mse / ntest
    p_test_mse = p_test_mse / ntest
    p_surf_test_mse = p_surf_test_mse / ntest
    nu_test_mse = nu_test_mse / ntest

    print("Test results without subsampling | Volume and Surface scores: \n")
    print(
        f"Test code: {code_test_loss}, vx: {vx_test_mse}, vy: {vy_test_mse}, p: {p_test_mse}, nu: {nu_test_mse}, p surf: {p_surf_test_mse}"
    )

    # next steps is to obtain the physics results directly

    return code_test_loss


if __name__ == "__main__":
    main()
