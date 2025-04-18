import os
import sys
from pathlib import Path

import json
import hydra
import numpy as np
import torch
import torch.nn as nn
import pdb
import wandb
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from infinity.data.dataset import set_seed
from infinity.graph_metalearning import outer_step
from infinity.fourier_features import ModulatedFourierFeatures
from infinity.data.dataset import GeometryDatasetFull, KEY_TO_INDEX
from infinity.utils.load_inr import create_inr_instance

@hydra.main(config_path="config/", config_name="fourier_features.yaml")
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
    lr_inr = cfg.optim.lr_inr
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs
    weight_decay = cfg.optim.weight_decay

    # inr
    model_type = cfg.inr.model_type
    latent_dim = cfg.inr.latent_dim
    loss_type = cfg.inr.loss_type

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

    RESULTS_DIR = (
        Path(os.getenv("WANDB_DIR")) / "airfrans" / task / "inr" / data_to_encode
    )
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

    set_seed(seed)

    run.tags = ("inr",) + (task,) + (data_to_encode,) + (model_type,)

    # train
    try:
        with open(Path(data_dir) / "Dataset/manifest.json", "r") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        print("No manifest.json file found. You should download the airfrans dataset first.")

    trainset = manifest[task + "_train"]
    testset = manifest[task + "_test"] if task != "scarce" else manifest["full_test"]
    # n = int(.9*len(manifest_train))

    print("len train", len(trainset))
    print("len test", len(testset))

    ntrain = len(trainset)
    # nval = len(valset)
    ntest = len(testset)

    input_dim = 2
    output_dim = 1
    sample = "mesh"
    if data_to_encode in ["sdf", "n", "nx", "ny"]:
        include_sdf = False
        if data_to_encode == "n":
            output_dim = output_dim + 1  # (nx, ny)
        if data_to_encode in ["n", "nx", "ny"]:
            sample = "surface"
    else:
        include_sdf = True
        input_dim = input_dim + 1
    
    return_dim_loss = False
    if data_to_encode=='all_physics_fields':
        output_dim = output_dim + 3
        return_dim_loss = True

    # default sample is none
    trainset = GeometryDatasetFull(
        trainset,
        key=data_to_encode,
        latent_dim=latent_dim,
        norm=True,
        sample=sample,
        n_boot=4000,
    )
    testset = GeometryDatasetFull(
        testset,
        key=data_to_encode,
        latent_dim=latent_dim,
        sample=sample,
        n_boot=4000,
        coef_norm=trainset.coef_norm,
    )

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # test
    test_loader = DataLoader(testset, batch_size=batch_size_val, shuffle=True)
    device = torch.device("cuda")

    inr = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )

    alpha = nn.Parameter(torch.Tensor([lr_code]).cuda())
    meta_lr_code = meta_lr_code
    weight_decay_lr_code = weight_decay_code

    optimizer = torch.optim.AdamW(
        [
            {"params": inr.parameters(), "lr": lr_inr},
            {"params": alpha, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
        ],
        lr=lr_inr,
        weight_decay=0,
    )

    # This Plateau scheduler looks suspicious actually. 
    # It might be better to use a simple Cosine at every step (rather than epoch).

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer,
    #    mode="min",
    #    factor=0.5,
    #    patience=500,
    #    threshold=0.01,
    #    threshold_mode="rel",
    #    cooldown=0,
    #    min_lr=1e-5,
    #    eps=1e-08,
    #    verbose=True,
    #)

    total_steps = epochs * len(train_loader)  # 100 * 200 = 20,000
    scheduler = scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_loss = np.inf

    for step in range(epochs):
        fit_train_mse = 0
        fit_test_mse = 0
        step_show = step % 100 == 0

        if data_to_encode == 'all_physics_fields':
            fit_train_mse_vx, fit_train_mse_vy, fit_train_mse_p, fit_train_mse_nu = 0, 0, 0, 0
            fit_test_mse_vx, fit_test_mse_vy, fit_test_mse_p, fit_test_mse_nu = 0, 0, 0, 0

        for substep, (graph, idx) in enumerate(train_loader):
            inr.train()

            if include_sdf:
                graph.pos = torch.cat([graph.pos, graph.sdf], axis=-1)

            if data_to_encode == "vx":
                graph.images = graph.vx
            elif data_to_encode == "vy":
                graph.images = graph.vy
            elif data_to_encode == "p":
                graph.images = graph.p
            elif data_to_encode == "nu":
                graph.images = graph.nu
            elif data_to_encode == "sdf":
                graph.images = graph.sdf
            elif data_to_encode == "n":
                graph.images = torch.cat([graph.nx, graph.ny], axis=-1)
            elif data_to_encode == "nx":
                graph.images = graph.nx
            elif data_to_encode == "ny":
                graph.images = graph.ny
            elif data_to_encode == "all_physics_fields":
                graph.images = torch.cat([graph.vx, graph.vy, graph.p, graph.nu], axis=-1)

            graph.modulations = torch.zeros((len(graph), latent_dim))
            graph = graph.cuda()
            n_samples = len(graph)

            outputs = outer_step(
                inr,
                graph,
                inner_steps,
                alpha,
                is_train=True,
                return_reconstructions=step_show,
                return_dim_loss=return_dim_loss,
                gradient_checkpointing=False,
                use_rel_loss=False,
            )

            optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)
            optimizer.step()
            scheduler.step()
            
            loss = outputs["loss"].cpu().detach()
            fit_train_mse += loss.item() * n_samples
            if data_to_encode == 'all_physics_fields':
                loss_dim_tot = outputs["dim_loss"].cpu().detach()
                loss_vx, loss_vy, loss_p, loss_nu = loss_dim_tot[0], loss_dim_tot[1], loss_dim_tot[2], loss_dim_tot[3]
                fit_train_mse_vx += loss_vx.item() * n_samples
                fit_train_mse_vy += loss_vy.item() * n_samples
                fit_train_mse_p += loss_p.item() * n_samples
                fit_train_mse_nu += loss_nu.item() * n_samples

        train_loss = fit_train_mse / (ntrain)
        #scheduler.step(train_loss) with plateau

        if data_to_encode == 'all_physics_fields':
            train_loss_vx = fit_train_mse_vx / (ntrain)
            train_loss_vy = fit_train_mse_vy / (ntrain)
            train_loss_p = fit_train_mse_p / (ntrain)
            train_loss_nu = fit_train_mse_nu / (ntrain)

        if step_show:
            for substep, (graph, idx) in enumerate(test_loader):
                inr.eval()
                if include_sdf:
                    graph.pos = torch.cat([graph.pos, graph.sdf], axis=-1)

                if data_to_encode == "vx":
                    graph.images = graph.vx
                elif data_to_encode == "vy":
                    graph.images = graph.vy
                elif data_to_encode == "p":
                    graph.images = graph.p
                elif data_to_encode == "nu":
                    graph.images = graph.nu
                elif data_to_encode == "sdf":
                    graph.images = graph.sdf
                elif data_to_encode == "n":
                    graph.images = torch.cat([graph.nx, graph.ny], axis=-1)
                elif data_to_encode == "nx":
                    graph.images = graph.nx
                elif data_to_encode == "ny":
                    graph.images = graph.ny
                elif data_to_encode == "all_physics_fields":
                    graph.images = torch.cat([graph.vx, graph.vy, graph.p, graph.nu], axis=-1)

                graph.modulations = torch.zeros((len(graph), latent_dim))
                graph = graph.cuda()
                n_samples = len(graph)

                outputs = outer_step(
                    inr,
                    graph,
                    test_inner_steps,
                    alpha,
                    is_train=False,
                    return_reconstructions=step_show,
                    return_dim_loss=return_dim_loss,
                    gradient_checkpointing=False,
                    use_rel_loss=False,
                )

                loss = outputs["loss"]
                fit_test_mse += loss.item() * n_samples
                if data_to_encode == 'all_physics_fields':
                    loss_dim_tot = outputs["dim_loss"].cpu().detach()
                    loss_vx, loss_vy, loss_p, loss_nu = loss_dim_tot[0], loss_dim_tot[1], loss_dim_tot[2], loss_dim_tot[3]
                    fit_test_mse_vx += loss_vx.item() * n_samples
                    fit_test_mse_vy += loss_vy.item() * n_samples
                    fit_test_mse_p += loss_p.item() * n_samples
                    fit_test_mse_nu += loss_nu.item() * n_samples

            test_loss = fit_test_mse / ntest
            if data_to_encode == 'all_physics_fields':
                test_loss_vx = fit_test_mse_vx / (ntest)
                test_loss_vy = fit_test_mse_vy / (ntest)
                test_loss_p = fit_test_mse_p / (ntest)
                test_loss_nu = fit_test_mse_nu / (ntest)

        if step_show:
            if data_to_encode == 'all_physics_fields':
                wandb.log(
                    {
                "test_loss_vx": test_loss_vx,
                "test_loss_vy": test_loss_vy,
                "test_loss_p": test_loss_p,
                "test_loss_nu": test_loss_nu,
                "train_loss_vx": train_loss_vx,
                "train_loss_vy": train_loss_vy,
                "train_loss_p": train_loss_p,
                "train_loss_nu": train_loss_nu,
                "test_loss": test_loss,
                "train_loss": train_loss,
                },
                step=step)

            else:
                wandb.log(
                {
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                },
                step=step
                )

        else:
            if data_to_encode == 'all_physics_fields':
                wandb.log(
                    {
                        "train_loss_vx": train_loss_vx,
                        "train_loss_vy": train_loss_vy,
                        "train_loss_p": train_loss_p,
                        "train_loss_nu": train_loss_nu,
                        "train_loss": train_loss,
                    },
                    step=step
                )
            else:
                wandb.log(
                {
                    "train_loss": train_loss,
                },
                step=step,
                commit=not step_show,
            )

        if train_loss < best_loss:
            best_loss = train_loss

            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "inr": inr.state_dict(),
                    "optimizer_inr": optimizer.state_dict(),
                    "loss": test_loss,
                    "alpha": alpha,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )

    return test_loss


if __name__ == "__main__":
    main()
