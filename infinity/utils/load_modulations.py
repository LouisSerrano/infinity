import torch
from torch_geometric.loader import DataLoader
from infinity.graph_metalearning import outer_step
import os


def load_modulations(
    trainset,
    valset,
    inr,
    run_dir,
    run_name,
    data_to_encode,
    num_points=None,
    input_dim=2,
    inner_steps=3,
    alpha=0.01,
    batch_size=2,
    latent_dim=256,
):
    try:
        return torch.load(run_dir / f"{data_to_encode}/{run_name}.pt")
    except:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

        if data_to_encode in ["sdf", "n"]:
            include_sdf = False
        else:
            include_sdf = True

        ntrain = len(train_loader.dataset)
        ntest = len(val_loader.dataset)
        #trainset.modulation_only = False
        #testset.modulation_only = False

        fit_train_mse = 0
        fit_test_mse = 0
        mod_tr = torch.zeros(ntrain, latent_dim)
        mod_te = torch.zeros(ntest, latent_dim)

        for substep, (graph, idx) in enumerate(train_loader):
            inr.eval()
            if num_points is not None:
                mask = torch.randperm(graph.pos.shape[0])[:num_points]
            else:
                mask = ...

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
                mask = graph.surface
            elif data_to_encode == "all_physics_fields":
                graph.images = torch.cat([graph.vx, graph.vy, graph.p, graph.nu], axis=-1)

            graph.pos = graph.pos[mask]
            graph.batch = graph.batch[mask]
            graph.images = graph.images[mask]
            graph.modulations = torch.zeros((len(graph), latent_dim))
            graph = graph.cuda()
            n_samples = len(graph)

            outputs = outer_step(
                inr,
                graph,
                inner_steps,
                alpha,
                is_train=False,
                return_reconstructions=True,
                gradient_checkpointing=False,
                use_rel_loss=False,
            )
            mod_tr[idx] = outputs["modulations"].cpu().detach()

            loss = outputs["loss"].cpu().detach()
            fit_train_mse += loss.item() * n_samples

        print("train", data_to_encode, fit_train_mse / ntrain)

        for substep, (graph, idx) in enumerate(val_loader):
            inr.eval()
            if num_points is not None:
                mask = torch.randperm(graph.pos.shape[0])[:num_points]
            else:
                mask = ...
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
                mask = graph.surface
            elif data_to_encode == "all_physics_fields":
                graph.images = torch.cat([graph.vx, graph.vy, graph.p, graph.nu], axis=-1)

            graph.pos = graph.pos[mask]
            graph.batch = graph.batch[mask]
            graph.images = graph.images[mask]
            graph.modulations = torch.zeros((len(graph), latent_dim))
            graph = graph.cuda()
            n_samples = len(graph)

            outputs = outer_step(
                inr,
                graph,
                inner_steps,
                alpha,
                is_train=False,
                return_reconstructions=True,
                gradient_checkpointing=False,
                use_rel_loss=False,
            )
            mod_te[idx] = outputs["modulations"].cpu().detach()

            loss = outputs["loss"].cpu().detach()
            fit_test_mse += loss.item() * n_samples

        print("test", data_to_encode, fit_test_mse / ntest)

        os.makedirs(run_dir / f"{data_to_encode}/", exist_ok=True)

        modulations = {"z_train": mod_tr, "z_test": mod_te}
        torch.save(modulations, run_dir / f"{data_to_encode}/{run_name}.pt")

        #trainset.modulation_only = True
        #testset.modulation_only = True

        return modulations
