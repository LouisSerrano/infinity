import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.data.dataset import GeometryDatasetGraph, KEY_TO_INDEX
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import einops
import json
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from coral.mlp import MLP, Derivative, ResNet
from omegaconf import DictConfig, OmegaConf
from torchdiffeq import odeint

from coral.losses import batch_mse_rel_fn
from coral.siren import ModulatedSiren, LatentToModulation
from coral.utils.data import (
    shape2coordinates,
    DatasetWithCode,
    set_seed,
)
from coral.utils.plot import show
from coral.utils.scheduler import learning_rate_scheduler
from graph_inr_metalearning import outer_step

# from wisp.models.embedders import PositionalEmbedder
from src.models.basic_conditioning import film, film_linear, film_translate

# from wisp.core import WispModule
from typing import Dict, Any
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.nn.unpool import knn_interpolate
from sklearn.cluster import KMeans, BisectingKMeans


class PositionalEmbedder2(nn.Module):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers."""

    def __init__(
        self,
        num_freq,
        max_freq_log2,
        log_sampling=True,
        include_input=True,
        input_dim=3,
        base_freq=2,
    ):
        """Initialize the module.
        Args:
            num_freq (int): The number of frequency bands to sample.
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.
        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()

        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        self.base_freq = base_freq

        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            self.bands = self.base_freq ** torch.linspace(
                0.0, max_freq_log2, steps=num_freq
            )
        else:
            self.bands = torch.linspace(
                1, self.base_freq**max_freq_log2, steps=num_freq
            )

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)

    def forward(self, coords):
        """Embeds the coordinates.
        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]
        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        N = coords.shape[0]
        winded = (coords[:, None] * self.bands[None, :, None]).reshape(
            N, coords.shape[1] * self.num_freq
        )
        encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    def name(self) -> str:
        """A human readable name for the given wisp module."""
        return "Positional Encoding"

    def public_properties(self) -> Dict[str, Any]:
        """Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {
            "Output Dim": self.out_dim,
            "Num. Frequencies": self.num_freq,
            "Max Frequency": f"2^{self.max_freq_log2}",
            "Include Input": self.include_input,
        }


class ModulatedPositionalEmbedder(nn.Module):
    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        max_frequencies=10,
        num_frequencies=32,
        num_nodes=64,
        latent_dim=16,
        width=256,
        depth=3,
        modulate_scale=False,
        modulate_shift=True,
        radius=1.0,
        max_num_neighbors=5,
        k=3,
    ):
        super().__init__()
        self.embedding = PositionalEmbedder2(
            num_frequencies,
            max_frequencies,
            log_sampling=True,
            include_input=False,
            input_dim=input_dim,
            base_freq=2,
        )
        # self.network = tcnn.Network(self.encoding.n_output_dims, output_dim, config["network"])
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.in_channels = [self.embedding.out_dim] + [width] * (depth - 1)
        self.out_channels = [width] * (depth - 1) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels[k], self.out_channels[k]) for k in range(depth)]
        )
        self.depth = depth
        self.hidden_dim = width
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.k = k

        self.num_modulations = self.hidden_dim * (self.depth - 1)
        if modulate_scale and modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            self.num_modulations *= 2
        # self.latent_to_modulation = LatentToModulation(8, self.num_modulations, dim_hidden=256, num_layers=1)
        self.latent_to_modulation = SAGEConv(
            in_channels=latent_dim, out_channels=self.num_modulations, project=False
        )

        if modulate_shift and modulate_scale:
            self.conditioning = film
        elif modulate_scale and not modulate_shift:
            self.conditioning = film_linear
        else:
            self.conditioning = film_translate

    def modulated_forward(self, graph, z):
        # z.edge_index = radius_graph(z.pos,
        #                            self.radius,
        #                            z.batch,
        #                            loop=False,
        #                            max_num_neighbors=self.max_num_neighbors,)

        edge_index = knn_graph(
            z.pos,
            k=8,
            batch=z.batch,
            loop=False,
            flow="source_to_target",
            cosine=False,
            num_workers=1,
        )
        features = self.latent_to_modulation(z.features, edge_index)
        features_interpolate = knn_interpolate(
            features, z.pos, graph.pos, z.batch, graph.batch, k=self.k
        )

        if self.input_dim == 3:
            inpt = torch.cat([graph.pos, graph.sdf], axis=-1)
        else:
            inpt = graph.pos
        position = self.embedding(inpt)
        pre_out = self.conditioning(
            position, features_interpolate, self.layers[:-1], torch.relu
        )
        out = self.layers[-1](pre_out)
        return out


@hydra.main(config_path="config/", config_name="graph.yaml")
def main(cfg: DictConfig) -> None:
    # submitit.JobEnvironment()
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
    num_nodes = cfg.inr.num_nodes
    latent_dim = cfg.inr.latent_dim
    depth = cfg.inr.depth
    hidden_dim = cfg.inr.hidden_dim
    w0 = cfg.inr.w0
    use_latent = cfg.inr.use_latent
    modulate_scale = cfg.inr.modulate_scale
    modulate_shift = cfg.inr.modulate_shift
    hypernet_depth = cfg.inr.hypernet_depth
    hypernet_width = cfg.inr.hypernet_width
    loss_type = cfg.inr.loss_type
    gamma = cfg.inr.gamma
    scale_factor = cfg.inr.scale_factor
    radius = cfg.inr.radius
    max_num_neighbors = cfg.inr.max_num_neighbors
    k_interpolate = cfg.inr.k_interpolate
    num_frequencies = cfg.inr.num_frequencies
    include_sdf = cfg.inr.include_sdf
    frequency_embedding = cfg.inr.frequency_embedding
    include_input = cfg.inr.include_input
    scale = cfg.inr.scale
    max_frequencies = cfg.inr.max_frequencies
    base_frequency = cfg.inr.base_frequency

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

    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / "airfrans" / "inr" / data_to_encode
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

    set_seed(seed)

    run.tags = ("inr-graph",) + (data_to_encode,) + (model_type,)

    data_dir = "/data/serrano/airfrans/"
    print("data_dir", data_dir)

    # train
    with open(Path(data_dir) / "Dataset/manifest.json", "r") as f:
        manifest = json.load(f)

    manifest_train = manifest[task + "_train"]
    testset = manifest[task + "_test"] if task != "scarce" else manifest["full_test"]
    n = int(0.9 * len(manifest_train))

    print("len manifest train", manifest_train, n)

    trainset = manifest_train[:n]  # was [:-n] ???
    valset = manifest_train[n:]

    ntrain = len(trainset)
    nval = len(valset)

    # default sample is none
    trainset = GeometryDatasetGraph(
        trainset,
        key=data_to_encode,
        num_nodes=num_nodes,
        latent_dim=latent_dim,
        scale_factor=scale_factor,
        norm=True,
        sample="mesh",
        n_boot=4000,
    )
    print("loaded train")
    valset = GeometryDatasetGraph(
        valset,
        key=data_to_encode,
        num_nodes=num_nodes,
        latent_dim=latent_dim,
        scale_factor=scale_factor,
        sample="mesh",
        n_boot=4000,
        coef_norm=trainset.coef_norm,
    )

    # create the nodes for the alpha
    clustering = BisectingKMeans(n_clusters=num_nodes, max_iter=1000)
    clustering.fit(trainset.modulation_pos.reshape(-1, 2))
    alpha_nodes = torch.from_numpy(clustering.cluster_centers_).float().cuda()

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # test
    val_loader = DataLoader(valset, batch_size=batch_size_val, shuffle=True)
    device = torch.device("cuda")

    input_dim = 2
    if model_type == "siren":
        inr = ModulatedSiren(
            dim_in=input_dim,
            dim_hidden=hidden_dim,
            dim_out=1,
            num_layers=depth,
            w0=w0,
            w0_initial=w0,
            use_bias=True,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
            use_latent=use_latent,
            latent_dim=latent_dim,
            modulation_net_dim_hidden=hypernet_width,
            modulation_net_num_layers=hypernet_depth,
            last_activation=None,
        ).cuda()
    elif model_type == "pos_embedder":
        input_dim = 2
        if include_sdf:
            input_dim = input_dim + 1
        inr = ModulatedPositionalEmbedder(
            input_dim=input_dim,
            output_dim=1,
            max_frequencies=max_frequencies,
            num_frequencies=num_frequencies,
            num_nodes=num_nodes,
            latent_dim=latent_dim,
            width=hidden_dim,
            depth=depth,
            modulate_scale=modulate_scale,
            modulate_shift=modulate_shift,
            radius=radius,
            max_num_neighbors=max_num_neighbors,
            k=k_interpolate,
        ).cuda()

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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epochs//10, gamma=0.75, last_epoch=- 1, verbose=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=100,
        threshold=0.01,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
        eps=1e-08,
        verbose=True,
    )

    best_loss = np.inf

    for step in range(epochs):
        rel_train_mse = 0
        rel_test_mse = 0
        fit_train_mse = 0
        fit_test_mse = 0
        use_rel_loss = step % 10 == 0
        step_show = step % 100 == 0

        for substep, (graph, modulations, idx) in enumerate(train_loader):
            inr.train()
            modulations = modulations.cuda()
            graph = graph.cuda()
            n_samples = len(graph)

            outputs = outer_step(
                inr,
                graph,
                modulations,
                inner_steps,
                alpha,
                is_train=True,
                return_reconstructions=step_show,
                gradient_checkpointing=False,
                use_rel_loss=use_rel_loss,
                loss_type=loss_type,
            )

            optimizer.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)
            optimizer.step()
            # alpha = alpha.clamp(-5, 5)
            loss = outputs["loss"].cpu().detach()
            fit_train_mse += loss.item() * n_samples

            # mlp regression
            if use_rel_loss:
                rel_train_mse += outputs["rel_loss"].item() * n_samples

            if step_show and substep == 0:
                pass
                # mask = graph.batch == 0
                # preds = outputs['reconstructions'].detach().cpu()
                # show(graph.images[mask].cpu().unsqueeze(0), preds[mask].unsqueeze(0), graph.pos[mask].unsqueeze(0), "train_reconstuction")

        train_loss = fit_train_mse / (ntrain)
        scheduler.step(train_loss)

        if use_rel_loss:
            rel_train_loss = rel_train_mse / (ntrain)

        if step_show:
            for substep, (graph, modulations, idx) in enumerate(val_loader):
                inr.eval()
                modulations = modulations.cuda()
                graph = graph.cuda()
                n_samples = len(graph)

                outputs = outer_step(
                    inr,
                    graph,
                    modulations,
                    test_inner_steps,
                    alpha,
                    is_train=False,
                    return_reconstructions=step_show,
                    gradient_checkpointing=False,
                    use_rel_loss=use_rel_loss,
                    loss_type=loss_type,
                )

                loss = outputs["loss"]
                fit_test_mse += loss.item() * n_samples

                if use_rel_loss:
                    rel_test_mse += outputs["rel_loss"].item() * n_samples

                if substep == 0:
                    pass
                    # mask = graph.batch == 0
                    # preds = outputs['reconstructions'].detach().cpu()
                    # show(graph.images[mask].cpu().unsqueeze(0), preds[mask].unsqueeze(0), graph.pos[mask].unsqueeze(0), "test_reconstuction")

            test_loss = fit_test_mse / (nval)

            if use_rel_loss:
                rel_test_loss = rel_test_mse / (nval)

        if step_show:
            wandb.log(
                {
                    "test_rel_loss": rel_test_loss,
                    "train_rel_loss": rel_train_loss,
                    "test_loss": test_loss,
                    "train_loss": train_loss,
                },
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
                    "data": cfg.data,
                    "cfg_inr": cfg.inr,
                    "epoch": step,
                    "inr": inr.state_dict(),
                    "optimizer_inr": optimizer.state_dict(),
                    "loss": test_loss,
                    "alpha": alpha,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )

    return rel_test_loss


if __name__ == "__main__":
    main()
