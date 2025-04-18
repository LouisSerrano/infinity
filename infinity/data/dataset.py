import numpy as np
import pyvista as pv

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import random
from tqdm import tqdm
import os

from infinity.utils.reorganize import reorganize
from infinity.graph_metalearning import outer_step

def set_seed(seed=33):
    """Set all seeds for the experiments.

    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


KEY_TO_INDEX = {
    "x": 0,
    "y": 1,
    "inlet_x": 2,
    "inlet_y": 3,
    "sdf": 4,
    "nx": 5,
    "ny": 6,
    "vx": 7,
    "vy": 8,
    "p": 9,
    "nu": 10,
}

# MIN_POS = torch.Tensor([-2.25, -1.65]).view(-1, 2)
# MAX_POS = torch.Tensor([4.25, 1.65]).view(-1,2)
# MIN_POS = torch.Tensor([-2.5, -1.75]).view(-1, 2)
# MAX_POS = torch.Tensor([4.5, 1.75]).view(-1,2)
MIN_POS = 0
MAX_POS = 1


def cell_sampling_2d(cell_points, cell_attr=None):
    """
    Sample points in a two dimensional cell via parallelogram sampling and triangle interpolation via barycentric coordinates. The vertices have to be ordered in a certain way.

    Args:
        cell_points (array): Vertices of the 2 dimensional cells. Shape (N, 4) for N cells with 4 vertices.
        cell_attr (array, optional): Features of the vertices of the 2 dimensional cells. Shape (N, 4, k) for N cells with 4 edges and k features.
            If given shape (N, 4) it will resize it automatically in a (N, 4, 1) array. Default: ``None``
    """
    # Sampling via triangulation of the cell and parallelogram sampling
    v0, v1 = (
        cell_points[:, 1] - cell_points[:, 0],
        cell_points[:, 3] - cell_points[:, 0],
    )
    v2, v3 = (
        cell_points[:, 3] - cell_points[:, 2],
        cell_points[:, 1] - cell_points[:, 2],
    )
    a0, a1 = np.abs(
        np.linalg.det(np.hstack([v0[:, :2], v1[:, :2]]).reshape(-1, 2, 2))
    ), np.abs(np.linalg.det(np.hstack([v2[:, :2], v3[:, :2]]).reshape(-1, 2, 2)))
    p = a0 / (a0 + a1)
    index_triangle = np.random.binomial(1, p)[:, None]
    u = np.random.uniform(size=(len(p), 2))
    sampled_point = index_triangle * (u[:, 0:1] * v0 + u[:, 1:2] * v1) + (
        1 - index_triangle
    ) * (u[:, 0:1] * v2 + u[:, 1:2] * v3)
    sampled_point_mirror = index_triangle * (
        (1 - u[:, 0:1]) * v0 + (1 - u[:, 1:2]) * v1
    ) + (1 - index_triangle) * ((1 - u[:, 0:1]) * v2 + (1 - u[:, 1:2]) * v3)
    reflex = u.sum(axis=1) > 1
    sampled_point[reflex] = sampled_point_mirror[reflex]

    # Interpolation on a triangle via barycentric coordinates
    if cell_attr is not None:
        t0, t1, t2 = (
            np.zeros_like(v0),
            index_triangle * v0 + (1 - index_triangle) * v2,
            index_triangle * v1 + (1 - index_triangle) * v3,
        )
        w = (t1[:, 1] - t2[:, 1]) * (t0[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[:, 0]) * (
            t0[:, 1] - t2[:, 1]
        )
        w0 = (t1[:, 1] - t2[:, 1]) * (sampled_point[:, 0] - t2[:, 0]) + (
            t2[:, 0] - t1[:, 0]
        ) * (sampled_point[:, 1] - t2[:, 1])
        w1 = (t2[:, 1] - t0[:, 1]) * (sampled_point[:, 0] - t2[:, 0]) + (
            t0[:, 0] - t2[:, 0]
        ) * (sampled_point[:, 1] - t2[:, 1])
        w0, w1 = w0 / w, w1 / w
        w2 = 1 - w0 - w1

        if len(cell_attr.shape) == 2:
            cell_attr = cell_attr[:, :, None]
        attr0 = (
            index_triangle * cell_attr[:, 0] + (1 - index_triangle) * cell_attr[:, 2]
        )
        attr1 = (
            index_triangle * cell_attr[:, 1] + (1 - index_triangle) * cell_attr[:, 1]
        )
        attr2 = (
            index_triangle * cell_attr[:, 3] + (1 - index_triangle) * cell_attr[:, 3]
        )
        sampled_attr = w0[:, None] * attr0 + w1[:, None] * attr1 + w2[:, None] * attr2

    sampled_point += (
        index_triangle * cell_points[:, 0] + (1 - index_triangle) * cell_points[:, 2]
    )

    return (
        np.hstack([sampled_point[:, :2], sampled_attr])
        if cell_attr is not None
        else sampled_point[:, :2]
    )


def cell_sampling_1d(line_points, line_attr=None):
    """
    Sample points in a one dimensional cell via linear sampling and interpolation.

    Args:
        line_points (array): Edges of the 1 dimensional cells. Shape (N, 2) for N cells with 2 edges.
        line_attr (array, optional): Features of the edges of the 1 dimensional cells. Shape (N, 2, k) for N cells with 2 edges and k features.
            If given shape (N, 2) it will resize it automatically in a (N, 2, 1) array. Default: ``None``
    """
    # Linear sampling
    u = np.random.uniform(size=(len(line_points), 1))
    sampled_point = u * line_points[:, 0] + (1 - u) * line_points[:, 1]

    # Linear interpolation
    if line_attr is not None:
        if len(line_attr.shape) == 2:
            line_attr = line_attr[:, :, None]
        sampled_attr = u * line_attr[:, 0] + (1 - u) * line_attr[:, 1]

    return (
        np.hstack([sampled_point[:, :2], sampled_attr])
        if line_attr is not None
        else sampled_point[:, :2]
    )


def get_dataset(
    set,
    datadir="/data/serrano/airfrans/Dataset/",
    norm=False,
    coef_norm=None,
    crop=None,
    sample=None,
    n_boot=int(5e5),
    surf_ratio=0.1,
):
    """
    Create a list of simulation to input in a PyTorch Geometric DataLoader. Simulation are transformed by keeping vertices of the CFD mesh or
    by sampling (uniformly or via the mesh density) points in the simulation cells.

    Args:
        set (list): List of geometry names to include in the dataset.
        norm (bool, optional): If norm is set to ``True``, the mean and the standard deviation of the dataset will be computed and returned.
            Moreover, the dataset will be normalized by these quantities. Ignored when ``coef_norm`` is not None. Default: ``False``
        coef_norm (tuple, optional): This has to be a tuple of the form (mean input, std input, mean output, std ouput) if not None.
            The dataset generated will be normalized by those quantites. Default: ``None``
        crop (list, optional): List of the vertices of the rectangular [xmin, xmax, ymin, ymax] box to crop simulations. Default: ``None``
        sample (string, optional): Type of sampling. If ``None``, no sampling strategy is applied and the nodes of the CFD mesh are returned.
            If ``uniform`` or ``mesh`` is chosen, uniform or mesh density sampling is applied on the domain. Default: ``None``
        n_boot (int, optional): Used only if sample is not None, gives the size of the sampling for each simulation. Defaul: ``int(5e5)``
        surf_ratio (float, optional): Used only if sample is not None, gives the ratio of point over the airfoil to sample with respect to point
            in the volume. Default: ``0.1``
    """
    if norm and coef_norm is not None:
        raise ValueError(
            "If coef_norm is not None and norm is True, the normalization will be done via coef_norm"
        )

    dataset = []

    for k, s in enumerate(tqdm(set)):
        # Get the 3D mesh, add the signed distance function and slice it to return in 2D
        internal = pv.read(datadir + s + "/" + s + "_internal.vtu")
        aerofoil = pv.read(datadir + s + "/" + s + "_aerofoil.vtp")
        internal = internal.compute_cell_sizes(length=False, volume=False)

        # Cropping if needed, crinkle is True.
        if crop is not None:
            bounds = (crop[0], crop[1], crop[2], crop[3], 0, 1)
            internal = internal.clip_box(bounds=bounds, invert=False, crinkle=True)

        # If sampling strategy is chosen, it will sample points in the cells of the simulation instead of directly taking the nodes of the mesh.
        if sample is not None:
            # Sample on a new point cloud
            if sample == "uniform":  # Uniform sampling strategy
                p = internal.cell_data["Area"] / internal.cell_data["Area"].sum()
                sampled_cell_indices = np.random.choice(
                    internal.n_cells, size=n_boot, p=p
                )
                surf_p = (
                    aerofoil.cell_data["Length"] / aerofoil.cell_data["Length"].sum()
                )
                sampled_line_indices = np.random.choice(
                    aerofoil.n_cells, size=int(n_boot * surf_ratio), p=surf_p
                )
            elif sample == "mesh":  # Sample via mesh density
                sampled_cell_indices = np.random.choice(internal.n_cells, size=n_boot)
                sampled_line_indices = np.random.choice(
                    aerofoil.n_cells, size=int(n_boot * surf_ratio)
                )

            cell_dict = internal.cells.reshape(-1, 5)[sampled_cell_indices, 1:]
            cell_points = internal.points[cell_dict]
            line_dict = aerofoil.lines.reshape(-1, 3)[sampled_line_indices, 1:]
            line_points = aerofoil.points[line_dict]

            # Geometry information
            geom = -internal.point_data["implicit_distance"][
                cell_dict, None
            ]  # Signed distance function
            Uinf, alpha = float(s.split("_")[2]), float(s.split("_")[3]) * np.pi / 180
            # u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*(internal.point_data['U'][cell_dict, :1] != 0)
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(
                1, 2
            ) * np.ones_like(internal.point_data["U"][cell_dict, :1])
            normal = np.zeros_like(u)

            surf_geom = np.zeros_like(aerofoil.point_data["U"][line_dict, :1])
            # surf_u = np.zeros_like(aerofoil.point_data['U'][line_dict, :2])
            surf_u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(
                1, 2
            ) * np.ones_like(aerofoil.point_data["U"][line_dict, :1])
            surf_normal = -aerofoil.point_data["Normals"][line_dict, :2]

            attr = np.concatenate(
                [
                    u,
                    geom,
                    normal,
                    internal.point_data["U"][cell_dict, :2],
                    internal.point_data["p"][cell_dict, None],
                    internal.point_data["nut"][cell_dict, None],
                ],
                axis=-1,
            )
            surf_attr = np.concatenate(
                [
                    surf_u,
                    surf_geom,
                    surf_normal,
                    aerofoil.point_data["U"][line_dict, :2],
                    aerofoil.point_data["p"][line_dict, None],
                    aerofoil.point_data["nut"][line_dict, None],
                ],
                axis=-1,
            )
            sampled_points = cell_sampling_2d(cell_points, attr)
            surf_sampled_points = cell_sampling_1d(line_points, surf_attr)

            # Define the inputs and the targets
            pos = sampled_points[:, :2]
            init = sampled_points[:, :7]
            target = sampled_points[:, 7:]
            surf_pos = surf_sampled_points[:, :2]
            surf_init = surf_sampled_points[:, :7]
            surf_target = surf_sampled_points[:, 7:]

            # Put everything in tensor
            surf = torch.cat([torch.zeros(len(pos)), torch.ones(len(surf_pos))], dim=0)
            pos = torch.cat(
                [
                    torch.tensor(pos, dtype=torch.float),
                    torch.tensor(surf_pos, dtype=torch.float),
                ],
                dim=0,
            )
            x = torch.cat(
                [
                    torch.tensor(init, dtype=torch.float),
                    torch.tensor(surf_init, dtype=torch.float),
                ],
                dim=0,
            )
            y = torch.cat(
                [
                    torch.tensor(target, dtype=torch.float),
                    torch.tensor(surf_target, dtype=torch.float),
                ],
                dim=0,
            )

        else:  # Keep the mesh nodes
            surf_bool = internal.point_data["U"][:, 0] == 0
            geom = -internal.point_data["implicit_distance"][
                :, None
            ]  # Signed distance function
            Uinf, alpha = float(s.split("_")[2]), float(s.split("_")[3]) * np.pi / 180
            # u = (np.array([np.cos(alpha), np.sin(alpha)])*Uinf).reshape(1, 2)*(internal.point_data['U'][:, :1] != 0)
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(
                1, 2
            ) * np.ones_like(internal.point_data["U"][:, :1])
            normal = np.zeros_like(u)
            normal[surf_bool] = reorganize(
                aerofoil.points[:, :2],
                internal.points[surf_bool, :2],
                -aerofoil.point_data["Normals"][:, :2],
            )

            attr = np.concatenate(
                [
                    u,
                    geom,
                    normal,
                    internal.point_data["U"][:, :2],
                    internal.point_data["p"][:, None],
                    internal.point_data["nut"][:, None],
                ],
                axis=-1,
            )

            pos = internal.points[:, :2]
            init = np.concatenate([pos, attr[:, :5]], axis=1)
            target = attr[:, 5:]

            # Put everything in tensor
            surf = torch.tensor(surf_bool)
            pos = torch.tensor(pos, dtype=torch.float)
            x = torch.tensor(init, dtype=torch.float)
            y = torch.tensor(target, dtype=torch.float)

        if norm and coef_norm is None:
            if k == 0:
                old_length = init.shape[0]
                mean_in = init.mean(axis=0, dtype=np.double)
                mean_out = target.mean(axis=0, dtype=np.double)
            else:
                new_length = old_length + init.shape[0]
                mean_in += (
                    init.sum(axis=0, dtype=np.double) - init.shape[0] * mean_in
                ) / new_length
                mean_out += (
                    target.sum(axis=0, dtype=np.double) - init.shape[0] * mean_out
                ) / new_length
                old_length = new_length

        # Graph definition
        # if cell_centers:
        #     data = Data(pos = pos, x = x, y = y, surf = surf.bool(), centers = centers.bool())
        # else:
        #     data = Data(pos = pos, x = x, y = y, surf = surf.bool())
        data = Data(pos=pos, x=x, y=y, surf=surf.bool())
        dataset.append(data)

    if norm and coef_norm is None:
        # Compute normalization
        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)
        # Umean = np.linalg.norm(data.x[:, 2:4], axis = 1).mean()
        for k, data in enumerate(dataset):
            # data.x = data.x/torch.tensor([6, 6, Umean, Umean, 6, 1, 1], dtype = torch.float)
            # data.y = data.y/torch.tensor([Umean, Umean, .5*Umean**2, Umean], dtype = torch.float)

            if k == 0:
                old_length = data.x.numpy().shape[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(
                    axis=0, dtype=np.double
                ) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(
                    axis=0, dtype=np.double
                ) / old_length
            else:
                new_length = old_length + data.x.numpy().shape[0]
                std_in += (
                    ((data.x.numpy() - mean_in) ** 2).sum(axis=0, dtype=np.double)
                    - data.x.numpy().shape[0] * std_in
                ) / new_length
                std_out += (
                    ((data.y.numpy() - mean_out) ** 2).sum(axis=0, dtype=np.double)
                    - data.x.numpy().shape[0] * std_out
                ) / new_length
                old_length = new_length

        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        # Normalize
        for data in dataset:
            data.x = (data.x - mean_in) / (std_in + 1e-8)
            data.y = (data.y - mean_out) / (std_out + 1e-8)

        coef_norm = (mean_in, std_in, mean_out, std_out)
        dataset = (dataset, coef_norm)

    elif coef_norm is not None:
        # Normalize
        for data in dataset:
            # data.x = data.x/torch.tensor([6, 6, coef_norm[-1], coef_norm[-1], 6, 1, 1], dtype = torch.float)
            # data.y = data.y/torch.tensor([coef_norm[-1], coef_norm[-1], .5*coef_norm[-1]**2, coef_norm[-1]], dtype = torch.float)
            data.x = (data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)
            data.y = (data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)

        dataset = (dataset, coef_norm)

    return dataset


class GeometryDataset(Dataset):
    def __init__(
        self,
        set,
        key=None,
        latent_dim=256,
        scale_factor=1.0,
        norm=False,
        coef_norm=None,
        crop=None,
        sample="mesh",
        n_boot=8000,
        surf_ratio=0.1,
    ):
        dataset, coef_norm = get_dataset(
            set,
            norm=norm,
            coef_norm=coef_norm,
            crop=crop,
            sample=None,
            n_boot=n_boot,
            surf_ratio=surf_ratio,
        )
        self.dataset = dataset
        self.coef_norm = coef_norm
        self.latent_dim = latent_dim
        print("coef_norm", coef_norm)
        self.modulations = torch.zeros(len(self.dataset), latent_dim)
        self.key = key
        self.scale_factor = scale_factor
        if key is not None:
            self.index = KEY_TO_INDEX[key]
            self.input_key = self.index <= 6
            self.relative_index = self.index if self.index <= 6 else self.index - 7
        self.sample = sample
        self.n_points = n_boot

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        if self.sample == "mesh":
            mask = torch.randperm(self.dataset[i].pos.shape[0])[: self.n_points]
            if self.key == None:
                graph = Data(pos=self.dataset[i].pos[mask])
                graph.images = self.dataset[i].x[
                    mask, 2:
                ]  # we do not take the x, y coordinates
                graph.surface = self.dataset[i].surf[mask]
                graph.modulations = self.modulations[i].unsqueeze(0)
                return (graph, i)

            else:
                scaled_pos = (
                    (self.dataset[i].pos[mask] - MIN_POS)
                    / (MAX_POS - MIN_POS)
                    * self.scale_factor
                )
                graph = Data(pos=scaled_pos, edge_index=None)
                graph.surface = self.dataset[i].surf[mask]
                graph.sdf = self.dataset[i].x[mask, KEY_TO_INDEX["sdf"]].unsqueeze(-1)
                graph.modulations = self.modulations[i].unsqueeze(0)

                if self.input_key:
                    graph.images = (
                        self.dataset[i].x[mask, self.relative_index].unsqueeze(-1)
                    )
                else:
                    graph.images = (
                        self.dataset[i].y[mask, self.relative_index].unsqueeze(-1)
                    )

                return (graph, i)
        else:
            if self.key == None:
                return (self.dataset[i], i)

            else:
                scaled_pos = (
                    (self.dataset[i].pos - MIN_POS)
                    / (MAX_POS - MIN_POS)
                    * self.scale_factor
                )
                graph = Data(pos=scaled_pos, edge_index=None)
                graph.surface = self.dataset[i].surf
                graph.sdf = self.dataset[i].x[..., KEY_TO_INDEX["sdf"]].unsqueeze(-1)
                graph.modulations = self.modulations[i].unsqueeze(0)

                if self.input_key:
                    graph.images = (
                        self.dataset[i].x[..., self.relative_index].unsqueeze(-1)
                    )
                else:
                    graph.images = (
                        self.dataset[i].y[..., self.relative_index].unsqueeze(-1)
                    )

                return (graph, i)


class GeometryDatasetFull(Dataset):
    def __init__(
        self,
        set,
        key=None,
        latent_dim=256,
        scale_factor=1.0,
        norm=False,
        coef_norm=None,
        crop=None,
        sample="mesh",
        n_boot=4000,
        surf_ratio=0.1,
        modulation_only=False
    ):
        dataset, coef_norm = get_dataset(
            set,
            norm=norm,
            coef_norm=coef_norm,
            crop=crop,
            sample=None,
            n_boot=n_boot,
            surf_ratio=surf_ratio,
        )
        super().__init__()
        self.dataset = dataset
        self.coef_norm = coef_norm
        self.latent_dim = latent_dim
        self.modulation_only = modulation_only

        print("coef_norm", coef_norm)
        self.in_modulations = {
            "sdf": torch.zeros(len(self.dataset), latent_dim),
            "n": torch.zeros(len(self.dataset), latent_dim),
        }
        self.out_modulations = {
            "vx": torch.zeros(len(self.dataset), latent_dim),
            "vy": torch.zeros(len(self.dataset), latent_dim),
            "p": torch.zeros(len(self.dataset), latent_dim),
            "nu": torch.zeros(len(self.dataset), latent_dim),
            "fields": torch.zeros(len(self.dataset), latent_dim),
        }
        self.key = key
        self.scale_factor = scale_factor
        self.sample = sample
        self.n_points = n_boot

    def len(self):
        return len(self.dataset)

    def get(self, i):
        if self.modulation_only:
            z_sdf = self.in_modulations["sdf"][i]
            z_n = self.in_modulations["n"][i]
            z_fields = self.out_modulations["fields"][i]
            inlet_x = self.dataset[i].x[0, 2].unsqueeze(-1)
            inlet_y = self.dataset[i].x[0, 3].unsqueeze(-1)

            return z_sdf, z_n, inlet_x, inlet_y, z_fields, i

        else:

            if self.sample == "mesh":
                mask = torch.randperm(self.dataset[i].pos.shape[0])[: self.n_points]
            elif self.sample == "surface":
                mask = self.dataset[i].surf
            else:
                mask = ...

            graph = Data(pos=self.dataset[i].pos[mask])

            # input
            # we take only the first element fot the inlets
            graph.inlet_x = self.dataset[i].x[0, 2].unsqueeze(-1)
            graph.inlet_y = self.dataset[i].x[0, 3].unsqueeze(-1)
            graph.sdf = self.dataset[i].x[mask, 4].unsqueeze(-1)
            graph.nx = self.dataset[i].x[mask, 5].unsqueeze(-1)
            graph.ny = self.dataset[i].x[mask, 6].unsqueeze(-1)

            graph.x = self.dataset[i].x[mask, 2:]

            # output
            graph.vx = self.dataset[i].y[mask, 0].unsqueeze(-1)
            graph.vy = self.dataset[i].y[mask, 1].unsqueeze(-1)
            graph.p = self.dataset[i].y[mask, 2].unsqueeze(-1)
            graph.nu = self.dataset[i].y[mask, 3].unsqueeze(-1)

            # output code
            graph.z_vx = self.out_modulations["vx"][i].unsqueeze(0)
            graph.z_vy = self.out_modulations["vy"][i].unsqueeze(0)
            graph.z_p = self.out_modulations["p"][i].unsqueeze(0)
            graph.z_nu = self.out_modulations["nu"][i].unsqueeze(0)
            graph.z_sdf = self.in_modulations["sdf"][i].unsqueeze(0)
            graph.z_n = self.in_modulations["n"][i].unsqueeze(0)

            graph.z_fields = self.out_modulations["fields"][i].unsqueeze(0)

            graph.z = torch.cat([graph.z_vx, graph.z_vy, graph.z_p, graph.z_nu], axis=-1)

            graph.surface = self.dataset[i].surf[mask]

            return (graph, i)
