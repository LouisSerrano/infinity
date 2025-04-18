import pdb
from math import sqrt
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch_geometric.nn import (
    MLP,
    PointConv,
    fps,
    global_max_pool,
    radius,
    knn_interpolate,
)
from typing import Dict, Any


class SAModule(torch.nn.Module):
    """
    adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
    """

    def __init__(self, ratio, r, nn, pos_embedder=None):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """
    adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
    """

    def __init__(self, nn, pos_embedder=None, input_dim=2):
        super().__init__()
        self.nn = nn
        self.input_dim = input_dim

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), self.input_dim))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2(torch.nn.Module):
    """
    adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
    """

    def __init__(
        self,
        input_dim=2,
        node_features=5,
        output_dim=1024,
        dropout=0.25,
        latent_dim=256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.node_features = node_features

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(
            0.5, 0.2, MLP([self.node_features + self.input_dim, 32, 64])
        )  # was 0.4, 0.2 and 32, 32
        self.sa2_module = SAModule(
            0.25, 0.4, MLP([64 + self.input_dim, 64, 128])
        )  # 0.25, 0.4 , 64,
        self.sa3_module = GlobalSAModule(
            MLP([128 + self.input_dim, 128, 256], dropout=self.dropout)
        )

        self.mlp = MLP([256, 256, self.output_dim], dropout=self.dropout, norm=None)

        print("sa1_module", self.sa1_module)
        print("sa2_module", self.sa2_module)
        print("sa3_module", self.sa3_module)
        print("mlp", self.mlp)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x)


class PointNet2SDF(torch.nn.Module):
    """
    adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
    """

    def __init__(
        self, input_dim=2, node_features=5, output_dim=1024, dropout=0.3, latent_dim=256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.node_features = node_features

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(
            0.5, 0.2, MLP([self.node_features + self.input_dim, 32, 64])
        )  # was 0.4, 0.2
        self.sa2_module = SAModule(
            0.25, 0.4, MLP([64 + self.input_dim, 64, 128])
        )  # 0.25, 0.4
        self.sa3_module = GlobalSAModule(MLP([128 + self.input_dim, 256, 256]))

        self.mlp = MLP(
            [256 + 256, 256, self.output_dim], dropout=self.dropout, norm=None
        )
        self.sdf_pre_layer = MLP(
            [256, 256, 256], dropout=self.dropout, norm=None
        )  # norm=batch_norm before

        print("sa1_module", self.sa1_module)
        print("sa2_module", self.sa2_module)
        print("sa3_module", self.sa3_module)
        print("mlp", self.mlp)
        print("pre_layer", self.sdf_pre_layer)
        print("self.dropout", dropout)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        sdf_encoded = self.sdf_pre_layer(data.z_sdf)

        # print('x', x.shape)
        # print('batch', batch.shape)
        # print('sdf_encoded', sdf_encoded.shape)

        return self.mlp(
            torch.cat([x, sdf_encoded[batch]], axis=-1)
        )  # x + sdf_encoded[batch])


class PositionalEmbedder(nn.Module):
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


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x).log_softmax(dim=-1)


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncodingTransformer(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc
