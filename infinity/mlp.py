import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


ACTIVATIONS = {
    "relu": partial(nn.ReLU),
    "sigmoid": partial(nn.Sigmoid),
    "tanh": partial(nn.Tanh),
    "selu": partial(nn.SELU),
    "softplus": partial(nn.Softplus),
    "gelu": partial(nn.GELU),
    "swish": partial(Swish),
    "elu": partial(nn.ELU),
    "leakyrelu": partial(nn.LeakyReLU),
}


class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self, latent_dim, num_modulations, dim_hidden, num_layers, activation=nn.SiLU
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.activation = activation

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), self.activation()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden), self.activation()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        drop_rate=0.0,
        activation="swish",
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.activation1 = ACTIVATIONS[activation]()
        self.activation2 = ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        eta = self.linear1(x)
        # eta = self.batch_norm1(eta)
        eta = self.linear2(self.activation1(eta))
        # no more dropout
        # out = self.activation2(x + self.dropout(eta))
        out = x + self.activation2(self.dropout(eta))
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        depth=2,
        dropout=0.0,
        activation="swish",
    ):
        super().__init__()
        net = [ResBlock(input_dim, hidden_dim, dropout, activation)]
        for _ in range(depth - 1):
            net.append(ResBlock(input_dim, hidden_dim, dropout, activation))

        self.net = nn.Sequential(*net)
        self.project_map = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        out = self.net(z)
        out = self.project_map(out)

        return out
