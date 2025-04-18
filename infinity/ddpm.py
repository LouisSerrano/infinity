import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


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

class ResMLPBlock(nn.Module):
    def __init__(self, in_channels, embed_channels, hidden_channels, out_channels, droprate=0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(embed_channels, hidden_channels)
        self.linear0 = Linear0(hidden_channels, out_channels)
        #self.linear0 = nn.Linear(hidden_channels, out_channels)

        self.activation = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_channels)
        self.layer_norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(droprate)
        self.activation = nn.SiLU()
        
    def forward(self, x_hidden, time_embedding):

        x = self.layer_norm1(x_hidden)
        x = self.activation(x)
        x = self.linear1(x)

        time_hidden = self.activation(time_embedding)
        time_hidden = self.linear2(time_hidden)

        x = x + time_hidden
        x = self.layer_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear0(x)

        return x_hidden + x

class Linear0(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        return self.linear(x)

class NameBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.resmlp1 = ResMLPBlock(in_channels, out_channels, hidden_channels)
        self.resmlp2 = ResMLPBlock(out_channels, out_channels, hidden_channels)
        
    def forward(self, x):
        x = self.resmlp1(x)
        x = self.resmlp2(x)
        return x

class DDPM(nn.Module):
    def __init__(self, in_channels, embed_channels, hidden_channels, out_channels, depth=3, droprate=0.3):
        super().__init__()
        #self.embedding = nn.Embedding(1000, in_channels)
        self.depth = depth

        self.embedding = SinusoidalPosEmb(embed_channels)
        self.linear_time_1 = nn.Linear(embed_channels, hidden_channels)
        self.linear_time_2 = nn.Linear(hidden_channels, hidden_channels)

        self.lift = nn.Linear(in_channels, hidden_channels)

        self.project = Linear0(hidden_channels, out_channels)
        #self.project = nn.Linear(hidden_channels, out_channels)

        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(hidden_channels)

        #self.residual_blocks = nn.ModuleList([ResMLPBlock(hidden_channels, embed_channels, hidden_channels, hidden_channels, droprate=droprate) for _ in range(depth)])
        self.residual_blocks = nn.ModuleList([ResMLPBlock(hidden_channels, hidden_channels, hidden_channels, hidden_channels, droprate=droprate) for _ in range(depth)])

    def forward(self, x, t):

        t = self.embedding(t)
        #print('t', t.shape)

        x = self.lift(x)

        #print('x', x.shape)

        for j in range(self.depth):
            x = self.residual_blocks[j](x, t)
            #print(j, 'x', x.shape)

        x = self.layer_norm(x)

        #print('layer norm', x.shape)

        x = self.activation(x)

        #print('act', x.shape)

        x = self.project(x)

        #print('out', x.shape)
        
        return x


class ConditionalResMLPBlock(nn.Module):
    def __init__(self, in_channels, y_channels, embed_channels, hidden_channels, out_channels, droprate=0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.lift_y = nn.Linear(y_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)  # Added y_channels input
        self.linear3 = nn.Linear(embed_channels, hidden_channels)
        self.linear0 = Linear0(hidden_channels, out_channels)

        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.layer_norm2 = nn.LayerNorm(hidden_channels)
        self.layer_norm3 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(droprate)
        self.activation = nn.SiLU()
        
    def forward(self, x_hidden, y_hidden, time_embedding):  # Added y_hidden
        x = self.layer_norm1(x_hidden)
        x = self.activation(x)
        x = self.linear1(x)

        #y = self.layer_norm2(y_hidden)
        #y = self.activation(y)
        y = self.lift_y(y_hidden)
        y = self.layer_norm2(y)  # Process the conditional input y
        y = self.activation(y)
        y = self.linear2(y)

        time_hidden = self.activation(time_embedding)
        time_hidden = self.linear3(time_hidden)

        x = x + y + time_hidden  # Combine x, y, and time_hidden
        x = self.layer_norm3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear0(x)

        return x_hidden + x

class ConditionalDDPM(nn.Module):
    def __init__(self, in_channels, y_channels, embed_channels, hidden_channels, out_channels, depth=3, droprate=0.3):
        super().__init__()
        self.depth = depth

        self.embedding = SinusoidalPosEmb(embed_channels)
        self.lift = nn.Linear(in_channels, hidden_channels)
        self.linear_time_1 = nn.Linear(embed_channels, hidden_channels)
        self.linear_time_2 = nn.Linear(hidden_channels, hidden_channels)

        self.project = Linear0(hidden_channels + y_channels, out_channels)

        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(hidden_channels)

        self.residual_blocks = nn.ModuleList([
            ConditionalResMLPBlock(hidden_channels, y_channels, hidden_channels, hidden_channels, hidden_channels, droprate=droprate)
            for _ in range(depth)
        ])

    def forward(self, x, y, t):  # Added y as input
        t = self.embedding(t)
        t = self.linear_time_1(t)
        t = self.activation(t)
        t = self.linear_time_2(t)

        x = self.lift(x)

        for j in range(self.depth):
            x = self.residual_blocks[j](x, y, t)  # Pass y along with x and t

        x = self.layer_norm(x)
        x = self.activation(x)
        x = self.project(torch.cat([x, y], -1))
        
        return x
