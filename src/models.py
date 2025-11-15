# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import device


class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.U_layer = nn.Linear(input_dim, hidden_dims[0])
        self.V_layer = nn.Linear(input_dim, hidden_dims[0])
        self.H_layer = nn.Linear(input_dim, hidden_dims[0])
        self.layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                                     for i in range(len(hidden_dims) - 1)])
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        for layer in [self.U_layer, self.V_layer, self.H_layer, *self.layers, self.output_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        U = torch.tanh(self.U_layer(x))
        V = torch.tanh(self.V_layer(x))
        H = torch.tanh(self.H_layer(x))
        for layer in self.layers:
            Z = torch.tanh(layer(H))
            H = (1 - Z) * U + Z * V
        return self.output_layer(H)


class SPINN(nn.Module):
    def __init__(self, hidden_dims=[40, 40, 40, 40], r=10, use_gated_mlp=True,
                 n_hidden=[10, 10], n_min=0.02, n_max=0.04):
        super().__init__()
        self.r = r
        self.use_gated_mlp = use_gated_mlp
        self.n_min, self.n_max = float(n_min), float(n_max)

        if use_gated_mlp:
            self.x_net = GatedMLP(1, hidden_dims, r)
            self.t_net = GatedMLP(1, hidden_dims, r)
        else:
            def mlp(inp, hds, out):
                dims = [inp] + hds
                layers = []
                for i in range(len(dims) - 1):
                    lin = nn.Linear(dims[i], dims[i + 1])
                    nn.init.xavier_uniform_(lin.weight)
                    nn.init.zeros_(lin.bias)
                    layers += [lin, nn.Tanh()]
                head = nn.Linear(dims[-1], out)
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)
                layers += [head]
                return nn.Sequential(*layers)

            self.x_net = mlp(1, hidden_dims, r)
            self.t_net = mlp(1, hidden_dims, r)

        self.output_layer = nn.Linear(r * r, 2)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        n_layers = []
        last = 1
        for h in n_hidden:
            lin = nn.Linear(last, h)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            n_layers += [lin, nn.Tanh()]
            last = h
        self.n_layers = nn.Sequential(*n_layers)
        self.n_head = nn.Linear(last, 1)
        nn.init.xavier_uniform_(self.n_head.weight)
        nn.init.zeros_(self.n_head.bias)

    def forward(self, x_norm, t_norm):
        phi_x = self.x_net(x_norm)
        phi_t = self.t_net(t_norm)
        outer = torch.einsum('bi,bj->bij', phi_x, phi_t).view(phi_x.size(0), -1)
        out = self.output_layer(outer)
        h = F.softplus(out[:, 0:1]) + 1e-3
        Q = F.softplus(out[:, 1:2]) + 1e-3
        return h, Q

    def n_field(self, x_norm):
        z = self.n_head(self.n_layers(x_norm))
        n01 = torch.sigmoid(z)
        return self.n_min + (self.n_max - self.n_min) * n01