# -*- coding: utf-8 -*-
import torch

# Device and random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Physical domain range
X_MIN, X_MAX = 0.0, 2000.0
T_MIN, T_MAX = 0.0, 12000.0
X_SCALE = 2.0 / (X_MAX - X_MIN)
T_SCALE = 2.0 / (T_MAX - T_MIN)
EPS = 1e-8

# Training parameters
DEFAULT_EPOCHS = 50000
DEFAULT_PRINT_FREQ = 1000
ADAM_LR = 1e-3
LBFGS_LR = 1e-3

# Model parameters
HIDDEN_DIMS = [40, 40, 40, 40 ]
RANK = 10
N_HIDDEN = [10, 10]
N_MIN, N_MAX = 0.01, 0.06