# -*- coding: utf-8 -*-
import torch
import numpy as np
from .config import X_MIN, X_MAX, T_MIN, T_MAX, X_SCALE, T_SCALE, EPS, device

def get_true_s0(x_phys, constant=False):
    """True bottom slope (can be constant or piecewise linear)"""
    if constant:
        if isinstance(x_phys, torch.Tensor):
            return torch.full_like(x_phys, 0.0025, device=device)
        else:
            return np.full_like(x_phys, 0.0025)
    else:
        if isinstance(x_phys, torch.Tensor):
            x_phys = x_phys.detach().cpu().numpy()
        s0_true = np.zeros_like(x_phys)
        mask1 = x_phys <= 1000
        s0_true[mask1] = 0.0025
        mask2 = (x_phys > 1000) & (x_phys <= 1600)
        slope2 = (0.0050 - 0.0025) / (1600 - 1000)
        s0_true[mask2] = 0.0025 + slope2 * (x_phys[mask2] - 1000)
        mask3 = x_phys > 1600
        slope3 = (0.0015 - 0.0050) / (2000 - 1600)
        s0_true[mask3] = 0.0050 + slope3 * (x_phys[mask3] - 1600)
        return torch.tensor(s0_true, device=device).float()

def get_true_n(x_phys):
    """True roughness coefficient (piecewise linear)"""
    if isinstance(x_phys, torch.Tensor):
        x_phys = x_phys.detach().cpu().numpy()
    n_true = np.zeros_like(x_phys)
    mask1 = x_phys <= 800
    n_true[mask1] = 2.5e-5 * x_phys[mask1] + 0.02
    mask2 = (x_phys > 800) & (x_phys <= 1200)
    n_true[mask2] = 0.04
    mask3 = x_phys > 1200
    n_true[mask3] = -1.25e-5 * (x_phys[mask3] - 1200) + 0.04
    return torch.tensor(n_true, device=device).float()

def get_width(x_phys, constant=False):
    """River width function (can be constant or varying)"""
    if constant:
        if isinstance(x_phys, torch.Tensor):
            return torch.full_like(x_phys, 2.0, device=device)
        else:
            return np.full_like(x_phys, 2.0)
    else:
        if isinstance(x_phys, torch.Tensor):
            x_phys = x_phys.detach().cpu().numpy()
        condition1 = x_phys < 500
        condition2 = x_phys > 1500
        t = (x_phys - 500) / 1000
        smooth_factor = 3 * t ** 2 - 2 * t ** 3
        transition_value = 2.0 + 2.0 * smooth_factor
        width = np.where(condition1, 2.0, transition_value)
        width = np.where(condition2, 4.0, width)
        return torch.tensor(width, device=device).float()

def normalize_coordinates(x, t):
    """Normalize coordinates to [-1, 1] range"""
    if not isinstance(x, torch.Tensor): x = torch.tensor(x, device=device)
    if not isinstance(t, torch.Tensor): t = torch.tensor(t, device=device)
    x_norm = 2 * (x - X_MIN) / (X_MAX - X_MIN) - 1
    t_norm = 2 * (t - T_MIN) / (T_MAX - T_MIN) - 1
    return x_norm, t_norm, x, t

def denorm_x(x_norm):
    """Denormalize x coordinate back to physical domain"""
    return (x_norm + 1.0) * 0.5 * (X_MAX - X_MIN) + X_MIN

def denorm_t(t_norm):
    """Denormalize t coordinate back to physical domain"""
    return (t_norm + 1.0) * 0.5 * (T_MAX - T_MIN) + T_MIN

def saint_venant_equations(h, Q, x_norm, t_norm, n_field, constant_s0=False, constant_width=False, g=9.81, eps=1e-8):
    """Saint-Venant equations implementation"""
    x_phys = denorm_x(x_norm)
    width = get_width(x_phys, constant=constant_width)
    A = h * width + eps
    P_w = 2 * h + width + eps
    R = A / (P_w + eps)

    S0_true = get_true_s0(x_phys, constant=constant_s0)

    dh_dx_n = torch.autograd.grad(h.sum(), x_norm, create_graph=True, retain_graph=True)[0]
    dQ_dx_n = torch.autograd.grad(Q.sum(), x_norm, create_graph=True, retain_graph=True)[0]
    dh_dt_n = torch.autograd.grad(h.sum(), t_norm, create_graph=True, retain_graph=True)[0]
    dQ_dt_n = torch.autograd.grad(Q.sum(), t_norm, create_graph=True, retain_graph=True)[0]
    Q2_A = Q ** 2 / (A + eps)
    d_Q2A_dx_n = torch.autograd.grad(Q2_A.sum(), x_norm, create_graph=True, retain_graph=True)[0]

    dh_dx = dh_dx_n * X_SCALE
    dQ_dx = dQ_dx_n * X_SCALE
    dh_dt = dh_dt_n * T_SCALE
    dQ_dt = dQ_dt_n * T_SCALE
    d_Q2A_dx = d_Q2A_dx_n * X_SCALE

    S_f = (n_field ** 2 * Q * torch.abs(Q)) / (A ** 2 * (R ** (4.0 / 3.0) + eps) + eps)

    f1 = width * dh_dt + dQ_dx
    f2 = dQ_dt + d_Q2A_dx + g * A * (dh_dx + S_f - S0_true)
    return f1, f2