# -*- coding: utf-8 -*-
import torch
import traceback
from torch.optim import LBFGS
from time import time
from .config import device, EPS
from .physics import saint_venant_equations, denorm_x, denorm_t
from .data_loader import load_training_data, generate_collocation_points, load_validation_data
from .models import SPINN



class CognitiveScheduler:

    def __init__(self, num_epochs, beta=0.001):
        self.num_epochs = num_epochs
        self.beta = beta

    def get_gamma_d(self, epoch):
        ve = 1 - (epoch - 1) / self.num_epochs
        vh = (epoch - 1) / self.num_epochs
        return (ve - vh) * self.beta

    def get_SPL_V(self, grad_x, grad_t, epoch):
        difficult = torch.sqrt(grad_x ** 2 + grad_t ** 2 + EPS)
        dmin, dmax = difficult.min(), difficult.max()
        normalized = (difficult - dmin) / (dmax - dmin + 1e-6)
        gamma_d = self.get_gamma_d(epoch)
        V = 1 - (1 / self.num_epochs) * (epoch - 1) - gamma_d * normalized
        return V, normalized


def loss_fn(model, x_train, t_train, h_train, Q_train, x_c, t_c,
            epoch, num_epochs, cognitive_scheduler=None,
            constant_s0=False, constant_width=False):
    """Calculate total loss function"""
    # Data loss
    h_pred, Q_pred = model(x_train, t_train)
    mse_h = torch.mean((h_pred - h_train) ** 2)
    mse_Q = torch.mean((Q_pred - Q_train) ** 2)
    # PDE residual loss
    h_c, Q_c = model(x_c, t_c)
    n_c = model.n_field(x_c)
    f1, f2 = saint_venant_equations(h_c, Q_c, x_c, t_c, n_c,
                                    constant_s0=constant_s0,
                                    constant_width=constant_width)
    pde_residual = torch.nan_to_num(f1 ** 2 + f2 ** 2, nan=1e6, posinf=1e6, neginf=1e6)

    V = None
    if cognitive_scheduler is not None:
        grad_x = torch.autograd.grad(pde_residual.sum(), x_c, retain_graph=True, create_graph=True)[0]
        grad_t = torch.autograd.grad(pde_residual.sum(), t_c, retain_graph=True, create_graph=True)[0]
        V, normalized = cognitive_scheduler.get_SPL_V(grad_x, grad_t, epoch)
        mse_pde = torch.mean(pde_residual * V)
    else:
        mse_pde = torch.mean(pde_residual)
        normalized = torch.zeros_like(pde_residual)

    # Boundary conditions
    upstream = (x_train == x_train.min()).squeeze()
    downstream = (x_train == x_train.max()).squeeze()
    initial = (t_train == t_train.min()).squeeze()

    mse_bc = torch.tensor(0.0, device=device)
    if upstream.any():
        mse_bc += torch.mean((h_pred[upstream] - h_train[upstream]) ** 2) + \
                  torch.mean((Q_pred[upstream] - Q_train[upstream]) ** 2)
    if downstream.any():
        mse_bc += torch.mean((h_pred[downstream] - h_train[downstream]) ** 2) + \
                  torch.mean((Q_pred[downstream] - Q_train[downstream]) ** 2)
    # Initial conditions
    mse_ic = torch.tensor(0.0, device=device)
    if initial.any():
        mse_ic += torch.mean((h_pred[initial] - h_train[initial]) ** 2) + \
                  torch.mean((Q_pred[initial] - Q_train[initial]) ** 2)

    total_loss = mse_h + mse_Q + mse_pde + mse_bc + mse_ic
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        total_loss = torch.tensor(1e6, device=device, requires_grad=True)

    return total_loss, {
        'mse_h': float(mse_h), 'mse_Q': float(mse_Q), 'mse_pde': float(mse_pde),
        'mse_bc': float(mse_bc), 'mse_ic': float(mse_ic),
        'normalized_difficulty': float(normalized.mean()) if isinstance(normalized, torch.Tensor) else 0.0,
        'V': V.detach() if isinstance(V, torch.Tensor) else None
    }


def validate(model, x_val_norm, t_val_norm, h_val_true, Q_val_true, *_):
    """Validate model performance"""
    with torch.no_grad():
        h_pred, Q_pred = model(x_val_norm, t_val_norm)
        rmse_h = torch.sqrt(torch.mean((h_pred - h_val_true) ** 2))
        rmse_Q = torch.sqrt(torch.mean((Q_pred - Q_val_true) ** 2))
        eps_h = torch.sqrt(torch.sum((h_pred - h_val_true) ** 2)) / (torch.sqrt(torch.sum(h_val_true ** 2)) + 1e-6)
        eps_Q = torch.sqrt(torch.sum((Q_pred - Q_val_true) ** 2)) / (torch.sqrt(torch.sum(Q_val_true ** 2)) + 1e-6)
    return rmse_h.item(), rmse_Q.item(), eps_h.item(), eps_Q.item()


def train_model(num_epochs=50000, print_freq=1000, adam_lr=1e-3, lbfgs_lr=1e-3,
                constant_s0=False, constant_width=False):
    print("Loading training data...")
    x_train, t_train, h_train, Q_train = load_training_data()
    if x_train is None:
        return (None,) * 7

    print("Generating collocation points...")
    x_c_full, t_c_full = generate_collocation_points(N_c=20000)
    x_c_full, t_c_full = x_c_full.detach(), t_c_full.detach()
    x_phys_full = denorm_x(x_c_full).cpu().numpy().squeeze()
    t_phys_full = denorm_t(t_c_full).cpu().numpy().squeeze()

    print("Loading validation data..")
    val = load_validation_data()
    if val is None:
        x_val_norm, t_val_norm, h_val, Q_val = x_train, t_train, h_train, Q_train
    else:
        x_val_norm, t_val_norm, h_val, Q_val, _, _ = val

    print("Initializing model...")
    model = SPINN(hidden_dims=[40, 40, 40,40], r=10, use_gated_mlp=True,
                  n_hidden=[10, 10], n_min=0.01, n_max=0.06).to(device)
    cognitive_scheduler = CognitiveScheduler(num_epochs, beta=0.001)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    optimizer_lbfgs = LBFGS(model.parameters(), lr=lbfgs_lr, max_iter=50000,
                            tolerance_grad=1e-6, tolerance_change=1e-10, history_size=100)

    losses, val_metrics, n_history, loss_components = [], [], [], []
    difficulty_history = []

    print(f"Starting Adam training (epochs={num_epochs})")
    print(f"Bottom slope form: {'constant' if constant_s0 else 'varying'}, "
          f"River width form: {'constant' if constant_width else 'varying'}")
    t0 = time()

    for epoch in range(num_epochs):
        x_c = x_c_full.clone().detach().requires_grad_(True)
        t_c = t_c_full.clone().detach().requires_grad_(True)

        optimizer_adam.zero_grad()
        total_loss, comp = loss_fn(model, x_train, t_train, h_train, Q_train,
                                   x_c, t_c, epoch, num_epochs, cognitive_scheduler,
                                   constant_s0=constant_s0, constant_width=constant_width)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_adam.step()

        losses.append(float(total_loss))
        loss_components.append(comp)
        difficulty_history.append(comp.get('normalized_difficulty', 0.0))

        with torch.no_grad():
            x_probe = torch.linspace(-1, 1, 401, device=device).unsqueeze(1)
            n_mean = model.n_field(x_probe).mean().item()
        n_history.append((epoch, n_mean))

        if epoch % print_freq == 0 or epoch == num_epochs - 1:
            rmse_h, rmse_Q, eps_h, eps_Q = validate(model, x_val_norm, t_val_norm, h_val, Q_val)
            val_metrics.append((epoch, rmse_h, rmse_Q, eps_h, eps_Q))
            print(
                f"Epoch {epoch}: Loss={float(total_loss):.4e} | RMSE_h={rmse_h:.4f}, RMSE_Q={rmse_Q:.4f} | ε_h={eps_h:.4f}, ε_Q={eps_Q:.4f} | <n>≈{n_mean:.5f}")


        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Loss is NaN/Inf, stopping early")
            break

    print(f"Adam completed, time elapsed: {time() - t0:.1f}s")

    print("Starting L-BFGS...")
    try:
        def closure():
            optimizer_lbfgs.zero_grad()
            x_c = x_c_full.clone().detach().requires_grad_(True)
            t_c = t_c_full.clone().detach().requires_grad_(True)
            tl, _ = loss_fn(model, x_train, t_train, h_train, Q_train, x_c, t_c,
                            num_epochs, num_epochs, None,
                            constant_s0=constant_s0, constant_width=constant_width)
            tl.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            return tl

        optimizer_lbfgs.step(closure)
        print("L-BFGS completed")
    except Exception as e:
        print(f"L-BFGS error: {e}")
        traceback.print_exc()

    try:
        final_epoch = 'final'
        with torch.no_grad():
            x_probe = torch.linspace(-1, 1, 401, device=device).unsqueeze(1)
            n_mean = model.n_field(x_probe).mean().item()
        n_history.append((final_epoch, n_mean))
        rmse_h, rmse_Q, eps_h, eps_Q = validate(model, x_val_norm, t_val_norm, h_val, Q_val)
        val_metrics.append((final_epoch, rmse_h, rmse_Q, eps_h, eps_Q))
        print(f"Final: RMSE_h={rmse_h:.4f}, RMSE_Q={rmse_Q:.4f}, ε_h={eps_h:.4f}, ε_Q={eps_Q:.4f}, <n>≈{n_mean:.5f}")
    except Exception as e:
        print(f"Final validation failed: {e}")

    return model, losses, val_metrics, n_history, loss_components, difficulty_history, None