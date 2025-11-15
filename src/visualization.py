# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from matplotlib.colors import LogNorm
from .physics import denorm_x, get_true_n
from .config import device



def plot_results(model, losses, val_metrics, n_history, difficulty_history):
    """Plot training results summary"""
    if not losses:
        print("Loss is empty, skipping plot")
        return

    plt.figure(figsize=(15, 12))

    # 1. Training loss curve
    plt.subplot(2, 2, 1)
    plt.semilogy(losses)
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss Value (log scale)')
    plt.title('Training Loss Evolution')
    plt.grid(True, alpha=0.3)

    # 2. Water depth validation error
    plt.subplot(2, 2, 2)
    if val_metrics:
        # Extract numeric data (exclude 'final' etc.)
        numeric_metrics = [m for m in val_metrics if isinstance(m[0], int)]
        if numeric_metrics:
            epochs = [m[0] for m in numeric_metrics]
            rmse_h = [m[1] for m in numeric_metrics]
            eps_h = [m[3] for m in numeric_metrics]

            plt.plot(epochs, rmse_h, 'o-', label='RMSE_h', markersize=4)
            plt.plot(epochs, eps_h, 's-', label='Relative Error ε_h', markersize=4)
            plt.legend()
            plt.xlabel('Training Epochs')
            plt.ylabel('Error')
            plt.title('Water Depth Validation Error')
            plt.grid(True, alpha=0.3)

    # 3. Discharge validation error
    plt.subplot(2, 2, 3)
    if val_metrics:
        numeric_metrics = [m for m in val_metrics if isinstance(m[0], int)]
        if numeric_metrics:
            epochs = [m[0] for m in numeric_metrics]
            rmse_Q = [m[2] for m in numeric_metrics]
            eps_Q = [m[4] for m in numeric_metrics]

            plt.plot(epochs, rmse_Q, 'o-', label='RMSE_Q', markersize=4)
            plt.plot(epochs, eps_Q, 's-', label='Relative Error ε_Q', markersize=4)
            plt.legend()
            plt.xlabel('Training Epochs')
            plt.ylabel('Error')
            plt.title('Discharge Validation Error')
            plt.grid(True, alpha=0.3)

    # 4. Roughness mean history
    plt.subplot(2, 2, 4)
    if n_history:
        # Extract numeric data
        numeric_history = [(e, v) for e, v in n_history if isinstance(e, int)]
        if numeric_history:
            epochs = [e for e, v in numeric_history]
            n_means = [v for e, v in numeric_history]

            plt.plot(epochs, n_means, 'o-', label='Mean Roughness <n(x)>', markersize=4)
            plt.axhline(y=0.03, color='r', linestyle='--', label='Reference Mean (~0.03)', alpha=0.7)
            plt.legend()
            plt.xlabel('Training Epochs')
            plt.ylabel('Mean Roughness')
            plt.title('Mean Roughness History')
            plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/app/results/training_results.png', dpi=300, bbox_inches='tight')
    print("Training results saved as /app/results/training_results.png")

    # Save training data to CSV
    try:
        # Save loss data
        loss_df = pd.DataFrame({'epoch': range(len(losses)), 'loss': losses})
        loss_df.to_csv('/app/results/training_loss.csv', index=False)

        # Save validation metrics
        if val_metrics:
            val_df = pd.DataFrame(val_metrics, columns=['epoch', 'rmse_h', 'rmse_Q', 'eps_h', 'eps_Q'])
            val_df.to_csv('/app/results/validation_metrics.csv', index=False)

        # Save roughness history
        if n_history:
            n_df = pd.DataFrame(n_history, columns=['epoch', 'mean_n'])
            n_df.to_csv('/app/results/n_history.csv', index=False)

        print("Training data saved to CSV files")
    except Exception as e:
        print(f"Error saving training data to CSV: {e}")


def plot_time_comparison(model, validation_data, ystep=0.05):
    """Plot spatial distribution comparison at specific time points"""

    def nice_limits(y, step=0.05, pad=0.02):
        """Calculate appropriate axis limits"""
        y = np.asarray(y)
        if len(y) == 0 or np.all(np.isnan(y)):
            return 0.0, step
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        if ymin == ymax:
            ymin -= step / 2
            ymax += step / 2
        rng = max(ymax - ymin, 1e-9)
        ymin -= pad * rng
        ymax += pad * rng
        lo = step * np.floor(ymin / step)
        hi = step * np.ceil(ymax / step)
        if np.isnan(lo) or np.isnan(hi):
            lo, hi = 0.0, step
        if hi - lo < step:
            hi = lo + step
        return lo, hi

    try:
        x_val_norm, t_val_norm, h_val_true, Q_val_true, x_val_orig, t_val_orig = validation_data

        # Select three typical time points for comparison
        target_times = [4000, 8000, 12000]
        plt.figure(figsize=(15, 10))

        comparison_data = []

        for i, target_time in enumerate(target_times):
            # Find the index corresponding to the time point
            time_idx = target_time // 10  # Assuming time step is 10 seconds
            start_idx = time_idx * 21  # Each time point has 21 spatial points
            end_idx = start_idx + 21

            # Extract true values
            h_true = h_val_true[start_idx:end_idx].cpu().numpy().squeeze()
            Q_true = Q_val_true[start_idx:end_idx].cpu().numpy().squeeze()

            # Get corresponding normalized coordinates
            xn = x_val_norm[start_idx:end_idx]
            tn = t_val_norm[start_idx:end_idx]

            # Model prediction
            with torch.no_grad():
                h_pred, Q_pred = model(xn, tn)
            h_pred = h_pred.cpu().numpy().squeeze()
            Q_pred = Q_pred.cpu().numpy().squeeze()

            # Physical coordinates
            x_orig = x_val_orig[start_idx:end_idx].cpu().numpy().squeeze()

            # Sort by distance
            sort_idx = np.argsort(x_orig)
            x_sorted = x_orig[sort_idx]

            # Collect data for saving
            for j in range(len(x_sorted)):
                comparison_data.append({
                    'time': target_time,
                    'distance': x_sorted[j],
                    'h_true': h_true[sort_idx][j],
                    'h_pred': h_pred[sort_idx][j],
                    'Q_true': Q_true[sort_idx][j],
                    'Q_pred': Q_pred[sort_idx][j]
                })

            # Plot water depth comparison subplot
            ax1 = plt.subplot(2, 3, i + 1)
            ax1.plot(x_sorted, h_true[sort_idx], 'b-', label='FDM True', linewidth=2)
            ax1.plot(x_sorted, h_pred[sort_idx], 'r--', label='PINN Predicted', linewidth=2)
            ax1.set_title(f'Water Depth at t = {target_time} s')
            ax1.set_xlabel('Distance (m)')
            ax1.set_ylabel('Depth (m)')
            ax1.grid(True, alpha=0.3)
            if i == 0:
                ax1.legend()

            # Set appropriate y-axis range
            y_data = np.concatenate([h_true[sort_idx], h_pred[sort_idx]])
            y_min, y_max = nice_limits(y_data, step=ystep)
            ax1.set_ylim(y_min, y_max)

            # Plot discharge comparison subplot
            ax2 = plt.subplot(2, 3, i + 4)
            ax2.plot(x_sorted, Q_true[sort_idx], 'b-', label='FDM True', linewidth=2)
            ax2.plot(x_sorted, Q_pred[sort_idx], 'r--', label='PINN Predicted', linewidth=2)
            ax2.set_title(f'Discharge at t = {target_time} s')
            ax2.set_xlabel('Distance (m)')
            ax2.set_ylabel('Discharge (m³/s)')
            ax2.grid(True, alpha=0.3)
            if i == 0:
                ax2.legend()

            # Set appropriate y-axis range
            y_data = np.concatenate([Q_true[sort_idx], Q_pred[sort_idx]])
            y_min, y_max = nice_limits(y_data, step=ystep)
            ax2.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig('/app/results/spatial_comparison.png', dpi=300, bbox_inches='tight')
        print("Spatial comparison plot saved as /app/results/spatial_comparison.png")

        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv('/app/results/spatial_comparison_data.csv', index=False)
        print("Spatial comparison data saved as /app/results/spatial_comparison_data.csv")

        return df_comparison

    except Exception as e:
        print(f"Error plotting spatial comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_roughness_field_1d(model, nx=401):
    """Plot 1D roughness field distribution"""
    model.eval()
    with torch.no_grad():
        x_norm = torch.linspace(-1.0, 1.0, nx, device=device).unsqueeze(1)
        n_pred = model.n_field(x_norm).cpu().numpy().squeeze()
        x_phys = denorm_x(x_norm).cpu().numpy().squeeze()

    n_true = get_true_n(x_phys).cpu().numpy().squeeze()

    plt.figure(figsize=(10, 5))
    plt.plot(x_phys, n_pred, 'b-', lw=2, label='PINN Predicted n(x)')
    plt.plot(x_phys, n_true, 'r--', lw=2, label='True n(x)')
    plt.hlines([model.n_min, model.n_max], x_phys.min(), x_phys.max(),
               colors='k', linestyles=':', alpha=0.35, label='n bounds')
    plt.xlabel('Distance (m)')
    plt.ylabel('Manning Roughness Coefficient n')
    plt.title('River Roughness Distribution Along Channel')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('/app/results/n_field.png', dpi=300, bbox_inches='tight')
    print("Roughness field plot saved:  /app/results/n_field.png")

    # Save data to CSV
    try:
        df = pd.DataFrame({
            'x(m)': x_phys,
            'n_pred': n_pred,
            'n_true': n_true,
            'error': np.abs(n_pred - n_true)
        })
        df.to_csv('/app/results/n_field.csv', index=False)
        print("Roughness field data saved: /app/results/n_field.csv")
    except Exception as e:
        print(f"Error saving roughness data to CSV: {e}")


def plot_error_heatmaps(model, validation_data):
    """Plot error heatmaps"""
    try:
        x_val_norm, t_val_norm, h_true, Q_true, x_val_orig, t_val_orig = validation_data

        with torch.no_grad():
            h_pred, Q_pred = model(x_val_norm, t_val_norm)

        x_unique = np.unique(x_val_orig.detach().cpu().numpy().squeeze())
        t_unique = np.unique(t_val_orig.detach().cpu().numpy().squeeze())
        nx, nt = len(x_unique), len(t_unique)

        # Reshape data to grid format
        h_pred_grid = h_pred.detach().cpu().numpy().reshape(nt, nx)
        Q_pred_grid = Q_pred.detach().cpu().numpy().reshape(nt, nx)
        h_true_grid = h_true.detach().cpu().numpy().reshape(nt, nx)
        Q_true_grid = Q_true.detach().cpu().numpy().reshape(nt, nx)

        # Calculate absolute errors
        h_error = np.abs(h_pred_grid - h_true_grid)
        Q_error = np.abs(Q_pred_grid - Q_true_grid)

        # Create grid for data saving
        t_mesh, x_mesh = np.meshgrid(t_unique, x_unique, indexing='ij')

        # Prepare data for CSV saving
        error_data = []
        for i in range(nt):
            for j in range(nx):
                error_data.append({
                    'time': t_mesh[i, j],
                    'distance': x_mesh[i, j],
                    'h_error': h_error[i, j],
                    'Q_error': Q_error[i, j],
                    'h_true': h_true_grid[i, j],
                    'h_pred': h_pred_grid[i, j],
                    'Q_true': Q_true_grid[i, j],
                    'Q_pred': Q_pred_grid[i, j]
                })

        # Plot water depth error heatmap
        plt.figure(figsize=(10, 6))
        plt.imshow(h_error, origin='lower', aspect='auto',
                   extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                   cmap='magma')
        plt.colorbar(label='Absolute Error of Depth (m)')
        plt.title('Heatmap of |h_PINN - h_FDM|')
        plt.xlabel('Distance x (m)')
        plt.ylabel('Time t (s)')
        plt.xticks(np.arange(0, 2001, 500))
        plt.yticks(np.arange(0, 12001, 2000))
        plt.tight_layout()
        plt.savefig('/app/results/err_heatmap_depth.png', dpi=300, bbox_inches='tight')
        print("Water depth error heatmap saved:/app/results/err_heatmap_depth.png")

        # Plotting a heatmap of flow errors
        plt.figure(figsize=(10, 6))
        plt.imshow(Q_error, origin='lower', aspect='auto',
                   extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                   cmap='plasma')
        plt.colorbar(label='Absolute Error of Discharge (m³/s)')
        plt.title('Heatmap of |Q_PINN - Q_FDM|')
        plt.xlabel('Distance x (m)')
        plt.ylabel('Time t (s)')
        plt.xticks(np.arange(0, 2001, 500))
        plt.yticks(np.arange(0, 12001, 2000))
        plt.tight_layout()
        plt.savefig('/app/results/err_heatmap_discharge.png', dpi=300, bbox_inches='tight')
        print("Discharge error heatmap saved: /app/results/err_heatmap_discharge.png")

        # Save error data to CSV
        df_errors = pd.DataFrame(error_data)
        df_errors.to_csv('/app/results/error_heatmap_data.csv', index=False)
        print("Error heatmap data saved as /app/results/error_heatmap_data.csv")

    except Exception as e:
        print(f"Error plotting error heatmaps: {e}")
        import traceback
        traceback.print_exc()