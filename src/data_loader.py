# -*- coding: utf-8 -*-
import os
import traceback
import torch
import numpy as np
import pandas as pd
from .physics import normalize_coordinates, denorm_x, denorm_t
from .config import device


def load_training_data(case_id="case3"):
    """Load training data

    Args:
        case_id: Case ID, corresponding to data/case1, data/case2, data/case3
    """
    try:
        # Build data directory path
        data_dir = os.path.join('/app/data', case_id)
        print(f"Loading data from directory: {data_dir}")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        # Initial condition file
        ic_file = os.path.join(data_dir, 'initial_condition.csv')
        if not os.path.exists(ic_file):
            raise FileNotFoundError(f"{ic_file} not found")

        ic_data = pd.read_csv(ic_file, header=None)
        x_ic = torch.tensor(ic_data.iloc[:, 0].values, device=device).float().unsqueeze(1)
        t_ic = torch.zeros_like(x_ic)
        h_ic = torch.tensor(ic_data.iloc[:, 1].values, device=device).float().unsqueeze(1)
        Q_ic = torch.tensor(ic_data.iloc[:, 2].values, device=device).float().unsqueeze(1)

        # Upstream boundary condition
        bc_file = os.path.join(data_dir, 'upstream_bc.csv')
        if not os.path.exists(bc_file):
            raise FileNotFoundError(f"{bc_file} not found")

        bc_data = pd.read_csv(bc_file, header=None)
        t_bc = torch.tensor(bc_data.iloc[:, 0].values, device=device).float().unsqueeze(1)
        x_bc_up = torch.zeros_like(t_bc)
        h_bc_up = torch.tensor(bc_data.iloc[:, 1].values, device=device).float().unsqueeze(1)
        Q_bc_up = torch.tensor(bc_data.iloc[:, 2].values, device=device).float().unsqueeze(1)

        # Downstream boundary condition
        down_bc_file = os.path.join(data_dir, 'down_bc.csv')
        if not os.path.exists(down_bc_file):
            raise FileNotFoundError(f"{down_bc_file} not found")

        down_bc_df = pd.read_csv(down_bc_file, header=None)
        h_bc_down = torch.tensor(down_bc_df.iloc[:, 0].values, device=device).float().unsqueeze(1)
        Q_bc_down = torch.tensor(down_bc_df.iloc[:, 1].values, device=device).float().unsqueeze(1)
        x_bc_down = torch.ones_like(t_bc) * 2000.0

        def load_obs_file(x_val, h_file, q_file):
            """Load observation data file"""
            h_file_path = os.path.join(data_dir, h_file)
            q_file_path = os.path.join(data_dir, q_file)

            if not os.path.exists(h_file_path):
                print(f"Warning: {h_file_path} does not exist")
                return np.empty((0, 4))
            if not os.path.exists(q_file_path):
                print(f"Warning: {q_file_path} does not exist")
                return np.empty((0, 4))

            try:
                h_data = np.loadtxt(h_file_path, skiprows=1, delimiter=',')
                q_data = np.loadtxt(q_file_path, skiprows=1, delimiter=',')

                # Ensure data lengths are consistent
                min_len = min(len(h_data), len(q_data))
                h_data = h_data[:min_len]
                q_data = q_data[:min_len]

                arr = np.column_stack((np.full(min_len, x_val),
                                       h_data[:, 0], h_data[:, 1], q_data[:, 1]))
                return arr[arr[:, 1] > 0]
            except Exception as e:
                print(f"Error loading observation file {h_file}/{q_file}: {e}")
                return np.empty((0, 4))

        # Load observation data from different locations - only load 500 and 1500
        obs_files = [
            (500, 'h_x_500.csv', 'q_total_x_500.csv'),
            (1500, 'h_x_1500.csv', 'q_total_x_1500.csv')
        ]

        obs_data_list = []
        for x_val, h_file, q_file in obs_files:
            obs_data = load_obs_file(x_val, h_file, q_file)
            if len(obs_data) > 0:
                obs_data_list.append(obs_data)
                print(f"Loaded {len(obs_data)} observation points from {x_val}m position")
            else:
                print(f"No valid observation data found at {x_val}m position")

        if not obs_data_list:
            print("Warning: No observation data found")
            obs_data = np.empty((0, 4))
        else:
            obs_data = np.vstack(obs_data_list)

        # Create empty arrays if no observation data
        if len(obs_data) == 0:
            x_obs = torch.empty((0, 1), device=device)
            t_obs = torch.empty((0, 1), device=device)
            h_obs = torch.empty((0, 1), device=device)
            Q_obs = torch.empty((0, 1), device=device)
        else:
            x_obs = torch.tensor(obs_data[:, 0], device=device).float().unsqueeze(1)
            t_obs = torch.tensor(obs_data[:, 1], device=device).float().unsqueeze(1)
            h_obs = torch.tensor(obs_data[:, 2], device=device).float().unsqueeze(1)
            Q_obs = torch.tensor(obs_data[:, 3], device=device).float().unsqueeze(1)

        # Normalize coordinates
        x_ic_norm, t_ic_norm, _, _ = normalize_coordinates(x_ic, t_ic)
        x_bc_up_norm, t_bc_norm, _, _ = normalize_coordinates(x_bc_up, t_bc)
        x_bc_down_norm, _, _, _ = normalize_coordinates(x_bc_down, t_bc)
        x_obs_norm, t_obs_norm, _, _ = normalize_coordinates(x_obs, t_obs)

        # Combine all training data
        x_train = torch.cat([x_ic_norm, x_bc_up_norm, x_bc_down_norm, x_obs_norm], dim=0)
        t_train = torch.cat([t_ic_norm, t_bc_norm, t_bc_norm, t_obs_norm], dim=0)
        h_train = torch.cat([h_ic, h_bc_up, h_bc_down, h_obs], dim=0)
        Q_train = torch.cat([Q_ic, Q_bc_up, Q_bc_down, Q_obs], dim=0)

        print(f"Training data loaded: {len(x_train)} points")
        print(f" - Initial conditions: {len(x_ic)} points")
        print(f"  - Boundary conditions: {len(x_bc_up) + len(x_bc_down)} points")
        print(f"  - Observation data: {len(x_obs)} points")

        return (x_train.requires_grad_(True),
                t_train.requires_grad_(True), h_train, Q_train)

    except Exception as e:
        print(f"Error loading training data: {e}")
        traceback.print_exc()
        return None, None, None, None


def generate_collocation_points(N_c=20000):
    """Generate collocation points"""
    from .config import X_MIN, X_MAX, T_MIN, T_MAX
    x_c = torch.rand(N_c, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    t_c = torch.rand(N_c, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    x_c_norm, t_c_norm, _, _ = normalize_coordinates(x_c, t_c)
    return x_c_norm.requires_grad_(True), t_c_norm.requires_grad_(True)


def load_validation_data(case_id="case3"):
    """Load validation data"""
    try:
        # Build data directory path
        data_dir = os.path.join('/app/data', case_id)
        print(f"Loading validation data from directory: {data_dir}")

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        # Create validation grid
        x_val = torch.arange(0, 2001, 100, device=device).float().unsqueeze(1)
        t_val = torch.arange(0, 12001, 10, device=device).float().unsqueeze(1)
        Tm, Xm = torch.meshgrid(t_val.squeeze(), x_val.squeeze(), indexing='ij')
        t_val_grid = Tm.reshape(-1, 1)
        x_val_grid = Xm.reshape(-1, 1)

        # Load validation data files
        h_file = os.path.join(data_dir, 'water_depth.csv')
        Q_file = os.path.join(data_dir, 'total_flow.csv')

        if not os.path.exists(h_file):
            raise FileNotFoundError(f"{h_file} not found")
        if not os.path.exists(Q_file):
            raise FileNotFoundError(f"{Q_file} not found")

        h_data = pd.read_csv(h_file, header=None).values
        Q_data = pd.read_csv(Q_file, header=None).values

        total_points = t_val_grid.shape[0]
        h_val = torch.zeros(total_points, 1, device=device)
        Q_val = torch.zeros(total_points, 1, device=device)

        for t_idx in range(min(1201, h_data.shape[0])):
            for x_idx in range(min(21, h_data.shape[1])):
                idx = t_idx * 21 + x_idx
                if idx < total_points:
                    h_val[idx] = h_data[t_idx, x_idx]
                    Q_val[idx] = Q_data[t_idx, x_idx]

        x_val_norm, t_val_norm, _, _ = normalize_coordinates(x_val_grid, t_val_grid)

        print(f"Validation data loaded: {len(x_val_norm)} points")
        return (x_val_norm.requires_grad_(True),
                t_val_norm.requires_grad_(True),
                h_val, Q_val, x_val_grid, t_val_grid)

    except Exception as e:
        print(f"Error loading validation data: {e}")
        traceback.print_exc()
        return None, None, None, None, None, None


def list_available_cases():
    """List available data cases"""
    data_dir = '/app/data'
    if not os.path.exists(data_dir):
        print(f"Data directory does not exist: {data_dir}")
        return []

    cases = [d for d in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('case')]
    cases.sort()
    print(f"Available cases: {cases}")
    return cases