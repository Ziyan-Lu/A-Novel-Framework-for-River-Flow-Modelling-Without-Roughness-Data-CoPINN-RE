# -*- coding: utf-8 -*-
"""
Main Entry Point - Saint-Venant Equation PINN Inversion
"""

import warnings

warnings.filterwarnings('ignore')

from src.training import train_model
from src.visualization import plot_time_comparison, plot_results , plot_error_heatmaps, plot_roughness_field_1d
from src.data_loader import load_validation_data, list_available_cases

def main():
    print("Starting training...")

    available_cases = list_available_cases()

    print(f"Available data cases: {available_cases}")

    if not available_cases:
        print("Error: No data cases found")
        return

    # Configuration parameters
    case_id = "case3"
    constant_s0 = False  # Bottom slope form: True-constant, False-varying
    constant_width = False  # River width form: True-constant, False-varying

    print(f"Using data case: {case_id}")

    # Train model
    model, losses, val_metrics, n_history, loss_components, difficulty_history, _ = train_model(
        num_epochs=50000,
        print_freq=1000,
        constant_s0=constant_s0,
        constant_width=constant_width
    )

    if model is None:
        print("Training failed!")
        return

    # Result visualization
    print("Plotting training results...")
    plot_results(model, losses, val_metrics, n_history, difficulty_history)

    print("Visualizing roughness field...")
    plot_roughness_field_1d(model)

    # 验证和对比
    print("Loading validation data...")
    validation_data = load_validation_data()

    if validation_data[0] is not None:
        print("Plotting spatiotemporal comparison...")
        plot_time_comparison(model, validation_data)

        print("Plotting error heatmaps...")
        plot_error_heatmaps(model, validation_data)
    else:
        print("Validation data loading failed, skipping comparison analysis")

    print("Training completed!")


if __name__ == "__main__":
    main()