# 使src成为Python包
from .models import SPINN, GatedMLP
from .physics import saint_venant_equations, get_true_s0, get_true_n, get_width
from .data_loader import load_training_data, load_validation_data, generate_collocation_points
from .training import train_model, CognitiveScheduler, loss_fn, validate
from .visualization import plot_results, plot_time_comparison, plot_roughness_field_1d, plot_error_heatmaps