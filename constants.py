from pathlib import Path

project_root = Path(__file__).resolve().parent


######################################################################
# DATA SPLIT
######################################################################

split_train_fraction = 0.60
split_val_fraction = 0.20


######################################################################
# DATASET AND MODEL FILE NAMES
######################################################################

dataset_prefix = "qubit_dataset_sim"


######################################################################
# CNN-SITE
######################################################################

cnn_site_decision_threshold = 0.5


######################################################################
# TINYCNN (LIGHTWEIGHT CNN-SITE)
######################################################################

tinycnn_method = "lightweight_cnn_site"
tinycnn_neighbor_method = "lightweight_cnn_site_neighbor"
tinycnn_neighbor_gaussian_method = "lightweight_cnn_site_neighbor_gaussian"

tinycnn_channels = (4, 8, 16)
tinycnn_learning_rate = 1e-4
tinycnn_epochs = 800
tinycnn_early_stopping_patience = 20
tinycnn_early_stopping_min_delta = 1e-5
tinycnn_batch_size = 64
tinycnn_eval_batch_size = 2048

tinycnn_neighbor_directions = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
tinycnn_neighbor_feature_count = len(tinycnn_neighbor_directions)


######################################################################
# THRESHOLD FILTERS: SQUARE AND GAUSSIAN
######################################################################

baselines_fit_size = 9


######################################################################
# EXPERIMENTAL DATASET: IMAGE KEYS
######################################################################

experimental_input_image_key = "image_2"
experimental_label_image_key = "image_1"


######################################################################
# EXPERIMENTAL DATASET: SITES AND LABELS
######################################################################

labeling_peak_min_distance = 18
labeling_peak_smooth_sigma = 2.0


######################################################################
# FIGURES: SHARED BY SEVERAL PLOT SCRIPTS
######################################################################

plot_default_data_dir = project_root / "dataset_3x3"
plot_default_model_dir = project_root / "models_3x3"
plot_default_output_dir = project_root / "output_3x3"
plot_model_dirs = (plot_default_model_dir,)

plot_baseline_method = "cnn_site"
plot_target_methods = (
    "lightweight_cnn_site",
    "lightweight_cnn_site_neighbor",
    "lightweight_cnn_site_neighbor_gaussian",
)

plot_delta_figsize = (8.8, 5.4)
plot_delta_dpi = 200
plot_delta_y_axis_limits = [-0.5, 0.5]


######################################################################
# FIGURES: COLORS SHARED BY THE EXPERIMENTAL AND NEIGHBOR FIGURES
######################################################################

figure_ink = "#0b0b0b"
figure_ink_secondary = "#52514e"
figure_grid = "#e2e1dc"
figure_site = "#2a78d6"
figure_site_dark = "#1c4f8f"
figure_fit = "#eb6834"
figure_neutral = "#8d8c86"


######################################################################
# FIGURE: MODEL COLORS
######################################################################

model_palette = (
    "#01468d",
    "#2a78d6",
    "#79adf2",
    "#a8a7a0",
    "#6b6a66",
    "#0b0b0b",
)

model_priority = (
    "cnn_site",
    "lightweight_cnn_site",
    "lightweight_cnn_site_neighbor",
    "lightweight_cnn_site_neighbor_gaussian",
    "mf_site",
    "mf_array",
    "gaussian",
    "square",
    "wei",
)
