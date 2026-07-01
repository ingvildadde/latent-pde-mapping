import os
import sys
import sys
import pandas as pd
from IPython.display import display, Markdown
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.evaluation_utils import get_training_times, get_average_inference_times_across_geometry_number, get_family, print_mean_std, get_predictions_for_all_PINNs
from src.utils.plot_utils import compute_times_boxplot, create_multiple_V_plots_3d
from src.utils.animation_utils import animate_multiple_V_plots_3d


def create_plots_3d():

    DOWNSAMPLE_FACTOR = 1 #30

    exp_paths_3d_single = [
        os.path.abspath("./outputs/CMAME_results/3D/PINNs/rot_x_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/3D/PINNs/rot_y_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/3D/PINNs/rot_z_reproduce_1")
        ]

    exp_paths_3d_deeponets = [
        os.path.abspath("./outputs/CMAME_results/3D/DeepONets/rot_x_42_sensors_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/3D/DeepONets/rot_y_42_sensors_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/3D/DeepONets/rot_z_42_sensors_reproduce_1"),
        ]

    exp_names_3d_single = [
        r"$\mathcal{H}_{rot}^{x}$",
        r"$\mathcal{H}_{rot}^{y}$",
        r"$\mathcal{H}_{rot}^{z}$"
    ]

    training_times = get_training_times(exp_paths_3d_single + exp_paths_3d_deeponets)
    inference_times = get_average_inference_times_across_geometry_number(exp_paths_3d_single + exp_paths_3d_deeponets)

    training_time_boxplot = compute_times_boxplot(training_times, y_label="Average Epoch Time [s]")
    training_time_boxplot.savefig("./outputs/figures/3D/3D_ani_mse_experiments_training_times_boxplot_reproduce_1_average_across_experiments_shape_10_4.pdf", bbox_inches='tight')

    all_internal_preds, all_external_preds = [], []

    for exp_path, exp_name, deeponet_path in zip(exp_paths_3d_single, exp_names_3d_single, exp_paths_3d_deeponets):
    # for exp_path, exp_name in zip(exp_paths_3d_single, exp_names_3d_single):
        internal_df, external_df, internal_preds, external_preds = get_predictions_for_all_PINNs(exp_path)

        internal_df_deeponet, external_df_deeponet, internal_preds_deeponet, external_preds_deeponet = get_predictions_for_all_PINNs(deeponet_path)

        if 'LPM-DeepONet' in internal_preds_deeponet:
            internal_preds_deeponet['LPM-DON'] = internal_preds_deeponet.pop('LPM-DeepONet')
        if 'LG-DeepONet' in internal_preds_deeponet:
            internal_preds_deeponet['LG-DON'] = internal_preds_deeponet.pop('LG-DeepONet')

        if 'LPM-DeepONet' in external_preds_deeponet:
            external_preds_deeponet['LPM-DON'] = external_preds_deeponet.pop('LPM-DeepONet')
        if 'LG-DeepONet' in external_preds_deeponet:
            external_preds_deeponet['LG-DON'] = external_preds_deeponet.pop('LG-DeepONet')

        if 'Affine-PINN' in internal_preds:
            internal_preds['PA-PINN'] = internal_preds.pop('Affine-PINN')
        if 'Affine-PINN' in external_preds:
            external_preds['PA-PINN'] = external_preds.pop('Affine-PINN')

        # Combine PINN and DeepONet predictions
        internal_preds.update(internal_preds_deeponet)
        external_preds.update(external_preds_deeponet)

        # Combined PINN and DeepONet dfs
        internal_df = pd.concat([internal_df, internal_df_deeponet], ignore_index=True)
        external_df = pd.concat([external_df, external_df_deeponet], ignore_index=True)

        all_internal_preds.append(internal_preds)
        all_external_preds.append(external_preds)

        display(Markdown(exp_name))
        print_mean_std(internal_df_deeponet, file="./outputs/RL2_errors/3D/anisotropic/center/internal_metrics_reproduce_1_deeponets_with_stats.txt", include_statistical_test=True)

        display(Markdown(exp_name[:-2] + r"*}$"))
        print_mean_std(external_df_deeponet, file="./outputs/RL2_errors/3D/anisotropic/center/external_metrics_reproduce_1_deeponets_with_stats.txt", include_statistical_test=True)

        print("\n")


    tqdm.write("Loading families for internal plots...")
    internal_3d_single_families = [get_family(folder_path, dim=3, use_external=False) for folder_path in exp_paths_3d_single]
    print("Finished loading families for internal plots.")

    tqdm.write("Loading families for external plots.", file=sys.stderr)
    external_3d_single_families = [get_family(folder_path, dim=3, use_external=True, downsample_factor=DOWNSAMPLE_FACTOR) for folder_path in exp_paths_3d_single]
    tqdm.write("Finished loading families for external plots.", file=sys.stderr)

    external_names_3d = [name[:-2] + '*}$' for name in exp_names_3d_single]

    print("Creating external 3D plots...")
    
    time_points = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for t in time_points:
        tqdm.write(f"Creating external 3D plot for time {t}...", file=sys.stderr)
        fig = create_multiple_V_plots_3d(all_external_preds, [t, t, t, t], [0, 3, 1], external_3d_single_families, use_all_domains=True, dir=['x', 'y', 'z'], family_names=external_names_3d, figsize=(24, 14), model_order=['LPM-PINN', 'LG-PINN', 'PA-PINN', 'Basic-PINN', 'LPM-DON', 'LG-DON'], downsample_factor=DOWNSAMPLE_FACTOR)
        fig.text(0.5, 0.24, ' ', transform=fig.transFigure)  # invisible bottom anchor
        
        fig.savefig(f"./outputs/figures/3D/3D_ani_mse_experiments_external_V_t_{t}_figsize_24_14_pinns_deeponets_reproduce_1_dpi1200.png", dpi=1200, bbox_inches='tight', pad_inches=0.05)

    # Create video
    # animation = animate_multiple_V_plots_3d(all_external_preds,
    #                                                  [0, 3, 1],
    #                                                  external_3d_single_families,
    #                                                  use_all_domains=True,
    #                                                  family_names=external_names_3d,
    #                                                  figsize=(24, 10),
    #                                                  model_order=['LPM-PINN', 'LG-PINN', 'PA-PINN', 'Basic-PINN', 'LPM-DON', 'LG-DON'],
    #                                                  downsample_factor=DOWNSAMPLE_FACTOR
    #                                                  )

    # animation.save("./outputs/videos/3D_ani_mse_experiments_external_V_figsize_24_10_pinns_deeponets_domains_0_3_1_time_interval_100_final.gif",
    #                writer='pillow',
    #                fps=animation._fps
    #                )




if __name__ == "__main__":
    create_plots_3d()