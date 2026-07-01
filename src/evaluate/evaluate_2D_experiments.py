import os
import sys
import pandas as pd
from IPython.display import display, Markdown
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.evaluation_utils import get_training_times, get_average_inference_times_across_geometry_number, get_family, print_mean_std, get_predictions_for_all_PINNs, get_boundary_terms
from src.utils.plot_utils import compute_times_boxplot, create_multiple_V_plots, plot_boundary_terms
from src.utils.animation_utils import animate_multiple_V_plots


def evaluate_2d_results():

    exp_paths_2d_single = [
        os.path.abspath("./outputs/CMAME_results/2D/PINNs/exp_rerun_with_epoch_times_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/2D/PINNs/shear_rerun_with_epoch_times_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/2D/PINNs/nonlin_rerun_with_epoch_times_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/2D/PINNs/rot_rerun_with_epoch_times_reproduce_1"),
    ]
    exp_paths_2d_deeponets = [
        os.path.abspath("./outputs/CMAME_results/2D/DeepONets/exp_14_sensors_rerun_with_epoch_times_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/2D/DeepONets/shear_14_sensors_rerun_with_epoch_times_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/2D/DeepONets/nonlin_14_sensors_rerun_with_epoch_times_reproduce_1"),
        os.path.abspath("./outputs/CMAME_results/2D/DeepONets/rot_14_sensors_rerun_with_epoch_times_reproduce_1"),
        ]

    exp_names_2d_single = [
        r"$\mathcal{G}_{exp}$",
        r"$\mathcal{G}_{shear}$",
        r"$\mathcal{G}_{nonlin}$",
        r"$\mathcal{G}_{rot}$"
    ]

    training_times = get_training_times(exp_paths_2d_single + exp_paths_2d_deeponets)
    inference_times = get_average_inference_times_across_geometry_number(exp_paths_2d_single + exp_paths_2d_deeponets)

    training_boxplot = compute_times_boxplot(training_times, y_label="Average Epoch Time [s]")
    training_boxplot.savefig("./outputs/figures/2D/2D_ani_mse_experiments_training_times_boxplot_reproduce_1_average_across_experiments_shape_10_6.pdf", bbox_inches='tight')

    all_internal_preds, all_external_preds = [], []

    for exp_path, exp_name, deeponet_path in zip(exp_paths_2d_single, exp_names_2d_single, exp_paths_2d_deeponets):
    # for exp_path, exp_name in zip(exp_paths_2d_single, exp_names_2d_single):
        tqdm.write(f"Loading predictions from: {exp_path}", file=sys.stderr)
        internal_df, external_df, internal_preds, external_preds = get_predictions_for_all_PINNs(exp_path)

        tqdm.write(f"Loading predictions from: {deeponet_path}", file=sys.stderr)
        internal_df_deeponet, external_df_deeponet, internal_preds_deeponet, external_preds_deeponet = get_predictions_for_all_PINNs(deeponet_path)

        # Replace LPM-DeepONet and LG-DeepONet key with LPM-DON and LG-DON for consistency
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

        tqdm.write(f"Combined predictions contain models: {list(internal_preds.keys())}", file=sys.stderr)

        all_internal_preds.append(internal_preds)
        all_external_preds.append(external_preds)

        display(Markdown(exp_name))
        print_mean_std(internal_df_deeponet, file="./outputs/RL2_errors/2D/anisotropic/center/internal_metrics_pca_reproduce_1_deeponets_with_stats.txt", include_statistical_test=True)

        display(Markdown(exp_name[:-1] + r"*}$"))
        print_mean_std(external_df_deeponet, file="./outputs/RL2_errors/2D/anisotropic/center/external_metrics_pca_reproduce_1_deeponets_with_stats.txt", include_statistical_test=True)

        print("\n")

    # fig = plot_boundary_terms_2D_exp(exp_paths_2d_single, exp_names_2d_single)
    # fig.savefig("./outputs/figures/2D/2D_ani_mse_experiments_boundary_terms_affine_pinn_reproduce_1_with_std_labels_std.png", bbox_inches='tight') #, pad_inches=0.5)
    # fig.savefig("./outputs/figures/2D/2D_ani_mse_experiments_boundary_terms_affine_pinn_reproduce_1_with_std_labels_std.pdf", bbox_inches='tight') #, pad_inches=0.5)

    tqdm.write("Loading families for internal plots...")
    internal_2d_single_families = [get_family(folder_path, dim=2, use_external=False) for folder_path in exp_paths_2d_single]
    print("Finished loading families for internal plots.")

    print("Loading families for external plots...")
    external_2d_single_families = [get_family(folder_path, dim=2, use_external=True) for folder_path in exp_paths_2d_single]
    print("Finished loading families for external plots.")

    external_names_2d = [name[:-1] + '^{*}$' for name in exp_names_2d_single]

    print("Creating external 2D plots...")
    # fig = create_multiple_V_plots(all_internal_preds, [50, 50, 50, 50], [0, 0, 2, 0], internal_2d_single_families, use_all_domains=False, family_names=exp_names_2d_single, figsize=(24, 14), model_order=['LPM-PINN', 'LG-PINN', 'PA-PINN', 'Basic-PINN', 'LPM-DON', 'LG-DON'], padding=0.09)
    fig = create_multiple_V_plots(all_external_preds, [50, 50, 50, 50], [0, 0, 0, 1], external_2d_single_families, use_all_domains=True, family_names=external_names_2d, figsize=(24, 14), model_order=['LPM-PINN', 'LG-PINN', 'PA-PINN', 'Basic-PINN', 'LPM-DON', 'LG-DON'], padding=0.09)

    # fig.savefig("./outputs/figures/2D/2D_ani_mse_experiments_internal_V_t_50_figsize_24_14_pinns_deeponets_domains_0_0_0_1_reproduce_1.png", bbox_inches='tight') #, pad_inches=0.5)
    # fig.savefig("./outputs/figures/2D/2D_ani_mse_experiments_internal_V_t_50_figsize_24_14_pinns_deeponets_domains_0_0_0_1_reproduce_1.svg", bbox_inches='tight') #, pad_inches=0.5)
    # fig.savefig("./outputs/figures/2D_ani_mse_experiments_external_V_t_50_figsize_24_14_pinns_deeponets.svg", bbox_inches='tight') #, pad_inches=0.5)


    # animation = animate_multiple_V_plots(all_internal_preds,
    #                                     [0, 0, 0, 1],
    #                                     internal_2d_single_families,
    #                                     use_all_domains=False,
    #                                     family_names=exp_names_2d_single,
    #                                     figsize=(24, 14),
    #                                     model_order=['LPM-PINN', 'LG-PINN', 'PA-PINN', 'Basic-PINN', 'LPM-DON', 'LG-DON'],
    #                                     padding=0.14)

    # animation.save("./outputs/videos/2D_ani_mse_experiments_internal_V_figsize_24_14_pinns_deeponets_domains_0_0_0_1_time_interval_100.gif", writer='pillow', fps=animation._fps)


        
            

if __name__ == "__main__":
    evaluate_2d_results()