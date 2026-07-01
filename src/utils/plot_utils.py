import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import torch
import numpy as np
import pathlib
from matplotlib.path import Path
import matplotlib.tri as tri
import seaborn as sns
from tqdm import tqdm
import sys
import os

from src.data.domain_family import DomainFamily
from src.data.domain import Domain
from src.training.loggers import TrainLogger, ValLogger
from src.utils.evaluation_utils import activation_times, repolarization_times, get_boundary_terms

# Set matplotlib font size
font_size = 24
ticks_label_size = 20
plt.rcParams.update({'font.size': font_size})


def create_multiple_V_plots(predictions: list, time: list, indices: list, families: list, use_all_domains, family_names: list, figsize=(18, 18), padding=0.08, model_order: list = ['LPM-PINN', 'PA-PINN', 'Basic-PINN']):
    
    plt.rcParams.update({'font.size': font_size})

    fig, axes = plt.subplots(len(predictions), len(predictions[0].keys())+1, figsize=figsize, constrained_layout=True, sharey='row')
    
    for row in range(len(predictions)):

        domains = families[row].domains if use_all_domains else families[row].test_domains
        selected_domain = domains[indices[row]]

        first_exp = list(predictions[row].keys())[0]
        gt = predictions[row][first_exp]['output'][str(indices[row])]['gt'][:, time[row]]

        min_at, max_at = -80, 20

        ax = axes[row][0] if len(predictions) > 1 else axes[0]
        _ = smooth_plot_tri(ax, selected_domain.x[:, 0, 0], selected_domain.x[:, 0, 1], selected_domain.x_bc, gt, vmin=min_at, vmax=max_at, padding=padding)

        ax.set_ylabel('y [mm]')
        ax.tick_params(axis='x', labelsize=ticks_label_size)
        ax.tick_params(axis='y', labelsize=ticks_label_size)
        if row == 0:
            ax.set_title("FEM")
        if row == len(predictions)-1:
            ax.set_xlabel('x [mm]')

        # Add time as text to the left of the row
        # ax.text(-0.35, 0.5, family_names[row] + rf'$, t = {time[row]}$ ms', va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=font_size)
        ax.text(-0.35, 0.5, family_names[row], va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=32)

        predictions_row = dict(sorted(predictions[row].items(), key=lambda item: len(item[0]), reverse=True))

        for j, exp in enumerate(model_order):
            
            try:
                print(f"Plotting {exp} for family {family_names[row]}")
                v_preds = predictions_row[exp]['output'][str(indices[row])]['preds'][:, time[row]]

                ax = axes[row][j+1] if len(predictions) > 1 else axes[j+1]
                _ = smooth_plot_tri(ax, selected_domain.x[:, 0, 0], selected_domain.x[:, 0, 1], selected_domain.x_bc, v_preds, vmin=min_at, vmax=max_at, padding=padding)
                
                # Set xticks to fontsize 18
                ax.tick_params(axis='x', labelsize=ticks_label_size)
                ax.tick_params(axis='y', labelsize=ticks_label_size)
                if row == 0:
                    ax.set_title(str(exp))
                if row == len(predictions)-1:
                    ax.set_xlabel('x [mm]')
            except Exception as e:
                print(f"Exception while plotting {exp} for family {family_names[row]}: {e}")
                print(f"v_preds.shape: {v_preds.shape}")
                print(f"selected_domain.x.shape: {selected_domain.x.shape}")
                continue

        norm = plt.Normalize(vmin=min_at, vmax=max_at)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('V [mV]')
        cbar.ax.tick_params(labelsize=ticks_label_size)
        
    return fig



def create_multiple_V_plots_3d(predictions: list, time: list, indices: list, families: list, use_all_domains: bool, dir: list, family_names: list, figsize=(18, 18), model_order=['LPM-PINN', 'PA-PINN', 'Basic-PINN'], downsample_factor=1):

    plt.rcParams.update({'font.size': font_size})

    n_rows = len(predictions)
    n_cols = len(model_order) + 1  # +1 for FEM column
    
    fig = plt.figure(figsize=figsize)
    plot_idx = 1
    tqdm.write(f"Number of predictions: {len(predictions)}", file=sys.stderr)
    for row in range(len(families)):
        domains = families[row].domains if use_all_domains else families[row].test_domains
        selected_domain = domains[indices[row]]

        first_exp = list(predictions[row].keys())[0]
        gt = predictions[row][first_exp]['output'][str(indices[row])]['gt'][::downsample_factor, time[row]]

        min_at, max_at = -80, 20
        
        ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
        _ = ax.scatter(selected_domain.x[:, 0, 0], selected_domain.x[:, 0, 1], selected_domain.x[:, 0, 2], c=gt, vmin=min_at, vmax=max_at, edgecolors='none')
        plot_idx += 1
        ax.view_init(elev=20, azim=60)

        ax.set_zlabel('z [mm]', labelpad=10)
        ax.set_xlabel('x [mm]', labelpad=10)
        ax.set_ylabel('y [mm]', labelpad=10)
        ax.tick_params(axis='x', labelsize=ticks_label_size)
        ax.tick_params(axis='y', labelsize=ticks_label_size)
        ax.tick_params(axis='z', labelsize=ticks_label_size)
        if row == 0:
            ax.set_title("FEM")

        # Add family name as text to the left of the row
        ax.text2D(-0.37, 0.55, family_names[row], va='center', ha='right', rotation=90, transform=ax.transAxes, fontsize=font_size)

        predictions_row = dict(sorted(predictions[row].items(), key=lambda item: len(item[0]), reverse=True))

        for _, exp in enumerate(model_order):

            v_preds = predictions_row[exp]['output'][str(indices[row])]['preds'][::downsample_factor, time[row]]
            
            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
            _ = ax.scatter(selected_domain.x[:, 0, 0], selected_domain.x[:, 0, 1], selected_domain.x[:, 0, 2], c=v_preds, vmin=min_at, vmax=max_at, edgecolors='none')

            plot_idx += 1
            ax.view_init(elev=20, azim=60)
            ax.set_zticklabels([])
            
            # Set xticks to fontsize 18
            ax.tick_params(axis='x', labelsize=ticks_label_size)
            ax.tick_params(axis='y', labelsize=ticks_label_size)
            ax.tick_params(axis='z', labelsize=ticks_label_size)
            if row == 0:
                ax.set_title(str(exp))

    # Add colorbar once at the end, attached to the figure
    norm = plt.Normalize(vmin=min_at, vmax=max_at)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    
    fig.subplots_adjust(hspace=-0.5, bottom=0.15, top=0.95, left=0.05, right=0.86)
    
    cbar_ax = fig.add_axes([0.88, 0.35, 0.02, 0.40])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical') #, fraction=0.02, pad=0.08)
    cbar.set_label('V [mV]')

    return fig


def smooth_plot_tri(ax, x, y, boundary_points, values, vmin, vmax, cmap='viridis', padding=0.08):
    triang = tri.Triangulation(x, y)

    boundary_points_x = boundary_points[:, 0, 0]  # x-coordinates of boundary points
    boundary_points_y = boundary_points[:, 0, 1]  # y-coordinates of boundary points

    boundary_path = Path(np.column_stack((boundary_points_x, boundary_points_y)))

    # Compute triangle centroids
    xc = x[triang.triangles].mean(axis=1)
    yc = y[triang.triangles].mean(axis=1)

    # Mask triangles outside boundary
    mask = ~boundary_path.contains_points(np.c_[xc, yc])
    triang.set_mask(mask)
    
    levels = np.linspace(vmin, vmax, 101)
    tcf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')

    x_min = int(x.min())
    x_max = int(x.max())
    y_min = int(y.min())
    y_max = int(y.max())

    # Preserve whitespace by setting axis limits with padding
    x_range = x_max - x_min
    y_range = y_max - y_min

    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    return tcf


def plot_average_boundary_movement(boundary_terms: list[dict], family_names: list[str]):

    fig = plt.figure(figsize=(8, 6))

    for i in range(len(boundary_terms)):
        plt.bar(family_names[i], boundary_terms[i][0]["mean_Vn"], alpha=0.7, color='teal')
    
    plt.ylabel("Average boundary movement")

    return fig


def plot_boundary_terms(boundary_terms: list[dict], family_names: list[str], show_std: bool = False):

    fig, axes = plt.subplots(1, len(boundary_terms), figsize=(20,5), constrained_layout=True, sharey='row')

    for col in range(len(boundary_terms)):

        epochs = list(boundary_terms[col].keys())
        x_epochs = range(0, len(epochs)*10, 10)

        ax = axes[col]
        ax.plot(x_epochs, [10**boundary_terms[col][epoch]['mean_B'] for epoch in epochs], label=r'Mean missing shape gradient information', linestyle='-', c='steelblue')
        ax.plot(x_epochs, [10**boundary_terms[col][epoch]['mean_V'] for epoch in epochs], label=r'Mean shape gradient in $\mathcal{L}_{phys}^{conv}$', linestyle='-', c='darkorange')

        if show_std:
            ax.fill_between(x_epochs, [10**(boundary_terms[col][epoch]['mean_B'] - boundary_terms[col][epoch]['std_B']) for epoch in epochs],
                            [10**(boundary_terms[col][epoch]['mean_B'] + boundary_terms[col][epoch]['std_B']) for epoch in epochs], alpha=0.2, color='steelblue', label=r'$\pm$ 1 std')
            ax.fill_between(x_epochs, [10**(boundary_terms[col][epoch]['mean_V'] - boundary_terms[col][epoch]['std_V']) for epoch in epochs],
                            [10**(boundary_terms[col][epoch]['mean_V'] + boundary_terms[col][epoch]['std_V']) for epoch in epochs], alpha=0.2, color='darkorange', label=r'$\pm$ 1 std')

        ax.text(0.88, 0.9, family_names[col], transform=ax.transAxes, fontsize=font_size, ha='center')

        ax.tick_params(axis='x', labelsize=ticks_label_size)
        ax.tick_params(axis='y', labelsize=ticks_label_size)

        if col == 0:
            ax.set_ylabel('Mean magnitude')
    
        ax.set_xlabel("Epochs")

    ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.3), #(0.5, 1.2),   # adjust vertical space
            ncol= 2, #len(labels),             # put all items in one row
            fontsize=font_size)

    return fig


def plot_boundary_terms_2D_exp(exp_paths, exp_names):

    processed_boundary_terms = [get_boundary_terms(os.path.join(path, "boundary_terms")) for path in exp_paths]

    fig = plot_boundary_terms(processed_boundary_terms, exp_names, show_std=True)

    return fig



def plot_V_1D(V_true: torch.Tensor,
              time: torch.Tensor,
              index_position: int,
              V_predicted: torch.Tensor = None,
              scatter_size: int = 80,
              predicted_color: str = 'orange',
              x_label: str = 't [ms]',
              y_label: str = 'V [AU]'):
    """Plots the potential (V) at a given position (index_position) over a time sequence (time)"""
    
    V_true = V_true[index_position, :].squeeze()
    plt.scatter(time, V_true, s=scatter_size, label="Ground truth")

    if V_predicted is not None:
        V_predicted = V_predicted[index_position, :].squeeze()
        plt.plot(time, V_predicted, c=predicted_color, label="Predicted")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


def plot_V_over_time(predictions: dict, family: DomainFamily, domain_index: int, index_position: int, models: list[str]):
    """
    Plots V at a specific (x,y) location over time for a given domain and model.
    """
    domain = family.domains[domain_index]
    times = domain.tau[index_position, :].cpu().numpy()*12.9 # Convert to ms

    fig = plt.figure(figsize=(6, 6))
    gt_values = predictions[models[0]]['output'][str(domain_index)]['gt'][index_position, :]
    plt.plot(times, gt_values, label=f"FEM", linestyle='--', c='black', linewidth=2)
    
    for model in models:
        V_values = predictions[model]['output'][str(domain_index)]['preds'][index_position, :]
        plt.plot(times, V_values, label=f"{model}")

    plt.xlabel("t [ms]", fontsize=18)
    plt.ylabel("V [mV]", fontsize=18)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    
    # get legend and handles
    handles, labels = plt.gca().get_legend_handles_labels()

    fig.legend(handles, labels,
            loc='upper center',
            bbox_to_anchor=(0.51, 1.1),   # adjust vertical space
            ncol=2,            # put all items in one row
            fontsize=18)
    
    return fig


def plot_random_predictions_1D(targets, samples, predictions, time, nrows, ncols):
    """Plots multiple predictions at given spatial locations (samples)"""
    plt.figure(figsize=(12, 12))

    nrows = nrows
    ncols = ncols

    targets = targets.detach().cpu().numpy()
    samples = samples.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()

    for i, sample in enumerate(samples):
        plt.subplot(nrows, ncols, i+1)
        plot_V_1D(V_true=targets, time=time, index_position=i, V_predicted=predictions, scatter_size=10, predicted_color='darkorange', y_label='V [A.U.]', x_label=r'$\tau$ [T.U.]')
        plt.title(f"(x, y) = ({(sample[0, 0]):.2f}, {sample[0, 1]:.2f})")


def plot_loss_curves(results: dict[str, list[float]] | TrainLogger | ValLogger):
    """Plots training curves of a results dictionary"""
    
    if type(results) == TrainLogger or type(results) == ValLogger:
        results = vars(results)

    epochs = range(len(list(results.values())[0]))
    
    fig = plt.figure(figsize=(10, 6))
    for key in results.keys():
        plt.plot(epochs, results[key], label=str(key))

    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.legend()

    return fig


def plot_worst_predictions(rmse_dict: dict, domains: list[Domain], predictions: dict, mapped_mode: bool, show_max: bool = True, show_rnd_idx: bool = False):

    for i, key in enumerate(rmse_dict.keys()):

        rmse_idx = np.argmax(rmse_dict[key]) if show_max else np.argmin(rmse_dict[key])

        if show_rnd_idx:
            rmse_idx = np.random.randint(0, len(rmse_dict[key]) - 1)

        spatial_location = int(rmse_idx)

        Vm_true = (domains[i].V[spatial_location, :])
        Vm_true = Vm_true.squeeze()

        t = domains[i].tau[0, :]
        x = domains[i].x_ref if mapped_mode else domains[i].x

        plt.figure(figsize=(8, 8))
        plt.scatter(t.cpu(), Vm_true.cpu(), s=10, label='Ground truth')
        plt.scatter(t.cpu(), predictions[key]["preds"][spatial_location, :].cpu(), s=10, c="darkorange", label="Predicted")
        plt.legend()
        plt.title(f"Domain: {key}, Position: (x, y) = ({x[spatial_location, 0, 0]:.2f}, {x[spatial_location, 0, 1]:.2f})")
        plt.xlabel(r'$\tau$ [T.U.]')
        plt.ylabel('V [A.U.]')


def create_AT_plots(predictions:dict, indices: list, family, use_all_domains):

    domains = family.domains if use_all_domains else family.test_domains
    selected_domains = [domains[idx] for idx in indices]

    fig, axes = plt.subplots(len(selected_domains), 3, figsize=(18, 18), constrained_layout=True, sharey='row')

    for row, (i, domain) in enumerate(zip(indices, selected_domains)):

        gt = predictions['affine'][str(i)]['gt']
        v_preds = predictions['affine'][str(i)]['preds']
        m_preds = predictions['refaffine'][str(i)]['preds']

        thr = 0

        gt_at = activation_times(gt, threshold=thr)
        v_at = activation_times(v_preds, threshold=thr)
        m_at = activation_times(m_preds, threshold=thr)


        sc1 = axes[row][0].scatter(domain.x[:, 0, 0], domain.x[:, 0, 1], c=gt_at)
        axes[row][0].set_ylabel('y [mm]')
        axes[row][0].set_xlabel('x [mm]')
        axes[0][0].set_title("FEM")
        fig.colorbar(sc1)

        sc2 = axes[row][1].scatter(domain.x[:, 0, 0], domain.x[:, 0, 1], c=v_at)
        axes[row][1].set_xlabel('x [mm]')
        axes[0][1].set_title("PA-PINN")
        fig.colorbar(sc2)

        sc3 = axes[row][2].scatter(domain.x[:, 0, 0], domain.x[:, 0, 1], c=m_at)
        axes[row][2].set_xlabel('x [mm]')
        axes[0][2].set_title("RefAffine-PINN")
        fig.colorbar(sc3, label="Activation times [ms]")

    return fig



def create_RT_plots(predictions:dict, indices: list, family, use_all_domains):

    domains = family.domains if use_all_domains else family.test_domains
    selected_domains = [domains[idx] for idx in indices]

    print(len(selected_domains))

    fig, axes = plt.subplots(len(selected_domains), 3, figsize=(18, 18), constrained_layout=True, sharey='row')

    for row, (i, domain) in enumerate(zip(indices, selected_domains)):

        gt = predictions['affine'][str(i)]['gt']
        v_preds = predictions['affine'][str(i)]['preds']
        m_preds = predictions['refaffine'][str(i)]['preds']

        thr = -70

        gt_at = repolarization_times(gt, threshold=thr)
        v_at = repolarization_times(v_preds, threshold=thr)
        m_at = repolarization_times(m_preds, threshold=thr)


        sc1 = axes[row][0].scatter(domain.x[:, 0, 0], domain.x[:, 0, 1], c=gt_at)
        axes[row][0].set_ylabel('y [mm]')
        axes[row][0].set_xlabel('x [mm]')
        axes[0][0].set_title("FEM")
        fig.colorbar(sc1)

        sc2 = axes[row][1].scatter(domain.x[:, 0, 0], domain.x[:, 0, 1], c=v_at)
        axes[row][1].set_xlabel('x [mm]')
        axes[0][1].set_title("PA-PINN")
        fig.colorbar(sc2)

        sc3 = axes[row][2].scatter(domain.x[:, 0, 0], domain.x[:, 0, 1], c=m_at)
        axes[row][2].set_xlabel('x [mm]')
        axes[0][2].set_title("RefAffine-PINN")
        fig.colorbar(sc3, label="Activation times [ms]")

    return fig


def plot_theta_distribution_3D(train_params, test_params, reference_axis, extr_params=None):

    train_thetas = []
    for i in range(train_params.shape[0]):
        print(f"Train params {i}: {train_params[i]}")
        R_train = train_params[i].reshape(3, 3)
        print(R_train)
        train_theta = rotation_angle_from_R(R_train, reference_axis)
        train_thetas.append(train_theta)
    
    test_thetas = []
    for i in range(test_params.shape[0]):
        print(f"Test params {i}: {test_params[i]}")
        R_test = test_params[i].reshape(3, 3)
        test_theta = rotation_angle_from_R(R_test, reference_axis)
        test_thetas.append(test_theta)

    print(train_thetas)
    print(test_thetas)
    fig = plt.figure(figsize=(12, 10))
    sns.kdeplot(train_thetas, label=r'Train', fill=True, alpha=0.5, clip=(min(train_thetas), max(train_thetas)))
    sns.kdeplot(test_thetas, label=r'Interpolation Test', fill=True, alpha=0.5, clip=(min(test_thetas), max(test_thetas)))
    if extr_params is not None:

        extr_thetas = []
        for i in range(extr_params.shape[0]):
            print(f"Extrapolation params {i}: {extr_params[i]}")
            R_extr = extr_params[i].reshape(3, 3)
            extr_theta = rotation_angle_from_R(R_extr, reference_axis)
            extr_thetas.append(extr_theta)
        
        print(extr_thetas)

        positive_extr_thetas = [theta for theta in extr_thetas if theta > 0]
        negative_extr_thetas = [theta for theta in extr_thetas if theta < 0]
        sns.kdeplot(extr_thetas, label=r'Extrapolation Test', fill=True, alpha=0.5, clip=(min(positive_extr_thetas), max(positive_extr_thetas)), color='green')
        sns.kdeplot(extr_thetas, fill=True, alpha=0.5, clip=(min(negative_extr_thetas), max(negative_extr_thetas)), color='green')
    plt.xlabel(r'$\theta$ (degrees)')
    plt.legend()
    plt.show()
    return fig, train_thetas, test_thetas, extr_thetas


def rotation_angle_from_R(R, reference_axis):
    # Assuming R is a 3x3 rotation matrix, the angle of rotation can be computed as:
    theta = np.arccos((np.trace(R) - 1) / 2)
    rotation_axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    rotation_axis = rotation_axis / (2*np.sin(theta))
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    sign = np.sign(np.dot(rotation_axis, reference_axis))

    return sign * theta  * 180 / np.pi



###########################
######## Boxplots #########
###########################

def boxplot(dataframe,
                y_range = None,
                fontsize = 12,
                labelsize = 12, 
                title = '',
                colnames = [],
                colors = [],
                filename = '',
                save=False,
                xtick_rotation = 0):
    
    fig, ax = plt.subplots(figsize=(11,8))

    plt.grid(color='k', alpha=0.2, fillstyle='left', axis='y', linestyle='solid')
    
    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rcParams['font.family'] = "serif"
    matplotlib.rcParams.update({'xtick.labelsize': labelsize})

    df_names = dataframe.keys()
    
    for i in range(len(colnames)):
        color = colors[i] if len(colors) == len(colnames) else colors[0]
        filtered_data = dataframe[df_names[i]][~np.isnan(dataframe[df_names[i]])]
        parts = ax.violinplot(filtered_data, positions=[i], showmeans=False, showmedians=False,showextrema=False)
        for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.4)

                ax.boxplot(filtered_data, positions=[i], showmeans=False, meanline=False, 
                    patch_artist=True, boxprops=dict(facecolor=color, color='k'), 
                    medianprops=dict(color='k'),meanprops=dict(color='k'))
        
    ax.set_xticklabels(colnames, rotation=xtick_rotation)
    
    plt.xlabel(None)
    plt.ylabel(r'RMSE')
    plt.title(title)
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    plt.show()

    if save:
        root_path = pathlib.Path(filename.split("figures")[0])
        fig_dir = pathlib.Path.joinpath(root_path, "figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename)
        print(f"Saved figure to {filename}")


def grouped_boxplot(dataframe,
                    selected_family = None,
                    y_range = None,
                    fontsize = 12,
                    labelsize = 12, 
                    title = '',
                    colnames = [],
                    colors = [],
                    filename = '',
                    save=False,
                    save_path = '',
                    include_violin = True,
                    figsize = (10,6),
                    all_domains = False,
                    x_labels = []):
    
    fig, ax = plt.subplots(figsize=figsize)


    plt.grid(color='k', alpha=0.2, fillstyle='left', axis='y', linestyle='solid')
    
    matplotlib.rcParams.update({'font.size': fontsize})
    matplotlib.rcParams.update({'xtick.labelsize': labelsize})

    # Get unique test domains, models and model types
    test_domains = dataframe["test_domain"].unique()
    families = sorted(dataframe["family"].unique())
    models = dataframe["model"].unique()

    model_colors = {"Basic-PINN":'#9ecae1', "RefAffine-PINN": '#31a354', "PA-PINN": "#FFA500"}
    
    # Set width for individual boxplots and spacing for groups
    box_width = 0.22
    spacing_factor = 1.5
    positions = np.arange(len(families))*spacing_factor  # X-axis positions for test domains

    for i, model in enumerate(models):

            if all_domains:
                data = []
                for family in families:
                    data_values = dataframe[(dataframe["model"] == model) & (dataframe["family"] == family)]["metric_value"].dropna().values
                    data.append(data_values)
            else:
                data = [dataframe[(dataframe["test_domain"] == domain) & (dataframe["model"] == model) & (dataframe["family"] == selected_family)]["metric_value"].dropna().values for domain in test_domains]
            
            offset = (i - len(models)/2) * box_width + box_width/2
            pos = positions + offset

            color = colors[i] if len(colors) == len(colnames) else colors[0]
            if not include_violin:
                 ax.boxplot(data, positions=pos, widths=0.2, showmeans=True, meanline=True, 
                        patch_artist=True, boxprops=dict(facecolor=color, color='k'), 
                        medianprops=dict(color='none'), meanprops=dict(color='k', linestyle='-'), label=type)
            else:
                parts = ax.violinplot(data, pos, showmeans=False, showmedians=False, showextrema=False)
                for pc in parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.4)

                        ax.boxplot(data, positions=pos, widths=0.2, showmeans=False, meanline=False, 
                            patch_artist=True, boxprops=dict(facecolor=color, color='k'), 
                            medianprops=dict(color='k'), meanprops=dict(color='k'), label=type)
    
    ax.set_xticks(positions)
    ax.set_xticklabels([r"$\mathcal{" + str(label)[0] + "}" + r"_{" + str(label)[1] + "}$" for label in families] if x_labels == [] else [label for label in x_labels])
    # ax.set_xticklabels([str(label) for label in families])
    ax.tick_params(axis='x', labelsize=labelsize) 
    plt.xlabel('Domain families')
    plt.ylabel(r'$\varepsilon_{L2}$')
    
    # Create legend with both color and hatching patterns
    legend_patches = []
    # for model, color in model_colors.items():
    for model, color in zip(colnames, colors):
            patch = mpatches.Patch(facecolor=color, edgecolor="k", label=f"{model}") # - {mode}")
            legend_patches.append(patch)
            
    ax.legend(
        handles=legend_patches
    )
    
    fig_title = title if selected_family is None else selected_family
    plt.title(f"{fig_title}")
    plt.tight_layout()
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])
    plt.show()

    if save:
        fig_dir = pathlib.Path(save_path)
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(pathlib.Path.joinpath(fig_dir, filename))
        print(f"Saved figure to {filename}")


def compute_times_boxplot(computational_times, y_label):
    
    colors = ['#023e8a', '#0077b6', '#0096c7', '#00b4d8', 'darkgreen', 'green']

    fig = plt.figure(figsize=(10, 4))
    bplot = plt.boxplot([computational_times['LPM-PINN'], computational_times['LG-PINN'], computational_times['PA-PINN'], computational_times['Basic-PINN'], computational_times['LPM-DeepONet'], computational_times['LG-DeepONet']],
                labels=['LPM-PINN', 'LG-PINN', 'PA-PINN', 'Basic-PINN', 'LPM-DON', 'LG-DON'],
                patch_artist=True,
                medianprops=dict(color='black', linewidth=1)
    )
    
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel(y_label)
    plt.xticks(rotation=15)

    return fig
