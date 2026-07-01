import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import torch
import os
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.animation as animation
import sys
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from pathlib import Path
from tqdm import tqdm

from src.data.domain import Domain
from src.utils.evaluation_utils import get_family
from src.utils.plot_utils import smooth_plot_tri

try:
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    times_new_roman = fm.FontProperties(fname=font_path)

    # Register it with matplotlib
    fm.fontManager.addfont(font_path)

    # Now matplotlib knows about the name
    mpl.rcParams['font.family'] = times_new_roman.get_name()

    print("Using:", mpl.rcParams['font.family'])
except:
    print("Using standard font")

# Set matplotlib font size
font_size = 20
ticks_label_size = 18
plt.rcParams.update({'font.size': font_size})



def create_vtu_files(dataset: Domain, predictions: torch.Tensor, output_folder_path: Path, use_ref_domain: bool):

    output_folder_path.mkdir(parents=True, exist_ok=True)
    x = dataset.x_ref if use_ref_domain else dataset.x

    vertices = np.hstack([x[:, 0, :], np.zeros((x.shape[0], 1))])
    elements = np.hstack([[3] + list(triangle) for triangle in dataset.x_elems])

    poly_data = pv.PolyData(vertices, elements)
    mesh = pv.UnstructuredGrid(poly_data)

    mesh['potential'] = predictions[:, 0]

    for i in range(predictions.shape[1]):
        mesh['potential'] = predictions[:, i]
        mesh.point_data["Time"] = np.full(mesh.n_points, i)

        vtu_path = Path.joinpath(output_folder_path, f"time_step_{i}.vtu")
        mesh.save(vtu_path)


def create_vtu_files_3D(dataset: Domain, predictions: torch.Tensor, output_folder_path: Path | str, use_ref_domain: bool):

    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    x = dataset.x_ref if use_ref_domain else dataset.x

    vertices = x[:, 0, :]

    cells = []
    cell_type = pv.CellType.TETRA

    for elem in dataset.x_elems:
        assert len(elem) == 4, f"Each element must have {4} points."
        cells.append([4] + list(elem))

    cells = np.hstack(cells)
    celltypes = np.full(len(dataset.x_elems), cell_type, dtype=np.uint8)

    vertices = np.array(vertices, dtype=np.float32)

    mesh = pv.UnstructuredGrid(cells, celltypes, vertices)

    for i in range(predictions.shape[1]):
        mesh['potential'] = predictions[:, i]
        mesh.point_data["Time"] = np.full(mesh.n_points, i)

        vtu_path = Path.joinpath(output_folder_path, f"time_step_{i}.vtu")
        mesh.save(vtu_path)



def create_multiple_videos(predictions: dict, experiment: str, experiment_folder_path: Path, dim: int, use_ref_domain = False, use_external: bool = False, additional_name_tag = ""):

    experiment_folder = os.path.join(experiment_folder_path, experiment)
    family = get_family(experiment_folder_path, use_external=use_external, specific_model=experiment_folder, dim=dim)
    domains = family.test_domains

    for i in range(len(domains)):
        if dim == 2:
            create_vtu_files(dataset=domains[i],
                            predictions=predictions[experiment]['output'][str(i)]["gt"],
                            output_folder_path=Path.joinpath(Path(experiment_folder), f"simulations/{i}_gt" + additional_name_tag),
                            use_ref_domain=use_ref_domain)
            
            create_vtu_files(dataset=domains[i],
                            predictions=predictions[experiment]['output'][str(i)]["preds"],
                            output_folder_path=Path.joinpath(Path(experiment_folder), f"simulations/{i}_preds" + additional_name_tag),
                            use_ref_domain=use_ref_domain)
        elif dim == 3:
            create_vtu_files_3D(dataset=domains[i],
                                predictions=predictions[experiment]['output'][str(i)]["gt"],
                                output_folder_path=Path.joinpath(Path(experiment_folder), f"simulations/{i}_gt" + additional_name_tag),
                                use_ref_domain=use_ref_domain)
            
            create_vtu_files_3D(dataset=domains[i],
                                predictions=predictions[experiment]['output'][str(i)]["preds"],
                                output_folder_path=Path.joinpath(Path(experiment_folder), f"simulations/{i}_preds" + additional_name_tag),
                                use_ref_domain=use_ref_domain)
        else:
            raise ValueError(f"Unsupported dimension: {dim}. Only 2D and 3D are supported.")
    



def create_vm_animation(input_data: np.ndarray,
                        target_data: np.ndarray,
                        grid_size: int = 100j,
                        number_of_frames: int = None,
                        colorbar_label: str = "V [mV]",
                        min_colorbar_value: int = -80,
                        max_colorbar_value: int = 20,
                        animation_speed: int = 100,
                        repeat_animation: bool = False,
                        contour_levels: int = 20,
                        interpolation_method: str = "cubic") -> FuncAnimation:
    """
    Creates an animation of the electrical potential (V) using matplotlib
    
    Arguments:
        - input_data: numpy ndarray containing spatial locations and time steps
                      must have shape: [number_of_spatial_locations, number_of_time_steps, 3]
        - target_data: numpy ndarray containing the electrical potential data
                       must have shape: [number_of_spatial_locations, number_of_time_steps, 1]
    """

    x_coords = input_data[:, :, 0]
    y_coords = input_data[:, :, 1]
    time_data = input_data[0, :, 2]
    v_data = target_data[:, :, 0]

    time_step = time_data[1] - time_data[0]
    n_frames = number_of_frames if number_of_frames is not None else len(time_data)

    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Define a grid for the contour plot
    grid_x, grid_y = np.mgrid[
        np.min(x_coords):np.max(x_coords):grid_size,
        np.min(y_coords):np.max(y_coords):grid_size
    ]

    # Set axis limits
    ax.clear()
    ax.set_xlim(np.min(x_coords), np.max(x_coords))
    ax.set_ylim(np.min(y_coords), np.max(y_coords))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    # Color bar to represent the value of V
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=min_colorbar_value, vmax=max_colorbar_value)),
        ax=ax,
        orientation='vertical'
    )
    cbar.set_label(colorbar_label)

    # Update function for animation
    def animate(frame):
        # Interpolate the electrical data for the current time step
        grid_z = griddata((x_coords[:, frame], y_coords[:, frame]), v_data[:, frame], (grid_x, grid_y), method=interpolation_method)

        # Draw new contours
        contour = ax.contourf(grid_x, grid_y, grid_z, levels=contour_levels, cmap='viridis', vmin=min_colorbar_value, vmax=max_colorbar_value)

        ax.set_title(f"t = {(frame*time_step):.0f} ms")
        return contour.get_paths()

    # Create the animation
    ani = FuncAnimation(fig, animate, frames=range(n_frames), interval=animation_speed, repeat=repeat_animation)
    plt.close()
    return ani

def animate_multiple_V_plots(predictions: list, indices: list, families: list, use_all_domains, family_names: list, time_steps: list = None, figsize=(18, 18), padding=0.08, model_order: list = ['LPM-PINN', 'Affine-PINN', 'Basic-PINN'], interval=100, fps=10, font_size: int = font_size):
    """
    Animation version of create_multiple_V_plots. Animates over time steps.

    Parameters
    ----------
    predictions   : same structure as in create_multiple_V_plots
    indices       : domain index for each row
    families      : DomainFamily for each row
    use_all_domains : whether to use all domains or test_domains
    family_names  : row labels
    time_steps    : list of integer time indices to animate over.
                    Defaults to all time steps in the first prediction.
    figsize       : figure size
    padding       : spatial padding passed to smooth_plot_tri
    model_order   : order of model columns
    interval      : milliseconds between frames
    fps           : frames per second (used when saving)

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    plt.rcParams.update({'font.size': font_size})

    # Determine time steps from the first prediction if not provided
    if time_steps is None:
        first_exp = list(predictions[0].keys())[0]
        n_time = predictions[0][first_exp]['output'][str(indices[0])]['gt'].shape[1]
        time_steps = list(range(n_time))

    n_rows = len(predictions)
    n_cols = len(model_order) + 1  # +1 for FEM ground truth
    min_at, max_at = -80, 20

    # Pre-compute domain metadata per row
    domains_list = []
    for row in range(n_rows):
        domains = families[row].domains if use_all_domains else families[row].test_domains
        domains_list.append(domains[indices[row]])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True, sharey='row')
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D indexing

    # Add a single shared colorbar axis
    norm = plt.Normalize(vmin=min_at, vmax=max_at)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    _t_width = len(str(max(time_steps)))
    title = fig.suptitle(f't = {time_steps[0]:0{_t_width}d} ms', fontsize=32)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.98))

    def draw_frame(t):
        for row in range(n_rows):
            selected_domain = domains_list[row]
            x_pts = selected_domain.x[:, 0, 0]
            y_pts = selected_domain.x[:, 0, 1]

            first_exp = list(predictions[row].keys())[0]
            gt = predictions[row][first_exp]['output'][str(indices[row])]['gt'][:, t]

            # --- FEM column ---
            ax = axes[row][0]
            ax.cla()
            smooth_plot_tri(ax, x_pts, y_pts, selected_domain.x_bc, gt,
                            vmin=min_at, vmax=max_at, padding=padding)
            ax.set_ylabel('y [mm]')
            ax.tick_params(axis='x', labelsize=ticks_label_size)
            ax.tick_params(axis='y', labelsize=ticks_label_size)
            if row == 0:
                ax.set_title("FEM")
            if row == n_rows - 1:
                ax.set_xlabel('x [mm]')
            ax.text(-0.35, 0.5, family_names[row], va='center', ha='right', rotation=90,
                    transform=ax.transAxes, fontsize=32)

            # --- Model columns ---
            predictions_row = dict(sorted(predictions[row].items(), key=lambda item: len(item[0]), reverse=True))
            for j, exp in enumerate(model_order):
                ax = axes[row][j + 1]
                ax.cla()
                try:
                    v_preds = predictions_row[exp]['output'][str(indices[row])]['preds'][:, t]
                    smooth_plot_tri(ax, x_pts, y_pts, selected_domain.x_bc, v_preds,
                                    vmin=min_at, vmax=max_at, padding=padding)
                except Exception:
                    pass
                ax.tick_params(axis='x', labelsize=ticks_label_size)
                ax.tick_params(axis='y', labelsize=ticks_label_size)
                if row == 0:
                    ax.set_title(str(exp))
                if row == n_rows - 1:
                    ax.set_xlabel('x [mm]')

    # Draw initial frame
    draw_frame(time_steps[0])

    # Add colorbars once, after the initial frame
    for row in range(n_rows):
        cbar = fig.colorbar(sm, ax=axes[row][-1], orientation='vertical')
        cbar.set_label('V [mV]')
        cbar.ax.tick_params(labelsize=ticks_label_size)

    def update(frame_idx):
        t = time_steps[frame_idx]
        title.set_text(f't = {t:0{_t_width}d} ms')
        tqdm.write(f"Animating frame {frame_idx + 1}/{len(time_steps)}: t = {t} ms", file=sys.stderr)
        draw_frame(t)
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(time_steps),
        interval=interval,
        blit=False,
    )
    anim._fps = fps  # store for convenience when saving
    return anim


def animate_multiple_V_plots_3d(predictions: list, indices: list, families: list, use_all_domains: bool, family_names: list, time_steps: list = None, figsize=(18, 18), model_order=['LPM-PINN', 'Affine-PINN', 'Basic-PINN'], downsample_factor=1, interval=100, fps=10, font_size: int = font_size):
    """
    Animation version of create_multiple_V_plots_3d_notebook. Animates over time steps.

    Parameters
    ----------
    predictions     : same structure as in create_multiple_V_plots_3d_notebook
    indices         : domain index for each row
    families        : DomainFamily for each row
    use_all_domains : whether to use all domains or test_domains
    family_names    : row labels
    time_steps      : list of integer time indices to animate over.
                      Defaults to all time steps in the first prediction.
    figsize         : figure size
    model_order     : order of model columns
    downsample_factor : spatial downsampling applied to scatter points
    interval        : milliseconds between frames
    fps             : frames per second (used when saving)
    font_size       : font size for titles and labels

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    plt.rcParams.update({'font.size': font_size})

    # Determine time steps from the first prediction if not provided
    if time_steps is None:
        first_exp = list(predictions[0].keys())[0]
        n_time = predictions[0][first_exp]['output'][str(indices[0])]['gt'].shape[1]
        time_steps = list(range(n_time))

    n_rows = len(families)
    n_cols = len(model_order) + 1
    min_at, max_at = -80, 20
    _t_width = len(str(max(time_steps)))

    # Pre-compute domain data per row
    domains_list = []
    for row in range(n_rows):
        domains = families[row].domains if use_all_domains else families[row].test_domains
        domains_list.append(domains[indices[row]])

    # Build the figure and all axes up front
    fig = plt.figure(figsize=figsize)
    axes_flat = []  # one entry per subplot, in row-major order
    plot_idx = 1
    t0 = time_steps[0]

    for row in range(n_rows):
        selected_domain = domains_list[row]
        x = selected_domain.x[:, 0, 0]
        y = selected_domain.x[:, 0, 1]
        z = selected_domain.x[:, 0, 2]

        first_exp = list(predictions[row].keys())[0]
        gt = predictions[row][first_exp]['output'][str(indices[row])]['gt'][::downsample_factor, t0]

        # FEM column
        ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
        sc = ax.scatter(x, y, z, c=gt, vmin=min_at, vmax=max_at, edgecolors='none')
        axes_flat.append((ax, sc))
        plot_idx += 1
        ax.view_init(elev=20, azim=60)
        ax.set_zlabel('z [mm]', labelpad=10)
        ax.set_xlabel('x [mm]', labelpad=10)
        ax.set_ylabel('y [mm]', labelpad=10)
        ax.tick_params(axis='x', labelsize=ticks_label_size)
        ax.tick_params(axis='y', labelsize=ticks_label_size)
        ax.tick_params(axis='z', labelsize=ticks_label_size)
        if row == 0:
            ax.set_title('FEM')
        ax.text2D(-0.37, 0.55, family_names[row], va='center', ha='right', rotation=90,
                  transform=ax.transAxes, fontsize=font_size)

        # Model columns
        predictions_row = dict(sorted(predictions[row].items(), key=lambda item: len(item[0]), reverse=True))
        for exp in model_order:
            v_preds = predictions_row[exp]['output'][str(indices[row])]['preds'][::downsample_factor, t0]
            ax = fig.add_subplot(n_rows, n_cols, plot_idx, projection='3d')
            sc = ax.scatter(x, y, z, c=v_preds, vmin=min_at, vmax=max_at, edgecolors='none')
            axes_flat.append((ax, sc))
            plot_idx += 1
            ax.view_init(elev=20, azim=60)
            ax.set_zticklabels([])
            ax.tick_params(axis='x', labelsize=ticks_label_size)
            ax.tick_params(axis='y', labelsize=ticks_label_size)
            ax.tick_params(axis='z', labelsize=ticks_label_size)
            if row == 0:
                ax.set_title(str(exp))

    # Colorbar
    norm = plt.Normalize(vmin=min_at, vmax=max_at)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    fig.subplots_adjust(hspace=0.0, bottom=0.05, top=0.90, left=0.07, right=0.91)
    cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.55])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('V [mV]')
    cbar.ax.tick_params(labelsize=ticks_label_size)

    # Suptitle
    title = fig.suptitle(f't = {t0:0{_t_width}d} ms', fontsize=32, y=0.97)
    # fig.text(0.5, 0.24, ' ', transform=fig.transFigure)  # invisible bottom anchor

    def update(frame_idx):
        t = time_steps[frame_idx]
        tqdm.write(f"Animating frame {frame_idx + 1}/{len(time_steps)}: t = {t} ms", file=sys.stderr)
        title.set_text(f't = {t:0{_t_width}d} ms')
        artist_idx = 0
        for row in range(n_rows):
            first_exp = list(predictions[row].keys())[0]
            gt = predictions[row][first_exp]['output'][str(indices[row])]['gt'][::downsample_factor, t]
            _, sc = axes_flat[artist_idx]
            sc.set_array(gt)
            artist_idx += 1

            predictions_row = dict(sorted(predictions[row].items(), key=lambda item: len(item[0]), reverse=True))
            for exp in model_order:
                v_preds = predictions_row[exp]['output'][str(indices[row])]['preds'][::downsample_factor, t]
                _, sc = axes_flat[artist_idx]
                sc.set_array(v_preds)
                artist_idx += 1
        return [sc for _, sc in axes_flat] + [title]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(time_steps),
        interval=interval,
        blit=False,
    )
    anim._fps = fps
    return anim