import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.data.helper_functions import *
from src.data.data_handlers import DataSplit, Input, Target

class Domain():
    def __init__(self, **kwargs):

        self.affine_params = kwargs.get('affine_params', None)
        self.x = kwargs.get('x')
        self.x_elems = kwargs.get('x_elems')
        self.F = kwargs.get('F')
        self.D = kwargs.get('D', None) * 12.9 if kwargs.get('D', None) is not None else np.zeros_like(self.x) # t_norm = 12.9
        
        t_init = 6
        self.V = (kwargs.get("Vm")[:, t_init:] - (-80)) / 100  # Make transmembrane potential dimensionless, and exclude I_app
        self.W = np.zeros_like(self.V) if kwargs.get("W", None) is None else kwargs.get("W")[:, t_init:]  # Load recovery variable data (W) if available
        self.tau = kwargs.get("t")[t_init:] / 12.9  # Make time units dimensionless, and exclude I_app
        self.V_init = np.expand_dims(self.V[:, 0], axis=-1)  # Initial condition
        self.F_compute_time = kwargs.get("F_compute_time", None)

        self.x_ref = kwargs.get("x_ref")
        self.x_ref_elems = kwargs.get("x_ref_elems")
        self.stim_indices = kwargs.get("stim_indices")
        self.pca_nodes = kwargs.get("pca_nodes", None)

        # Boundary data (computed later if not provided)
        self.x_bc = kwargs.get("x_bc", None)
        self.x_ref_bc = kwargs.get("x_ref_bc", None)
        self.V_bc = kwargs.get("Vm_bc", None)
        self.F_bc = kwargs.get("F_bc", None)
        self.D_bc = kwargs.get("D_bc", None) * 12.9 if kwargs.get("D_bc", None) is not None else np.zeros_like(self.x_bc)  # t_norm = 12.9
        self.x_bc_normals = kwargs.get("x_bc_normals", None)
        self.x_ref_bc_normals = kwargs.get("x_ref_bc_normals", None)

        self.downsample_factor = kwargs.get("downsample_factor")

    @classmethod
    def create(cls, dim: int, **kwargs):
        """
        Factory class method to create the appropriate Domain subclass.
        
        Args:
            dim (int): Spatial dimension (2 or 3)
            **kwargs: Arguments to pass to Domain constructor
            
        Returns:
            Domain2D or Domain3D instance
        """
        if dim == 2:
            return Domain2D(**kwargs)
        elif dim == 3:
            return Domain3D(**kwargs)
        else:
            raise ValueError(f"Unsupported dimension: {dim}. Must be 2 or 3.")

    
    def _complete_setup(self):
        self.u = self.x - self.x_ref
        self.u_bc = self.x_bc - self.x_ref_bc

        self.__create_torch_tensors()

        self.train_data = DataSplit()
        self.mapped_train_data = DataSplit()

        # Sensor points
        self.sensor_points = None
        self.V_sensor_points = None
        self.geometric_descriptor_sensor_points = None
    

    def __create_torch_tensors(self):
        """
        Makes sure that all attributes are tensors and have correct shape.

        Downsamples data if downsample_factor is specified when creating Domain object.

        Example shapes
            x: (N_points, N_time_steps, spatial_dim)
            D: (N_points, N_time_steps, spatial_dim, spatial_dim)
            tau: (N_time_steps, time_dim)
            affine_params: (N_points, N_time_steps, N_affine_params)
        """
        attributes = list(vars(self).keys())

        for attr in attributes:
            data = getattr(self, attr)
            if data is not None:
                setattr(self, attr, torch.tensor(data))

        self.x = self.x.unsqueeze(dim=1).expand(self.x.shape[0], self.tau.shape[0], self.x.shape[1])
        self.x_bc = self.x_bc.unsqueeze(dim=1).expand(self.x_bc.shape[0], self.tau.shape[0], self.x_bc.shape[1])

        self.x_ref = self.x_ref.unsqueeze(dim=1).expand(self.x_ref.shape[0], self.tau.shape[0], self.x_ref.shape[1])
        self.x_ref_bc = self.x_ref_bc.unsqueeze(dim=1).expand(self.x_ref_bc.shape[0], self.tau.shape[0], self.x_ref_bc.shape[1])

        self.F = self.F.unsqueeze(dim=1).expand(self.F.shape[0], self.tau.shape[0], self.F.shape[1])
        self.F_bc = self.F_bc.unsqueeze(dim=1).expand(self.F_bc.shape[0], self.tau.shape[0], self.F_bc.shape[1])
        self.x_bc_normals = self.x_bc_normals.unsqueeze(dim=1).expand(self.x_bc_normals.shape[0], self.tau.shape[0], self.x_bc_normals.shape[1])
        self.x_ref_bc_normals = self.x_ref_bc_normals.unsqueeze(dim=1).expand(self.x_ref_bc_normals.shape[0], self.tau.shape[0], self.x_ref_bc_normals.shape[1])
    
        self.affine_params = self.affine_params.unsqueeze(dim=0).unsqueeze(dim=1).expand(self.x.shape[0], self.tau.shape[0], self.affine_params.shape[0])

        # Ensure that D has correct shape for both 2D and 3D cases, and expand to match x and tau dimensions
        if self.D.shape[1] == 2:
            self.D = self.D.unsqueeze(dim=1).expand(self.D.shape[0], self.tau.shape[0], self.D.shape[1], self.D.shape[2])
            self.D_bc = self.D_bc.unsqueeze(dim=1).expand(self.D_bc.shape[0], self.tau.shape[0], self.D_bc.shape[1], self.D_bc.shape[2])
        elif self.D.shape[1] == 3:
            self.D = self.D[0, :, :] # In 3D, D is given as (1, 3, 3) tensor since it is constant throughout the domain
            self.D_bc = self.D_bc[0, :, :]
            self.D = self.D.unsqueeze(dim=0).unsqueeze(dim=1).expand(self.x.shape[0], self.tau.shape[0], self.D.shape[0], self.D.shape[1])
            self.D_bc = self.D_bc.unsqueeze(dim=0).unsqueeze(dim=1).expand(self.x_bc.shape[0], self.tau.shape[0], self.D_bc.shape[0], self.D_bc.shape[1])
        else:
            self.D = torch.zeros_like(self.x)
            self.D_bc = torch.zeros_like(self.x_bc)

        self.u = self.u.unsqueeze(dim=1).expand(self.u.shape[0], self.tau.shape[0], self.u.shape[1])
        self.u_bc = self.u_bc.unsqueeze(dim=1).expand(self.u_bc.shape[0], self.tau.shape[0], self.u_bc.shape[1])

        self.tau = self.tau.unsqueeze(dim=0).expand(self.x.shape[0], self.tau.shape[0])

        # Downsample data if downsample_factor is specified
        if self.downsample_factor is not None:
            self.x = self.x[::self.downsample_factor]
            self.x_elems = self.x_elems[::self.downsample_factor]
            self.x_bc = self.x_bc[::self.downsample_factor]
            self.x_ref = self.x_ref[::self.downsample_factor]
            self.x_ref_elems = self.x_ref_elems[::self.downsample_factor]
            self.x_ref_bc = self.x_ref_bc[::self.downsample_factor]
            self.F = self.F[::self.downsample_factor]
            self.F_bc = self.F_bc[::self.downsample_factor]
            self.x_bc_normals = self.x_bc_normals[::self.downsample_factor]
            self.x_ref_bc_normals = self.x_ref_bc_normals[::self.downsample_factor]
            self.affine_params = self.affine_params[::self.downsample_factor]
            self.tau = self.tau[::self.downsample_factor]
            self.V = self.V[::self.downsample_factor]
            self.V_bc = self.V_bc[::self.downsample_factor]
            self.V_init = self.V_init[::self.downsample_factor]
            self.D = self.D[::self.downsample_factor]
            self.D_bc = self.D_bc[::self.downsample_factor]

    
    def select_train_indices(self, splits_config: dict):
        """Selects indices used for supervised points, initial points, boundary points and collocation points in a Domain object."""
        RANDOM_SEED = 1234

        num_supervised = splits_config["num_supervised_points"]
        
        indices = {}

        indices["supervised_indices"], _ = split_data(input_data=self.x,
                                                      number_of_points=num_supervised,
                                                      random_state=RANDOM_SEED)
        
        # We resample from all available data points for the initial, boundary and collocation losses
        indices["init_indices"] = range(self.x.shape[0]) # Make all data points available for initial lossl (data resampled in each epoch based on these indices)
        indices["bc_indices"] = range(self.x_bc.shape[0]) # Make all boundary points available for boundary loss (data resampled in each epoch based on these indices)
        indices["collocation_indices"] = range(self.x.shape[0]) # Make all data points available for collocation loss (data resampled in each epoch based on these indices)
        
        return indices


    def set_train_data(self, split_obj: DataSplit, use_mapped_data: bool, supervised_indices, init_indices, collocation_indices, bc_indices):
        """
        Sets the training data splits (supervised, initial, collocation, boundary) in the provided DataSplit object.
        """
        x = self.x_ref if use_mapped_data else self.x
        x_bc = self.x_ref_bc if use_mapped_data else self.x_bc
        x_bc_normals = self.x_ref_bc_normals if use_mapped_data else self.x_bc_normals

        time = self.tau[0] # The time does not depend on the spatial locations

        split_obj.supervised_inputs = [Input(x=loc, t=time, geometric_descriptor=geometric_descriptor) for (loc, geometric_descriptor) in zip(x[supervised_indices], self.geometric_descriptor[supervised_indices])]
        split_obj.initial_inputs = [Input(x=loc, t=time, geometric_descriptor=geometric_descriptor) for (loc, geometric_descriptor) in zip(x[init_indices], self.geometric_descriptor[init_indices])] if len(init_indices) > 0 else []
        
        split_obj.collocation_inputs = [Input(x=loc, t=time, F=F, D=D, u=u, geometric_descriptor=geometric_descriptor) for (loc, F, D, u, geometric_descriptor) in zip(x[collocation_indices], self.F[collocation_indices], self.D[collocation_indices], self.u[collocation_indices], self.geometric_descriptor[collocation_indices])] if len(collocation_indices) > 0 else []
        split_obj.boundary_inputs = [Input(x=loc, t=time, F=F, D=D, u=u, normal=normal, geometric_descriptor=geometric_descriptor) for (loc, F, D, u, normal, geometric_descriptor) in zip(x_bc[bc_indices], self.F_bc[bc_indices], self.D_bc[bc_indices], self.u_bc[bc_indices], x_bc_normals[bc_indices], self.geometric_descriptor[bc_indices])]  if len(bc_indices) > 0 else []
        
        split_obj.supervised_targets = [Target(target) for target in self.V[supervised_indices]]
        split_obj.initial_targets = [Target(target) for target in self.V_init[init_indices]] if len(init_indices) > 0 else []
        split_obj.collocation_targets = [Target(target) for target in self.V[collocation_indices]] if len(collocation_indices) > 0 else []
        split_obj.boundary_targets = [Target(target) for target in self.V_bc[bc_indices]] if len(bc_indices) > 0 else []

        if self.sensor_points is not None:
            split_obj.sensor_inputs = [Input(x=loc, t=time, geometric_descriptor=geometric_descriptor) for (loc, geometric_descriptor) in zip(self.sensor_points, self.geometric_descriptor_sensor_points)]
            split_obj.sensor_targets = [Target(target) for target in self.V_sensor_points]        


    def set_affine_parameters(self):
        self.geometric_descriptor = (self.affine_params).detach().clone()

    def set_pca(self, pca_components: np.ndarray):
        self.geometric_descriptor = torch.tensor(pca_components)
        self.geometric_descriptor = self.geometric_descriptor.unsqueeze(dim=0).unsqueeze(dim=1).expand(self.x.shape[0], self.x.shape[1], self.geometric_descriptor.shape[0])
        

    def get_inner_points(self):
        """
        Filters out and returns the inner vertices with corresponding deformation gradients (Fs) and target values (V_inner)
        """
        domain_x = self.x
        domain_x_bc = self.x_bc

        inner_points_mask = np.ones(domain_x.shape[0], dtype=bool)

        for i in range(domain_x.shape[0]):
            # Check if the point at domain_x[i] matches any row in domain_x_bc
            if any(np.array_equal(domain_x[i], domain_x_bc[j]) for j in range(domain_x_bc.shape[0])):
                inner_points_mask[i] = False  # Mark as boundary if a match is found

        inner_points_x = self.x[inner_points_mask]
        inner_points_ref_x = self.x_ref[inner_points_mask]
        Fs = self.F[inner_points_mask]
        V_inner = self.V[inner_points_mask]

        return inner_points_x, inner_points_ref_x, Fs, V_inner


class Domain2D(Domain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.x_bc is None or self.x_bc.shape[0] == 0:
            self.x_bc, self.x_ref_bc, self.V_bc, self.F_bc, self.D_bc = self.sort_boundary_points()
            self.x_bc_normals = self.compute_boundary_normals(self.x_bc)
            self.x_ref_bc_normals = self.compute_boundary_normals(self.x_ref_bc)

        self._complete_setup()


    def compute_boundary_normals(self, boundary_points):
        unit_normal_vectors = np.zeros_like(boundary_points)
        for i in range(boundary_points.shape[0]):

            prev_point = boundary_points[i - 1]
            next_point = boundary_points[(i + 1) % len(boundary_points)] # Accounts for last point in boundary array as well

            tangent = np.array(next_point) - np.array(prev_point)
            normal = np.array([-tangent[1], tangent[0]])
            unit_normal_vectors[i, :2] = normal / np.linalg.norm(normal)

        return unit_normal_vectors
    
    def sort_boundary_points(self):

        ref_temp_elems = np.hstack([[3] + list(triangle) for triangle in self.x_ref_elems])
        deformed_temp_elems = np.hstack([[3] + list(triangle) for triangle in self.x_elems])

        ref_boundary_points, _ = get_boundary_points(self.x_ref, ref_temp_elems)
        deformed_boundary_points, bc_indices = get_boundary_points(self.x, deformed_temp_elems)

        sorted_boundary_indices = sort_boundary_indices(ref_boundary_points)
        V_boundary = self.V[bc_indices]
        F_boundary = self.F[bc_indices]
        D_boundary = self.D[bc_indices]

        sorted_deformed_boundary_points = np.array([deformed_boundary_points[i] for i in sorted_boundary_indices])
        sorted_ref_boundary_points = np.array([ref_boundary_points[i] for i in sorted_boundary_indices])
        sorted_bc_targets = np.array([V_boundary[i, :] for i in sorted_boundary_indices])
        sorted_bc_Fs = np.array([F_boundary[i, :] for i in sorted_boundary_indices])
        sorted_bc_Ds = np.array([D_boundary[i, :] for i in sorted_boundary_indices])

        return sorted_deformed_boundary_points, sorted_ref_boundary_points, sorted_bc_targets, sorted_bc_Fs, sorted_bc_Ds


    # Visualization methods
    def visualize_data(self, use_mapped_data: bool, scale_arrows: float = 0.2, scale_arrow_width: float = 0.02, point_size: float = 10, s_color: str = 'lightgray', show_arrows: bool = True, ax = None, show_stim: bool = True, stim_color: str = 'yellow'):
        """
        Visualizes vertices and boundary normals.
        """
        x = self.x_ref if use_mapped_data else self.x
        x_bc = self.x_ref_bc if use_mapped_data else self.x_bc
        x_bc_normals = self.x_ref_bc_normals if use_mapped_data else self.x_bc_normals

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 10))
            created_figure = True
        else:
            created_figure = False

        ax.scatter(x[:, 0, 0], x[:, 0, 1], s=point_size, color=s_color)

        if show_stim:
            stim_verts = x[self.stim_indices]
            ax.scatter(stim_verts[:, 0, 0], stim_verts[:, 0, 1], s=point_size, color=stim_color, marker='*')

        if show_arrows:
            for i in range(x_bc.shape[0]):
                ax.arrow(x=x_bc[i, 0, 0].item(),
                        y=x_bc[i, 0, 1].item(),
                        dx=x_bc_normals[i, 0, 0].item()*scale_arrows,
                        dy=x_bc_normals[i, 0, 1].item()*scale_arrows, width=scale_arrow_width)

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect('equal', adjustable='box')

        if created_figure:
            plt.show()
            return ax.figure
    

    def visualize_data_splits(self,
                              data_split: DataSplit,
                                legend_y_loc: float = 1.0,
                                scale_arrows: float = 0.2,
                                scale_arrow_width: float = 0.02):
        """
        Visualizes vertices, boundary normals and which vertices are used for the different losses during training.
        """
        def get_data(obj_list):
            return (np.vectorize(lambda obj: obj.x[0, 0])(obj_list)), (np.vectorize(lambda obj: obj.x[0, 1])(obj_list))

        def get_normal(obj_list):
            return (np.vectorize(lambda obj: obj.normal[0, 0])(obj_list)), (np.vectorize(lambda obj: obj.normal[0, 1])(obj_list))

        point_size = 10

        fig = plt.figure(figsize=(12,10))

        plt.scatter(get_data(data_split.all_inputs)[0], get_data(data_split.all_inputs)[1], s=point_size, color='lightgray')

        plt.scatter(get_data(data_split.initial_inputs)[0], get_data(data_split.initial_inputs)[1], s=point_size, color='pink', label='Initial data points (supervised)')
        plt.scatter(get_data(data_split.boundary_inputs)[0], get_data(data_split.boundary_inputs)[1], s=point_size, color='orange', label='Boundary data points (physics)')
        plt.scatter(get_data(data_split.collocation_inputs)[0], get_data(data_split.collocation_inputs)[1], s=point_size, color='goldenrod', label='Collocation data points (physics)')
        plt.scatter(get_data(data_split.supervised_inputs)[0], get_data(data_split.supervised_inputs)[1], s=point_size, color='darkblue', label='Supervised data points (supervised)')

        for i in range(len(data_split.boundary_inputs)):
            plt.arrow(
                    x=(get_data(data_split.boundary_inputs)[0])[i],
                    y=(get_data(data_split.boundary_inputs)[1])[i],
                    dx=(get_normal(data_split.boundary_inputs)[0])[i]*scale_arrows,
                    dy=(get_normal(data_split.boundary_inputs)[1])[i]*scale_arrows, width=scale_arrow_width)


        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(bbox_to_anchor=(0.5, legend_y_loc), loc='lower center', ncol=2)
        plt.show()

        return fig

    def visualize_pyvista_mesh(self,
                               show_mapped_domain = False,
                               show_boundary_points: bool = False,
                               show_boundary_normals: bool = False,
                               show_elements: bool = False,
                               save_path: str = None):
        """Visualizes the meshed geometry in PyVista."""
        vertices = np.array(self.x_ref[:, 0, :]) if show_mapped_domain else np.array(self.x[:, 0, :])
        elements = self.x_ref_elems if show_mapped_domain else self.x_elems
        boundary_points = self.x_ref_bc if show_mapped_domain else self.x_bc
        boundary_normals = self.x_ref_bc_normals if show_mapped_domain else self.x_bc_normals

        elements = np.hstack([[3] + list(triangle) for triangle in elements])

        if vertices.shape[1] == 2:
            vertices = np.hstack([vertices, np.zeros((vertices.shape[0], 1))])
        if boundary_points.shape[1] == 2:
            boundary_points = np.hstack([boundary_points, np.zeros((boundary_points.shape[0], 1))])
        if boundary_normals.shape[1] == 2:
            boundary_normals = np.hstack([boundary_normals, np.zeros((boundary_normals.shape[0], 1))])

        mesh = pv.PolyData(vertices, elements)

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=show_elements)

        if show_boundary_points:
            plotter.add_mesh(boundary_points, color='red', point_size=3)

        if show_boundary_normals:
            for point, normal in zip(boundary_points, boundary_normals):
                plotter.add_arrows(point, normal*0.5, color='orange')
        
        plotter.view_xy()  # Set view to XY plane for better visualization of 2D geometry

        if save_path is not None:
            plotter.show(auto_close=False)
            plotter.save_graphic(save_path)
        else:
            plotter.show()
        
        plotter.close()




class Domain3D(Domain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.affine_params = self.affine_params[:-3] # Exclude M parameters included in data

        if self.x_bc is None or self.x_bc.shape[0] == 0:
            self.x_bc, self.x_bc_normals, self.F_bc, self.V_bc, self.D_bc = self.compute_boundary_normals(self.x, self.x_elems)
            self.x_ref_bc, self.x_ref_bc_normals, _, _, _ = self.compute_boundary_normals(self.x_ref, self.x_ref_elems)

        self._complete_setup()
        
    
    def compute_boundary_normals(self, vertices, elements):

        mesh = pv.UnstructuredGrid({pv.CellType.TETRA: elements}, vertices)
        surface = mesh.extract_surface()
        surface_points = np.array(surface.points)

        # Build KDTree from original vertices
        tree = cKDTree(vertices)

        # Get closest original vertex to each surface point
        distances, indices = tree.query(surface_points, k=1)

        # Filter based on a tolerance to ensure accuracy
        tolerance = 1e-8
        if np.any(distances > tolerance):
            raise ValueError("Some surface points do not match original vertices within tolerance")

        return surface_points, np.array(surface.point_normals), self.F[indices], self.V[indices], self.D[indices]


    # Visualization methods
    def visualize_data(self, use_mapped_data: bool, scale_arrows: float = 0.2, point_size: float = 5, s_color: str = 'lightgray', show_arrows: bool = True):
        """
        Visualizes vertices and boundary normals in 3D interactively using Plotly.
        """
        x = self.x_ref if use_mapped_data else self.x
        x_bc = self.x_ref_bc if use_mapped_data else self.x_bc
        x_bc_normals = self.x_ref_bc_normals if use_mapped_data else self.x_bc_normals

        # Create a scatter plot for the vertices
        scatter = go.Scatter3d(
            x=x[:, 0, 0],
            y=x[:, 0, 1],
            z=x[:, 0, 2],
            mode='markers',
            marker=dict(size=point_size, color=s_color),
            name='Vertices'
        )

        # Create quivers (arrows) for boundary normals if enabled
        arrow_traces = []
        if show_arrows:
            bc_scatter = go.Scatter3d(
                x=x_bc[:, 0, 0],
                y=x_bc[:, 0, 1],
                z=x_bc[:, 0, 2],
                mode='markers',
                marker=dict(size=point_size, color='red'),
                name='Vertices'
            )
            for i in range(x_bc.shape[0]):
                arrow_traces.append(go.Cone(
                    x=[x_bc[i, 0, 0].item()],
                    y=[x_bc[i, 0, 1].item()],
                    z=[x_bc[i, 0, 2].item()],
                    u=[x_bc_normals[i, 0, 0].item() * scale_arrows],
                    v=[x_bc_normals[i, 0, 1].item() * scale_arrows],
                    w=[x_bc_normals[i, 0, 2].item() * scale_arrows],
                    sizemode="scaled",
                    sizeref=0.1,
                    anchor="tail",
                    colorscale=[[0, 'blue'], [1, 'blue']],
                    showscale=False,
                    name='Boundary Normals'
                ))

        # Combine all traces
        fig = go.Figure(data=[scatter, bc_scatter] + arrow_traces)

        # Set axis labels
        fig.update_layout(
            scene=dict(
                xaxis_title="x [mm]",
                yaxis_title="y [mm]",
                zaxis_title="z [mm]",
                aspectmode='data'  # Equal aspect ratio
            ),
            title="3D Interactive Visualization"
        )

        # Show the interactive plot
        fig.show()

    def visualize_data_splits(self,
                              data_split: DataSplit,
                                legend_y_loc: float = 1.0,
                                scale_arrows: float = 0.2,
                                scale_arrow_width: float = 0.02):
        """
        Visualizes vertices, boundary normals and which vertices are used for the different losses during training.
        """
        def get_data(obj_list):
            return (np.vectorize(lambda obj: obj.x[0, 0])(obj_list)), (np.vectorize(lambda obj: obj.x[0, 1])(obj_list))

        def get_normal(obj_list):
            return (np.vectorize(lambda obj: obj.normal[0, 0])(obj_list)), (np.vectorize(lambda obj: obj.normal[0, 1])(obj_list))

        point_size = 10

        fig = plt.figure(figsize=(12,10))

        plt.scatter(get_data(data_split.all_inputs)[0], get_data(data_split.all_inputs)[1], s=point_size, color='lightgray')

        plt.scatter(get_data(data_split.initial_inputs)[0], get_data(data_split.initial_inputs)[1], s=point_size, color='pink', label='Initial data points (supervised)')
        plt.scatter(get_data(data_split.boundary_inputs)[0], get_data(data_split.boundary_inputs)[1], s=point_size, color='orange', label='Boundary data points (physics)')
        plt.scatter(get_data(data_split.collocation_inputs)[0], get_data(data_split.collocation_inputs)[1], s=point_size, color='goldenrod', label='Collocation data points (physics)')
        plt.scatter(get_data(data_split.supervised_inputs)[0], get_data(data_split.supervised_inputs)[1], s=point_size, color='darkblue', label='Supervised data points (supervised)')

        for i in range(len(data_split.boundary_inputs)):
            plt.arrow(
                    x=(get_data(data_split.boundary_inputs)[0])[i],
                    y=(get_data(data_split.boundary_inputs)[1])[i],
                    dx=(get_normal(data_split.boundary_inputs)[0])[i]*scale_arrows,
                    dy=(get_normal(data_split.boundary_inputs)[1])[i]*scale_arrows, width=scale_arrow_width)


        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(bbox_to_anchor=(0.5, legend_y_loc), loc='lower center', ncol=2)
        plt.show()

        return fig
    