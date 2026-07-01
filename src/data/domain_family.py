import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm
import sys

from src.data.domain import Domain
from src.data.data_handlers import DataLoaderGroup, DataSplit, Input, Target
from src.data.helper_functions import select_random_indices_from_data, stack_data
from src.utils.file_utils import load_data_family_from_hdf5
from src.enums.geometric_descriptor import GeometricDescriptor

class BaseFamily():
    def __init__(self, model_config):
        self.batch_size = model_config["training"]["batch_size"]
        self.data_splits = model_config["training"]["splits"]
        self.seed = model_config["random_seed"]
        self.dim = model_config["dimensions"]
        self.mapped_mode = model_config.get("use_LPM", False)

        self.num_collocation_points_each_domain = model_config["training"]["splits"]["num_collocation_points"]
        self.num_boundary_points_each_domain = model_config["training"]["splits"]["num_boundary_points"]
        self.num_initial_points_each_domain = model_config["training"]["splits"]["num_initial_points"]

        self.geometric_descriptor = model_config.get("geometric_descriptor", None)
        self.epoch = 0

    def iter_train(self):
        if self.num_collocation_points_each_domain > 0 and self.num_boundary_points_each_domain > 0 and self.num_initial_points_each_domain > 0:
            for domain in self.train_domains:
                updated_col_inputs, updated_boundary_inputs, updated_initial_inputs, updated_initial_targets = self.resample_collocation_points(self.epoch, domain=domain)
                yield (domain, updated_col_inputs, updated_boundary_inputs, updated_initial_inputs, updated_initial_targets)
            self.epoch += 1
        else:
            return iter(self.train_domains)
        
    def resample_collocation_points(self, epoch: int, domain):
        # Resample initial, collocation, and boundary points for each epoch
        seed = self.seed + epoch

        train_data = domain.train_data if not self.mapped_mode else domain.mapped_train_data


        initial_indices = select_random_indices_from_data(input_data=domain.x,
                                                                                    number_of_points=self.num_initial_points_each_domain,
                                                                                    random_seed=seed)

        collocation_indices = select_random_indices_from_data(input_data=domain.x,
                                                                                    number_of_points=self.num_collocation_points_each_domain,
                                                                                    random_seed=seed)
        
        boundary_indices = select_random_indices_from_data(input_data=domain.x_bc,
                                                                                    number_of_points=self.num_boundary_points_each_domain,
                                                                                    random_seed=seed)
        
        resampled_collocation_inputs, _ = self.set_resampled_data(train_data.collocation_inputs, collocation_indices)
        resampled_boundary_inputs, _ = self.set_resampled_data(train_data.boundary_inputs, boundary_indices)
        resampled_initial_inputs, resampled_initial_targets = self.set_resampled_data(train_data.initial_inputs, initial_indices, target_data=train_data.initial_targets)

        return resampled_collocation_inputs, resampled_boundary_inputs, resampled_initial_inputs, resampled_initial_targets

    def set_resampled_data(self, train_data: Input, indices: list, target_data: Target = None):

        updated_train_data = Input(x=torch.zeros(0), t=torch.zeros(0))
        updated_target_data = Target(V=torch.zeros(0)) if target_data is not None else None

        for attr in vars(train_data).keys():
            input_data = getattr(train_data, attr)
            resampled_data = input_data[indices]
            setattr(updated_train_data, attr, resampled_data)
            
        if target_data is not None:
            for attr in vars(target_data).keys():
                target = getattr(target_data, attr)
                resampled_target = target[indices]
                setattr(updated_target_data, attr, resampled_target)

        return updated_train_data, updated_target_data

        

    def create_train_batches(self, use_sensors: bool = False):

        for domain in self.train_domains:

            indices = domain.select_train_indices(self.data_splits)

            domain.set_train_data(domain.train_data, False, **indices)
            domain.set_train_data(domain.mapped_train_data, True, **indices)
        
        all_train_data, all_mapped_train_data = self.combine_data_splits(self.train_domains)

        num_collocation_points = self.num_collocation_points_each_domain*len(self.train_domains)
        num_boundary_points = self.num_boundary_points_each_domain*len(self.train_domains)
        num_initial_points = self.num_initial_points_each_domain*len(self.train_domains)

        train_dataloader = DataLoaderGroup(all_train_data, batch_size=self.batch_size, num_collocation_points=num_collocation_points, num_bc_points=num_boundary_points, num_initial_points=num_initial_points, seed=self.seed, use_sensors=use_sensors)
        mapped_train_dataloader = DataLoaderGroup(all_mapped_train_data, batch_size=self.batch_size, num_collocation_points=num_collocation_points, num_bc_points=num_boundary_points, num_initial_points=num_initial_points, seed=self.seed, use_sensors=use_sensors)

        return train_dataloader, mapped_train_dataloader
    
    
    def organize_train_data_by_domain(self):

        for domain in self.train_domains:

            indices = domain.select_train_indices(self.data_splits)

            domain.set_train_data(domain.train_data, False, **indices)
            domain.set_train_data(domain.mapped_train_data, True, **indices)

            input_attrs = [attr for attr in vars(domain.train_data).keys() if attr.endswith('_inputs')]
            target_attrs = [attr for attr in vars(domain.train_data).keys() if attr.endswith('_targets')]

            for attr in input_attrs:

                input_data = getattr(domain.train_data, attr)
                mapped_input_data = getattr(domain.mapped_train_data, attr)

                updated_input_data = Input(x=torch.zeros(0), t=torch.zeros(0))
                updated_mapped_input_data = Input(x=torch.zeros(0), t=torch.zeros(0))

                for input_attr in vars(input_data[0]).keys():

                    combined_data = torch.stack([getattr(data, input_attr) for data in input_data])
                    setattr(updated_input_data, input_attr, combined_data)

                    mapped_combined_data = torch.stack([getattr(data, input_attr) for data in mapped_input_data])
                    setattr(updated_mapped_input_data, input_attr, mapped_combined_data)

                setattr(domain.train_data, attr, updated_input_data)
                setattr(domain.mapped_train_data, attr, updated_mapped_input_data)

                del input_data, mapped_input_data, updated_input_data, updated_mapped_input_data

            
            for attr in target_attrs:

                target_data = getattr(domain.train_data, attr)
                mapped_target_data = getattr(domain.mapped_train_data, attr)

                updated_target_data = Target(V=torch.zeros(0))
                updated_mapped_target_data = Target(V=torch.zeros(0))

                for target_attr in vars(target_data[0]).keys():

                    combined_data = torch.stack([getattr(data, target_attr) for data in target_data])
                    setattr(updated_target_data, target_attr, combined_data)
                    mapped_combined_data = torch.stack([getattr(data, target_attr) for data in mapped_target_data])
                    setattr(updated_mapped_target_data, target_attr, mapped_combined_data)

                setattr(domain.train_data, attr, updated_target_data)
                setattr(domain.mapped_train_data, attr, updated_mapped_target_data)

                del target_data, mapped_target_data, updated_target_data, updated_mapped_target_data
            
            
    

    def create_sensor_points(self, num_sensor_points: int = 14, create_sensor_points: bool = False):

        for i, domain in enumerate(self.domains):
            if i == 0 and create_sensor_points:
                print(f"Sample fixed sensor points from domain {i}")
                idx = np.random.RandomState(seed=1234).choice(domain.x_ref.shape[0], size=num_sensor_points, replace=False)
                sensor_points = domain.x_ref[idx]
                domain.sensor_points = sensor_points
                domain.V_sensor_points = domain.V[idx]
                np.save(f"sensor_points_{num_sensor_points}.npy", sensor_points.cpu().numpy()) if self.dim == 2 else np.save(f"sensor_points_3d_{num_sensor_points}.npy", sensor_points.cpu().numpy())
                tqdm.write(f"Saved sensor points to sensor_points_{num_sensor_points}.npy", file=sys.stderr) if self.dim == 2 else tqdm.write(f"Saved sensor points to sensor_points_3d_{num_sensor_points}.npy", file=sys.stderr)
            else:
                tqdm.write(f"Using fixed sensor points for domain {i}", file=sys.stderr)

                domain.sensor_points = torch.from_numpy(np.load(f"sensor_points_{num_sensor_points}.npy")).float() if self.dim == 2 else torch.from_numpy(np.load(f"sensor_points_3d_{num_sensor_points}.npy")).float()

                x_ref = domain.x_ref[:, 0, :].cpu().numpy()
                V = domain.V.cpu().numpy()
                geom_desc = domain.geometric_descriptor.cpu().numpy()
                sensor_pts = domain.sensor_points.cpu().numpy()

                interp = LinearNDInterpolator(x_ref, V)
                interp_affine = LinearNDInterpolator(x_ref, geom_desc)

                # Use from_numpy instead of tensor() to avoid copying
                domain.V_sensor_points = torch.from_numpy(interp(sensor_pts)).float()[:, 0, :]
                domain.geometric_descriptor_sensor_points = torch.from_numpy(interp_affine(sensor_pts)).float()[:, 0, :]

                # Clear intermediate arrays if needed
                del x_ref, V, geom_desc, sensor_pts, interp, interp_affine

    

    def geometric_descriptor_type(self, use_external):

        if self.geometric_descriptor is None:
            tqdm.write("No geometric descriptor specified.", file=sys.stderr)
            for domain in self.domains:
                domain.geometric_descriptor = torch.zeros_like(domain.x)
        else:
            match GeometricDescriptor(self.geometric_descriptor):
                case GeometricDescriptor.AFFINE_PARAMETERS:
                    tqdm.write("Using affine parameters as geometric descriptor.", file=sys.stderr)
                    # Implement logic for setting affine parameters as geometric descriptor in each domain
                    for domain in self.domains:
                        domain.set_affine_parameters()
                case GeometricDescriptor.PCA:
                    tqdm.write("Using PCA as geometric descriptor.", file=sys.stderr)
                    # Implement logic for computing PCA over domains and setting as geometric descriptor in each domain
                    self.compute_pca_and_set_descriptor(use_external=use_external)
                case GeometricDescriptor.NONE:
                    tqdm.write("No geometric descriptor specified.", file=sys.stderr)
                    for domain in self.domains:
                        domain.geometric_descriptor = torch.zeros_like(domain.x)
        
    

    def compute_pca_and_set_descriptor(self, use_external):

        internal_domains, external_domains = self.get_all_domains()

        all_domains = internal_domains + external_domains

        tqdm.write("Running PCA", file=sys.stderr)
        X = np.array([np.array((domain.pca_nodes).cpu()).flatten() for domain in all_domains])

        n_pca_components = 2
        pca = PCA(n_components=n_pca_components)
        pca.fit(X)
        pca_result = pca.transform(X)

        self.pca_mean = pca.mean_
        self.pca_components = pca.components_
        self.pca_variances = pca.explained_variance_
        self.pca_variance_percentage = pca.explained_variance_ratio_

        pca_result /= np.sqrt(self.pca_variances)

        pca_result_dict = {'internal_pca_result': pca_result[:len(internal_domains)], 'external_pca_result': pca_result[len(internal_domains):]}

        for i, domain in enumerate(self.domains):
            if use_external:
                domain.set_pca(pca_result_dict['external_pca_result'][i])
            else:
                domain.set_pca(pca_result_dict['internal_pca_result'][i])

    def get_all_domains(self):
        external_data_path = os.path.join(self.data_config["root_path"], self.data_config["external_family_file"])
        internal_data_path = os.path.join(self.data_config["root_path"], self.data_config["internal_family_file"])

        external_domains_dict = np.load(external_data_path, allow_pickle=True).item() if external_data_path.endswith('.npy') else load_data_family_from_hdf5(external_data_path)
        internal_domains_dict = np.load(internal_data_path, allow_pickle=True).item() if internal_data_path.endswith('.npy') else load_data_family_from_hdf5(internal_data_path)

        external_domains = [Domain.create(dim=self.dim, **external_domains_dict[domain], downsample_factor=10) for domain in external_domains_dict.keys()]
        internal_domains = [Domain.create(dim=self.dim, **internal_domains_dict[domain], downsample_factor=10) for domain in internal_domains_dict.keys()]
        
        del external_domains_dict, internal_domains_dict

        return internal_domains, external_domains

    def combine_data_splits(self, domains: list[Domain]):
        """Combines data points from the list of domain into one split for each specific data type (supervised, collocation, initial, boundary)"""
        train_combined_data = DataSplit()
        mapped_train_combined_data = DataSplit()

        attributes = list(vars(train_combined_data).keys())

        for attr in attributes:

            train_data = stack_data([ds.train_data for ds in domains], attr)
            setattr(train_combined_data, attr, train_data)

            mapped_train_data = stack_data([ds.mapped_train_data for ds in domains], attr)
            setattr(mapped_train_combined_data, attr, mapped_train_data)

        return train_combined_data, mapped_train_combined_data
    
    
    def combine_data_domain(self, data_split: list):

        attributes = list(vars(data_split[0]).keys())

        updated_data_split = Input(x=torch.zeros(0), t=torch.zeros(0))

        for attr in attributes:
            combined_data = torch.stack([getattr(ds, attr) for ds in data_split])
            setattr(updated_data_split, attr, combined_data)
        
        return updated_data_split



    
    def print_batches(self, dataloader):

        num_batches = 0
        batch_idx = 0

        for (s_batch, i_batch, c_batch, b_batch) in dataloader:

            supervised_inputs_batch = s_batch["input"]["x"]
            initial_inputs_batch = i_batch["input"]["x"]
            collocation_inputs_batch = c_batch["input"]["x"]
            boundary_inputs_batch = b_batch["input"]["x"]
            boundary_normals_batch = b_batch["input"]["normal"]

            print(f"Batch {batch_idx}")
            print(f"Supervised points: {supervised_inputs_batch.shape}")
            print(f"Initial points: {initial_inputs_batch.shape}")
            print(f"Collocation points: {collocation_inputs_batch.shape}")
            print(f"Boundary points: {boundary_inputs_batch.shape}")
            print(f"Boundary normals: {boundary_normals_batch.shape}")
            print(f"Sum {supervised_inputs_batch.shape[0] + initial_inputs_batch.shape[0] + collocation_inputs_batch.shape[0]+ boundary_inputs_batch.shape[0]}")
            print("\n")

            num_batches +=1
            batch_idx +=1
        
        print(f"Number of batches: {num_batches}")

    
    def get_x_scaling_params(self):
        scaler = StandardScaler()
        all_x = torch.cat([domain.x for domain in self.train_domains], dim=0)
        scaler.fit(all_x[:, 0, :].numpy())
        x_offset = torch.tensor(scaler.mean_, dtype=torch.float32)
        x_scale = torch.tensor(np.sqrt(scaler.var_), dtype=torch.float32)

        return x_offset, x_scale

    def get_t_scaling_params(self):
        scaler = StandardScaler()
        all_t = torch.cat([domain.tau for domain in self.train_domains], dim=0)
        scaler.fit(all_t.numpy().reshape(-1, 1))
        t_offset = torch.tensor(scaler.mean_, dtype=torch.float32)
        t_scale = torch.tensor(np.sqrt(scaler.var_), dtype=torch.float32)

        return t_offset, t_scale

    def get_geometric_descriptor_scaling_params(self):
       
        if not self.geometric_descriptor:
            return None, None

        scaler = StandardScaler()
        all_geom = torch.cat([domain.geometric_descriptor for domain in self.train_domains], dim=0)
        scaler.fit(all_geom[:, 0, :].numpy())
        geom_offset = torch.tensor(scaler.mean_, dtype=torch.float32)
        geom_scale = torch.tensor(np.sqrt(scaler.var_), dtype=torch.float32)

        return geom_offset, geom_scale
    

class DomainFamily(BaseFamily):
    def __init__(self, name, model_config, data_config, dim: int, downsample_factor: int = None, use_external_family: bool = False, use_sensor_data: bool = False, recreate_sensor_points: bool = False, num_sensors: int = 14):
        super().__init__(model_config)

        self.data_config = data_config
        data_family_name = data_config["internal_family_file"] if not use_external_family else data_config["external_family_file"]

        data_file_path = os.path.join(data_config["root_path"], data_family_name)
        domain_splits = data_config["splits"]
        data_seed = data_config["random_seed"] if "random_seed" in data_config else model_config["random_seed"]

        # Load data from .npy file
        domain_family_dict = np.load(data_file_path, allow_pickle=True).item() if data_file_path.endswith('.npy') else load_data_family_from_hdf5(data_file_path)

        self.name = name
        self.num_domains = len(domain_family_dict.keys())
        self.domain_dim = dim

        self.domains = [Domain.create(dim=self.domain_dim, **domain_family_dict[domain], downsample_factor=downsample_factor) for domain in domain_family_dict.keys()]
        tqdm.write(f"Loaded {type(self.domains[0]).__name__} objects", file=sys.stderr)
        del domain_family_dict

        self.geometric_descriptor_type(use_external=use_external_family)

        if use_sensor_data:
            self.create_sensor_points(num_sensor_points=num_sensors, create_sensor_points=recreate_sensor_points)

        indices = np.arange(len(self.domains))

        train_idx, temp_idx = train_test_split(indices, test_size=domain_splits["train_test_size"], random_state=data_seed)
        val_idx, test_idx = train_test_split(temp_idx, test_size=domain_splits["test_val_size"], random_state=data_seed)

        train_idx = [int(idx) for idx in train_idx]
        val_idx = [int(idx) for idx in val_idx]
        test_idx = [int(idx) for idx in test_idx]

        self.split_indices = {"train_idx": list(train_idx), "val_idx": list(val_idx), "test_idx": list(test_idx)}
        self.train_domains = [self.domains[i] for i in train_idx]
        self.val_domains = [self.domains[i] for i in val_idx]
        self.test_domains = [self.domains[i] for i in test_idx]

        self.tau = self.domains[0].tau

        if use_sensor_data:
            self.organize_train_data_by_domain()
        else:
            self.train_dataloader, self.mapped_train_dataloader = self.create_train_batches(use_sensors=use_sensor_data)
        tqdm.write("Created DomainFamily", file=sys.stderr)


class CombinedFamily(BaseFamily):
    def __init__(self, families: list, model_config):
        super().__init__(model_config)

        self.families = families

        self.train_domains = [domain for family in self.families for domain in family.train_domains]
        self.val_domains = [domain for family in self.families for domain in family.val_domains]
        self.test_domains = [domain for family in self.families for domain in family.test_domains]
        self.domains = [*self.train_domains, *self.val_domains, *self.test_domains]

        test_idx, val_idx, train_idx = self.get_split_indices()

        self.split_indices = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}

        self.tau = self.train_domains[0].tau

        self.train_dataloader, self.mapped_train_dataloader = self.create_train_batches()

    
    def get_split_indices(self):

        test_idx, val_idx, train_idx = [], [], []

        for _, family in enumerate(self.families):
            test_idx.extend(family.split_indices["test_idx"])
            val_idx.extend(family.split_indices["val_idx"])
            train_idx.extend(family.split_indices["train_idx"])
        
        return test_idx, val_idx, train_idx