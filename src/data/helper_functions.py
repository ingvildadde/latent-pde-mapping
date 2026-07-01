import torch
import math
import numpy as np
import pyvista as pv
from sklearn.model_selection import train_test_split


def split_data(input_data: torch.Tensor, number_of_points: float, random_state: int, use_indices: bool = True):
        """Splits the dataset into a training set and test set."""        
        if use_indices:
            test_size = 1 - number_of_points / input_data.shape[0]
            train_indices, test_indices = train_test_split(range(input_data.shape[0]), test_size=test_size, random_state=random_state)
        else:
            test_size = 1 - number_of_points / len(input_data)
            train_indices, test_indices = train_test_split(input_data, test_size=test_size, random_state=random_state)
        return train_indices, test_indices
    

def select_random_indices_from_data(input_data: torch.Tensor, number_of_points: float, random_seed: int):
        """Randomly selects a number of indices given by number_of_points."""
        random_range = np.random.default_rng(seed=random_seed)
        indices = random_range.choice(input_data.shape[0], number_of_points, replace=False)
        return indices
    

def select_random_values_from_indices_list(indices: torch.Tensor, number_of_points: float, random_seed: int):
        """Randomly selects a number of indices given by data_size."""
        random_range = np.random.default_rng(seed=random_seed)
        indices = random_range.choice(indices, number_of_points, replace=False)
        indices = [int(idx) for idx in indices]
        return indices


def stack_data(data_list, attribute):
    values = [getattr(data, attribute) for data in data_list if getattr(data, attribute) is not None]
    return np.concatenate(values, axis=0)


def get_boundary_points(vertices, elements):
    """
    Uses PyVista to extract boundary points.
    """
    # Add extra dimension if data is in 2D (PyVista expects data in 3D format)
    if vertices.shape[1] == 2:
        vertices = np.hstack([vertices, np.zeros((vertices.shape[0], 1))])

    mesh = pv.PolyData(vertices, elements)
    boundary = mesh.extract_feature_edges(boundary_edges=True)
    boundary_points = (boundary.points[:, :2]).tolist()
    boundary_indices = [np.where((vertices[:, :2] == point).all(axis=1))[0][0] for point in boundary_points]

    return np.array(boundary_points), np.array(boundary_indices)


def sort_boundary_indices(org_points_boundary):

    org_points_boundary = np.array(org_points_boundary)
    center_of_square = np.array([org_points_boundary[:, 0].max()/2, org_points_boundary[:, 1].max()/2, 0])

    # Sort data over rectangular shape
    sorted_indices = clockwise_sort(org_points_boundary, center_of_square)

    return sorted_indices

def clockwise_sort(points, center):

    def polar_angle(point):
        x, y = point[0] - center[0], point[1] - center[1]
        angle = math.atan2(y, x)
        distance = math.sqrt(x**2 + y**2)
        return angle, -distance

    sorted_indices = [idx for idx, _ in sorted(enumerate(points), key=lambda pair: polar_angle(pair[1]), reverse=True)]
    return sorted_indices
