import torch
import pandas as pd
from typing import Callable

def create_metric_dict_and_df(predictions: dict, metric: Callable):

    rmse_dict = {}

    for key in predictions.keys():
        
        all_preds = torch.Tensor(predictions[key]['preds'].flatten())
        all_gt = torch.Tensor(predictions[key]['gt'].flatten())

        rmse_dict[key] = compute_evaluation_metric(predictions=all_preds,
                                                   ground_truth=all_gt,
                                                   metric=metric)
        
    rmse_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in rmse_dict.items()]))
    
    return rmse_dict, rmse_df


def compute_evaluation_metric(predictions: torch.Tensor, ground_truth: torch.Tensor, metric: Callable):
    evaluation = metric(predictions, ground_truth).item()
    return evaluation


def RMSE(predictions: torch.Tensor, ground_truth: torch.Tensor):    
    mse = torch.nn.MSELoss()
    return torch.sqrt(mse(predictions, ground_truth))

def MSE(predictions: torch.Tensor, ground_truth: torch.Tensor):    
    mse = torch.nn.MSELoss()
    return mse(predictions, ground_truth)

def MAE(predictions: torch.Tensor, ground_truth: torch.Tensor):    
    mae = torch.nn.L1Loss()
    return mae(predictions, ground_truth)

def L1(predictions: torch.Tensor, ground_truth: torch.Tensor):
    L1 = torch.sum(abs(predictions - ground_truth))
    return L1

def L2(predictions: torch.Tensor, ground_truth: torch.Tensor):
    L2 = torch.sqrt(torch.sum((abs(predictions - ground_truth))**2))
    return L2


def relative_L2(predictions: torch.Tensor, ground_truth: torch.Tensor):
    # Only makes sense if V is not normalized
    l2 = L2(predictions, ground_truth)
    relative_l2 = l2 / torch.sqrt(torch.sum((abs(ground_truth))**2))
    return relative_l2


def relative_L1(predictions: torch.Tensor, ground_truth: torch.Tensor):
    # Only makes sense if V is not normalized
    l1 = L1(predictions, ground_truth)
    relative_l1 = l1 / torch.sum(abs(ground_truth))
    return relative_l1
