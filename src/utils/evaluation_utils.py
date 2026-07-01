import numpy as np
import os
import copy
import pandas as pd
from scipy.stats import shapiro, wilcoxon

from src.data.domain_family import DomainFamily, CombinedFamily
from src.utils.file_utils import load_config
from src.inference.metrics import create_metric_dict_and_df
from src.inference.metrics import relative_L2
from src.inference.predictions import load_predictions_from_hdf5


def convert_predictions_to_Vm(predictions: dict):
    for i in predictions['output'].keys():
        predictions['output'][str(i)]['gt'] = (predictions['output'][str(i)]['gt']*100) + (-80)
        predictions['output'][str(i)]['preds'] = (predictions['output'][str(i)]['preds']*100) + (-80)
    return predictions


def activation_times(data, threshold=0):
    activation_times = []
    for location in range(data.shape[0]):
        activated = np.where(data[location, :] > threshold)
        if len(activated[0]) == 0:
            activation_times.append(np.nan)
        else:
            activation_times.append(int(activated[0][0]))
    return activation_times


def repolarization_times(data, threshold=-70):
    rt_times = []
    for location in range(data.shape[0]):
        activated = np.argmax(data[location, :])
        repolarized = np.where(data[location, activated:] < threshold)
        rt_times.append(repolarized[0][0])
    return rt_times


def print_mean_std(df, file: str = None, include_statistical_test: bool = False):

    model_means = {}

    for model in df["model"].unique():
        mean_value = df[df["model"] == model]["metric_value"].mean()
        std_value = df[df["model"] == model]["metric_value"].std()

        model_means[model] = mean_value

        if file is not None:
            with open(file, 'a') as f:
                f.write(f"{model}: mean = {mean_value:.6f}, std = {std_value:.6f}\n")
        else:
            print(f"{model}: mean = {mean_value:.3f}, std = {std_value:.3f}")

    if include_statistical_test:

        # Sort models by mean value from lowest to highest
        sorted_models = sorted(model_means, key=model_means.get)

        best_model, second_best_model = sorted_models[0], sorted_models[1]

        best_model_values = df[df["model"] == best_model]["metric_value"].tolist()
        second_best_model_values = df[df["model"] == second_best_model]["metric_value"].tolist()

        diff = np.array(best_model_values) - np.array(second_best_model_values)
        _, p_value_diff = shapiro(diff)

        _, p_value_wilcoxon = wilcoxon(np.array(best_model_values), np.array(second_best_model_values))

        if file is not None:
            with open(file, 'a') as f:
                f.write(f"\nStatistical test comparing {best_model} and {second_best_model}\n")
                f.write(f"  diff p-value = {p_value_diff:.6f}\n")
                f.write(f"  wilcoxon p-value = {p_value_wilcoxon:.6f}\n")

    if file is not None:
        with open(file, 'a') as f:
            f.write("\n")


def create_dataframe(predictions: dict, metric, name: str = None):
    rmse_dict, rmse_df = create_metric_dict_and_df(predictions['output'], metric=metric)
    rmse_df["model"] = predictions["name"] if name is None else name
    return rmse_dict, rmse_df

def get_predictions_for_all_PINNs(folder_root_path: str, downsample_factor: int = None):

    internal_predictions = {}
    external_predictions = {}

    internal_dfs = []
    external_dfs = []

    for f in os.listdir(folder_root_path):

        if not f.endswith(".h5"): # or f.endswith("predictions_with_time.h5"):
            continue

        predictions = load_predictions_from_hdf5(f, folder_root_path, downsample_factor=downsample_factor)
        predictions = convert_predictions_to_Vm(predictions)
        predictions['name'] = f.split('_')[0]
        _, df = create_dataframe(predictions, metric=relative_L2)

        if "internal" in f:
            internal_predictions[f.split('_')[0]] = predictions
            internal_dfs.append(df)
        elif "external" in f:
            external_predictions[f.split('_')[0]] = predictions
            external_dfs.append(df)

    if len(internal_dfs) > 0:
        final_internal_df = pd.concat(internal_dfs, ignore_index=True)
        internal_df = final_internal_df.melt(id_vars=["model"], var_name="test_domain", value_name="metric_value")
    
    if len(external_dfs) > 0:
        final_external_df = pd.concat(external_dfs, ignore_index=True)
        external_df = final_external_df.melt(id_vars=["model"], var_name="test_domain", value_name="metric_value")
    
    return internal_df, external_df, internal_predictions, external_predictions


def get_boundary_terms(boundary_path):

    affine_boundary_path = os.path.join(boundary_path, "PA-PINN_training_boundary_terms.npy")

    if not os.path.exists(affine_boundary_path):
        print(f"Boundary terms not found in {boundary_path}")
        return
    
    b_train = np.load(affine_boundary_path, allow_pickle=True).item()
    domains = list(b_train[0]['Ib'].keys())
    processed_boundary_terms = copy.deepcopy(b_train)

    for epoch in b_train.keys():
        Ib = b_train[epoch]['Ib']
        Iv = b_train[epoch]['Iv']
        for domain in domains:
            Ib[domain]['s'] = np.linalg.norm([Ib[domain][s]['B'] for s in Ib[domain].keys()], ord=2)
            Iv[domain]['s'] = np.linalg.norm([Iv[domain][s] for s in Iv[domain].keys()], ord=2)

            s_list = list(Ib[domain].keys())[:-1]
            Vn_values = []
            
            for i in range(len(Ib[domain][s_list[0]]['Vn'])):
                Vn_i = np.linalg.norm([Ib[domain][s]['Vn'][i] for s in s_list])
                Vn_values.append(Vn_i.item())

            Ib[domain]['Vn'] = np.mean(Vn_values)
            
        processed_boundary_terms[epoch]['mean_B'] = np.mean([np.log10(Ib[domain]['s']) for domain in domains])
        processed_boundary_terms[epoch]['mean_V'] = np.mean([np.log10(Iv[domain]['s']) for domain in domains])
        processed_boundary_terms[epoch]['std_B'] = np.std([np.log10(Ib[domain]['s']) for domain in domains])
        processed_boundary_terms[epoch]['std_V'] = np.std([np.log10(Iv[domain]['s']) for domain in domains])

        processed_boundary_terms[epoch]['mean_Vn'] = np.mean([Ib[domain]['Vn'] for domain in domains])


    return processed_boundary_terms




def get_family_result(predictions: dict, metric_values: dict, selected_family: str):
    result = {}
    for exp in predictions.keys():
        if predictions[exp]['family'] == selected_family:
            result[exp] = predictions[exp]['output']
            result[exp]['metric_value'] = metric_values[exp]
    return result


def get_family(folder_path: str, dim: int, use_external: bool, specific_model: str = None, downsample_factor = None):
    
    if specific_model is not None:
        exp = specific_model
    else:
        exp = os.listdir(folder_path)
        # filter out .h5 files and directories
        exp = [e for e in exp if not e.endswith(".h5") and e != 'figures']
        exp = exp[0]
        
    config = load_config(os.path.join(folder_path, exp, "model_config.yaml"))
    # config["data"]["root_path"] = "." + config["data"]["root_path"]
    config["data"]["root_path"] = config["data"]["root_path"]

    families = []

    if type(config["data"]["internal_family_file"]) == list:

        for internal_family_file, external_family_file in zip(config["data"]["internal_family_file"], config["data"]["external_family_file"]):
            data_config = copy.deepcopy(config["data"])
            data_config["internal_family_file"] = internal_family_file
            data_config["external_family_file"] = external_family_file

            family = DomainFamily("", config["pinn"], data_config, dim=dim, use_external_family=use_external, downsample_factor=downsample_factor)
            families.append(family)   
        family = CombinedFamily(families, config["pinn"])
    else:
        family = DomainFamily("", config["pinn"], config["data"], dim=dim, use_external_family=use_external, downsample_factor=downsample_factor)

    return family


def get_training_times(exp_paths: list[str]):

    training_times = {}

    for exp_path in exp_paths:
        
        for model in os.listdir(exp_path):

            if not model.startswith('boundary') and os.path.isdir(os.path.join(exp_path, model)):
                training_file_path = os.path.join(exp_path, model, "training_loss.csv")
                model_name = model.split('-')[0] + '-' + model.split('-')[1]  # e.g., LPM-PINN

                # read index column of training loss file
                training_df = pd.read_csv(training_file_path)
                epoch_times = training_df['epoch_durations'].values

                if model_name not in training_times:
                    training_times[model_name] = []

                training_times[model_name].append(epoch_times.mean())  # average time per epoch

    return training_times


def get_inference_times(exp_paths: list[str]):

    pred_times = {}
    for exp_path in exp_paths:

        _, _, internal_preds, external_preds = get_predictions_for_all_PINNs(exp_path)

        for model in internal_preds.keys():
            internal_pred_times = [internal_preds[model]['output'][i]['inference_time'] for i in internal_preds[model]['output'].keys()]
            external_pred_times = [external_preds[model]['output'][i]['inference_time'] for i in external_preds[model]['output'].keys()]
            total_pred_time = sum(internal_pred_times) + sum(external_pred_times)
            average_pred_time = total_pred_time / (len(internal_preds[model]['output']) + len(external_preds[model]['output']))
            if model not in pred_times:
                pred_times[model] = [average_pred_time]
            else:
                pred_times[model].append(average_pred_time)

    return pred_times


def get_average_inference_times_across_geometry_number(exp_paths: list[str]):
    """Returns {model_name: [avg_inference_time_per_family, ...]}, averaged over geometry numbers."""

    times_collector = {}
    for exp_path in exp_paths:
        _, _, internal_preds, external_preds = get_predictions_for_all_PINNs(exp_path)

        for model in internal_preds.keys():
            model_name = model.split('-')[0] + '-' + model.split('-')[1]  # e.g., LPM-PINN
            if model_name not in times_collector:
                times_collector[model_name] = {}

            for g in internal_preds[model]['output'].keys():
                if g not in times_collector[model_name]:
                    times_collector[model_name][g] = []
                times_collector[model_name][g].append(internal_preds[model]['output'][g]['inference_time'])
                times_collector[model_name][g].append(external_preds[model]['output'][g]['inference_time'])

    # Average across families for each geometry number
    pred_times = {
        model_name: {
            g: sum(times) / len(times)
            for g, times in geometry_times.items()
        }
        for model_name, geometry_times in times_collector.items()
    }

    return pred_times