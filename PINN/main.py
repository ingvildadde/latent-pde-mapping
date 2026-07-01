import os
import torch
import argparse
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.file_utils import load_config, load_trained_model
from PINN.train_models import train_pinn
from src.training.tune_parameters import tune_parameters
from src.inference.predictions import make_prediction
from src.data.domain_family import DomainFamily, CombinedFamily

def run_experiment(args):

    print("Starting experiment")

    # Load configurations
    model_config = load_config(os.path.abspath(args.model_config))
    sim_config = load_config(os.path.abspath("configs/system_dynamics.yaml"))

    dim = model_config["dimensions"]

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    torch.cuda.empty_cache()

    # Set random seed for reproducibility
    RANDOM_SEED = model_config["random_seed"]

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    internal_families = []
    internal_data_files = []
    external_data_files = []
    for i, data_path in enumerate(args.data_config):
        data_config = (load_config(os.path.abspath(data_path)))
        internal_family = DomainFamily(str(i), model_config, data_config, dim=dim)
        internal_families.append(internal_family)
        internal_data_files.append(data_config["internal_family_file"])
        external_data_files.append(data_config["external_family_file"])

    if len(internal_families) > 1:
        internal_family = CombinedFamily(internal_families, model_config)
        data_config["internal_family_file"] = internal_data_files
        data_config["external_family_file"] = external_data_files
        print(f"Domain family used for training: CombinedFamily with {len(internal_family.train_domains)} domains")
    else:
        internal_family = internal_families[0]

    if args.tune:
        best_params = tune_parameters(
                                train_func=train_pinn,
                                train_data=internal_family,
                                sim_config=sim_config,
                                model_config=model_config,
                                data_config=data_config,
                                device=device)
        print(f"Best parameters found: {best_params}")
    else:
        print(f"\nTraining PINN using LPM and with {dim}D data") if model_config['use_LPM'] else print(f"\nTraining PINN with {dim}D data")
        trained_model, _, _ = train_pinn(
                model_config=model_config,
                sim_config=sim_config,
                data_config=data_config,
                family=internal_family,
                device=device,
                save_results=args.save,
                noise_level=args.noise,
                output_path=args.output_path,
                checkpoint_folder_path=args.checkpoint_folder_path,
                start_epoch=args.start_epoch,
                map_pde=args.map_pde
                )
        

    if args.make_internal_predictions:
        make_prediction(trained_model, internal_family, extrapolation=False, model_config=model_config, device=device, dim=dim, output_path=args.output_path)

    if args.make_external_predictions:

        print("\nLoading external data for predictions...")
        external_families = []

        for i, data_path in enumerate(args.data_config):
            data_config = (load_config(os.path.abspath(data_path)))
            external_family = DomainFamily(str(i), model_config, data_config, dim=dim, use_external_family=True)
            external_families.append(external_family)

        if len(external_families) > 1:
            external_family = CombinedFamily(external_families, model_config)
        else:
            external_family = external_families[0]

        make_prediction(trained_model, external_family, extrapolation=True, model_config=model_config, device=device, dim=dim, output_path=args.output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("model_config", type=str, help="Path to model config file (.yaml).")
    parser.add_argument("data_config", nargs="+", help="One or more data config file paths (.yaml).")

    parser.add_argument("--output_path", type=str, default="", help="Path to save the output files. Default is output directory.")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Add Gaussian noise to training data. Value is noise level (e.g., 0.01 for 1% noise).")
    parser.add_argument("--save", action="store_true", help="Saves trained models if provided.")
    parser.add_argument("--tune", action="store_true", help="Tunes hyperparameters for the given models if provided.")
    parser.add_argument("--make_internal_predictions", action="store_true", help="Make predictions on internal test domains if provided.")
    parser.add_argument("--make_external_predictions", action="store_true", help="Make predictions on external test domains if provided.")
    parser.add_argument("--map_pde", action="store_true", help="Use mapped PDE loss calculation if provided.")

    parser.add_argument("--checkpoint_folder_path", type=str, default=None, help="Path to model checkpoint folder to load (.pt). Default is None.")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting epoch for training when resuming from checkpoint. Default is 0.")
    args = parser.parse_args()

    run_experiment(args)