from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor):

        self.inputs = input_tensor
        self.targets = target_tensor


    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):

        input_obj = self.inputs[index]
        target_obj = self.targets[index]

        return {'input': input_obj.__dict__,
                'target': target_obj.V}