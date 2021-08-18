import torch
from torch.utils.data import Dataset
import os

class CustomMolLoader(Dataset):
    def __init__(self, voxel_dir, smile_array, smile_lengths, voxel_transform=None, smile_transform=None):
        self.voxel_dir = voxel_dir
        self.voxel_transform = voxel_transform
        self.smile_array = smile_transform(smile_array) if smile_transform else smile_array
        self.smile_lengths = smile_lengths
    
    def __len__(self) -> int:
        return self.smile_array.shape[0]

    def __getitem__(self, index: int):
        index_A = index%self.smile_array.shape[0]
        index_B = (index_A + 1)%self.smile_array.shape[0]
        voxel_file_path_A = os.path.join(self.voxel_dir,
                "voxel_tensor_"+str(index_A)+".pt")
        voxel_file_path_B = os.path.join(self.voxel_dir, "voxel_tensor_" +
                str(index_B) + ".pt")
        target_voxel_tensor_A = torch.load(voxel_file_path_A)
        target_voxel_tensor_B = torch.load(voxel_file_path_B)

        if self.voxel_transform:
            target_voxel_tensor_A = self.voxel_transform(target_voxel_tensor_A) 
            target_voxel_tensor_B = self.voxel_transform(target_voxel_tensor_B)


        # fetch the ith row from the smile_array and ith entry in smile_lengths
        target_smile_A = self.smile_array[index_A,:]
        target_length_A = self.smile_lengths[index_A]

        target_smile_B = self.smile_array[index_B,:]
        target_length_B = self.smile_lengths[index_B]

        return {"A": [target_voxel_tensor_A, target_smile_A, target_length_A],
                "B": [target_voxel_tensor_B, target_smile_B, target_length_B]}

