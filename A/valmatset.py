import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from torchvision import transforms

#########################
# SIDDMatDataset – MATLAB-based Dataset for Validation
#########################


class SIDDMatDataset(data.Dataset):
    """
    A PyTorch Dataset for SIDD+ validation data stored in MATLAB (.mat) files.

    Two separate MAT files are provided: one for the noisy sRGB patches and one for the ground truth.
    The keys in the MAT files should be 'siddplus_valid_noisy_srgb' and 'siddplus_valid_gt_srgb', respectively.
    """

    def __init__(self, noisy_mat_file, gt_mat_file, transform=None):
        noisy_dict = sio.loadmat(noisy_mat_file)
        gt_dict = sio.loadmat(gt_mat_file)
        if "siddplus_valid_noisy_srgb" not in noisy_dict:
            raise KeyError(
                "The noisy MAT file does not contain 'siddplus_valid_noisy_srgb'"
            )
        if "siddplus_valid_gt_srgb" not in gt_dict:
            raise KeyError("The GT MAT file does not contain 'siddplus_valid_gt_srgb'")
        self.noisy = noisy_dict["siddplus_valid_noisy_srgb"]  # shape: [N, H, W, 3]
        self.gt = gt_dict["siddplus_valid_gt_srgb"]  # shape: [N, H, W, 3]
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return self.noisy.shape[0]

    def __getitem__(self, idx):
        noisy_patch = self.noisy[idx].astype(np.float32) / 255.0
        gt_patch = self.gt[idx].astype(np.float32) / 255.0
        # Convert from [H, W, C] to [C, H, W]
        noisy_tensor = torch.from_numpy(noisy_patch).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(gt_patch).permute(2, 0, 1)
        return noisy_tensor, gt_tensor
