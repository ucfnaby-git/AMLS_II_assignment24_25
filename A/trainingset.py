#########################
# Standard Library Imports
#########################
import os
import random
import sys

import torch.utils.data as data
from PIL import Image
from torchvision import transforms

# Import functional transforms.
from torchvision.transforms import functional as F

#########################
# Synchronized Augmentation Helper
#########################


def synchronized_transform(noisy_img, gt_img, angle_range=45, use_expand=True):
    """
    Applies the same random horizontal flip, vertical flip, and rotation
    to both the full noisy and GT images. Rotation is applied with expand=True
    to keep the entire rotated image, so cropping afterward can avoid black padding.

    Args:
        noisy_img (PIL.Image): The noisy image.
        gt_img (PIL.Image): The ground truth image.
        angle_range (int): Maximum rotation angle in degrees.
        use_expand (bool): If True, rotate with expand=True.
    Returns:
        (PIL.Image, PIL.Image): The transformed noisy and GT images.
    """
    # Random horizontal flip.
    if random.random() < 0.5:
        noisy_img = F.hflip(noisy_img)
        gt_img = F.hflip(gt_img)
    # Random vertical flip.
    if random.random() < 0.5:
        noisy_img = F.vflip(noisy_img)
        gt_img = F.vflip(gt_img)
    # Random rotation.
    angle = random.uniform(-angle_range, angle_range)
    noisy_img = noisy_img.rotate(angle, resample=Image.BILINEAR, expand=use_expand)
    gt_img = gt_img.rotate(angle, resample=Image.BILINEAR, expand=use_expand)
    return noisy_img, gt_img


#########################
# SIDTrainDataset – Folder-based Dataset for Training
#########################


class SIDTrainDataset(data.Dataset):
    """
    A PyTorch Dataset for SIDD training data (folder-based).

    Directory structure:
       root_dir/
         <scene_instance>/
            noisy_image.png
            ground_truth_image.png

    When in "train" mode with augmentation enabled, the entire noisy and GT images
    are first augmented (flip and rotate) using a synchronized transform. Then, a crop
    is extracted from the augmented images. In "val" mode, a center crop is extracted.
    """

    def __init__(
        self, root_dir, patch_size=256, mode="train", augment=True, transform=None
    ):
        self.root_dir = root_dir
        self.patch_size = patch_size
        if mode not in ["train", "val"]:
            raise ValueError("mode should be 'train' or 'val'")
        self.mode = mode
        # Only apply augmentation if in "train" mode.
        self.augment = augment if mode == "train" else False
        self.scene_dirs = sorted(
            [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        )
        self.transform = transform if transform is not None else transforms.ToTensor()

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_dir = os.path.join(self.root_dir, self.scene_dirs[idx])
        files = sorted([f for f in os.listdir(scene_dir) if f.lower().endswith(".png")])
        noisy_file, gt_file = None, None
        for f in files:
            lf = f.lower()
            if "nois" in lf:
                noisy_file = f
            elif "gt" in lf or "clean" in lf:
                gt_file = f
        if noisy_file is None or gt_file is None:
            if len(files) >= 2:
                noisy_file, gt_file = files[0], files[1]
            else:
                raise ValueError("Expected at least two PNG files in " + scene_dir)

        noisy_path = os.path.join(scene_dir, noisy_file)
        gt_path = os.path.join(scene_dir, gt_file)
        # open + convert inside a context manager to ensure file‑handles are closed
        with Image.open(noisy_path) as img:
            noisy_img = img.convert("RGB")
        with Image.open(gt_path) as img:
            gt_img = img.convert("RGB")
        if noisy_img.size != gt_img.size:
            raise ValueError(
                "Noisy and ground truth images must have the same dimensions."
            )

        # Get image size after augmentation.
        w, h = noisy_img.size
        if w < 725 or h < 725:
            raise ValueError(
                "Image size after augmentation is smaller than the patch size."
            )

        left = random.randint(0, w - 725)
        top = random.randint(0, h - 725)
        right = left + 725
        bottom = top + 725
        # Crop the images.
        noisy_img = noisy_img.crop((left, top, right, bottom))
        gt_img = gt_img.crop((left, top, right, bottom))

        # If augmentation is enabled, apply synchronized transform on the full images.
        if self.augment:
            noisy_img, gt_img = synchronized_transform(
                noisy_img, gt_img, angle_range=15, use_expand=True
            )

        # Get image size after augmentation.
        w, h = noisy_img.size
        if w < self.patch_size or h < self.patch_size:
            raise ValueError(
                "Image size after augmentation is smaller than the patch size."
            )

        # Determine crop coorinates.
        if self.augment:
            left = (w - self.patch_size) // 2
            top = (h - self.patch_size) // 2
        right = left + self.patch_size
        bottom = top + self.patch_size

        # Crop the images.
        noisy_crop = noisy_img.crop((left, top, right, bottom))
        gt_crop = gt_img.crop((left, top, right, bottom))

        # Convert to tensor.
        noisy_tensor = self.transform(noisy_crop)
        gt_tensor = self.transform(gt_crop)
        return noisy_tensor, gt_tensor
