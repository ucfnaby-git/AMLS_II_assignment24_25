#!/usr/bin/env python
"""
Improved Image Denoising Training with MWCNN (Modified for sRGB)

This script loads training data from image folders and validation data from MATLAB
files (with ground truth). To avoid excessive validation overhead, only a random
subset of the validation patches is used each epoch to calculate validation loss,
PSNR, and SSIM.

The training dataset now applies random flip and rotation on the full image before cropping,
which avoids black padding issues after rotation.
"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from A.indexcalc import evaluate_metrics

# Import MWCNN model
from A.model import MWCNN

# Import functional transforms.
from torchvision.transforms import functional as F
from A.trainingset import SIDTrainDataset
from A.valmatset import SIDDMatDataset

#########################
# Training and Evaluation Functions
#########################


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Trains the model for one epoch and returns the average training loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0.0
    for i, (noisy, gt) in enumerate(dataloader):
        dev_str = "cuda" if torch.cuda.is_available() else str(device)
        noisy = noisy.to(dev_str)
        gt = gt.to(dev_str)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        batch_size = noisy.size(0)
        # convert mean loss → sum over this batch
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        if (i + 1) % 10 == 0:
            print(f"Train Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.6f}")
    # average loss per sample over the entire epoch
    return total_loss / total_samples


def evaluate(model, dataloader, criterion, device):
    """Computes the average loss on the given dataloader (expected to yield GT)."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for noisy, gt in dataloader:
            dev_str = f"cuda" if torch.cuda.is_available() else str(device)
            noisy = noisy.to(dev_str)
            gt = gt.to(dev_str)
            output = model(noisy)
            loss = criterion(output, gt)
            total_loss += loss.item() * noisy.size(0)
    return total_loss / len(dataloader.dataset)


#########################
# Main Function
#########################


def main():

    # Set a base seed for reproducibility.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set up device (prefer GPU if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device count: {torch.cuda.device_count()}")
    print("Using device:", device)
    print()

    # Create the MWCNN model instance.
    model = MWCNN(n_feats=64)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 8, 9])
    model.to(device)

    if device.type == "cuda":
        print("Current CUDA device:", torch.cuda.current_device())
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(1) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(2) / 1024**3, 1), "GB")

    # Optionally load a pre-trained checkpoint.
    if os.path.exists("./A/results/mwcnn_trained.pth"):
        checkpoint = torch.load("./A/results/mwcnn_trained.pth", map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("Pre-trained data loaded.")

    # ---------------------
    # DataLoader Setup
    # ---------------------
    # Training data: folder-based (images)
    sidd_train_dir = "Datasets/train/sRGB/"
    train_dataset = SIDTrainDataset(
        root_dir=sidd_train_dir, patch_size=512, mode="train", augment=True
    )
    train_loader = data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=8
    )

    # Validation data: MAT file-based. (Both noisy and GT MAT files are provided.)
    noisy_mat_file = "Datasets/valid/siddplus_valid_noisy_srgb.mat"
    gt_mat_file = "Datasets/valid/siddplus_valid_gt_srgb.mat"
    val_mat_dataset = SIDDMatDataset(noisy_mat_file, gt_mat_file)
    # Define the subset size to use each epoch (e.g., 128 patches).
    val_subset_size = 128
    # For validation, select a random subset of patches.
    total_val = len(val_mat_dataset)
    subset_indices = np.random.choice(total_val, size=val_subset_size, replace=False)
    val_subset = data.Subset(val_mat_dataset, subset_indices)
    val_loader = data.DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=8)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    num_epochs = 3
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=5e-6
    )
    criterion = nn.L1Loss()

    train_losses, val_losses = [], []
    psnr_values, ssim_values = [], []
    train_lr = []
    print("Starting training on SIDD training data...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.6f}")
        # Print current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        train_lr.append(current_lr)
        print(f"Epoch {epoch+1}, LR: {current_lr:.9f}")

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.6f}")

        # Compute PSNR and SSIM metrics on the validation subset.
        psnr, ssim_val = evaluate_metrics(model, val_loader, device)
        psnr_values.append(psnr)
        ssim_values.append(ssim_val)
        print(f"Validation PSNR: {psnr:.3f} dB | SSIM: {ssim_val:.4f}")

        scheduler.step()

    print("Training completed.")
    torch.save({"state_dict": model.state_dict()}, "A/results/mwcnn_trained.pth")
    print("model saved.")

    # save training curve data
    with open("A/results/training-curve.csv", "w+") as f:
        f.write("epoch, train_loss, val_loss, LR, PSNR, SSIM\n")
        for i in range(num_epochs):
            f.write(
                f"{i + 1}, {train_losses[i]:.6f}, {val_losses[i]:.6f}, {train_lr[i]:.9f}, {psnr_values[i]:.3f}, {ssim_values[i]:.4f}\n"
            )
        f.close()

    # Plot the learning curve (Loss, PSNR, SSIM).
    plt.figure(figsize=(10, 8))
    epochs_range = range(1, num_epochs + 1)
    plt.subplot(3, 1, 1)
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(epochs_range, psnr_values, "b-", label="Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(epochs_range, ssim_values, "r-", label="Validation SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("A/results/training_metrics_curve.png")
    plt.show()
    print("Training metrics curve saved to training_metrics_curve.png")


if __name__ == "__main__":
    main()
