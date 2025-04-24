import numpy as np
import torch

# Import the SSIM implementation.
from A.myssim import compare_ssim as ssim

#########################
# Evaluation Metric Functions (PSNR and SSIM)
#########################


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def mean_psnr_srgb(ref_mat, res_mat):
    n_blk = ref_mat.shape[0]
    mean_psnr = 0
    for b in range(n_blk):
        psnr_val = output_psnr_mse(ref_mat[b], res_mat[b])
        mean_psnr += psnr_val
    return mean_psnr / n_blk


def mean_ssim_srgb(ref_mat, res_mat):
    n_blk = ref_mat.shape[0]
    mean_ssim_val = 0
    for b in range(n_blk):
        ssim_val = ssim(
            ref_mat[b],
            res_mat[b],
            gaussian_weights=True,
            use_sample_covariance=False,
            multichannel=True,
        )
        mean_ssim_val += ssim_val
    return mean_ssim_val / n_blk


def evaluate_metrics(model, dataloader, device):
    """
    Runs the model on the given dataloader, collects outputs, and computes mean PSNR and SSIM.
    """
    model.eval()
    results_list = []
    gt_list = []
    with torch.no_grad():
        for noisy, gt in dataloader:
            dev_str = "cuda" if torch.cuda.is_available() else str(device)
            noisy = noisy.to(dev_str)
            output = model(noisy)
            # Clamp and convert output to CPU numpy arrays.
            output_np = output.cpu().clamp(0, 1).numpy()  # [B, C, H, W]
            gt_np = gt.numpy()  # [B, C, H, W]
            output_np = output_np.transpose(0, 2, 3, 1)  # [B, H, W, C]
            gt_np = gt_np.transpose(0, 2, 3, 1)
            results_list.append(output_np)
            gt_list.append(gt_np)
    results_array = np.concatenate(results_list, axis=0)
    gt_array = np.concatenate(gt_list, axis=0)
    psnr = mean_psnr_srgb(gt_array, results_array)
    ssim_val = mean_ssim_srgb(gt_array, results_array)
    return psnr, ssim_val
