import os
import wget
import scipy.io
import numpy as np
import pandas as pd
import base64
import torch
from collections import OrderedDict

# Import your MWCNN model (assume it's defined in model.py)
from model import MWCNN  # adjust the import as needed

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load your pretrained denoising model.
model_checkpoint = "mwcnn_trained.pth"  # path to your trained model checkpoint
model = MWCNN(n_feats=64).to(device)
# Load the checkpoint.
checkpoint = torch.load(model_checkpoint, map_location=device)
# If the checkpoint is stored as a dictionary with key "state_dict", extract it.
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]
# Remove "module." prefix if present.
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k
    if k.startswith("module."):
        name = k[7:]  # remove 'module.' prefix
    new_state_dict[name] = v

# Load the modified state dict into the model.
model.load_state_dict(new_state_dict)
model.eval()  # set model to evaluation mode

def my_srgb_denoiser(x):
    """
    Denoises an sRGB image block using the pretrained MWCNN model.
    
    Args:
        x (numpy.ndarray): Input image block with shape [H, W, C] and dtype uint8.
        
    Returns:
        numpy.ndarray: Denoised image block with the same shape and dtype.
    """
    # Convert the input image to float [0, 1]
    x_float = x.astype(np.float32) / 255.0
    # Permute dimensions to [C, H, W] and add a batch dimension: [1, C, H, W]
    x_tensor = torch.from_numpy(x_float).permute(2, 0, 1).unsqueeze(0).to(device)
    x_tensor = x_tensor.float()
    # Denoise using the model
    with torch.no_grad():
        out_tensor = model(x_tensor)
    # Remove batch dimension and convert back to numpy array.
    out_tensor = out_tensor.squeeze(0)
    out_np = out_tensor.cpu().numpy()
    out_np = np.clip(out_np, 0, 1)
    # Convert float in [0,1] back to uint8 and change shape to [H, W, C]
    out_np = (out_np * 255.0).astype(np.uint8)
    out_np = out_np.transpose(1, 2, 0)
    return out_np

def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode("utf-8")
    return base64_string

def base64string_to_array(base64string, array_dtype, array_shape):
    decoded_bytes = base64.b64decode(base64string)
    decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
    decoded_array = decoded_array.reshape(array_shape)
    return decoded_array

# Download input file, if needed.
url = "https://competitions.codalab.org/my/datasets/download/0d8a1e68-155d-4301-a8cd-9b829030d719"
input_file = "BenchmarkNoisyBlocksSrgb.mat"
if os.path.exists(input_file):
    print(f"{input_file} exists. No need to download it.")
else:
    print("Downloading input file BenchmarkNoisyBlocksSrgb.mat...")
    wget.download(url, input_file)
    print("Downloaded successfully.")

# Read inputs.
key = "BenchmarkNoisyBlocksSrgb"
data = scipy.io.loadmat(input_file)
if key not in data:
    raise KeyError(f"The variable '{key}' is not found in {input_file}.")
inputs = data[key]  # Expected shape: [n1, n2, H, W, C]
print(f"inputs.shape = {inputs.shape}")
print(f"inputs.dtype = {inputs.dtype}")

# Denoising.
output_blocks_base64string = []
for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        in_block = inputs[i, j, :, :, :]   # shape: [H, W, C]
        out_block = my_srgb_denoiser(in_block)
        # Ensure output has same shape and dtype as input.
        assert in_block.shape == out_block.shape, "Output shape mismatch!"
        assert in_block.dtype == out_block.dtype, "Output dtype mismatch!"
        out_block_base64string = array_to_base64string(out_block)
        output_blocks_base64string.append(out_block_base64string)

# Save outputs to .csv file.
output_file = "SubmitSrgb.csv"
print(f"Saving outputs to {output_file}")
output_df = pd.DataFrame()
n_blocks = len(output_blocks_base64string)
print(f"Number of blocks = {n_blocks}")
output_df["ID"] = np.arange(n_blocks)
output_df["BLOCK"] = output_blocks_base64string
output_df.to_csv(output_file, index=False)

print("Submit the output file SubmitSrgb.csv at")
print("kaggle.com/competitions/sidd-benchmark-srgb-psnr")
print("Done.")

