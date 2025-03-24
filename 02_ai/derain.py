import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import os
from runpy import run_path
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
import cv2
import argparse
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Parsing arguments
parser = argparse.ArgumentParser(description='Demo MPRNet')
parser.add_argument('--input_dir', default='./samples/input/', type=str, help='Input images')
parser.add_argument('--result_dir', default='./samples/output/', type=str, help='Directory for results')
parser.add_argument('--task', required=True, type=str, help='Task to run', choices=['Deblurring', 'Denoising', 'Deraining'])

args = parser.parse_args()

# Save image function
def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Load model checkpoint
def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

# Task and directories setup
task = args.task
inp_dir = args.input_dir
out_dir = args.result_dir

print(f"Input directory: {inp_dir}")
print(f"Output directory: {out_dir}")

os.makedirs(out_dir, exist_ok=True)

# List input files
files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
                + glob(os.path.join(inp_dir, '*.JPG'))
                + glob(os.path.join(inp_dir, '*.png'))
                + glob(os.path.join(inp_dir, '*.PNG'))
                + glob(os.path.join(inp_dir, '*.jpeg'))
                + glob(os.path.join(inp_dir, '*.tiff')))

# Check if files were found
if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

print(f"Found {len(files)} images in {inp_dir}")

# Load model architecture and weights
print("Loading model...")
load_file = run_path('C:/Users/saaan/Documents/MPRNet/Deraining/MPRNet.py')
model = load_file['MPRNet']()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model loaded on {device}")

weights = os.path.join(task, "pretrained_models", "model_" + task.lower() + ".pth")
print(f"Loading weights from: {weights}")
load_checkpoint(model, weights)
model.eval()

img_multiple_of = 8

# Lists to track SSIM values (confidence levels)
ssim_values = []

# Process each file
for file_ in files:
    print(f"Processing file: {file_}")

    # Open image and convert to tensor
    img = Image.open(file_).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).to(device)

    # Pad the input if not a multiple of 8
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - h if h % img_multiple_of != 0 else 0
    padw = W - w if w % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    # Inference (Deraining)
    with torch.no_grad():
        restored = model(input_)
    restored = restored[0]
    restored = torch.clamp(restored, 0, 1)

    # Unpad the output
    restored = restored[:, :, :h, :w]

    # Convert to image and save
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    # Get filename and save result
    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img(os.path.join(out_dir, f + '.jpg'), restored)
    print(f"Saved derained image: {f}.jpg")

    # **Calculate SSIM for Confidence Level**
    original_img = np.array(img)
    restored_img = restored

    # Calculate SSIM as a measure of confidence (similarity)
    ssim_value = ssim(original_img, restored_img, multichannel=True)

    # Append the SSIM value to the list
    ssim_values.append(ssim_value)

    # Display side by side before and after
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(original_img)
    ax[0].set_title(f"Original Image\nSSIM: {ssim_value:.3f}")
    ax[0].axis('off')

    ax[1].imshow(restored_img)
    ax[1].set_title(f"Restored Image\nSSIM: {ssim_value:.3f}")
    ax[1].axis('off')

    plt.show()

# **Graph Confidence Level (SSIM) Comparison**
plt.plot(ssim_values, label='SSIM (Confidence Level)')
plt.xlabel('Image Index')
plt.ylabel('SSIM')
plt.title(f'{task} Performance (Confidence Level Comparison)')
plt.legend()
plt.show()

print(f"Files saved at {out_dir}")
