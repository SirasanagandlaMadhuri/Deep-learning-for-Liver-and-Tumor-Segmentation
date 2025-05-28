import torch
import numpy as np
import nibabel as nib
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import torch.nn.functional as F
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

# ================================
# Liver Tumor Segmentation Model
# ================================
class LiverSegmentationModel(UNet):
    def __init__(self):
        super().__init__(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            norm="instance",
            dropout=0.1
        )

# ================================
# Load Model Function
# ================================
def load_model(model_path):
    model = LiverSegmentationModel()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()
    return model

# ================================
# Load and Preprocess NIfTI Image
# ================================
def load_nifti_image(file_path):
    """ Load a NIfTI file and return as a NumPy array """
    nii_img = nib.load(file_path)
    return nii_img.get_fdata()

def preprocess_image(nii_data):
    """ Normalize and prepare the CT scan for visualization """
    nii_data = (nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data))
    return nii_data

# ================================
# Process NIfTI File through Model
# ================================
def process_nii_file(nii_path, model):
    nii_data = load_nifti_image(nii_path)
    
    # Expand dimensions for model input
    nii_data = np.expand_dims(nii_data, axis=(0, 1))  # Shape: (1, 1, D, H, W)
    nii_data = np.transpose(nii_data, (0, 1, 4, 2, 3))  # Ensure correct format (B, C, D, H, W)
    nii_data = torch.tensor(nii_data, dtype=torch.float32)

    # Resize to (64, 256, 256) for model input
    nii_data = F.interpolate(nii_data, size=(64, 256, 256), mode='trilinear', align_corners=False)

    with torch.no_grad():
        output = sliding_window_inference(nii_data, roi_size=(64, 256, 256), sw_batch_size=1, predictor=model, overlap=0.25)
        output = torch.sigmoid(output).squeeze().cpu().numpy()

    return output, nii_data.squeeze().cpu().numpy()

# ================================
# Tumor Detection & Highlighting
# ================================
def detect_tumor(model_output):
    return np.sum(model_output > 0.5) > 100  # Threshold-based tumor detection

def highlight_abnormalities(model_output):
    return np.where(model_output > 0.5, 255, 0).astype(np.uint8)

# ================================
# Visualization Function
# ================================
def display_result(nii_data, tumor_mask):
    """ Display the liver scan and tumor segmentation overlay """
    slice_idx = nii_data.shape[0] // 2  # Middle slice
    liver_slice = preprocess_image(nii_data[slice_idx])  # Normalize for visualization
    tumor_slice = highlight_abnormalities(tumor_mask[slice_idx])

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(liver_slice, cmap="gray")
    ax[0].set_title("Original Liver Scan")
    ax[0].axis("off")

    ax[1].imshow(liver_slice, cmap="gray")  # Base image
    ax[1].imshow(tumor_slice, cmap="jet", alpha=0.5)  # Overlay mask
    ax[1].set_title("Tumor Segmentation")
    ax[1].axis("off")

    plt.show()

# ================================
# GUI for File Selection
# ================================
def select_files():
    model_path = filedialog.askopenfilename(title="Select Model (.pth)", filetypes=[("PyTorch Model", "*.pth")])
    if not model_path:
        messagebox.showerror("Error", "No model file selected!")
        return

    nii_path = filedialog.askopenfilename(title="Select Liver NIfTI (.nii)", filetypes=[("NIfTI Files", "*.nii")])
    if not nii_path:
        messagebox.showerror("Error", "No NIfTI file selected!")
        return

    model = load_model(model_path)
    tumor_mask, nii_data = process_nii_file(nii_path, model)
    tumor_detected = detect_tumor(tumor_mask)

    messagebox.showinfo("Result", "🔴 Tumor Detected!" if tumor_detected else "🟢 No Tumor Found!")
    display_result(nii_data, tumor_mask)

# ================================
# Tkinter GUI
# ================================
root = tk.Tk()
root.title("Liver Tumor Detection")
root.geometry("400x200")

tk.Label(root, text="Liver Tumor Detection System", font=("Arial", 14)).pack(pady=10)
tk.Button(root, text="Select Model & NIfTI File", command=select_files, font=("Arial", 12), bg="lightblue").pack(pady=10)

root.mainloop()
