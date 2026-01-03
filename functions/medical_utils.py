"""
Medical Image Preprocessing Utilities for SDEdit MR-to-CT Synthesis

This module provides utilities to:
1. Load and preprocess medical images (NIfTI, DICOM formats)
2. Convert medical images to SDEdit input format (.pth files)
3. Apply appropriate windowing for CT images
"""

import torch
import numpy as np
import os
from typing import Tuple, Optional

try:
    from scipy.ndimage import zoom as scipy_zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def normalize_ct_image(image: np.ndarray, 
                       window_center: float = 40, 
                       window_width: float = 400) -> np.ndarray:
    """
    Apply CT windowing (window level/width) and normalize to [0, 1].
    
    Args:
        image: CT image in Hounsfield Units (HU)
        window_center: Window center (default: 40 for soft tissue)
        window_width: Window width (default: 400 for soft tissue)
        
    Returns:
        Normalized image in range [0, 1]
    """
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val)
    
    return image.astype(np.float32)


def normalize_mr_image(image: np.ndarray, 
                       percentile_low: float = 1, 
                       percentile_high: float = 99) -> np.ndarray:
    """
    Normalize MR image to [0, 1] using percentile clipping.
    
    Args:
        image: MR image
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping
        
    Returns:
        Normalized image in range [0, 1]
    """
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)
    
    image = np.clip(image, p_low, p_high)
    image = (image - p_low) / (p_high - p_low + 1e-8)
    
    return image.astype(np.float32)


def resize_image(image: np.ndarray, target_size: int = 256) -> np.ndarray:
    """
    Resize 2D image to target size using bilinear interpolation.
    
    Args:
        image: 2D image array
        target_size: Target size (image will be square)
        
    Returns:
        Resized image
        
    Raises:
        ImportError: If scipy is not installed
        ValueError: If image dimensions are invalid
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required for image resizing. Install with: pip install scipy")
    
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: height={h}, width={w}")
    
    zoom_factor = (target_size / h, target_size / w)
    
    if len(image.shape) == 3:
        zoom_factor = zoom_factor + (1,)
    
    return scipy_zoom(image, zoom_factor, order=1)


def load_nifti_slice(filepath: str, slice_idx: int) -> np.ndarray:
    """
    Load a single slice from a NIfTI file.
    
    Args:
        filepath: Path to .nii or .nii.gz file
        slice_idx: Index of the axial slice to load
        
    Returns:
        2D image array
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required to load NIfTI files. Install with: pip install nibabel")
    
    nii = nib.load(filepath)
    data = nii.get_fdata()
    
    # Assuming axial slices are in the third dimension
    if slice_idx < 0 or slice_idx >= data.shape[2]:
        raise ValueError(f"slice_idx {slice_idx} out of range [0, {data.shape[2]})")
    
    return data[:, :, slice_idx]


def load_dicom_slice(filepath: str) -> np.ndarray:
    """
    Load a DICOM file.
    
    Args:
        filepath: Path to DICOM file
        
    Returns:
        2D image array in Hounsfield Units (for CT)
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required to load DICOM files. Install with: pip install pydicom")
    
    dcm = pydicom.dcmread(filepath)
    image = dcm.pixel_array.astype(np.float32)
    
    # Apply rescale slope and intercept if available (for CT)
    if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
        image = image * dcm.RescaleSlope + dcm.RescaleIntercept
    
    return image


def prepare_sdedit_input(image: np.ndarray, 
                         mask: Optional[np.ndarray] = None,
                         target_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare image and mask for SDEdit input format.
    
    Args:
        image: 2D image array, normalized to [0, 1]
        mask: Optional binary mask. If None, creates a mask where all pixels are to be edited
        target_size: Target size for the image
        
    Returns:
        Tuple of (mask_tensor, image_tensor) ready for SDEdit
    """
    # Resize image
    image = resize_image(image, target_size)
    
    # Convert to tensor [C, H, W]
    if len(image.shape) == 2:
        # Grayscale - repeat to 3 channels if needed for RGB model
        # For medical images with 1-channel model, keep as 1 channel
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
    else:
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)
    
    # Create mask
    if mask is None:
        # All zeros means entire image will be edited
        mask_tensor = torch.zeros(image_tensor.shape[0], target_size, target_size)
    else:
        mask = resize_image(mask.astype(np.float32), target_size)
        mask_tensor = torch.from_numpy(mask).float()
        if len(mask_tensor.shape) == 2:
            mask_tensor = mask_tensor.unsqueeze(0).repeat(image_tensor.shape[0], 1, 1)
    
    return mask_tensor, image_tensor


def save_sdedit_input(image: np.ndarray, 
                      output_path: str,
                      mask: Optional[np.ndarray] = None,
                      target_size: int = 256):
    """
    Save image in SDEdit input format (.pth file).
    
    Args:
        image: 2D image array, normalized to [0, 1]
        output_path: Output .pth file path
        mask: Optional binary mask
        target_size: Target size for the image
    """
    mask_tensor, image_tensor = prepare_sdedit_input(image, mask, target_size)
    torch.save([mask_tensor, image_tensor], output_path)
    print(f"Saved SDEdit input to {output_path}")


def convert_mr_to_sdedit_input(mr_filepath: str, 
                               output_path: str,
                               slice_idx: int = None,
                               target_size: int = 256):
    """
    Convert an MR image file to SDEdit input format.
    
    Args:
        mr_filepath: Path to MR image file (.nii, .nii.gz, or DICOM)
        output_path: Output .pth file path
        slice_idx: Slice index for 3D volumes (required for NIfTI)
        target_size: Target size for the image
    """
    # Load image
    if mr_filepath.endswith('.nii') or mr_filepath.endswith('.nii.gz'):
        if slice_idx is None:
            raise ValueError("slice_idx is required for NIfTI files")
        image = load_nifti_slice(mr_filepath, slice_idx)
    elif mr_filepath.endswith('.dcm'):
        image = load_dicom_slice(mr_filepath)
    else:
        raise ValueError(f"Unsupported file format: {mr_filepath}")
    
    # Normalize
    image = normalize_mr_image(image)
    
    # Save
    save_sdedit_input(image, output_path, mask=None, target_size=target_size)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert medical images to SDEdit input format")
    parser.add_argument("--input", type=str, required=True, help="Input image file path")
    parser.add_argument("--output", type=str, required=True, help="Output .pth file path")
    parser.add_argument("--slice", type=int, default=None, help="Slice index for 3D volumes")
    parser.add_argument("--size", type=int, default=256, help="Target image size")
    parser.add_argument("--modality", type=str, choices=["mr", "ct"], default="mr", help="Image modality")
    
    args = parser.parse_args()
    
    # Load image
    if args.input.endswith('.nii') or args.input.endswith('.nii.gz'):
        if args.slice is None:
            raise ValueError("--slice is required for NIfTI files")
        image = load_nifti_slice(args.input, args.slice)
    elif args.input.endswith('.dcm'):
        image = load_dicom_slice(args.input)
    else:
        raise ValueError(f"Unsupported file format: {args.input}")
    
    # Normalize based on modality
    if args.modality == "mr":
        image = normalize_mr_image(image)
    else:
        image = normalize_ct_image(image)
    
    # Save
    save_sdedit_input(image, args.output, mask=None, target_size=args.size)
    print(f"Converted {args.input} to SDEdit format: {args.output}")
