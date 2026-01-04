"""
Medical Image Preprocessing Utilities for SDEdit MR-to-CT Synthesis

This module provides utilities to:
1. Load and preprocess medical images (NIfTI, DICOM, JPG/PNG formats)
2. Convert medical images to SDEdit input format (.pth files)
3. Apply appropriate windowing for CT images
4. Batch process images from a directory
"""

import torch
import numpy as np
import os
from typing import Tuple, Optional, List
from glob import glob

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

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


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
        mr_filepath: Path to MR image file (.nii, .nii.gz, DICOM, or JPG/PNG)
        output_path: Output .pth file path
        slice_idx: Slice index for 3D volumes (required for NIfTI)
        target_size: Target size for the image
    """
    # Load image
    image = load_image(mr_filepath, slice_idx)
    
    # Normalize
    image = normalize_mr_image(image)
    
    # Save
    save_sdedit_input(image, output_path, mask=None, target_size=target_size)


def load_image(filepath: str, slice_idx: int = None) -> np.ndarray:
    """
    Load an image from various formats (JPG, PNG, NIfTI, DICOM).
    
    Args:
        filepath: Path to image file
        slice_idx: Slice index for 3D volumes (required for NIfTI)
        
    Returns:
        2D image array
    """
    filepath_lower = filepath.lower()
    
    if filepath_lower.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
        return load_standard_image(filepath)
    elif filepath_lower.endswith('.nii') or filepath_lower.endswith('.nii.gz'):
        if slice_idx is None:
            raise ValueError("slice_idx is required for NIfTI files")
        return load_nifti_slice(filepath, slice_idx)
    elif filepath_lower.endswith('.dcm'):
        return load_dicom_slice(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def load_standard_image(filepath: str) -> np.ndarray:
    """
    Load a standard image file (JPG, PNG, etc.).
    
    Args:
        filepath: Path to image file
        
    Returns:
        2D image array (grayscale) with values normalized to [0, 1]
    """
    if not HAS_PIL:
        raise ImportError("PIL is required to load JPG/PNG files. Install with: pip install Pillow")
    
    img = Image.open(filepath)
    # Convert to grayscale for medical images
    if img.mode != 'L':
        img = img.convert('L')
    
    # Normalize to [0, 1] range for consistency with other loading functions
    return np.array(img, dtype=np.float32) / 255.0


def load_images_from_directory(input_dir: str, 
                               extensions: List[str] = None) -> List[Tuple[str, np.ndarray]]:
    """
    Load all images from a directory.
    
    Args:
        input_dir: Directory containing images
        extensions: List of file extensions to include (default: jpg, png)
        
    Returns:
        List of (filename, image_array) tuples with normalized [0, 1] values
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    
    images = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f'*{ext}')
        files = glob(pattern)
        pattern_upper = os.path.join(input_dir, f'*{ext.upper()}')
        files.extend(glob(pattern_upper))
        
        for filepath in sorted(files):
            try:
                img = load_standard_image(filepath)
                filename = os.path.basename(filepath)
                images.append((filename, img))
            except IOError as e:
                print(f"Warning: Could not read file {filepath}: {e}")
            except ValueError as e:
                print(f"Warning: Invalid image format {filepath}: {e}")
    
    return images


def batch_convert_directory(input_dir: str,
                           output_dir: str,
                           target_size: int = 256,
                           modality: str = "mr",
                           extensions: List[str] = None):
    """
    Batch convert all images in a directory to SDEdit format.
    
    Args:
        input_dir: Directory containing input images (JPG/PNG)
        output_dir: Directory to save .pth files
        target_size: Target image size
        modality: Image modality ('mr' or 'ct')
        extensions: List of file extensions to include
        
    Returns:
        List of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    images = load_images_from_directory(input_dir, extensions)
    output_files = []
    
    print(f"Found {len(images)} images in {input_dir}")
    
    for filename, image in images:
        # For standard images (JPG/PNG), already normalized to [0,1] by load_standard_image
        # Skip additional normalization for standard image formats
        # Only apply normalization for raw medical formats (NIfTI, DICOM)
        
        # Create output filename
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.pth")
        
        # Save
        save_sdedit_input(image, output_path, mask=None, target_size=target_size)
        output_files.append(output_path)
    
    print(f"Converted {len(output_files)} images to SDEdit format in {output_dir}")
    return output_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert medical images to SDEdit input format")
    parser.add_argument("--input", type=str, help="Input image file path (for single file conversion)")
    parser.add_argument("--input_dir", type=str, help="Input directory containing images (for batch conversion)")
    parser.add_argument("--output", type=str, help="Output .pth file path (for single file conversion)")
    parser.add_argument("--output_dir", type=str, help="Output directory for .pth files (for batch conversion)")
    parser.add_argument("--slice", type=int, default=None, help="Slice index for 3D volumes")
    parser.add_argument("--size", type=int, default=256, help="Target image size")
    parser.add_argument("--modality", type=str, choices=["mr", "ct"], default="mr", help="Image modality (used for NIfTI/DICOM normalization)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input_dir. Use one or the other.")
    
    # Batch mode
    if args.input_dir:
        if args.output_dir is None:
            args.output_dir = os.path.join(args.input_dir, "sdedit_input")
        batch_convert_directory(
            args.input_dir, 
            args.output_dir,
            target_size=args.size,
            modality=args.modality
        )
    # Single file mode
    elif args.input:
        if args.output is None:
            # Auto-generate output path
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = f"{base_name}.pth"
            print(f"No --output specified, using: {args.output}")
        
        # Load image
        image = load_image(args.input, args.slice)
        
        # For standard images (JPG/PNG), already normalized to [0,1]
        # For medical formats (NIfTI/DICOM), apply modality-specific normalization
        filepath_lower = args.input.lower()
        if filepath_lower.endswith(('.nii', '.nii.gz', '.dcm')):
            if args.modality == "mr":
                image = normalize_mr_image(image)
            else:
                image = normalize_ct_image(image)
        
        # Save
        save_sdedit_input(image, args.output, mask=None, target_size=args.size)
        print(f"Converted {args.input} to SDEdit format: {args.output}")
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  Single file: python medical_utils.py --input mr.jpg --output mr_input.pth")
        print("  Batch mode:  python medical_utils.py --input_dir ./test_images --output_dir ./sdedit_inputs")
