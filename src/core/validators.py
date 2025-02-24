from typing import List
import os
import tifffile
import numpy as np

def validate_tiff_file(file_path: str) -> bool:
    """Validate if file is a proper TIFF image."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
        
    try:
        with tifffile.TiffFile(file_path) as tiff:
            # Check if file is readable
            data = tiff.asarray()
            # Check if dimensions are valid
            if not (2 <= len(data.shape) <= 5):
                raise ValueError("Image must have between 2 and 5 dimensions")
        return True
    except Exception as e:
        raise ValueError(f"Invalid TIFF file: {str(e)}")

def validate_slice_params(shape: tuple, time: int, z: int, channel: int) -> None:
    """Validate slice parameters against image dimensions."""
    if time >= shape[0]:
        raise ValueError(f"Time index {time} out of bounds (max: {shape[0]-1})")
    if z >= shape[1]:
        raise ValueError(f"Z index {z} out of bounds (max: {shape[1]-1})")
    if channel >= shape[2]:
        raise ValueError(f"Channel index {channel} out of bounds (max: {shape[2]-1})") 