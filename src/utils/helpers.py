import os
from typing import Optional
import numpy as np
from pathlib import Path

def ensure_directory(directory: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def validate_file_extension(filename: str, allowed_extensions: list) -> bool:
    """Validate file extension."""
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)

def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"

def validate_dimensions(shape: tuple, min_dims: int = 2, max_dims: int = 5) -> bool:
    """Validate image dimensions."""
    return min_dims <= len(shape) <= max_dims

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0-1 range."""
    img_min = image.min()
    img_max = image.max()
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.float32)
    return (image - img_min) / (img_max - img_min)

def create_thumbnail(image: np.ndarray, max_size: int = 256) -> np.ndarray:
    """Create a thumbnail version of an image slice."""
    if image.shape[0] > max_size or image.shape[1] > max_size:
        scale = max_size / max(image.shape[0], image.shape[1])
        new_shape = (
            int(image.shape[0] * scale),
            int(image.shape[1] * scale)
        )
        # Implement resize logic here
        return image  # Placeholder
    return image

def validate_channel_index(channel: int, max_channels: int) -> bool:
    """Validate channel index."""
    return 0 <= channel < max_channels

class ImageValidationError(Exception):
    """Custom exception for image validation errors."""
    pass

def validate_image_metadata(metadata: dict) -> bool:
    """Validate image metadata structure."""
    required_keys = {'dimensions', 'dtype', 'size_bytes', 'shape_description'}
    if not all(key in metadata for key in required_keys):
        return False
    
    shape_keys = {
        'time_frames', 'z_slices', 'channels', 'height', 'width'
    }
    return all(key in metadata['shape_description'] for key in shape_keys)