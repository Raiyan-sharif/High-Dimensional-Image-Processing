import numpy as np
import dask.array as da
import tifffile
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from typing import Tuple, Dict, Union
import logging
from sklearn.cluster import KMeans
from skimage import filters

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.image_data = None
        self.metadata = None

    def load_image(self, file_path):
        """Load and validate a TIFF image, ensuring 5D structure."""
        try:
            # Load the image
            self.image_data = tifffile.imread(file_path)
            
            # Ensure the image is at least 2D
            if len(self.image_data.shape) < 2:
                raise ValueError("Image must be at least 2D")
            
            # Convert to 5D if necessary
            while len(self.image_data.shape) < 5:
                self.image_data = np.expand_dims(self.image_data, axis=0)
            
            # Create metadata
            self.metadata = {
                'dimensions': list(self.image_data.shape),
                'dtype': str(self.image_data.dtype),
                'size_bytes': self.image_data.nbytes,
                'shape_description': {
                    'time_frames': self.image_data.shape[0],
                    'z_slices': self.image_data.shape[1],
                    'channels': self.image_data.shape[2],
                    'height': self.image_data.shape[3],
                    'width': self.image_data.shape[4]
                }
            }
            return self.metadata
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise ValueError(f"Failed to load image: {str(e)}")

    def get_slice(self, time=0, z=0, channel=0):
        """Extract a specific 2D slice from the image."""
        if self.image_data is None:
            raise ValueError("No image loaded")
            
        try:
            # Ensure indices are within bounds
            if (time >= self.image_data.shape[0] or
                z >= self.image_data.shape[1] or
                channel >= self.image_data.shape[2]):
                raise ValueError("Slice indices out of range")
                
            return self.image_data[time, z, channel, :, :].copy()
        except Exception as e:
            logger.error(f"Error getting slice: {str(e)}")
            raise ValueError(f"Failed to get slice: {str(e)}")

    def run_pca(self, n_components=3):
        """Perform PCA on the image data."""
        if self.image_data is None:
            raise ValueError("No image loaded")
        
        try:
            # Reshape to 2D array (samples x features)
            original_shape = self.image_data.shape
            flattened = self.image_data.reshape(-1, np.prod(original_shape[2:]))
            
            # Perform PCA
            pca = PCA(n_components=min(n_components, flattened.shape[1]))
            reduced_data = pca.fit_transform(flattened)
            
            # Reshape back to original dimensions
            reduced_shape = list(original_shape[:2]) + [n_components]
            return reduced_data.reshape(reduced_shape)
        except Exception as e:
            logger.error(f"Error performing PCA: {str(e)}")
            raise ValueError(f"Failed to perform PCA: {str(e)}")

    def calculate_statistics(self):
        """Calculate basic statistics for each channel."""
        if self.image_data is None:
            raise ValueError("No image loaded")
        
        try:
            # Calculate statistics across spatial dimensions (height and width)
            stats = {
                'mean': np.mean(self.image_data, axis=(3, 4)).tolist(),
                'std': np.std(self.image_data, axis=(3, 4)).tolist(),
                'min': np.min(self.image_data, axis=(3, 4)).tolist(),
                'max': np.max(self.image_data, axis=(3, 4)).tolist(),
                'global_stats': {
                    'mean': float(np.mean(self.image_data)),
                    'std': float(np.std(self.image_data)),
                    'min': float(np.min(self.image_data)),
                    'max': float(np.max(self.image_data))
                }
            }
            return stats
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            raise ValueError(f"Failed to calculate statistics: {str(e)}")

    def segment_channel(self, time=0, z=0, channel=0, method='otsu'):
        """Segment a specific channel using either Otsu or K-means."""
        try:
            slice_data = self.get_slice(time, z, channel)
            
            if method.lower() == 'otsu':
                threshold = filters.threshold_otsu(slice_data)
                return slice_data > threshold
            
            elif method.lower() == 'kmeans':
                kmeans = KMeans(n_clusters=2, random_state=42)
                flattened = slice_data.reshape(-1, 1)
                labels = kmeans.fit_predict(flattened)
                return labels.reshape(slice_data.shape)
            
            else:
                raise ValueError("Unsupported segmentation method")
        except Exception as e:
            logger.error(f"Error in segmentation: {str(e)}")
            raise ValueError(f"Failed to segment image: {str(e)}")