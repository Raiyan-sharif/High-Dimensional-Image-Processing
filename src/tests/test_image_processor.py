import pytest
import numpy as np
from src.core.image_processor import ImageProcessor

def test_load_image(processor, test_image):
    """Test loading an image."""
    metadata = processor.load_image(test_image)
    
    assert processor.image_data is not None
    assert len(processor.image_data.shape) == 5
    assert metadata['dimensions'] == list(processor.image_data.shape)

def test_get_slice(loaded_processor):
    """Test getting a specific slice."""
    slice_data = loaded_processor.get_slice(time=0, z=0, channel=0)
    
    assert slice_data.shape == (100, 100)
    assert slice_data.dtype == np.uint8

def test_run_pca(loaded_processor):
    """Test PCA analysis."""
    n_components = 2
    reduced_data = loaded_processor.run_pca(n_components=n_components)
    
    assert reduced_data.shape[-1] == n_components

def test_calculate_statistics(loaded_processor):
    """Test statistics calculation."""
    stats = loaded_processor.calculate_statistics()
    
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert 'global_stats' in stats

def test_segment_channel_otsu(loaded_processor):
    """Test Otsu segmentation."""
    segmented = loaded_processor.segment_channel(method='otsu')
    assert segmented.dtype == bool

def test_segment_channel_kmeans(loaded_processor):
    """Test K-means segmentation."""
    segmented = loaded_processor.segment_channel(method='kmeans')
    assert segmented.shape == (100, 100)

def test_invalid_slice_indices(loaded_processor):
    """Test error handling for invalid slice indices."""
    with pytest.raises(ValueError):
        loaded_processor.get_slice(time=999, z=0, channel=0)

def test_invalid_segmentation_method(loaded_processor):
    """Test error handling for invalid segmentation method."""
    with pytest.raises(ValueError):
        loaded_processor.segment_channel(method='invalid')