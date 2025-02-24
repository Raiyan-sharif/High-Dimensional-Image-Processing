import pytest
import numpy as np
import tifffile
import tempfile
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ..src.db.database import Base
from ..src.api.main import app
from ..src.db.database import get_db
from src.core.image_processor import ImageProcessor

@pytest.fixture
def test_db():
    # Create a temporary database for testing
    test_db_url = "sqlite:///./test.db"
    engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
            
    app.dependency_overrides[get_db] = override_get_db
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    os.remove("./test.db")

@pytest.fixture
def test_client(test_db):
    return TestClient(app)

@pytest.fixture
def sample_tiff_file():
    # Create a temporary TIFF file for testing
    with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
        # Create a 5D array (T=2, Z=3, C=2, Y=64, X=64)
        data = np.random.rand(2, 3, 2, 64, 64).astype(np.float32)
        tifffile.imwrite(tmp.name, data)
        yield tmp.name
    os.unlink(tmp.name)

@pytest.fixture
def test_image():
    """Create a test 5D TIFF image."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple 5D test image (time, z, channel, height, width)
        image = np.random.randint(0, 255, (2, 3, 4, 100, 100), dtype=np.uint8)
        
        # Save the image
        file_path = os.path.join(tmpdir, "test_image.tiff")
        tifffile.imwrite(file_path, image)
        
        yield file_path

@pytest.fixture
def processor():
    """Create an ImageProcessor instance."""
    return ImageProcessor()

@pytest.fixture
def loaded_processor(processor, test_image):
    """Create an ImageProcessor instance with a loaded test image."""
    processor.load_image(test_image)
    return processor 