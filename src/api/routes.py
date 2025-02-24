from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from ..core.image_processor import ImageProcessor
from ..db.models import ImageMetadata, AnalysisResult
from typing import Optional
import os
import uuid
import tempfile
import shutil
from fastapi import Depends
from ..db.database import engine, SessionLocal
from sqlalchemy.orm import Session

router = APIRouter()
UPLOAD_DIR = "uploads"
processor = ImageProcessor()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a multi-dimensional TIFF image."""
    if not file.filename.endswith(('.tiff', '.tif')):
        raise HTTPException(status_code=400, detail="Only TIFF files are supported")
    
    try:
        # Create unique ID and filename
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.tiff"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process image and get metadata
        metadata = processor.load_image(file_path)
        
        # Create database entry
        db_image = ImageMetadata(
            id=image_id,
            filename=filename,
            file_path=file_path,
            image_metadata=metadata
        )
        
        # Add and commit to database
        db.add(db_image)
        db.commit()
        db.refresh(db_image)  # Refresh to ensure we have the latest data
        
        return {
            "message": "Image uploaded successfully",
            "image_id": db_image.id,  # Use the database object's ID
            "metadata": metadata
        }
        
    except Exception as e:
        # Cleanup on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        db.rollback()
        print(f"Upload error: {str(e)}")  # Add logging for debugging
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metadata")
async def get_metadata():
    """Retrieve image metadata."""
    if processor.metadata is None:
        raise HTTPException(status_code=404, detail="No image loaded")
    return processor.metadata

@router.get("/slice")
async def get_slice(time: int = 0, z: int = 0, channel: int = 0):
    """Extract a specific slice from the image."""
    try:
        slice_data = processor.get_slice(time, z, channel)
        return {"slice_data": slice_data.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/analyze")
async def analyze_image(n_components: int = 3):
    """Run PCA on the image data."""
    try:
        reduced_data = processor.run_pca(n_components)
        return {"reduced_data": reduced_data.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/statistics")
async def get_statistics():
    """Return calculated image statistics."""
    try:
        if processor.image_data is None:
            raise HTTPException(status_code=404, detail="No image loaded")
        
        stats = processor.calculate_statistics()
        if stats is None:
            raise HTTPException(status_code=500, detail="Failed to calculate statistics")
            
        return stats
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/segment")
async def segment_image(time: int = 0, z: int = 0, channel: int = 0, method: str = 'otsu'):
    """Segment a specific channel."""
    try:
        segmented = processor.segment_channel(time, z, channel, method)
        return {"segmented_data": segmented.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/metadata/{image_id}")
async def get_metadata_by_id(
    image_id: str,
    db: Session = Depends(get_db)
):
    """Retrieve image metadata by ID."""
    image = db.query(ImageMetadata).filter(ImageMetadata.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image.image_metadata

@router.get("/slice/{image_id}")
async def get_slice_by_id(
    image_id: str,
    time: int = 0,
    z: int = 0,
    channel: int = 0,
    db: Session = Depends(get_db)
):
    """Extract a specific slice from the image by ID."""
    image = db.query(ImageMetadata).filter(ImageMetadata.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    processor = ImageProcessor()
    processor.load_image(image.file_path)
    
    try:
        slice_data = processor.get_slice(time, z, channel)
        return {"slice_data": slice_data.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze/{image_id}")
async def analyze_image_by_id(
    image_id: str,
    n_components: int = 3,
    background_tasks: BackgroundTasks = None
):
    """Run PCA analysis on the image."""
    # Implement PCA analysis
    pass

@router.get("/statistics/{image_id}")
async def get_statistics_by_id(
    image_id: str,
    channel: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get image statistics."""
    try:
        # Get image from database
        image = db.query(ImageMetadata).filter(ImageMetadata.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Load image and calculate statistics
        processor = ImageProcessor()
        processor.load_image(image.file_path)
        
        # Calculate statistics
        stats = processor.calculate_statistics()
        if stats is None:
            raise HTTPException(status_code=500, detail="Failed to calculate statistics")
        
        # If channel is specified, return only that channel's statistics
        if channel is not None:
            if channel >= processor.image_data.shape[2]:
                raise HTTPException(status_code=400, detail="Channel index out of range")
            
            channel_stats = {
                'mean': stats['mean'][channel],
                'std': stats['std'][channel],
                'min': stats['min'][channel],
                'max': stats['max'][channel]
            }
            return channel_stats
            
        return stats
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test_db")
async def test_db(db: Session = Depends(get_db)):
    """Test database connection"""
    try:
        # Try to query the database
        images = db.query(ImageMetadata).all()
        return {"message": "Database connection successful", "image_count": len(images)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")