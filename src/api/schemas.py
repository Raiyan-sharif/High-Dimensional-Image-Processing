from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class ImageMetadataSchema(BaseModel):
    dimensions: List[int]
    dtype: str
    size_bytes: int
    shape_description: Dict[str, int]

class SliceRequest(BaseModel):
    time: int = 0
    z: int = 0
    channel: int = 0

class AnalysisRequest(BaseModel):
    n_components: int = 3
    method: str = "pca"

class AnalysisResponse(BaseModel):
    image_id: int
    analysis_type: str
    result: Dict
    created_at: datetime

    class Config:
        orm_mode = True 