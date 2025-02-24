from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy.sql import func
from .database import Base

class ImageMetadata(Base):
    __tablename__ = "images"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    image_metadata = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, nullable=False)
    analysis_type = Column(String, nullable=False)
    result = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow) 