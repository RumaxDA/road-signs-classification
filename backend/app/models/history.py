from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base

class DetectionHistory(Base):
    __tablename__ = "detection_history"

    id = Column(Integer, primary_key= True, index = True)
    filename = Column(String, index = True)
    model_version = Column(String, index = True)
    inference_time_ms = Column(Float)
    detected_count = Column(Integer)
    detections = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    