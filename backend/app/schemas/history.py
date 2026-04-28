from pydantic import BaseModel, JsonValue
from datetime import datetime

class History(BaseModel):
    id : int
    filename : str
    model_version: str
    inference_time_ms: float
    detected_count: int
    detections: JsonValue
    created_at: datetime

class UpdateHistory(BaseModel):
    model_version: str