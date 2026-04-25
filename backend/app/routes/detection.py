import time
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy import select, func
import cv2
import numpy as np
from app.services.traffic_logic import TrafficSignSystem
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.history import DetectionHistory
from app.core.config import settings

system = TrafficSignSystem(settings.YOLO_MODEL_PATH, settings.CNN_MODEL_PATH)

router = APIRouter(prefix="/detection", tags=["AI Detection"])

@router.post("/predict")
async def predict_signs(
    file: UploadFile = File(...),
    model_version: str = "CNN_48_v1",
    db: AsyncSession = Depends(get_db)
    ):
    # Odczyt 
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Start czas
    start_time = time.perf_counter()

    # Detekcja
    detections = system.predict(frame)

    # Stop czas i konwersja na ms
    end_time = time.perf_counter()
    inference_time_ms = round((end_time - start_time) * 1000, 2)

    # Zapis do bazy
    db_record = DetectionHistory(
        filename = file.filename,
        model_version = model_version,
        inference_time_ms = inference_time_ms,
        detected_count = len(detections),
        detections = detections
    )

    db.add(db_record)
    await db.commit()

    return {
        "count": len(detections),
        "inference_time_ms": inference_time_ms,
        "detections": detections
    }

@router.get("/history")
async def get_history(
    page: int = 1, 
    size: int = 10, 
    sort_by: str = "created_at", 
    order: str = "desc",       
    db: AsyncSession = Depends(get_db)
):
    offset = (page - 1) * size
    
    allowed_columns = {
        "id": DetectionHistory.id,
        "filename": DetectionHistory.filename,
        "model_version": DetectionHistory.model_version,
        "inference_time_ms": DetectionHistory.inference_time_ms,
        "detected_count": DetectionHistory.detected_count,
        "created_at": DetectionHistory.created_at
    }

    column = allowed_columns.get(sort_by, DetectionHistory.created_at)
    sort_func = column.desc() if order == "desc" else column.asc()

    count_query = select(func.count(DetectionHistory.id))
    total_result = await db.execute(count_query)
    total_count = total_result.scalar()

    query = select(DetectionHistory).order_by(sort_func).limit(size).offset(offset)
    result = await db.execute(query)
    records = result.scalars().all()

    return {
        "items": records,
        "total": total_count,
        "page": page,
        "size": size,
        "pages": (total_count + size - 1) // size,
        "sort_by": sort_by,
        "order": order
    }