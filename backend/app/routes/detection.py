import time
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy import select
import cv2
import numpy as np
from app.services.traffic_logic import TrafficSignSystem
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.history import DetecionHistory
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
    db_record = DetecionHistory(
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
async def get_history(db: AsyncSession = Depends(get_db)):
    query = select(DetecionHistory).order_by(DetecionHistory.created_at.desc()).limit(50)
    result = await db.execute(query)
    records = result.scalars().all()

    return records