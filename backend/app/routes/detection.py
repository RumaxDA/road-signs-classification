import time
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query
from sqlalchemy import select, func, update
import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.history import DetectionHistory
from app.services.model_manager import ModelVersion, model_manager
from app.schemas.history import History, UpdateHistory

router = APIRouter(prefix="/detection", tags=["AI Detection"])

@router.post("/predict")
async def predict_signs(
    file: UploadFile = File(...),
    model_version: ModelVersion = Query(ModelVersion.CNN_48),
    db: AsyncSession = Depends(get_db)
    ):
    # Odczyt 
    active_system = model_manager.get_model(model_version)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Start czas
    start_time = time.perf_counter()

    # Detekcja
    detections = active_system.predict(frame)

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

@router.get("/history_list", response_model = list[History])
async def get_all_history(db: AsyncSession= Depends(get_db)):
    query = select(DetectionHistory).order_by(DetectionHistory.id)
    result = await db.execute(query)
    return result.scalars().all()

@router.get("/history/{history_id}", response_model = History)
async def get_single_history(history_id : int, db: AsyncSession = Depends(get_db)):
    query = await db.execute(select(DetectionHistory).where(DetectionHistory.id == history_id))
    return query.scalars().first()

@router.delete("/delete/{history_id}")
async def delete_history(history_id: int,  db: AsyncSession = Depends(get_db)):
    result = await db.get(DetectionHistory, history_id)
    if not result:
        raise HTTPException(f"Wrong history_id: {history_id}")

    await db.delete(result)

    await db.commit()
    return {"message": f"Deleted history {history_id}"}

@router.put("update/{history_id}")
async def update_history(history_id: int, update_history: UpdateHistory, db: AsyncSession = Depends(get_db)):
    query = select(DetectionHistory).where(DetectionHistory.id == history_id)
    result = await db.execute(query)
    user = result.scalars().first()

    if not user:
        raise HTTPException(f"Wrong history id: {history_id}")

    update_data = update_history.model_dump() 
    for field, value in update_data.items():
        setattr(user, field, value)
    

    await db.commit()
    await db.refresh(user)
    return {"message": "Data updated successfully"}



