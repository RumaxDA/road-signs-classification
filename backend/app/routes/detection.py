from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
from app.services.model_loader import ModelService

router = APIRouter(prefix="/detection", tags=["AI Detection"])

@router.post("/predict")
async def predict_signs(file: UploadFile = File(...)):
    print(f"DEBUG: Otrzymano plik: {file.filename}, rozmiar: {file.size} bajtów")
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="To nie jest obraz")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Błąd dekodowania")

    # Pobieramy singleton modelu i robimy predykcję
    model = ModelService.get_model()
    results = model.predict(img)

    return {"count": len(results), "detections": results}