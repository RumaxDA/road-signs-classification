from app.services.traffic_logic import TrafficSignSystem
from app.core.config import settings

class ModelService:
    _instance = None

    @classmethod
    def get_model(cls):
        if cls._instance is None:
            yolo_path = settings.YOLO_MODEL_PATH
            cnn_path = settings.CNN_MODEL_PATH
            cls._instance = TrafficSignSystem(yolo_path, cnn_path)
        return cls._instance
    

