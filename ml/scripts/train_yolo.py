from ultralytics import YOLO
import os

# === KONFIGURACJA ===
YAML_PATH = '/home/rumaxx/road-signs-project/ml/YOLO/YOLOv2-GTSDB/sign_detector.yaml'
MODEL_NAME = 'yolov8n.pt' 

def train_universal_detector():
    print("=== START TRENINGU: UNIWERSALNY DETEKTOR ZNAKÓW ===")
    
    if not os.path.exists(YAML_PATH):
        print(f"BŁĄD: Nie znaleziono pliku {YAML_PATH}!")
        return

    model = YOLO(MODEL_NAME) 


    model.train(
        data=YAML_PATH,
        epochs=70,
        imgsz=640,
        batch=16,
        project='/home/rumaxx/road-signs-project/ml/YOLO/YOLOv2-GTSDB', 
        name='yolo_universal_sign_det',
        device=0,
        patience=15,
        box=7.5,
        cls=0.5,
        augment=True,
        close_mosaic=10,
        optimizer='AdamW',
        lr0=0.001
    )
    
    print("\n=== TRENING ZAKOŃCZONY ===")
    print("Szukaj wag w: runs/detect/yolo_universal_sign_det/weights/best.pt")

if __name__ == "__main__":
    train_universal_detector()