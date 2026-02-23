from ultralytics import YOLO
import os

YAML_PATH = '/home/rumaxx/road-signs-project/ml/data/polish_traffic.yaml'

def train_yolo():
    print("=== START TRENINGU YOLO (WERSJA PRECYZYJNA) ===")
    
    model = YOLO('yolov8n.pt') 

    # Trening
    results = model.train(
        data=YAML_PATH,
        epochs=100,         
        imgsz=640,          
        batch=16,           
        name='yolo_polish_final', 
        device=0,           
        patience=15,       
        verbose=True,
        

        close_mosaic=10,    
        
        optimizer='AdamW',
        lr0=0.001
    )
    
    print("Trening zakończony.")
    print(f"Najlepszy model: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    if not os.path.exists(YAML_PATH):
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono pliku {YAML_PATH}")
        print("Sprawdź czy ścieżka jest poprawna!")
        exit()
        
    train_yolo()