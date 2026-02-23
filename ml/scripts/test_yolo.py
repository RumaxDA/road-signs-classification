from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

MODEL_PATH = '/home/rumaxx/road-signs-project/runs/detect/yolo_polish_final/weights/best.pt'
IMAGE_PATH = '/home/rumaxx/road-signs-project/ml/images/sign21.jpg' 

if not os.path.exists(MODEL_PATH):
    print(f"BŁĄD: Nie ma pliku {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)

print(f"Testowanie modelu na: {os.path.basename(IMAGE_PATH)}")
results = model(IMAGE_PATH, conf=0.25)

for result in results:
    img_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.title("Wynik Nowego YOLO (100 epok)")
    plt.axis('off')
    plt.show()
    
    if len(result.boxes) == 0:
        print("SUKCES! YOLO nie wykryło żadnego obiektu (Most jest bezpieczny).")
    else:
        print(f"YOLO wykryło {len(result.boxes)} obiektów:")
        for box in result.boxes:
            print(f"- Klasa: {int(box.cls[0])}, Pewność: {float(box.conf[0]):.2f}")