import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ================== KONFIGURACJA ==================

IMAGE_PATH = '/home/rumaxx/road-signs-project/ml/images/sign20.jpg'
CNN_MODEL_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/final_model_v1.keras'



MIN_CONFIDENCE = 50.0  # CNN

# ================== KLASY CNN ==================

CLASSES = {
    0: 'Ograniczenie prędkości (20km/h)', 1: 'Ograniczenie prędkości (30km/h)',
    2: 'Ograniczenie prędkości (50km/h)', 3: 'Ograniczenie prędkości (60km/h)',
    4: 'Ograniczenie prędkości (70km/h)', 5: 'Ograniczenie prędkości (80km/h)',
    6: 'Koniec ograniczenia prędkości (80km/h)', 7: 'Ograniczenie prędkości (100km/h)',
    8: 'Ograniczenie prędkości (120km/h)', 9: 'Zakaz wyprzedzania',
    10: 'Zakaz wyprzedzania przez pojazdy ciężarowe', 11: 'Skrzyżowanie z drogą podporządkowaną',
    12: 'Droga z pierwszeństwem', 13: 'Ustąp pierwszeństwa', 14: 'Stop',
    15: 'Zakaz ruchu', 16: 'Zakaz wjazdu pojazdów ciężarowych', 17: 'Zakaz wjazdu',
    18: 'Inne niebezpieczeństwo', 19: 'Niebezpieczny zakręt w lewo',
    20: 'Niebezpieczny zakręt w prawo', 21: 'Podwójny zakręt, pierwszy w lewo',
    22: 'Nierówna droga', 23: 'Śliska jezdnia',
    24: 'Zwężenie jezdni – prawostronne', 25: 'Roboty drogowe',
    26: 'Sygnalizacja świetlna', 27: 'Przejście dla pieszych',
    28: 'Dzieci', 29: 'Rowerzyści',
    30: 'Oszronienie jezdni', 31: 'Dzikie zwierzęta',
    32: 'Koniec zakazów', 33: 'Nakaz jazdy w prawo',
    34: 'Nakaz jazdy w lewo', 35: 'Nakaz jazdy prosto',
    36: 'Nakaz jazdy prosto lub w prawo', 37: 'Nakaz jazdy prosto lub w lewo',
    38: 'Nakaz jazdy z prawej strony znaku', 39: 'Nakaz jazdy z lewej strony znaku',
    40: 'Rondo', 41: 'Koniec zakazu wyprzedzania',
    42: 'Koniec zakazu wyprzedzania ciężarowe',
    43: 'NIE_ZNAK (TŁO)'
}

# ================== PIPELINE ==================

def run_pipeline():
    print("=== START PIPELINE: YOLO (Traffic Signs) → CNN ===")

    # 1. YOLO
    print("1. Pobieranie wag YOLO z HuggingFace...")
    detector = YOLO("keremberke/yolov8n-gtsrb")

    # 2. CNN
    print("2. Ładowanie CNN...")
    classifier = tf.keras.models.load_model(CNN_MODEL_PATH)

    # 3. Obraz
    original_img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # 4. Detekcja
    results = detector(original_img, conf=0.25, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("Brak detekcji YOLO.")
        plt.imshow(img_rgb); plt.axis("off"); plt.show()
        return

    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    ax = plt.gca()

    print(f"YOLO wykryło {len(boxes)} obiektów")

    # 5. Klasyfikacja
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        tensor = tf.image.resize_with_pad(crop, 224, 224)
        tensor = tf.expand_dims(tf.cast(tensor, tf.float32), 0)

        preds = classifier.predict(tensor, verbose=0)
        class_id = int(np.argmax(preds))
        conf = float(np.max(preds) * 100)

        if class_id == 43 or conf < MIN_CONFIDENCE:
            continue

        label = CLASSES[class_id]
        print(f"[OK] {label} ({conf:.1f}%)")

        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                              fill=False, color='lime', linewidth=3)
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, f"{label} ({conf:.0f}%)",
                color='yellow', fontsize=11,
                bbox=dict(facecolor='black', alpha=0.7))

    plt.axis("off")
    plt.title("YOLO Traffic Signs + Custom CNN")
    plt.savefig("result_final.jpg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_pipeline()
