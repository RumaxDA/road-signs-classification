import os
import cv2
import time
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque, Counter
import uuid

# === KONFIGURACJA ===
VIDEO_SOURCE = '/home/rumaxx/road-signs-project/ml/videos/test.mp4' 
# VIDEO_SOURCE = 0  # Odkomentuj dla kamery

OUTPUT_FILENAME = 'processed_output.mp4'

YOLO_MODEL_PATH = '/home/rumaxx/road-signs-project/runs/detect/yolo_polish_signs/weights/best.pt'
CNN_MODEL_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/best_model.keras'

YOLO_CONFIDENCE = 0.25 
CNN_MIN_CONFIDENCE = 60.0 

# Ignorowane klasy (Car, Lights, Pedestrian, Truck)
IGNORED_CLASSES = [0, 2, 3, 4, 7, 9]

CLASSES = {
    0: 'Ograniczenie predkosci (20km/h)', 1: 'Ograniczenie predkosci (30km/h)', 
    2: 'Ograniczenie predkosci (50km/h)', 3: 'Ograniczenie predkosci (60km/h)', 
    4: "Ograniczenie predkosci (70km/h)", 5: "Ograniczenie predkosci (80km/h)", 
    6: "Koniec ograniczenia predkosci (80km/h)", 7: "Ograniczenie predkosci (100km/h)", 
    8: "Ograniczenie predkosci (120km/h)", 9: "Zakaz wyprzedzania", 
    10: "Zakaz wyprzedzania przez pojazdy ciężarowe", 11: "Skrzyżowanie z drogą podporządkowaną", 
    12: "Droga z pierwszeństwem", 13: "Ustąp pierwszeństwa", 14: "Stop", 
    15: "Zakaz ruchu", 16: "Zakaz wjazdu pojazdów ciężarowych", 17: "Zakaz wjazdu", 
    18: "Inne niebezpieczeństwo", 19: "Niebezpieczny zakręt w lewo", 
    20: "Niebezpieczny zakręt w prawo", 21: "Podwójny zakręt, pierwszy w lewo", 
    22: "Nierówna droga", 23: "Śliska jezdnia", 24: "Zagrożenie zwężeniem jezdni - prawostronne", 
    25: "Roboty drogowe", 26: "Sygnalizacja świetlna", 27: "Przejście dla pieszych", 
    28: "Dzieci", 29: "Rowerzyści", 30: "Oszronienie jezdni", 
    31: "Dzikie zwierzęta", 32: "Koniec zakazów", 33: "Nakaz jazdy w prawo", 
    34: "Nakaz jazdy w lewo", 35: "Nakaz jazdy prosto", 36: "Nakaz jazdy prosto lub w prawo", 
    37: "Nakaz jazdy prosto lub w lewo", 38: "Nakaz jazdy z prawej strony znaku", 
    39: "Nakaz jazdy z lewej strony znaku", 40: "Rondo", 
    41: "Koniec zakazu wyprzedzania", 42: "Koniec zakazu wyprzedzania przez pojazdy ciężarowe",
}
def run_video_pipeline():
    print("=== START WIDEO PIPELINE (ZE STABILIZACJĄ) ===")

    # 1. Ładowanie Modeli
    print("1. Ładowanie modeli...")
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"BŁĄD: Brak modelu YOLO w {YOLO_MODEL_PATH}"); return
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    try:
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        print("   -> Modele załadowane.")
    except Exception as e:
        print(f"BŁĄD CNN: {e}"); return

    # 2. INICJALIZACJA WIDEO (Tego brakowało!)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"BŁĄD: Nie można otworzyć źródła wideo: {VIDEO_SOURCE}")
        return

    # Pobranie parametrów wideo
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    # Konfiguracja zapisu
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'videos', OUTPUT_FILENAME)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print(f"   -> Start przetwarzania: {width}x{height} @ {fps} FPS")
    
    # Folder na debug
    os.makedirs('debug_crops', exist_ok=True)

    # === HISTORIA DETEKCJI (Dla stabilizacji) ===
    detection_history = {} 
    HISTORY_LEN = 5 

    frame_count = 0
    start_time = time.time()

    # === GŁÓWNA PĘTLA ===
    while True:
        ret, frame = cap.read()
        if not ret: break 

        frame_count += 1
        
        # --- DETEKCJA YOLO ---
        results = yolo_model(frame, conf=YOLO_CONFIDENCE, verbose=False)
        detections = results[0].boxes

        # --- PRZETWARZANIE OBIEKTÓW ---
        for i, box in enumerate(detections):
            cls_id = int(box.cls[0])
            if cls_id in IGNORED_CLASSES: continue

            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            cropped_sign = frame[y1:y2, x1:x2]
            if cropped_sign.size == 0: continue

            # --- CNN ---
            cropped_rgb = cv2.cvtColor(cropped_sign, cv2.COLOR_BGR2RGB)
            tensor_crop = tf.convert_to_tensor(cropped_rgb, dtype=tf.float32)
            tensor_resized = tf.image.resize_with_pad(tensor_crop, 224, 224, method='bicubic')
            tensor_clipped = tf.clip_by_value(tensor_resized, 0.0, 255.0)
            input_batch = tf.expand_dims(tensor_clipped, 0)

            preds = cnn_model.predict(input_batch, verbose=0)
            pred_id = np.argmax(preds)
            confidence = np.max(preds) * 100

            # Surowy wynik
            raw_label = CLASSES.get(pred_id, 'Unknown')

            # --- STABILIZACJA WYNIKU (VOTING) ---
            center_x, center_y = (x1+x2)//2, (y1+y2)//2
            matched_key = None
            
            # Tracker pozycyjny
            for key in list(detection_history.keys()):
                hx, hy = key
                if abs(hx - center_x) < 50 and abs(hy - center_y) < 50:
                    matched_key = key
                    break
            
            if matched_key:
                dq = detection_history.pop(matched_key)
                dq.append(raw_label)
                detection_history[(center_x, center_y)] = dq
                # Głosowanie
                most_common = Counter(dq).most_common(1)[0][0]
                final_label = most_common
            else:
                dq = deque(maxlen=HISTORY_LEN)
                dq.append(raw_label)
                detection_history[(center_x, center_y)] = dq
                final_label = raw_label

            # --- DEBUGOWANIE: ZAPISZ CROP ---
            if confidence > 10: 
                unique_id = str(uuid.uuid4())[:8]
                safe_label = raw_label.split('(')[0].strip().replace(' ', '_')
                debug_filename = f"debug_crops/{safe_label}_conf{int(confidence)}_{unique_id}.jpg"
                cv2.imwrite(debug_filename, cropped_sign)

            # FILTR 2: Wyświetlanie
            if confidence < CNN_MIN_CONFIDENCE: continue

            # --- RYSOWANIE ---
            # Wyświetlamy UŚREDNIONĄ etykietę i BIEŻĄCĄ pewność
            label_text = f"{final_label} ({confidence:.0f}%)"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # --- LICZNIK FPS ---
        fps_realtime = frame_count / (time.time() - start_time)
        if (time.time() - start_time) > 0:
             cv2.putText(frame, f"FPS: {fps_realtime:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)
        if frame_count % 30 == 0: print(f"Przetworzono klatkę {frame_count}...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nGOTOWE! Film zapisano w: {save_path}")

if __name__ == "__main__":
    run_video_pipeline()