import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from app.core.config import Settings

class TrafficSignSystem:
    def __init__(self, yolo_path, cnn_path):
        print("--- Inicjalizacja systemów rozpoznawania ---")
        self.detector = YOLO(yolo_path)
        self.classifier = tf.keras.models.load_model(cnn_path)
        self.img_size = 48
        self.min_confidence = 0.5 

        self.classes = {
            0: 'Ograniczenie prędkości (20km/h)', 1: 'Ograniczenie prędkości (30km/h)', 
            2: 'Ograniczenie prędkości (50km/h)', 3: 'Ograniczenie prędkości (60km/h)', 
            4: "Ograniczenie prędkości (70km/h)", 5: "Ograniczenie prędkości (80km/h)", 
            6: "Koniec ograniczenia prędkości (80km/h)", 7: "Ograniczenie prędkości (100km/h)", 
            8: "Ograniczenie prędkości (120km/h)", 9: "Zakaz wyprzedzania", 
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

    def preprocess_for_cnn(self, cropped_img):
        """Przygotowanie wyciętego znaku pod wejście CNN."""
        img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, frame):
        """Główna metoda przetwarzająca obraz."""
        results_list = []
        h, w, _ = frame.shape

        # 1. Detekcja YOLO
        yolo_results = self.detector(frame, conf=0.1, verbose=True)
        detections = yolo_results[0].boxes

        for box in detections:
            # Koordynaty
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            # Zabezpieczenie wymiarów
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            box_w, box_h = x2 - x1, y2 - y1

            # --- DEBUG: co wycięło YOLO ---
            print(f"DEBUG: YOLO znalazło obiekt: {x1, y1, x2, y2}")
            

            if box_h == 0 or box_w == 0: continue

            # Prosty filtr proporcji 
            aspect_ratio = box_w / box_h
            if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                continue

            # 2. Wycięcie i Klasyfikacja CNN
            cropped = frame[y1:y2, x1:x2]
            input_batch = self.preprocess_for_cnn(cropped)
            
            prediction = self.classifier.predict(input_batch, verbose=0)
            class_id = np.argmax(prediction)
            confidence = float(np.max(prediction))

            # 3. Filtr pewności
            if confidence >= self.min_confidence:
                results_list.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": int(class_id),
                    "label": self.classes.get(class_id, f"ID {class_id}"),
                    "confidence": round(confidence, 2)
                })

        return results_list

if __name__ == "__main__":
    YOLO_PATH = Settings.YOLO_MODEL_PATH
    CNN_PATH = Settings.CNN_MODEL_PATH
    
    system = TrafficSignSystem(YOLO_PATH, CNN_PATH)
    
    test_img = cv2.imread('ml/others/sign33.jpg')
    if test_img is not None:
        predictions = system.predict(test_img)
        print("\nWyniki rozpoznawania (JSON):")
        print(predictions)