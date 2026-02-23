import os
import cv2
import pandas as pd
from tqdm import tqdm

# === KONFIGURACJA ===
DATASET_ROOT = '/home/rumaxx/road-signs-project/ml/data/road_detection/road_detection'
OUTPUT_DIR = '/home/rumaxx/road-signs-project/ml/data/candidates_warning'

# Klasa w polskim datasecie, którą chcemy wyciąć
TARGET_CLASS_ID = 10 

def extract_candidates():
    print("=== EKSTRAKCJA KANDYDATÓW DO FINE-TUNINGU ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Ścieżki do treningowych
    images_dir = os.path.join(DATASET_ROOT, 'train', 'images')
    labels_dir = os.path.join(DATASET_ROOT, 'train', 'labels')
    
    # Lista plików
    image_files = os.listdir(images_dir)
    print(f"Przeszukiwanie {len(image_files)} zdjęć...")
    
    count = 0
    
    for img_file in tqdm(image_files):
        if not img_file.endswith(('.jpg', '.png')): continue
        
        # Znajdź odpowiadający plik txt
        label_file = img_file.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path): continue
        
        # Wczytaj obraz
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img, _ = img.shape
        
        # Czytaj etykiety
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            cls_id = int(parts[0])
            
            if cls_id == TARGET_CLASS_ID:
                cx, cy, bw, bh = map(float, parts[1:])
                
                # Konwersja na piksele
                w_box = int(bw * w_img)
                h_box = int(bh * h_img)
                x_center = int(cx * w_img)
                y_center = int(cy * h_img)
                
                x1 = int(x_center - w_box / 2)
                y1 = int(y_center - h_box / 2)
                x2 = x1 + w_box
                y2 = y1 + h_box
                
                # Clip (żeby nie wyjść poza obraz)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                # Wytnij
                crop = img[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Zapisz
                save_name = f"warning_{count}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), crop)
                count += 1

    print(f"\nGotowe! Wycięto {count} znaków ostrzegawczych.")
    print(f"Sprawdź folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_candidates()