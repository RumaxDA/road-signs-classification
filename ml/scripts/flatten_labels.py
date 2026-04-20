import os
import glob

# === ŚCIEŻKA DO TWOJEGO ZBIORU GTSDB ===
DATASET_PATH = "/home/rumaxx/road-signs-project/ml/data/GTSDB"

def flatten_yolo_labels():
    print("=== ROZPOCZYNAM SPŁASZCZANIE ETYKIET DO 1 KLASY ===")
    
    txt_files = glob.glob(os.path.join(DATASET_PATH, "**", "*.txt"), recursive=True)
    
    if not txt_files:
        print("BŁĄD: Nie znaleziono żadnych plików .txt. Sprawdź ścieżkę DATASET_PATH!")
        return

    files_modified = 0

    for txt_file in txt_files:
        if "readme" in txt_file.lower() or "classes" in txt_file.lower():
            continue

        with open(txt_file, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5: 
                parts[0] = "0" 
                new_lines.append(" ".join(parts) + "\n")
        
        if new_lines:
            with open(txt_file, "w") as f:
                f.writelines(new_lines)
            files_modified += 1

    print(f"Zakończono sukcesem! Przetworzono i nadpisano {files_modified} plików .txt.")
    print("Wszystkie znaki na zdjęciach to teraz klasa '0'. Można odpalać YOLO.")

if __name__ == "__main__":
    flatten_yolo_labels()