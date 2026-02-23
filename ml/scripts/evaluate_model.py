import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# === KONFIGURACJA GPU ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Znaleziono {len(gpus)} GPU – memory growth włączony.")
    except RuntimeError as e:
        print(e)

# === STAŁE ===
NUM_CATEGORIES = 43
TEST_CSV_PATH = '/home/rumaxx/road-signs-project/ml/data/Test.csv'
TEST_ROOT_DIR = '/home/rumaxx/road-signs-project/ml/data'
PLOTS_DIR = '/home/rumaxx/road-signs-project/ml/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
BATCH_SIZE = 32
IMG_HEIGHT, IMG_WIDTH = 224, 224

# === WYBÓR MODELU ===
MODEL_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/best_model.keras'
HISTORY_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/training_history.json'

# Opcja 2: MobileNetV2 
# MODEL_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/mobilenet_v2_best.keras'
# HISTORY_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/mobilenet_history.json'


# === FUNKCJE POMOCNICZE (WYKRESY) ===
def plot_history(history, name):
    acc = history.get('accuracy', history.get('acc', []))
    val_acc = history.get('val_accuracy', history.get('val_acc', []))
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])

    if not acc:
        print("Pusta historia lub zły format kluczy.")
        return

    plt.figure(figsize=(12, 5))
    
    # Wykres Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title(f"{name} - Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    
    # Wykres Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f"{name} - Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"{name}_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Zapisano wykres historii: {save_path}")

def plot_conf_mat(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(NUM_CATEGORIES), yticklabels=range(NUM_CATEGORIES))
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title(f"{name} - Confusion Matrix", fontsize=20)
    
    save_path = os.path.join(PLOTS_DIR, f"{name}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Zapisano macierz pomyłek: {save_path}")

# === PREPROCESSING ===
def load_crop_and_preprocess(path, x1, y1, x2, y2, label):
    # 1. Wczytanie
    file_content = tf.io.read_file(path)
    img = tf.io.decode_image(file_content, channels=3, expand_animations=False)
    
    # 2. Bezpieczne koordynaty (Slicing Logic)
    shape = tf.shape(img)
    src_h, src_w = shape[0], shape[1]

    x1 = tf.clip_by_value(tf.cast(x1, tf.int32), 0, src_w - 1)
    y1 = tf.clip_by_value(tf.cast(y1, tf.int32), 0, src_h - 1)
    x2 = tf.clip_by_value(tf.cast(x2, tf.int32), x1 + 1, src_w)
    y2 = tf.clip_by_value(tf.cast(y2, tf.int32), y1 + 1, src_h)

    # 3. Wycięcie (Slicing)
    img_cropped = img[y1:y2, x1:x2, :]

    # 4. Resize with Pad 
    img_resized = tf.image.resize_with_pad(img_cropped, IMG_HEIGHT, IMG_WIDTH, method='bicubic')
    
    # 5. Konwersja na float32 [0-255]
    img_final = tf.cast(img_resized, tf.float32)
    img_final = tf.clip_by_value(img_final, 0.0, 255.0)

    return img_final, label

# === MAIN ===
if __name__ == "__main__":
    
    filename = os.path.basename(MODEL_PATH)
    print(f"=== ROZPOCZYNAM EWALUACJĘ MODELU: {filename} ===")

    # Automatyczne nadawanie nazwy do wykresów
    if 'mobilenet' in filename.lower():
        model_display_name = "MobileNetV2"
    elif 'best_model' in filename.lower():
        model_display_name = "Custom_CNN"
    else:
        model_display_name = "Unknown_Model"
        print("⚠️ Uwaga: Nierozpoznana nazwa pliku, używam domyślnych ustawień.")

    # 1. Wczytanie CSV
    print('Wczytywanie ścieżek z Test.csv...')
    try:
        df = pd.read_csv(TEST_CSV_PATH)
        df = df.sort_values(by=['Path']) 
        
        image_paths = [os.path.join(TEST_ROOT_DIR, p) for p in df["Path"]]
        y_true = df["ClassId"].values
        rois = df[['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']].values
        print(f"Załadowano {len(y_true)} próbek testowych.")
    except Exception as e:
        print(f"Błąd wczytywania {TEST_CSV_PATH}: {e}"); exit()

    # 2. Tworzenie Datasetu
    print("Tworzenie generatora danych testowych...")
    test_ds = tf.data.Dataset.from_tensor_slices((
        image_paths, 
        rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], 
        y_true
    ))
    
    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.map(load_crop_and_preprocess, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # 3. Wczytywanie Modelu
    print(f'Wczytywanie modelu z {MODEL_PATH}...')
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model załadowany pomyślnie.")
    except Exception as e:
        print(f"KRYTYCZNY BŁĄD: Nie można wczytać modelu.\n{e}"); exit()

    # 4. Wczytaj Historię (do wykresów)
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
                plot_history(history, model_display_name)
        except Exception as e:
            print(f"Nie udało się wczytać historii: {e}")
    else:
        print(f"Brak pliku historii: {HISTORY_PATH}")

    # 5. Ewaluacja (Loss & Accuracy)
    print("\n=== WYNIKI NA ZBIORZE TESTOWYM ===")
    results = model.evaluate(test_ds)
    print(f"Test Loss:     {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")

    # 6. Predykcje i Czas
    print("\nGenerowanie predykcji...")
    t0 = time.perf_counter()
    y_pred_probs = model.predict(test_ds)
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    per_image_ms = (total_time / len(y_true)) * 1000
    print(f"Średni czas inferencji: {per_image_ms:.3f} ms / zdjęcie")

    y_pred = np.argmax(y_pred_probs, axis=1)

    # 7. Raport Klasyfikacji
    print("\n=== Szczegółowy Raport Klasyfikacji ===")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    # 8. Macierz Pomyłek
    plot_conf_mat(y_true, y_pred, model_display_name)

    print("\nEwaluacja zakończona sukcesem.")