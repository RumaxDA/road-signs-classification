import os
import tensorflow as tf
from keras.models import load_model, Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

# === KONFIGURACJA ===
DATA_DIR = '/home/rumaxx/road-signs-project/ml/data/polish_finetune'
MODEL_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/best_model.keras'
SAVE_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/best_model_finetuned_v2.keras'

BATCH_SIZE = 8
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 44 

def ensure_all_folders_exist():
    """Tworzy puste foldery 0-43."""
    print("Sprawdzanie struktury folderów...")
    for i in range(NUM_CLASSES):
        cls_folder = os.path.join(DATA_DIR, str(i))
        os.makedirs(cls_folder, exist_ok=True)

def modify_model_architecture(model):
    print("Modyfikacja architektury modelu (metoda Sequential)...")

    model.pop()
    

    model.add(Dense(NUM_CLASSES, activation='softmax', name='new_output_44'))
    
    return model

def finetune():
    print(f"=== START FINE-TUNINGU (Z KLASĄ ŚMIETNIK: {NUM_CLASSES-1}) ===")
    
    ensure_all_folders_exist()
    class_names_list = [str(i) for i in range(NUM_CLASSES)]

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        brightness_range=[0.5, 1.3], 
        channel_shift_range=40.0, 
        validation_split=0.2         
    )

    print("Generowanie danych...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',        
        classes=class_names_list,
        subset='training',
        seed=123
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=class_names_list,
        subset='validation',
        seed=123
    )

    # Wczytaj stary model
    print(f"Wczytywanie modelu bazowego: {MODEL_PATH}")
    original_model = load_model(MODEL_PATH)

    model = modify_model_architecture(original_model)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Rozpoczynam douczanie nowej warstwy...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15
    )

    model.save(SAVE_PATH)
    print(f"Zapisano model 44-klasowy: {SAVE_PATH}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"BŁĄD: Folder {DATA_DIR} nie istnieje.")
    else:
        finetune()