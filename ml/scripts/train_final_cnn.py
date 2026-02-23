import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

# === KONFIGURACJA ===
DATA_DIR = '/home/rumaxx/road-signs-project/ml/data/full_dataset_mixed/Train' 
SAVE_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/final_model_v1.keras'

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 15

NUM_CLASSES = 44 

def train_final():
    print(f"=== START TRENINGU FINALNEGO (Oczekiwane klasy: {NUM_CLASSES}) ===")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.5, 1.3],
        channel_shift_range=40.0, 
        validation_split=0.2
    )

    print(f"Szukam danych w: {DATA_DIR}")
    
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training',
        seed=42
    )

    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        seed=42
    )
    

    print(f"Znalezione klasy: {list(train_generator.class_indices.keys())[:5]} ... (razem {len(train_generator.class_indices)})")
    
    if len(train_generator.class_indices) != NUM_CLASSES:
        print(f"!!! UWAGA !!! Wykryto {len(train_generator.class_indices)} folderów, ale w configu masz {NUM_CLASSES}.")
        print("Zaktualizuj NUM_CLASSES w kodzie!")
        return

    # Budowa Modelu
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    model.save(SAVE_PATH)
    print(f"Model zapisany jako: {SAVE_PATH}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"BŁĄD: Folder {DATA_DIR} nie istnieje.")
    else:
        train_final()