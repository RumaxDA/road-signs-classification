import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import RandomFlip, RandomRotation, RandomZoom
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# === CONFIG ===
TRAIN_CSV_PATH = "/home/rumaxx/road-signs-project/ml/data/Train.csv"
TRAIN_ROOT_DIR = "/home/rumaxx/road-signs-project/ml/data"
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CATEGORIES = 43
BATCH_SIZE = 32
EPOCHS = 30 
SEED = 123

# === GPU CONFIG ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# === ARCHITEKTURA MOBILENET V2 ===
def build_mobilenet_model():
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = False

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Augmentacja (Taka sama jak w Custom CNN dla uczciwego porównania)
    x = RandomRotation(0.1)(inputs)
    x = RandomZoom(0.1)(x)
    x = tf.keras.layers.RandomContrast(0.1)(x)

    x = preprocess_input(x)

    # Przepuszczamy przez bazę
    x = base_model(x, training=False) 
    
    x = GlobalAveragePooling2D()(x) 
    x = Dropout(0.5)(x) # Dropout dla redukcji overfittingu
    x = Dense(256, activation='relu')(x)
    outputs = Dense(NUM_CATEGORIES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# === PREPROCESSING ===
def load_and_preprocess_image(path, x1, y1, x2, y2, label):
    file_content = tf.io.read_file(path)
    img = tf.io.decode_image(file_content, channels=3, expand_animations=False)
    
    shape = tf.shape(img)
    img_h, img_w = shape[0], shape[1]

    x1 = tf.clip_by_value(tf.cast(x1, tf.int32), 0, img_w - 1)
    y1 = tf.clip_by_value(tf.cast(y1, tf.int32), 0, img_h - 1)
    x2 = tf.clip_by_value(tf.cast(x2, tf.int32), x1 + 1, img_w)
    y2 = tf.clip_by_value(tf.cast(y2, tf.int32), y1 + 1, img_h)

    # Slicing
    img_cropped = img[y1:y2, x1:x2, :]

    # Resize with pad
    img_resized = tf.image.resize_with_pad(img_cropped, IMG_HEIGHT, IMG_WIDTH, method='bicubic')
    
    img_resized = tf.cast(img_resized, tf.float32)
    img_resized.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    
    return img_resized, label

def create_dataset(df, is_train=True):
    paths = [os.path.join(TRAIN_ROOT_DIR, p) for p in df["Path"]]
    labels = df["ClassId"].values
    rois = df[['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']].values

    ds = tf.data.Dataset.from_tensor_slices((
        paths, rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], labels
    ))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        ds = ds.shuffle(buffer_size=1000)
    
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    print("=== START TRANSFER LEARNING (MobileNetV2) ===")
    
    df = pd.read_csv(TRAIN_CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['ClassId'])
    
    train_ds = create_dataset(train_df, is_train=True)
    val_ds = create_dataset(val_df, is_train=False)

    model = build_mobilenet_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trained_models')
    os.makedirs(save_dir, exist_ok=True)
    
    # Zapis jako mobilenet_v2.keras
    model_path = os.path.join(save_dir, 'mobilenet_v2_best.keras')

    callbacks = [
        ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
    ]

    print("Rozpoczynam trening...")
    try:
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=callbacks 
        )
    except KeyboardInterrupt:
        print("\nPrzerwano trening ręcznie.")

    # Zapis historii
    history_path = os.path.join(save_dir, 'mobilenet_history.json')
    if 'history' in locals():
        history_dict = {key: [float(val) for val in value_list] for key, value_list in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f)
    
    print("KONIEC TRANSFER LEARNINGU.")