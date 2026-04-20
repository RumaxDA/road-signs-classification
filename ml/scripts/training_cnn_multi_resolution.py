import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, 
                          Activation, Dense, Dropout, GlobalAveragePooling2D, 
                          RandomRotation, RandomZoom, RandomTranslation, RandomContrast)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

# === CONFIG ===
TRAIN_CSV_PATH = "/home/rumaxx/road-signs-project/ml/data/Train.csv"
ROOT_DIR = "/home/rumaxx/road-signs-project/ml/data"
SAVE_DIR = "/home/rumaxx/road-signs-project/ml/new_trained_models/experiments"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CLASSES = 43
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42
LR = 1e-3

RESOLUTIONS = [224] 

def load_and_preprocess(path, x1, y1, x2, y2, label, img_size):
    file_content = tf.io.read_file(path)
    img = tf.io.decode_image(file_content, channels=3, expand_animations=False)
    img = img[y1:y2, x1:x2, :]
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def create_dataset(df, img_size, training=True):
    paths = [os.path.join(ROOT_DIR, p) for p in df["Path"]]
    labels = df["ClassId"].values
    rois = df[['Roi.X1','Roi.Y1','Roi.X2','Roi.Y2']].values

    ds = tf.data.Dataset.from_tensor_slices((paths, rois[:,0], rois[:,1], rois[:,2], rois[:,3], labels))
    ds = ds.map(lambda p, x1, y1, x2, y2, l: load_and_preprocess(p, x1, y1, x2, y2, l, img_size), 
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_cnn(img_size):
    inputs = Input(shape=(img_size, img_size, 3))
    x = RandomRotation(0.05)(inputs)
    x = RandomZoom(0.1)(x)
    x = RandomTranslation(0.1, 0.1)(x)
    x = RandomContrast(0.1)(x)

    def conv_block(x, filters):
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # Blok 1
    x = conv_block(x, 32)
    x = conv_block(x, 32)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Blok 2
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Blok 3 - tylko jeśli obraz jest wystarczająco duży
    if img_size >= 48:
        x = conv_block(x, 128)
        x = conv_block(x, 128)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.4)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    return Model(inputs, outputs)

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_CSV_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["ClassId"], random_state=SEED)

    for res in RESOLUTIONS:
        print(f"\n{'='*20}\nSTART TRAINING: {res}x{res}\n{'='*20}")
        
        m_path = os.path.join(SAVE_DIR, f"cnn_{res}_v1.keras")
        h_path = os.path.join(SAVE_DIR, f"cnn_{res}_history_v1.json")

        train_ds = create_dataset(train_df, res, True)
        val_ds = create_dataset(val_df, res, False)

        model = build_cnn(res)
        model.compile(optimizer=tf.keras.optimizers.Adam(LR), 
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        callbacks = [
            ModelCheckpoint(filepath=m_path, monitor="val_accuracy", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
        ]

        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
        
        with open(h_path, "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)
            
    print("\nWSZYSTKIE EKSPERYMENTY ZAKOŃCZONE.")