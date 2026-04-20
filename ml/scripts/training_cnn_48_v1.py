import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D,
                          BatchNormalization, Activation,
                          Dense, Dropout, GlobalAveragePooling2D, 
                          RandomRotation, RandomZoom, RandomTranslation, RandomContrast)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

# === CONFIG ===
TRAIN_CSV_PATH = "/home/rumaxx/road-signs-project/ml/data/Train.csv"
ROOT_DIR = "/home/rumaxx/road-signs-project/ml/data"
SAVE_DIR = "/home/rumaxx/road-signs-project/ml/new_trained_models"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_HEIGHT = 48
IMG_WIDTH = 48
NUM_CLASSES = 43
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42
LR = 1e-3

MODEL_PATH = os.path.join(SAVE_DIR, "cnn_48_v1.keras")
HISTORY_PATH = os.path.join(SAVE_DIR, "cnn_48_history_v1.json")

# === DATA LOADING ===
def load_and_preprocess(path, x1, y1, x2, y2, label):
    file_content = tf.io.read_file(path)
    img = tf.io.decode_image(file_content, channels=3, expand_animations=False)
    
    img = img[y1:y2, x1:x2, :]
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def create_dataset(df, training=True):
    paths = [os.path.join(ROOT_DIR, p) for p in df["Path"]]
    labels = df["ClassId"].values
    rois = df[['Roi.X1','Roi.Y1','Roi.X2','Roi.Y2']].values

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, rois[:,0], rois[:,1], rois[:,2], rois[:,3], labels)
    )
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def conv_block(x, filters, kernel_size=(3, 3), l2_reg=1e-4):
    """
    Dodana regularyzacja L2, aby zapobiec rozrostowi wag.
    """
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False, 
               kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_cnn():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # === AUGMENTACJA ===
    x = RandomRotation(0.05)(inputs) 
    x = RandomZoom(0.1)(x)
    x = RandomTranslation(0.1, 0.1)(x)
    x = RandomContrast(0.1)(x)

    # Blok 1: 48x48 -> 24x24
    x = conv_block(x, 32)
    x = conv_block(x, 32)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Blok 2: 24x24 -> 12x12
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Blok 3: 12x12 -> 6x6
    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.4)(x)


    x = GlobalAveragePooling2D()(x) 
    
    x = Dense(256, use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs, outputs)

# === TRAINING ===
if __name__ == "__main__":
    print("=== START TRAINING: Optimized Custom CNN 48x48 ===")

    df = pd.read_csv(TRAIN_CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["ClassId"], random_state=SEED
    )

    train_ds = create_dataset(train_df, True)
    val_ds = create_dataset(val_df, False)

    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(filepath=MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1), # Wydłużone patience ze względu na agresywniejszy Dropout
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f)

    print(f"Model zapisany: {MODEL_PATH}")