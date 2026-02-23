import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input


DATA_DIR = "/home/rumaxx/road-signs-project/ml/data/Train"
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CATEGORIES = 43


def load_gtsrb_data(data_dir, img_width, img_height):
    images = []
    labels = []

    print(f"Loading data from: {data_dir}")

    for category in tqdm(range(NUM_CATEGORIES), desc = "Loading classes"):
        category_path = os.path.join(data_dir, str(category))

        if not os.path.isdir(category_path):
            print(f"Warning: Directory for class {category} does not exist. Skipping.")
            continue

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)

            try:
                image = cv2.imread(img_path)
                image_resized = cv2.resize(image, (img_width, img_height))
                images.append(image_resized)
                labels.append(category)

            except Exception as e:
                print(f"Error loading file {img_path}: {e}")

    return (np.array(images), np.array(labels))

def load_gtsrb_test_data(csv_path, images_base_dir, img_width, img_height):
    images = []
    labels = []
    try:
        test_data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File doesn't exist {csv_path}")
        return None, None

    print(f"Data loading from {csv_path}")
    for index, row in tqdm(test_data.iterrows(), total = len(test_data), desc = "Loading test data"):
        img_path = os.path.join(images_base_dir, row['Path'])
        label = row['ClassId']
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: The image couldn't be load {img_path}. Skip")
                continue
            image_resized = cv2.resize(image, (img_width, img_height))
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            images.append(image_rgb)
            labels.append(label)
        except Exception as e:
            print("Error during load file")

    if not images:
        print("Error: No images loaded")
        return None, None 

    return np.array(images), np.array(labels)   


def load_and_preprocess_test_image(path, x1, y1, x2, y2, label, img_height, img_width):
    """Wczytuje, KADRUJE i skaluje obraz testowy."""
    image_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(image_bytes, channels=3)
    
    # --- KROK KADROWANIA (Crop) ---
    offset_height = y1
    offset_width = x1
    target_height = y2 - y1
    target_width = x2 - x1
    
    img_cropped = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    
    # Skalujemy wykadrowany obraz
    img_resized = tf.image.resize(img_cropped, (img_height, img_width))
    
    return img_resized, label


def load_and_preprocess_image(path, x1, y1, x2, y2, label):
    """Wczytuje, KADRUJE i skaluje obraz."""
    image_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(image_bytes, channels=3) 
    
    offset_height = tf.cast(y1, tf.int32)
    offset_width = tf.cast(x1, tf.int32)
    target_height = tf.cast(y2 - y1, tf.int32)
    target_width = tf.cast(x2 - x1, tf.int32)
    
    img_cropped = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
    
    img_resized = tf.image.resize(img_cropped, (IMG_HEIGHT, IMG_WIDTH))
    
    return img_resized, label

if __name__ == "__main__":
    images, labels = load_gtsrb_data(DATA_DIR)

    print("\n--- Data Loading Summary ---")
    print(f"Total images loaded: {len(images)}")
    print(f"Total labels loaded: {len(labels)}")
    print(f"Images array shape: {images.shape}")
    print(f"Labels array shape: {labels.shape}")
    print(f"----------------------------------")
