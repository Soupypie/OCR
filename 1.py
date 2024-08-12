import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 32))  # Resize to a fixed size
    image = image / 255.0  # Normalize pixel values
    return image