import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import cv2
import tensorflow_datasets as tfds
import array_record
from tqdm import tqdm  # Import tqdm for progress bar
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import string
import random


# Initialize lists to hold the image data and labels
trainX, trainy = [], []
testX, testy = [], []

# Check if preprocessed data already exists
try:
    # Attempt to load preprocessed data from disk
    trainX = np.load('trainX.npy')
    trainy = np.load('trainy.npy')
    testX = np.load('testX.npy')
    testy = np.load('testy.npy')
    print("Preprocessed data loaded from disk.")
except FileNotFoundError:
    print("Download Data First")

# Verify the data shapes
print(f"trainX shape: {trainX.shape}, trainy shape: {trainy.shape}")
print(f"testX shape: {testX.shape}, testy shape: {testy.shape}")

# Find the unique classes in trainy
unique_classes = np.unique(trainy)

# Get the number of unique classes
num_classes = len(unique_classes)

print(f"Number of unique classes in trainy: {num_classes}")
print(f"Unique classes: {unique_classes}")

def preprocess_image(image_path, save_path):
    """
    Preprocess the image to the required input format for the model.
    - Load the image as grayscale.
    - Resize it to 28x28 pixels.
    - Rotate and flip to match EMNIST orientation.
    - Invert colors (assuming white on black).
    - Normalize to range [0, 1].
    - Binarize to have only 0 or 1 values.
    - Save the processed image.
    """
    # Load the image as grayscale
    img = image_path
    img_array = img_to_array(img)

    # Convert to a PIL Image to perform rotation and flip transformations
    img_pil = Image.fromarray(np.uint8(img_array.squeeze()), mode='L')
    # Rotate 90 degrees counterclockwise and flip horizontally
    # img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    # img_pil = img_pil.rotate(90, expand=True)
    
    # Convert back to a NumPy array for model compatibility
    img_array = np.array(img_pil)

	# Reshape to (28, 28, 1) if your model expects 3D shape input
    img_array = img_array.reshape((-1, 28, 28, 1))
    
	# Invert colors (white background to black, black text to white)
    # img_array = 255 - img_array  

    # Normalize to range [0, 1]
    img_array = img_array / 255.0

    # Binarize: set values > 0.5 to 1, others to 0
    img_array = np.where(img_array > 0.5, 1.0, 0.0)

    # Convert back to uint8 format for saving
    save_img = (img_array.squeeze() * 255).astype(np.uint8)

    # Save the processed image as a .png file
    Image.fromarray(save_img, mode='L').save(save_path)

    return img_array  # Return the processed array if needed

def display_random_image(data, labels, preprocess=False):
    """
    Display a random image from the given dataset.
    Parameters:
        - data: The dataset to sample from (trainX or testX).
        - labels: The labels corresponding to the dataset.
        - preprocess: If True, preprocess the image using the preprocess_image function.
    """
    # Select a random index
    random_index = random.randint(0, len(data) - 1)
    label = 100
    # Get the image and label at the random index
    while int(label) != 3:
        random_index = random.randint(0, len(data) - 1)
        image = data[random_index]
        label = labels[random_index]

    # Preprocess if needed
    image = preprocess_image(image, 'array.png')  # Skip saving if not necessary
    
    # Display the image
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()

# Call the function with trainX and trainy, with or without preprocessing
display_random_image(trainX, trainy, preprocess=False)