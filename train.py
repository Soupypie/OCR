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

# Define a more complex model architecture with Batch Normalization and Dropout
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Define the learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr  # No change for the first 10 epochs
    else:
        return lr * 0.9  # Reduce learning rate by 10% after epoch 10

# Create the LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# One-hot encode the labels
trainy_one_hot = tf.keras.utils.to_categorical(trainy, num_classes=62)
testy_one_hot = tf.keras.utils.to_categorical(testy, num_classes=62)

# Compile the model using categorical_crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(
    trainX, trainy_one_hot,  # Use one-hot encoded labels
    validation_split = 0.1,
    epochs=15,
    batch_size=64,
    callbacks=[lr_scheduler, early_stopping]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(testX, testy)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model in .keras format
model.save('emnist_ocr_model.keras')
