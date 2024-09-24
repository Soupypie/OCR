import cv2
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras
import tensorflow as tf
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def build_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu', padding='VALID'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='VALID'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(62, activation='softmax')  # 62 classes for uppercase, lowercase, and digits
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Load the data
with open('emnist_train.pkl', 'rb') as f:
    emnist = pickle.load(f)

with open('emnist_test.pkl', 'rb') as g:
    validation = pickle.load(g)

# Extract data and labels
X_train = emnist['data']
y_train = emnist['labels']
X_val = validation['data']
y_val = validation['labels']

# Reshape the data to match the input shape of your model
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=62)
y_val = to_categorical(y_val, num_classes=62)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

class_weights = {i: 1 for i in range(62)}
class_weights[7] = 2  # Example: Increase weight for class 'H'
class_weights[1] = 2  # Example: Increase weight for class 'B'

# Build and compile your model
model = build_model()
model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=64, batch_size=64, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights)

# Save the model
model.save('model.keras')

model.summary()
val_loss, val_accuracy = model.evaluate(X_val, y_val)

print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')