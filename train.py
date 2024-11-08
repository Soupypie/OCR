# Import necessary libraries
import numpy as np
import pickle
import keras
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# Function to build and compile the CNN model
def build_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu', padding='VALID'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='VALID'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(62, activation='softmax')  # Output layer for 62 classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary
    return model


# Load and preprocess the EMNIST dataset
with open('emnist_train.pkl', 'rb') as f:
    emnist_train = pickle.load(f)
with open('emnist_test.pkl', 'rb') as g:
    emnist_test = pickle.load(g)

# Extract data and labels
X_train, y_train = emnist_train['data'], emnist_train['labels']
X_val, y_val = emnist_test['data'], emnist_test['labels']

# Reshape and normalize image data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_val = X_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Check data shapes for debugging
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Convert labels to categorical format
y_train = keras.utils.to_categorical(y_train, num_classes=62)
y_val = keras.utils.to_categorical(y_val, num_classes=62)

# Set up data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

# Build and compile the model
model = build_model()
model.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with augmented data
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Save the trained model
model.save('emnist_model.keras')

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')
