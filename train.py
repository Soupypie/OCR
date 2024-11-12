import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Load the EMNIST dataset
def load_data(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
    return data_dict['data'], data_dict['labels']

# Load training and testing data
train_data, train_labels = load_data('emnist_train.pkl')
test_data, test_labels = load_data('emnist_test.pkl')

# Preprocess the data
train_data = np.array(train_data).reshape(-1, 28, 28, 1).astype('float32')
test_data = np.array(test_data).reshape(-1, 28, 28, 1).astype('float32')

# Normalize pixel values (0-1 range)
train_data /= 255.0
test_data /= 255.0

# Convert labels to categorical (for multi-class classification)
num_classes = len(np.unique(train_labels))
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)
"""
# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)
datagen.fit(train_data)
"""
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
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch % 10 == 0: 
        return lr * 0.5
    else:
        return lr

# Early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data, train_labels,  # Use the original training data directly
    validation_data=(test_data, test_labels),
    epochs=15,
    batch_size=64,
    callbacks=[LearningRateScheduler(lr_schedule), early_stopping]
)


# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model in .keras format
model.save('emnist_ocr_model.keras')
