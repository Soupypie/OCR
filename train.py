import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
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

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, 
          validation_data=(test_data, test_labels), 
          epochs=10, 
          batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model for future use
model.save('emnist_ocr_model.h5')
