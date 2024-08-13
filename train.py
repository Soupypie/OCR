import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np

def build_model():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(62, activation='softmax')  # 62 classes for uppercase, lowercase, and digits
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

def output(image):
    output = (image * 255).astype(np.uint8)  # Convert back to [0, 255] range and uint8 type
    cv2.imwrite('output.jpg', output)

image = preprocess_image("screenshot.png")
output(image)

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

# Build and compile your model
model = build_model()
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('model.keras')
