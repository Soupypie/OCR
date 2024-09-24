import cv2
from tensorflow.keras.models import load_model
import numpy as np
import string

# Load the pre-trained model
model = load_model('model.keras')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Unable to load image at path: {path}")
    image = cv2.resize(image, (28, 28))  # Resize to a fixed size (width, height)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return image

def predict_text(image):
    predictions = model.predict(image)
    return predictions

# Preprocess the image
image = preprocess_image("screenshot.png")

# Predict the text
predicted_text = predict_text(image)

# Create an array with numbers 0-9, capital letters A-Z, and lowercase letters a-z
char_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Decode the text
decoded_text = ''.join([char_set[np.argmax(row)] for row in predicted_text.reshape(-1, 62)])

print("Predicted Text:", decoded_text)
