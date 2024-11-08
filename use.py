import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model
model = load_model('emnist_model.keras')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Character set
char_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def preprocess_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Unable to load image at path: {path}")
    image = cv2.resize(image, (28, 28))  # Resize to a fixed size (width, height)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.reshape(1, 28, 28, 1) 
    return image.astype('float32')

def predict_text(image):
    predictions = model.predict(image)
    return predictions

# Preprocess the image
try:
    image = preprocess_image("input.png")
    image_output = image.reshape(28, 28) * 255.0
    cv2.imwrite("output.png", image_output)

    # Predict the text
    predicted_text = predict_text(image)

    # Decode the text
    decoded_text = ''.join([char_set[np.argmax(row)] for row in predicted_text.reshape(-1, len(char_set))])
    print("Predicted Text:", decoded_text)

except Exception as e:
    print(f"An error occurred: {e}")
