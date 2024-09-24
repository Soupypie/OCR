import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np
import string

# Load the pre-trained model
model = load_model('model.keras')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # Resize to a fixed size (width, height)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    image = np.expand_dims(image, axis=-1)  # Add batch dimension
    return image

def output(image):
    output = (image * 255).astype(np.uint8)  # Convert back to [0, 255] range and uint8 type
    cv2.imwrite('output.jpg', output)

def decode_predictions(predictions):
    # Assuming predictions is a 2D array with shape (batch_size, num_classes)
    predicted_indices = np.argmax(predictions, axis=-1)
    predicted_text = ''.join(index_to_char[idx] for idx in predicted_indices)
    return predicted_text

def predict_text(image):
    predictions = model.predict(image)
    # predicted_text = decode_predictions(predictions)
    return predictions

# Preprocess the image
image = preprocess_image("screenshot.png")
output(image)

# Predict the text
predicted_text = predict_text(image)

import numpy as np
import string

# Example prediction output
# predictions = predicted_text
# Define the index to character mapping
# index_to_char = {i: char for i, char in enumerate(string.ascii_lowercase + string.digits)}

# Decode the predictions
#
# predicted_indices = np.argmax(predictions, axis=-1)
# predicted_text = ''.join(index_to_char[idx] for idx in predicted_indices)

print("Predicted Text:", predicted_text)


