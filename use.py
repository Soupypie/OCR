import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import string

# Load the trained model
model = tf.keras.models.load_model('emnist_ocr_model.keras')

# Define possible character mappings for labels (0-9, A-Z, a-z variations)
# characters1 = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
characters2 = list(string.digits + string.ascii_lowercase + string.ascii_uppercase)
"""
characters3 = list(string.ascii_uppercase + string.digits + string.ascii_lowercase)
characters4 = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
characters5 = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)
characters6 = list(string.ascii_lowercase + string.digits + string.ascii_uppercase)
"""

def preprocess_image(image_path):
    """
    Preprocess the image to the required input format for the model.
    - Load the image as grayscale.
    - Resize it to 28x28 pixels.
    - Invert colors (assuming white on black).
    - Normalize to range [0, 1].
    """
    # Load the image as grayscale
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    
    # Convert the image to a numpy array and invert colors if needed
    img_array = img_to_array(img)
    img_array = 255 - img_array  # Assuming white background, black text
    
    # Normalize to range [0, 1]
    img_array /= 255.0

    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def predict_top_characters(image_path, top_k=3):
    """
    Predicts the top K characters in the given image across all mappings.
    """
    # Preprocess the input image
    processed_image = preprocess_image(image_path)
    
    # Make a prediction
    prediction = model.predict(processed_image).flatten()
    
    # Get the indices of the top K highest probabilities
    top_indices = np.argsort(prediction)[-top_k:][::-1]
    
    # Print top K predictions across all mappings
    print("Top predictions for each mapping:")
    for i, char_map in enumerate([characters2]):
        print(f"\nMapping {i+1}:")
        for idx in top_indices:
            print(f"Character: {char_map[idx]}, Confidence: {prediction[idx]:.4f}")

# Test the function on 'input.png'
predict_top_characters('4.png', top_k=10)
