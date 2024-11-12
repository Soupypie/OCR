import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import string

# Load the trained model
model = tf.keras.models.load_model('emnist_ocr_model.keras')

# Define the character mapping for labels (0-9, A-Z, a-z)
characters1 = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
characters2 = list(string.digits + string.ascii_lowercase + string.ascii_uppercase)
characters3 = list(string.ascii_uppercase + string.digits + string.ascii_lowercase)
characters4 = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
characters5 = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)
characters6 = list(string.ascii_lowercase + string.digits + string.ascii_uppercase)

def preprocess_image(image_path):
    """
    Preprocess the image to the required input format for the model.
    - Load the image as grayscale.
    - Resize it to 28x28 pixels.
    - Invert colors (if necessary, assuming white on black).
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

def predict_character(image_path):
    """
    Predicts the character in the given image.
    """
    # Preprocess the input image
    processed_image = preprocess_image(image_path)
    
    # Make a prediction
    prediction = model.predict(processed_image)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(prediction)
    
    # Map the index to the corresponding character
    predicted_characters = [characters1[predicted_index],characters2[predicted_index],characters3[predicted_index],characters4[predicted_index],characters5[predicted_index],characters6[predicted_index]]
    
    return predicted_characters

# Test the function on 'input.png'
predicted_characters = predict_character('input.png')
print("The predicted character is: ")
for i in range(predicted_characters):
    print(predicted_characters[i])
