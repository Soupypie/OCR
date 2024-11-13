import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import string
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('emnist_ocr_model.keras')

# Define possible character mappings for labels (0-9, A-Z, a-z variations)
characters1 = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
characters2 = list(string.digits + string.ascii_lowercase + string.ascii_uppercase)
# """
characters3 = list(string.ascii_uppercase + string.digits + string.ascii_lowercase)
characters4 = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
characters5 = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)
characters6 = list(string.ascii_lowercase + string.digits + string.ascii_uppercase)
# """

from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path, save_path):
    """
    Preprocess the image to the required input format for the model.
    - Load the image as grayscale.
    - Resize it to 28x28 pixels.
    - Rotate and flip to match EMNIST orientation.
    - Invert colors (assuming white on black).
    - Normalize to range [0, 1].
    - Binarize to have only 0 or 1 values.
    - Save the processed image.
    """
    # Load the image as grayscale
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    img_array = img_to_array(img)

    # Convert to a PIL Image to perform rotation and flip transformations
    img_pil = Image.fromarray(np.uint8(img_array.squeeze()), mode='L')

    # Show the original image for debugging
    # img_pil.show()
    #input()

    # Flip horizontally
    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    
    # img_pil.show()
    # input() 
    
    # Rotate 90 degrees counterclockwise (with expand=True to avoid cropping)
    img_pil = img_pil.transpose(Image.ROTATE_90)

    # Show the image after flip and rotate for debugging
    # img_pil.show()
    # input()

    # Convert back to a NumPy array for model compatibility
    img_array = np.array(img_pil)

    # Reshape to (28, 28, 1) if your model expects 3D shape input
    img_array = img_array.reshape((-1, 28, 28, 1))
    
    # Invert colors (white background to black, black text to white)
    img_array = 255 - img_array  

    # Normalize to range [0, 1]
    img_array = img_array / 255.0

    # Binarize: set values > 0.5 to 1, others to 0
    img_array = np.where(img_array > 0.5, 1.0, 0.0)

    # Convert back to uint8 format for saving
    save_img = (img_array.squeeze() * 255).astype(np.uint8)

    # Save the processed image as a .png file
    Image.fromarray(save_img, mode='L').save(save_path)

    return img_array  # Return the processed array if needed



def predict_top_characters(image_path, top_k=3):
    """
    Predicts the top K characters in the given image across all mappings.
    """
    
    # Make a prediction
    prediction = model.predict(preprocess_image(image_path, 'array.png')).flatten()
    
    # Get the indices of the top K highest probabilities
    top_indices = np.argsort(prediction)[-top_k:][::-1]
    
    # Print top K predictions across all mappings
    print("Top predictions for each mapping:")
    for i, char_map in enumerate([characters1, characters2, characters3, characters4, characters5, characters6]):
        print(f"\nMapping {i+1}:")
        for idx in top_indices:
            print(f"Character: {char_map[idx]}, Confidence: {prediction[idx]:.4f}")

# Test the function on 'input.png'
predict_top_characters('P.png', top_k=10)
