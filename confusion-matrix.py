import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import string
import pickle

# Load the EMNIST dataset
def load_data(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
    return data_dict['data'], data_dict['labels']

# Load training and testing data
train_data, train_labels = load_data('emnist_train.pkl')
test_data, test_labels = load_data('emnist_test.pkl')

# Preprocess the test data (reshape and normalize)
# Assuming test_data is flat, reshape it to (num_samples, 28, 28, 1) and normalize to range [0, 1]
test_data = test_data.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Define character labels
class_labels = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)

# Load your model (if not already loaded)
model = tf.keras.models.load_model('emnist_ocr_model.keras')

# Generate predictions on the test data
predictions = model.predict(test_data)

# Convert predictions and true labels to class indices
predicted_labels = np.argmax(predictions, axis=1)
true_labels = test_labels  # Assumes test_labels are already class indices

# Generate the normalized confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, normalize='true')

# Plot the normalized confusion matrix with increased figure size
plt.figure(figsize=(24, 24))  # Increase size to make it more readable
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues', xticks_rotation='vertical', values_format=".2f")
plt.title("Normalized Confusion Matrix for OCR Model")
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Save as a high-resolution image for detailed viewing
plt.savefig("confusion_matrix_high_res.png", dpi=300)  # Save with higher resolution

plt.show()
