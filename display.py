import pickle
import matplotlib.pyplot as plt
import pprint
import numpy as np

# Load the pickle file
with open('emnist_train.pkl', 'rb') as file:
    images = pickle.load(file)
    file.close()
np.set_printoptions(threshold=np.inf)

print("Loaded")

for i in range(5):  # Display first 5 samples
    pprint.pprint(f"Sample {i+1}:")
    pprint.pprint(images['data'][i])
    pprint.pprint(f"Label: {images['labels'][i]}")