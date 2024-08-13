import pickle
import matplotlib.pyplot as plt
import pprint
import numpy as np

# Load the pickle file
with open('emnist_train.pkl', 'rb') as file:
    images = pickle.load(file)
np.set_printoptions(threshold=np.inf)

with open('print.txt', 'w') as txt_file:
    pprint.pprint(images['data'], stream=txt_file)
