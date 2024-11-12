import numpy as np
import pickle

# Load the EMNIST dataset
def load_data(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file)
    return data_dict['data'], data_dict['labels']

# Load training and testing data
train_data, train_labels = load_data('emnist_train.pkl')

print(len(np.unique(train_labels)))