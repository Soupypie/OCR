import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tqdm

# Load the EMNIST dataset
ds_train = tfds.load('emnist', split='train', shuffle_files=True)
ds_test = tfds.load('emnist', split='test', shuffle_files=True)

# Convert the dataset to numpy arrays
ds_train = tfds.as_numpy(ds_train)
ds_test = tfds.as_numpy(ds_test)

# Initialize lists to hold the image data and labels
trainX, trainy = [], []
testX, testy = [], []

# Check if preprocessed data already exists
try:
    # Attempt to load preprocessed data from disk
    trainX = np.load('trainX.npy')
    trainy = np.load('trainy.npy')
    testX = np.load('testX.npy')
    testy = np.load('testy.npy')
    print("Preprocessed data loaded from disk.")
except FileNotFoundError:
    # If data files do not exist, preprocess and save them
    print("Preprocessing data, this may take some time...")

    # Iterate over the train dataset with progress bar
    for example in tqdm(ds_train, desc="Processing training data", total=len(ds_train)):
        trainX.append(example['image'])  # Add the image to trainX
        trainy.append(example['label'])  # Add the label to trainy

    # Convert lists to numpy arrays
    trainX = np.array(trainX)
    trainy = np.array(trainy)

    # Iterate over the test dataset with progress bar
    for example in tqdm(ds_test, desc="Processing test data", total=len(ds_test)):
        testX.append(example['image'])  # Add the image to testX
        testy.append(example['label'])  # Add the label to testy

    # Convert lists to numpy arrays
    testX = np.array(testX)
    testy = np.array(testy)

    # Save the preprocessed data to disk
    np.save('trainX.npy', trainX)
    np.save('trainy.npy', trainy)
    np.save('testX.npy', testX)
    np.save('testy.npy', testy)
    print("Preprocessed data saved to disk.")