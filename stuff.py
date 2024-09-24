import pickle
import pprint

# Load the data from the pickle file
with open('emnist_train.pkl', 'rb') as file:
    e = pickle.load(file)

# Assuming e is a dictionary with 'data' and 'labels' keys
data = e['data']
labels = e['labels']

# Open a file to write the output
with open('output.txt', 'w') as output_file:
    for i in range(10):
        output_file.write(f"Image {i+1}:\n")
        pprint.pprint(data[i], stream=output_file)
        output_file.write(f"Label {i+1}: {labels[i]}\n\n")

print("First 10 images and labels have been saved to output.txt")
