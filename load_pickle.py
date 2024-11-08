import pickle

fin = 'emnist_test.pkl'

data_dict = pickle.load(open(fin, 'rb'), encoding='latin1')

print(data_dict)