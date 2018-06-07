import os

input_size = 42 # 6-dimensional user CS, 8-dimensional agent CS, 2-dimensional previous rapport values, 26-dimensional agent task intention
hidden_size = 8
output_size = 7 # 6-dimensional user CS (at next step), 1-dimensional rapport value
window = 2 # How many previous user and agent CSs and rapport values we take into account for prediction
leaky_slope = 0.1 # Hyperparameter for Leaky ReLU
parent_path = os.path.abspath('../')
data_path = parent_path + '/data/davos/'
train_data_fname = 'train_data_full.pkl'
frac_valid = 0.1
frac_test = 0.1
lr = 1e-3
# n_epochs = [1000, 5000]
n_epochs = [1000]
print_every = 1000
data_types = ['train', 'valid', 'test']
thresh = 0.4
social_reasoner = 1
social_output_size = 6
social_input_size = 42

c = 'all'