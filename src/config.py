import os

ALL_STR = 'all'

parent_path = os.path.abspath('../')
data_path = parent_path + '/data/davos/'
train_data_fname = 'train_data_full.pkl'
frac_valid = 0.1
frac_test = 0.1
lr = 1e-3
print_every = 1000
data_types = ['train', 'valid', 'test']
max_window = 2
num_user_cs = 6
num_agent_cs = 8
num_agent_ts = 26

testing = False
c = 1
model_type = 'sr'  # Possible values = ['re', 'sr']
window_type = 2  # Possible values = [1, 2, 'linear_combination']
feature_type = 'cs + rapport'  # Possible values = ['cs_only', 'cs + rapport', 'cs + rapport + ts']

# Hyperparameters
num_leaky_iter = 5
leaky_min = 0.05
leaky_step = 0.05
n_epochs = [5000, 10000, 20000]
hidden_sizes = [8, 16]
thresh = [0.35, 0.40, 0.45, 0.50]

# Test parameters
test_hidden_dim = 16
test_leaky_slope = 0.1
test_thresh = 0.4
test_epochs = 20000


def get_input_size():
    if feature_type == 'cs_only':
        input_size = num_user_cs + num_agent_cs
    elif feature_type == 'cs + rapport':
        input_size = num_user_cs + num_agent_cs + 1
    else:
        input_size = num_user_cs + num_agent_cs + num_agent_ts + 1

    if window_type == 2:
        input_size *= 2

    return input_size


def get_output_size():
    if model_type == 're':
        output_size = 1
    else:
        output_size = num_user_cs

    return output_size

