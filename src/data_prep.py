# Get data from the pickle file.
import torch
import numpy as np
from sklearn.utils import resample


def prepare_data(data):
    # Input (un-processed) to the joint model.
    U_full = {}
    A_full = {}
    R_full = {}

    # True values.
    u_true = {}
    r_true = {}

    # Total number of data points.
    n_tot = {}

    input_variables = ['user_cs_inp', 'agent_cs_inp', 'rapp_inp']
    output_variables = ['user_cs_outp', 'rapp_outp']

    for k, val in data.items():
        all_keys = list(val.keys())
        u_true[k] = val[all_keys[0]]
        r_true[k] = val[all_keys[1]]
        n_tot[k] = u_true[k].shape[0]
        U_full[k] = np.concatenate(( add_axis(val[all_keys[2]]), add_axis(val[all_keys[3]]) ))
        A_full[k] = np.concatenate(( add_axis(val[all_keys[4]]), add_axis(val[all_keys[5]]) ))
        R_full[k] = np.concatenate(( add_axis(val[all_keys[6]]), add_axis(val[all_keys[7]]) ))

    return u_true, r_true, n_tot, U_full, A_full, R_full


def upsampled_tuple(X):
    other_us = []
    for i in range(num_cs - 1):
        u_other = np.where(X[:, i] == 1)[0]
        other_us.append(re_sample(X[u_other], ns))
    return other_us


def prep_data(u_true, r_true, R_full, U_full, A_full):
    u_tr_f = u_true
    r_tr_f = r_true
    R_f = R_full.squeeze().transpose()
    U_f = numpy_view(U_full)
    A_f = numpy_view(A_full)
    return u_tr_f, r_tr_f, R_f, U_f, A_f


def upsample_data(u_true, r_true, R_full, U_full, A_full, window):
    # Last index corresponds to NONE conversational strategy
    u_none = np.where(u_true[:, -1] == 1)[0]
    ns = u_none.shape[0]
    # u_not_none = np.where(u_true[c][:, -1] == 0)[0]
    # Sample non-NONE CSs to equal the number of samples for NONE CS
    # u_SD = np.where(u_true[:, 0] == 1)[0]
    # n_SD = u_SD.shape[0]
    # Outputs: True user CS, true rapport value
    num_cs = u_true.shape[1]

    # us_other = [np.where(u_true[:, i] == 1)[0] for i in range(num_cs - 1)]
    us_other = [np.where(u_true[:, -1] != 1)[0]]
    # us_other = [np.where(u_true[:, 0] == 1)[0]]
    # for i in [1, 2, 3]:
    #     us_other.append(np.where(u_true[:, i] == 1)[0])

    # u_tr_f = np.concatenate((re_sample(u_true[c][u_not_none], ns), u_true[c][u_none]))
    # r_tr_f = np.concatenate((re_sample(r_true[c][u_not_none], ns), r_true[c][u_none]))

    tup = [re_sample(u_true[uo], ns) for uo in us_other]
    tup.append(u_true[u_none])
    u_tr_f = np.concatenate(tuple(tup))

    tup = [re_sample(r_true[uo], ns) for uo in us_other]
    tup.append(r_true[u_none])
    r_tr_f = np.concatenate(tuple(tup))

    # Inputs: Rapport, user CS, agent CS
    R_temp = R_full.squeeze().transpose()
    tup = [re_sample(R_temp[uo, :], ns) for uo in us_other]
    tup.append(R_temp[u_none, :])
    R_f = np.concatenate(tuple(tup))
    # R_f = np.concatenate((re_sample(R_temp[u_not_none, :], ns), R_temp[u_none, :]))

    # U_temp = numpy_view(U_full[c])
    # U_f = np.concatenate((re_sample(U_temp[u_not_none, :, :], ns), U_temp[u_none, :, :]))

    U_temp = numpy_view(U_full)
    tup = [re_sample(U_temp[uo, :, :], ns) for uo in us_other]
    tup.append(U_temp[u_none, :, :])
    U_f = np.concatenate(tuple(tup))

    # A_temp = numpy_view(A_full[c])
    # A_f = np.concatenate((re_sample(A_temp[u_not_none, :, :], ns), A_temp[u_none, :, :]))

    A_temp = numpy_view(A_full)
    tup = [re_sample(A_temp[uo, :, :], ns) for uo in us_other]
    tup.append(A_temp[u_none, :, :])
    A_f = np.concatenate(tuple(tup))

    assert R_f.shape[-1] == U_f.shape[-1] == A_f.shape[-1] == window
    assert u_tr_f.shape[0] == r_tr_f.shape[0] == R_f.shape[0] == U_f.shape[0] == A_f.shape[0]

    return torch.Tensor(u_tr_f), torch.Tensor(r_tr_f), torch.Tensor(R_f), torch.Tensor(U_f), torch.Tensor(A_f)


def get_train_valid_test_indices(frac_valid, frac_test, N):
    assert frac_valid < 1 and frac_test < 1 and frac_valid + frac_test < 1
    n_valid = int(frac_valid * N)
    n_test = int(frac_test * N)
    n_train = N - (n_valid + n_test)

    all_indices = range(N)
    valid_indices = np.random.choice(all_indices, size=n_valid, replace=False).tolist()
    rem_indices = subtract(all_indices, valid_indices)
    test_indices = np.random.choice(rem_indices, size=n_test, replace=False).tolist()
    train_indices = subtract(rem_indices, test_indices)

    return train_indices, valid_indices, test_indices


def get_final_data(data_types, train_indices, valid_indices, test_indices, u_tr_f, r_tr_f, R_f, U_f, A_f):
    U = {}
    A = {}
    R = {}
    u_tr = {}
    r_tr = {}

    indices = [train_indices, valid_indices, test_indices]

    for i, dt in enumerate(data_types):
        u_tr[dt] = torch.Tensor(u_tr_f[indices[i], :])
        r_tr[dt] = torch.Tensor(r_tr_f[indices[i], :])
        R[dt] = torch.Tensor(R_f[indices[i], :])
        U[dt] = torch.Tensor(U_f[indices[i], :])
        A[dt] = torch.Tensor(A_f[indices[i], :])

    return u_tr, r_tr, R, U, A


def numpy_view(np_array):
    d1, d2, d3 = np_array.shape
    return torch.Tensor(np_array).view(d2, d3, d1).cpu().numpy()


def re_sample(X, ns):
    return resample(X, replace=True, n_samples=ns, random_state=0)


def add_axis(mat):
    return mat[np.newaxis, :, :]


def convert_to_tensor(np_array):
    d1, d2, d3 = np_array.shape
    return torch.Tensor(np_array).view(d2, d3, d1)


def re_sample(X, ns):
    return resample(X, replace=True, n_samples=ns)


def subtract(list1, list2):
    """Subtract list2 from list1"""
    return [i for i in list1 if i not in list2]