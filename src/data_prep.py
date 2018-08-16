# Get data from the pickle file.
import torch
import numpy as np
from sklearn.utils import resample
import config


def prepare_data(data):
    # Possible inputs
    U_full = {}  # User CS
    A_full = {}  # Agent CS
    R_full = {}  # Rapport
    AT_full = {}  # Agent Task Strategy (TS)

    # True values.
    u_true = {}
    r_true = {}

    # Total number of data points.
    n_tot = {}

    for k, val in data.items():

        if config.sr_type == 'user':
            u_true[k] = val['user_cs_outp']
            A_full[k] = np.concatenate((add_axis(val['agent_cs_inp_t-0']), add_axis(val['agent_cs_inp_t-1'])))
        else:
            u_true[k] = val['agent_cs_inp_t-0']
            A_full[k] = np.concatenate((add_axis(val['agent_cs_inp_t-1']), add_axis(val['agent_cs_inp_t-2'])))

        r_true[k] = val['rapp_outp']
        n_tot[k] = u_true[k].shape[0]
        U_full[k] = np.concatenate((add_axis(val['user_cs_inp_t-1']), add_axis(val['user_cs_inp_t-2'])))
        A_full[k] = np.concatenate((add_axis(val['agent_cs_inp_t-0']), add_axis(val['agent_cs_inp_t-1'])))
        R_full[k] = np.concatenate((add_axis(val['rapp_inp_t-1']), add_axis(val['rapp_inp_t-2'])))
        AT_full[k] = np.concatenate((add_axis(val['agent_intention_inp_t-0']), add_axis(val['agent_intention_inp_t-1'])))

    return u_true, r_true, n_tot, U_full, A_full, R_full, AT_full


def upsampled_tuple(X):
    other_us = []
    for i in range(num_cs - 1):
        u_other = np.where(X[:, i] == 1)[0]
        other_us.append(re_sample(X[u_other], ns))
    return other_us


def prep_data(u_true, r_true, R_full, U_full, A_full, AT_full):
    u_tr_f = u_true
    r_tr_f = r_true
    R_f = R_full.squeeze().transpose()
    U_f = numpy_view(U_full)
    A_f = numpy_view(A_full)
    AT_f = numpy_view(AT_full)
    return u_tr_f, r_tr_f, R_f, U_f, A_f, AT_f


def upsample_data(u_true, r_true, R_full, U_full, A_full, AT_full, window):
    # Last index corresponds to NONE conversational strategy
    u_none = np.where(u_true[:, -1] == 1)[0]
    ns = u_none.shape[0]

    # Outputs: True user CS, true rapport value
    us_other = [np.where(u_true[:, -1] != 1)[0]]
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

    U_temp = numpy_view(U_full)
    tup = [re_sample(U_temp[uo, :, :], ns) for uo in us_other]
    tup.append(U_temp[u_none, :, :])
    U_f = np.concatenate(tuple(tup))

    A_temp = numpy_view(A_full)
    tup = [re_sample(A_temp[uo, :, :], ns) for uo in us_other]
    tup.append(A_temp[u_none, :, :])
    A_f = np.concatenate(tuple(tup))

    AT_temp = numpy_view(AT_full)
    tup = [re_sample(AT_temp[uo, :, :], ns) for uo in us_other]
    tup.append(AT_temp[u_none, :, :])
    AT_f = np.concatenate(tuple(tup))

    assert R_f.shape[-1] == U_f.shape[-1] == A_f.shape[-1] == AT_f.shape[-1] == window
    assert u_tr_f.shape[0] == r_tr_f.shape[0] == R_f.shape[0] == U_f.shape[0] == A_f.shape[0] == AT_f.shape[0]

    return torch.Tensor(u_tr_f), torch.Tensor(r_tr_f), torch.Tensor(R_f), torch.Tensor(U_f), torch.Tensor(A_f), torch.Tensor(AT_f)


def get_train_valid_test_indices(frac_valid, frac_test, N):
    assert frac_valid < 1 and frac_test < 1 and frac_valid + frac_test < 1
    n_valid = int(frac_valid * N)
    n_test = int(frac_test * N)

    all_indices = range(N)
    valid_indices = np.random.choice(all_indices, size=n_valid, replace=False).tolist()
    rem_indices = subtract(all_indices, valid_indices)
    test_indices = np.random.choice(rem_indices, size=n_test, replace=False).tolist()
    train_indices = subtract(rem_indices, test_indices)

    return train_indices, valid_indices, test_indices


def get_final_data(data_types, train_indices, valid_indices, test_indices, u_tr_f, r_tr_f, R_f, U_f, A_f, AT_f):
    U = {}
    A = {}
    R = {}
    AT = {}
    u_tr = {}
    r_tr = {}

    indices = [train_indices, valid_indices, test_indices]

    for i, dt in enumerate(data_types):
        u_tr[dt] = torch.Tensor(u_tr_f[indices[i], :])
        r_tr[dt] = torch.Tensor(r_tr_f[indices[i], :])
        R[dt] = torch.Tensor(R_f[indices[i], :])
        U[dt] = torch.Tensor(U_f[indices[i], :])
        A[dt] = torch.Tensor(A_f[indices[i], :])
        AT[dt] = torch.Tensor(AT_f[indices[i], :])

    return u_tr, r_tr, R, U, A, AT


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