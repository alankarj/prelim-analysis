import torch
import numpy as np
import random
import config
from model import JointEstimator
import pickle
import data_prep

from trainer import Trainer
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def main():
    # Data preparation
    data = pickle.load(open(config.data_path + config.train_data_fname, 'rb'))
    # Unpack the data
    u_true, r_true, n_tot, U_full, A_full, R_full, AT_full = data_prep.prepare_data(data)
    clusters = list(n_tot.keys())
    c = config.c

    train_indices = {}
    valid_indices = {}
    test_indices = {}

    train_indices[config.ALL_STR] = []
    valid_indices[config.ALL_STR] = []
    test_indices[config.ALL_STR] = []

    indexing_list = [0]

    # Create training, validation and test sets for all clusters. 'all' is everything combined.
    for i, cluster_id in enumerate(clusters[:-1]):
        num_samples = u_true[cluster_id].shape[0]
        indexing_list.append(num_samples)
        train_indices[cluster_id], valid_indices[cluster_id], test_indices[cluster_id] = data_prep.get_train_valid_test_indices(config.frac_valid, config.frac_test, num_samples)

        train_indices[config.ALL_STR] += [ind + indexing_list[i] for ind in train_indices[cluster_id]]
        valid_indices[config.ALL_STR] += [ind + indexing_list[i] for ind in valid_indices[cluster_id]]
        test_indices[config.ALL_STR] += [ind + indexing_list[i] for ind in test_indices[cluster_id]]

    # Get data for the given cluster
    new_data = data_prep.prep_data(u_true[c], r_true[c], R_full[c], U_full[c], A_full[c], AT_full[c])

    # Resample the data
    train_data = data_prep.upsample_data(
        u_true[c][train_indices[c]],
        r_true[c][train_indices[c]],
        R_full[c][:, train_indices[c], :],
        U_full[c][:, train_indices[c], :],
        A_full[c][:, train_indices[c], :],
        AT_full[c][:, train_indices[c], :],
        config.max_window)

    tr_ind = train_indices[c]

    if config.testing:
        if config.neural:
            eval_data_type = config.data_types[2]

            je, trainer, _, _ = train_only(config.test_hidden_dim, config.test_leaky_slope,
                                     config.test_thresh, config.test_epochs, train_data)

            if c == 'all':
                for i, cluster_id in enumerate(clusters[:-1]):
                    val_ind = valid_indices[cluster_id]
                    te_ind = [ind + indexing_list[i] for ind in test_indices[cluster_id]]
                    eval_data = get_eval_data(tr_ind, val_ind, te_ind, new_data, eval_data_type)
                    loss = eval_only(eval_data, je, trainer, save_model=False)
                    print("Loss for cluster-%d is %.3f" % (cluster_id, loss))

            val_ind = valid_indices[c]
            te_ind = test_indices[c]
            eval_data = get_eval_data(tr_ind, val_ind, te_ind, new_data, eval_data_type)
            loss = eval_only(eval_data, je, trainer, save_model=True)
            print("Overall loss is %.3f" % loss)

        else:
            eval_data_type = config.data_types[2]

            X_train, Y_train = prepare_training_data_non_nn(train_data)
            print("Train data shape: ", X_train.shape)
            print("Train data shape: ", Y_train.shape)
            clf = train_only_non_nn(X_train, Y_train)

            if c == 'all':
                for i, cluster_id in enumerate(clusters[:-1]):
                    val_ind = valid_indices[cluster_id]
                    te_ind = [ind + indexing_list[i] for ind in test_indices[cluster_id]]
                    eval_data = get_eval_data(tr_ind, val_ind, te_ind, new_data, eval_data_type)
                    X_eval, Y_eval = prepare_training_data_non_nn(eval_data)
                    print("Eval data shape: ", X_eval.shape)
                    print("Eval data shape: ", Y_eval.shape)
                    eval_only_non_nn(clf, X_eval, Y_eval, print_info=True)

            val_ind = valid_indices[c]
            te_ind = test_indices[c]
            eval_data = get_eval_data(tr_ind, val_ind, te_ind, new_data, eval_data_type)
            X_eval, Y_eval = prepare_training_data_non_nn(eval_data)
            print("Eval data shape: ", X_eval.shape)
            print("Eval data shape: ", Y_eval.shape)
            eval_only_non_nn(clf, X_eval, Y_eval, print_info=True)

    else:
        val_ind = valid_indices[c]
        te_ind = test_indices[c]
        eval_data_type = config.data_types[1]
        eval_data = get_eval_data(tr_ind, val_ind, te_ind, new_data, eval_data_type)

        if not config.neural:
            train_and_evaluate_non_nn(train_data, eval_data)

        else:
            if config.model_type == 're':
                loss = float("inf")

                thresh = 0.4
                best_leaky_slope = None
                best_hidden_dim = None
                best_epoch = None

                for leaky_iter in range(config.num_leaky_iter):
                    leaky_slope = config.leaky_min + leaky_iter * config.leaky_step
                    for hidden_dim in config.hidden_sizes:
                        print("###########################################################")
                        print("Leaky slope: %.2f, Hidden dim: %d" % (leaky_slope, hidden_dim))
                        print("###########################################################")

                        seed = 0
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        random.seed(seed)

                        _, _, temp_loss, temp_epoch = train_only(hidden_dim, leaky_slope, thresh, config.n_epochs, train_data, eval_data)

                        loss = min(loss, temp_loss)
                        if loss == temp_loss:
                            best_leaky_slope = leaky_slope
                            best_hidden_dim = hidden_dim
                            best_epoch = temp_epoch

                print("###########################################################")
                print("Best loss: %.3f, best leaky slope: %.2f, best hidden dim: %d, best epoch: %d" % (
                    loss, best_leaky_slope, best_hidden_dim, best_epoch))

            else:
                loss = float("inf")

                best_epoch = None
                best_thresh = None
                best_leaky_slope = None
                best_hidden_dim = None

                for thresh in config.thresh:
                    for leaky_iter in range(config.num_leaky_iter):
                        leaky_slope = config.leaky_min + leaky_iter * config.leaky_step
                        for hidden_dim in config.hidden_sizes:
                            print("###########################################################")
                            print("Threshold: %.2f, Leaky slope: %.2f, Hidden dim: %d" %
                                  (thresh, leaky_slope, hidden_dim))
                            print("###########################################################")

                            seed = 0
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            random.seed(seed)

                            _, _, temp_loss, temp_epoch = train_only(hidden_dim, leaky_slope, thresh, config.n_epochs, train_data, eval_data)

                            loss = min(loss, temp_loss)
                            if loss == temp_loss:
                                best_leaky_slope = leaky_slope
                                best_hidden_dim = hidden_dim
                                best_epoch = temp_epoch
                                best_thresh = thresh

                print("###########################################################")
                print("Best loss: %.3f, best n_epochs: %d, best thresh: %.2f, best leaky slope: %.2f, best hidden dim: %d" % (
                    loss, best_epoch, best_thresh, best_leaky_slope, best_hidden_dim))


def get_eval_data(tr_ind, val_ind, te_ind, new_data, eval_data_type):
    u_tr_f, r_tr_f, R_f, U_f, A_f, AT_f = data_prep.get_final_data(
        config.data_types, tr_ind, val_ind, te_ind, *new_data)
    R_eval = R_f[eval_data_type]
    U_eval = U_f[eval_data_type]
    eval_data = (u_tr_f[eval_data_type], r_tr_f[eval_data_type], R_eval, U_eval,
                 A_f[eval_data_type], AT_f[eval_data_type])
    return eval_data


def train_only(hidden_dim, leaky_slope, thresh, n_epochs, train_data, eval_data=None):
    input_size = config.get_input_size()
    output_size = config.get_output_size()
    je = JointEstimator(input_size, hidden_dim, output_size, leaky_slope,
                        config.window_type, config.feature_type, config.model_type)
    trainer = Trainer(je, config.lr, n_epochs, config.print_every, thresh, config.model_type, eval_data)
    loss, best_epoch = trainer.train(*train_data)
    return je, trainer, loss, best_epoch


def eval_only(eval_data, je, trainer, save_model=False):
    temp_loss = trainer.eval(eval_data, print_info=True)
    if save_model:
        torch.save(je.state_dict(), 'weights_' + str(config.c) + '.t7')
    return temp_loss


def train_and_evaluate_non_nn(train_data, eval_data):
    X_train, Y_train = prepare_training_data_non_nn(train_data)
    X_eval, Y_eval = prepare_training_data_non_nn(eval_data)
    print("Train data shape: ", X_train.shape)
    print("Train data shape: ", Y_train.shape)
    print("Eval data shape: ", X_eval.shape)
    print("Eval data shape: ", Y_eval.shape)
    clf = train_only_non_nn(X_train, Y_train)
    acc = eval_only_non_nn(clf, X_eval, Y_eval, print_info=True)
    return acc


def prepare_training_data_non_nn(train_data):
    u_tr, r_tr, R, U, A, AT = train_data

    if config.window_type == 1:
        u_input = U[:, :, 0]
        a_input = A[:, :, 0]
        at_input = AT[:, :, 0]
        r_input = R[:, 0][:, None]

    elif config.window_type == 2:
        R = R[:, :, None]
        u_input = torch.cat([U[:, :, 0], U[:, :, 1]], dim=1)
        a_input = torch.cat([A[:, :, 0], A[:, :, 1]], dim=1)
        at_input = torch.cat([AT[:, :, 0], AT[:, :, 1]], dim=1)
        r_input = torch.cat([R[:, 0, :], R[:, 1, :]], dim=1)

    if config.feature_type == 'cs_only':
        full_input = torch.cat([u_input, a_input], 1)

    elif config.feature_type == 'cs + rapport':
        full_input = torch.cat([u_input, a_input, r_input], 1)

    else:
        full_input = torch.cat([u_input, a_input, at_input, r_input], 1)

    if config.model_type == 're':
        output = r_tr
    else:
        output = u_tr

    X = full_input.cpu().data.numpy()
    Y = output.cpu().data.numpy()

    return X, Y


def train_only_non_nn(X, Y):
    clf = ExtraTreesClassifier(random_state=0, n_estimators=63)
    clf.fit(X, Y)
    return clf


def eval_only_non_nn(clf, X, Y, print_info=False):
    Y_pred = clf.predict(X)
    acc = f1_score(Y, Y_pred, average='micro')

    if print_info:
        print("True y sum: ", np.sum(Y))
        print("Predicted y sum: ", np.sum(Y_pred))
        print("True y sum col: ", np.sum(Y, axis=0))
        print("Predicted y sum col: ", np.sum(Y_pred, axis=0))
        conf_mat = classification_report(Y, Y_pred)
        print("Confusion matrix: ", conf_mat)
        print("Accuracy: ", acc)

    return acc


if __name__ == '__main__':
    main()
