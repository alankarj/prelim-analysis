import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, je, lr, n_epochs, print_every, thresh, model_type, eval_data=None):
        self.je = je
        self.lr = lr
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.thresh = thresh
        self.model_type = model_type

        self.loss_fn = None
        self.optimizer = None
        self.eval_data = eval_data

        self.set_losses()

    def set_losses(self):
        if self.model_type == 're':
            self.loss_fn = torch.nn.MSELoss()
        else:
            self.loss_fn = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.je.parameters(), lr=self.lr, weight_decay=0)

    def accuracy(self, prob_pred, y_true, print_info=False):
        """
        :param prob_pred: numpy array of probabilities for each CS
        :param y_true: true CS numpy array
        :return: accuracy of prediction, y_pred
        """
        y_pred = prob_pred.copy()
        y_pred[prob_pred >= self.thresh] = 1
        y_pred[prob_pred < self.thresh] = 0
        acc = f1_score(y_true, y_pred, average='micro')

        if print_info:
            print("True y sum: ", np.sum(y_true))
            print("Predicted y sum: ", np.sum(y_pred))
            print("True y sum col: ", np.sum(y_true, axis=0))
            print("Predicted y sum col: ", np.sum(y_pred, axis=0))
            conf_mat = classification_report(y_true, y_pred)
            print("Confusion matrix: ", conf_mat)
        return y_pred, acc

    def train(self, u_tr, r_tr, R, U, A, AT):
        print("Rapport shape: ", R.shape)
        print("User CS shape: ", U.shape)
        print("Agent CS shape: ", A.shape)
        print("Agent TS shape: ", AT.shape)

        best_epoch = 0
        eval_loss = float("inf")

        for i in range(self.n_epochs):
            self.optimizer.zero_grad()
            output = self.je(U, A, R, AT)

            if self.model_type == 're':
                loss = self.loss_fn(output, r_tr)
            else:
                loss = self.loss_fn(output, u_tr)

            loss.backward()
            self.optimizer.step()

            if self.eval_data is not None:
                temp_loss = self.eval(self.eval_data)
                eval_loss = min(eval_loss, temp_loss)
                if eval_loss == temp_loss:
                    best_epoch = i

            if i % self.print_every == 0:
                if self.model_type == 're':
                    print("Epoch: %d, Loss: %.3f" % (i, loss))
                else:
                    _, acc = self.accuracy(output.cpu().data.numpy(), u_tr.cpu().data.numpy())
                    print("Epoch: %d, Loss: %.3f, " % (i, loss), end='')
                    print("CS prediction accuracy: %.3f" % acc)

        if self.eval_data is not None:
            print("Best eval loss: %.3f at epoch: %d." % (eval_loss, best_epoch))

        return eval_loss, best_epoch

    def eval(self, eval_data, print_info=False):
        u_tr, r_tr, R, U, A, AT = eval_data
        self.je.eval()
        output = self.je(U, A, R, AT)

        if self.model_type == 're':
            loss = self.loss_fn(output, r_tr)
        else:
            loss = self.loss_fn(output, u_tr)
            _, acc = self.accuracy(output.cpu().data.numpy(), u_tr.cpu().data.numpy(), print_info=print_info)

        if print_info:
            print("Validation stats:")
            if self.model_type == 're':
                print("Loss: %.3f" % loss)
            else:
                print("Loss: %.3f, CS prediction accuracy: %.3f" % (loss, acc))

        if self.model_type == 're':
            return loss
        else:
            return -1 * acc
