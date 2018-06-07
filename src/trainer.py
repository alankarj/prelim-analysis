import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, je, lr, n_epochs, print_every, thresh, social=False):
        self.je = je
        self.lr = lr
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.thresh = thresh
        self.social = social

        self.loss_fn_rapp = None
        self.loss_fn_cs = None
        self.optimizer = None

        self.set_losses()

    def set_losses(self):
        self.loss_fn_rapp = torch.nn.MSELoss()
        self.loss_fn_cs = torch.nn.BCELoss()
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
            # print("Predicted probability: ", prob_pred)
            # print("True y: ", y_true)
            # print("Predicted y: ", y_pred)
            print("True y sum: ", np.sum(y_true))
            print("Predicted y sum: ", np.sum(y_pred))
            print("True y sum col: ", np.sum(y_true, axis=0))
            print("Predicted y sum col: ", np.sum(y_pred, axis=0))
            conf_mat = classification_report(y_true, y_pred)
            print("Confusion matrix: ", conf_mat)
        return y_pred, acc

    def train(self, u_tr, r_tr, R, U, A, AT):
        print(R.shape)
        print(U.shape)
        print(A.shape)
        print(AT.shape)
        for i in range(self.n_epochs):
            self.optimizer.zero_grad()
            rapp_pred, cs_pred = self.je(U, A, R, AT)
            # print(cs_pred)
            if not self.social:
                loss_rapp = self.loss_fn_rapp(rapp_pred, r_tr.squeeze(1))
                # tot_loss = loss_rapp
                loss_cs = self.loss_fn_cs(cs_pred, u_tr)
                tot_loss = loss_rapp + loss_cs
            else:
                loss_cs = self.loss_fn_cs(cs_pred, u_tr)
                tot_loss = loss_cs

            tot_loss.backward()
            self.optimizer.step()
            if i % self.print_every == 0:
                # print("Epoch: %d, Rapport loss: %.3f, CS loss: %.3f, CS prediction accuracy: %.2f" % (i, loss_rapp, loss_cs, acc*100))
                if not self.social:
                    # print("Epoch: %d, Total loss: %.3f" % (i, tot_loss))
                    _, acc = self.accuracy(cs_pred.cpu().data.numpy(), u_tr.cpu().data.numpy())
                    print("Epoch: %d, Rapport loss: %.3f, CS loss: %.3f, CS prediction accuracy: %.2f" % (i, loss_rapp, loss_cs, acc * 100))
                    #print("Epoch: %d, Rapport loss: %.3f" % (i, tot_loss))

                else:
                    _, acc = self.accuracy(cs_pred.cpu().data.numpy(), u_tr.cpu().data.numpy())
                    print("Epoch: %d, Total loss: %.3f, CS prediction accuracy: %.2f" % (i, loss_cs, acc * 100))

    def eval(self, u_tr, r_tr, R, U, A, AT):
        self.je.eval()
        rapp_pred, cs_pred = self.je(U, A, R, AT)
        if not self.social:
            loss_rapp = self.loss_fn_rapp(rapp_pred, r_tr.squeeze(1))
            loss_cs = self.loss_fn_cs(cs_pred, u_tr)
            tot_loss = loss_rapp + loss_cs
            _, acc = self.accuracy(cs_pred.cpu().data.numpy(), u_tr.cpu().data.numpy(), print_info=True)
        else:
            loss_cs = self.loss_fn_cs(cs_pred, u_tr)
            _, acc = self.accuracy(cs_pred.cpu().data.numpy(), u_tr.cpu().data.numpy(), print_info=True)
        print("Validation stats:")
        if not self.social:
            print("Rapp loss: %.3f" % (loss_rapp))
            print("CS loss: %.3f" % (loss_cs))
            print("Total loss: %.3f, CS prediction accuracy: %.2f" % (tot_loss, acc * 100))
        else:
            print("Total loss: %.3f, CS prediction accuracy: %.2f" % (loss_cs, acc * 100))

