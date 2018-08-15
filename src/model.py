import torch
import torch.nn as nn
import torch.nn.functional as F


class JointEstimator(nn.Module):
    """Joint model for rapport estimator and social reasoner for social user simulator"""

    def __init__(self, input_size, hidden_size, output_size, leaky_slope, window_type, feature_type, model_type):
        super(JointEstimator, self).__init__()

        self.window_type = window_type
        self.feature_type = feature_type
        self.model_type = model_type

        window = 1
        if window_type in [2, 'linear_combination']:
            window = 2

        self.alpha = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))
        self.beta = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))
        self.gamma = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))
        self.delta = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))

        self.map1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.LeakyReLU(leaky_slope)
        self.map2 = nn.Linear(hidden_size, output_size)

    def forward(self, U, A, R, AT):
        if self.window_type == 1:
            u_input = U[:, :, 0]
            a_input = A[:, :, 0]
            at_input = AT[:, :, 0]
            r_input = R[:, 0][:, None]

        elif self.window_type == 2:
            R = R[:, :, None]
            u_input = torch.cat([U[:, :, 0], U[:, :, 1]], dim=1)
            a_input = torch.cat([A[:, :, 0], A[:, :, 1]], dim=1)
            at_input = torch.cat([AT[:, :, 0], AT[:, :, 1]], dim=1)
            r_input = torch.cat([R[:, 0, :], R[:, 1, :]], dim=1)

        else:
            u_input = torch.matmul(U, self.alpha[None, :, :]).squeeze(-1)
            a_input = torch.matmul(A, self.beta[None, :, :]).squeeze(-1)
            at_input = torch.matmul(AT, self.gamma[None, :, :]).squeeze(-1)
            r_input = torch.matmul(R, self.delta)

        if self.feature_type == 'cs_only':
            full_input = torch.cat([u_input, a_input], 1)

        elif self.feature_type == 'cs + rapport':
            full_input = torch.cat([u_input, a_input, r_input], 1)

        else:
            full_input = torch.cat([u_input, a_input, at_input, r_input], 1)

        output = self.map2(self.activation1(self.map1(full_input)))

        if self.model_type == 're':
            rapp = output
            return rapp
        else:
            cs_prob = F.sigmoid(output)
            return cs_prob
