import torch
import torch.nn as nn
import torch.nn.functional as F


class JointEstimator(nn.Module):
    """Joint model for rapport estimator and social reasoner for social user simulator"""

    def __init__(self, input_size, hidden_size, output_size, leaky_slope, window):
        super(JointEstimator, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))
        self.beta = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))
        self.map1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.LeakyReLU(leaky_slope)
        self.map2 = nn.Linear(hidden_size, output_size)
        self.gamma = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))

    def forward(self, U, A, R, AT):
        u_input = torch.matmul(U, self.alpha[None, :, :]).squeeze(-1)
        a_input = torch.matmul(A, self.beta[None, :, :]).squeeze(-1)
        at_input = torch.matmul(AT, self.gamma[None, :, :]).squeeze(-1)

        # u_input = U[:, :, 0]
        # a_input = A[:, :, 0]
        # at_input = AT[:, :, 0]

        full_input = torch.cat([u_input, a_input, R], 1)
        output = self.map2(self.activation1(self.map1(full_input)))
        rapp = output[:, 0]
        cs_prob = F.sigmoid(output[:, 1:])
        # rapp = output
        # cs_prob = None
        return rapp, cs_prob


class SocialReasoner(nn.Module):
    """Model for social reasoner for social user simulator"""

    def __init__(self, input_size, hidden_size, output_size, leaky_slope, window):
        super(SocialReasoner, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))
        self.beta = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))
        self.map1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.LeakyReLU(leaky_slope)
        self.map2 = nn.Linear(hidden_size, output_size)
        self.gamma = nn.Parameter(torch.FloatTensor(window, 1).fill_(0.5))

    def forward(self, U, A, R, AT):
        u_input = torch.matmul(U, self.alpha[None, :, :]).squeeze(-1)
        a_input = torch.matmul(A, self.beta[None, :, :]).squeeze(-1)
        at_input = torch.matmul(AT, self.gamma[None, :, :]).squeeze(-1)

        # u_input = U[:, :, 0]
        # a_input = A[:, :, 0]
        # at_input = AT[:, :, 0]

        full_input = torch.cat([u_input, a_input, R, at_input], 1)
        output = self.map2(self.activation1(self.map1(full_input)))
        cs_prob = F.sigmoid(output)
        return None, cs_prob
