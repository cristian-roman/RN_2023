import torch.nn as nn


class NeuralNetwork(nn.Module):

    @staticmethod
    def init_weights(m):
        if type(m) is nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 532),
            nn.ReLU(),
            nn.Linear(532, 532),
            nn.ReLU(),
            nn.Linear(532, 10),
            nn.Softmax(dim=1)
        )

        self.apply(self.init_weights)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        y = self.linear_relu_stack(x)
        return y
