import torch.nn as nn


class NeuralNetwork(nn.Module):
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

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        y = self.linear_relu_stack(x)
        return y
