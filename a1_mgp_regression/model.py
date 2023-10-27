import torch.nn as nn


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.linear = nn.Linear(9, 1)
    def forward(self, x):
        out = self.linear(x)
        return out