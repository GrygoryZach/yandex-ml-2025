from torch import nn


class MiniSimpleton(nn.Module):
    def __init__(self):
        super(MiniSimpleton, self).__init__()

        self.flatten = nn.Flatten(start_dim=-3)
        self.fc1 = nn.Linear(28 * 28, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
