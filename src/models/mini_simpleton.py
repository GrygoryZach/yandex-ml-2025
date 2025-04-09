from torch import nn


class MiniSimpleton(nn.Module):
    def __init__(self, device):
        super(MiniSimpleton, self).__init__()
        self.relu = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(2, 2)
        self.maxpool_4 = nn.MaxPool2d(4, 4)

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(32 * 16 * 32, 20)

        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool_2(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool_2(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool_4(x)

        x = self.flatten(x)

        x = self.fc(x)

        return x
