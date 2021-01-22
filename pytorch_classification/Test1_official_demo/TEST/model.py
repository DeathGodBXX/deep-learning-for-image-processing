import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)  # input: (3, 32, 32) output: (16, 28, 28)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # output: (16, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, 5)  # output: (32, 10, 10)
        self.pool2 = nn.MaxPool2d(2, 2)  # output: (32, 5, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)  # output: (120)
        self.fc2 = nn.Linear(120, 84)  # output: (84)
        self.fc3 = nn.Linear(84, 10)  # output: (10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
