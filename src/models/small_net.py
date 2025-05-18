import torch.nn as nn
import torch.nn.functional as F

class SmallLeNet(nn.Module):
    def __init__(self, num_channels: int = 3, num_classes: int = 10, activation_function: nn.Module = nn.ReLU()):
        super(SmallLeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 96)
        self.fc2 = nn.Linear(96, 67)
        self.fc3 = nn.Linear(67, num_classes)

        self.ac = activation_function()

    def forward(self, x):
        out = self.ac(self.conv1(x))
        out = F.avg_pool2d(out, 2)
        out = self.ac(self.conv2(out))
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.ac(self.fc1(out))
        out = self.ac(self.fc2(out))
        out = self.fc3(out)
        return out