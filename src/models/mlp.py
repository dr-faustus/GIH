import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, num_channels: int = 3, hidden_size: int = 200, activation_function: nn.Module = nn.ReLU):
        super(MLP, self).__init__()

        self.layer_1 = nn.Linear(num_channels * 32 * 32, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.classification = nn.Linear(hidden_size, num_classes)

        self.ac = activation_function()

    def forward(self, x):
        out = self.ac(self.layer_1(x.reshape(x.shape[0], -1)))
        out = self.ac(self.layer_2(out))
        return self.classification(out)