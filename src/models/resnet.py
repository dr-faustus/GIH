import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, norm_layer: nn.Module = nn.BatchNorm2d, activation_function: nn.Module = nn.ReLU):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

        self.ac = activation_function()

    def forward(self, x):
        out = self.ac(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.ac(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: nn.Module, num_blocks: int, num_channels: int = 3, num_classes: int = 10, 
                 norm_layer: nn.Module = nn.BatchNorm2d, activation_function: nn.Module = nn.ReLU):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm_layer=norm_layer)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.ac = activation_function()

    def _make_layer(self, block, planes, num_blocks, stride, norm_layer):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.ac(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet18(ResNet):

    def __init__(self, num_channels: int = 3, num_classes: int = 10, 
                 norm_layer: nn.Module = nn.BatchNorm2d, activation_function: nn.Module = nn.ReLU):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_channels=num_channels, 
                                       num_classes=num_classes, norm_layer=norm_layer, activation_function=activation_function)
