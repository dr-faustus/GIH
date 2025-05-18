import torch.nn as nn
import math

vgg_cfg = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']}


class VGG(nn.Module):
    def __init__(self, vgg_name: str, batch_norm: bool = False, num_channels: int = 3, num_classes: int = 10, activation_function: nn.Module = nn.ReLU):
        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.batch_norm = batch_norm
        self.num_channels = num_channels
        self.features = self._make_layers(vgg_cfg[vgg_name], activation_function)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            activation_function(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            activation_function(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, vgg_cfg, activation_function):
        layers = []
        in_channels = self.num_channels
        for x in vgg_cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               activation_function(inplace=True)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               activation_function(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG11(VGG):
    def __init__(self, num_channels=3, num_classes=10, batch_norm: bool = True, activation_function: nn.Module = nn.ReLU):
        super(VGG11, self).__init__(vgg_name='VGG11', num_channels=num_channels, num_classes=num_classes, batch_norm=batch_norm, activation_function=activation_function)
