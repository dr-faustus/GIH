import torch.nn as nn

def conv_block(in_channels: int, out_channels: int, pool: bool = False, norm_layer: nn.Module = nn.BatchNorm2d,
               activation_function: nn.Module = nn.ReLU):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              norm_layer(out_channels), 
              activation_function(inplace=True)]
    if pool: layers.append(nn.AvgPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, norm_layer: nn.Module = nn.BatchNorm2d, 
                 activation_function: nn.Module = nn.ReLU):
        super().__init__()
        
        self.conv1 = conv_block(num_channels, 64, norm_layer=norm_layer, activation_function=activation_function)
        self.conv2 = conv_block(64, 128, pool=True, norm_layer=norm_layer, activation_function=activation_function)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True, norm_layer=norm_layer, activation_function=activation_function)
        self.conv4 = conv_block(256, 512, pool=True, norm_layer=norm_layer, activation_function=activation_function)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out)
        out = self.classifier(out)
        return out
