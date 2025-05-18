import torch.nn as nn

from .models.lenet import LeNet
from .models.mlp import MLP
from .models.resnet import ResNet18
from .models.resnet9 import ResNet9
from .models.vgg import VGG11
from .models.vit import VisionTransformer
from .models.small_net import SmallLeNet

class NoNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)

def model_gen_fun(model_name: str, activation_function: nn.Module, num_classes: int = 2, num_channels: int = 3):
    assert model_name in ['LeNet', 'SmallLeNet', 'VGG11', 'VGG11_wo_bn', 
                          'ResNet18', 'ResNet18_wo_bn', 'ResNet9', 'ResNet9_wo_bn',
                          'MLP', 'ViT', 'ViT_wo_ln']
    if model_name == 'LeNet':
        model = LeNet(num_classes=num_classes, num_channels=num_channels, activation_function=activation_function).eval()
    elif model_name == 'SmallLeNet':
        model = SmallLeNet(num_classes=num_classes, num_channels=num_channels, activation_function=activation_function).eval()
    elif model_name == 'VGG11':
        model = VGG11(num_classes=num_classes, num_channels=num_channels, batch_norm=True, activation_function=activation_function).eval()
    elif model_name == 'VGG11_wo_bn':
        model = VGG11(num_classes=num_classes, num_channels=num_channels, batch_norm=False, activation_function=activation_function).eval()
    elif model_name == 'ResNet18':
        model = ResNet18(num_classes=num_classes, num_channels=num_channels, activation_function=activation_function).eval()
    elif model_name == 'ResNet18_wo_bn':
        model = ResNet18(num_classes=num_classes, num_channels=num_channels, norm_layer=NoNorm, activation_function=activation_function).eval()
    elif model_name == 'ResNet9':
        model = ResNet9(num_classes=num_classes, num_channels=num_channels, activation_function=activation_function).eval()
    elif model_name == 'ResNet9_wo_bn':
        model = ResNet9(num_classes=num_classes, num_channels=num_channels, norm_layer=NoNorm, activation_function=activation_function).eval()
    elif model_name == 'MLP':
        model = MLP(num_classes=num_classes, num_channels=num_channels, hidden_size=100, activation_function=activation_function).eval()
    elif model_name == 'ViT':
        model = VisionTransformer(num_classes=num_classes, num_channels=num_channels, image_size=32, 
                                  patch_size=8, num_layers=9, num_heads=12, hidden_dim=192, mlp_dim=768, activation_function=activation_function).eval()
    elif model_name == 'ViT_wo_ln':
        model = VisionTransformer(num_classes=num_classes, num_channels=num_channels, image_size=32, 
                                  patch_size=8, num_layers=9, num_heads=12, hidden_dim=192, mlp_dim=768, norm_layer=NoNorm, activation_function=activation_function).eval()
    return model