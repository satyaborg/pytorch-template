import torch
import torch.nn as nn
from torchvision import models

class SimpleConvNet(nn.Module):
    """Simple CNN; returns raw logits"""
    def __init__(self, n_classes: int):
        super(SimpleConvNet, self).__init__()
        self.n_classes = n_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=32, 
                      kernel_size=5, 
                      stride=1, 
                      bias=True
                     ), # [32, 220, 220]
            nn.MaxPool2d(kernel_size=2,
                         stride=2 # Default value: stride=kernel_size
                        ), # [32, 110, 110] 
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32) # channels
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=5, 
                      stride=1, 
                      bias=True
                     ), # [64, 106, 106]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2
                        ), # [64, 53, 53]
            nn.BatchNorm2d(num_features=64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, 
                      out_channels=256, 
                      kernel_size=5, 
                      stride=8, 
                      bias=True
                     ), # [256, 7, 7]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5,
                         stride=2
                        ), # [256, 2, 2]
            nn.BatchNorm2d(num_features=256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256*2*2, # calculated from get_out_size()
                      out_features=1024, 
                      bias=True
                     ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1024, 
                      out_features=self.n_classes, 
                      bias=True
                     )
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1) # flatten -> [bs, 1024]
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    """Pretrained VGG19_bn model"""
    def __init__(self, n_classes: int, 
                use_pretrained: bool=False, 
                feature_extract: bool=False
                ):
        super(VGG, self).__init__()
        self.n_classes = n_classes
        self.model = models.vgg11_bn(pretrained=use_pretrained, progress=True)
        self.set_parameter_requires_grad(feature_extract)
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, n_classes)

    def set_parameter_requires_grad(self, feature_extract):
        if feature_extract:
            # Freeze training for all layers
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)