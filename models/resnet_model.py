import torch
import torch.nn as nn
import torchvision

class ResNetModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        backbone = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V2)

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.backbone(x))