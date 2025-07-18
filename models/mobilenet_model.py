import torch
import torch.nn as nn
import torchvision

class MobileNetModel(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super().__init__()
        backbone = torchvision.models.mobilenet_v3_large(pretrained=False)
        # backbone = torchvision.models.mobilenet_v3_small(torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        self.features = backbone.features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(960, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.head(self.features(x))