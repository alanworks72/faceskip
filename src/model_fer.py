import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV3(nn.Module):
    def __init__(self, num_classes, pretrained=True, feature_extract=False):
        super(MobileNetV3, self).__init__()

        self.model = models.MobileNetV3(pretrained=pretrained)
        
        if feature_extract:
            self.FeatureExtract()

        in_features = self.model.classifier[3].in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    
    def FeatureExtract(self):
        for param in self.model.parameters():
            param.requires_grad = False


def loadModel(num_classes, pretrained=True, feature_extract=True):
    model = MobileNetV3(num_classes=num_classes, pretrained=pretrained, feature_extract=feature_extract)

    return model