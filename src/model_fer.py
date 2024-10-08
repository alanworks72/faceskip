import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetV3(nn.Module):
    def __init__(self, num_classes, weights=models.MobileNet_V3_Large_Weights.DEFAULT, feature_extract=False):
        super(MobileNetV3, self).__init__()

        self.model = models.mobilenet_v3_large(weights=weights)
        
        if feature_extract:
            self.FeatureExtract()

        in_features = self.model.classifier[0].in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.model(x)

        return x
    
    def FeatureExtract(self):
        for param in self.model.parameters():
            param.requires_grad = False


def loadModel(num_classes, weights=models.MobileNet_V3_Large_Weights.DEFAULT, feature_extract=False):
    model = MobileNetV3(num_classes=num_classes, weights=weights, feature_extract=feature_extract)

    return model