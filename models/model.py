import torch
from torch import nn
from torch.nn import Sigmoid


class FocusClassifier(nn.Module):
    def __init__(self, num_classes, img_size=(40, 32)):
        super().__init__()
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        temp_tensor = torch.zeros(img_size).unsqueeze(0).unsqueeze(0)
        temp_out = self.cnn(temp_tensor)
        n_flat_features = torch.flatten(temp_out, 1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(n_flat_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FocusPredictor(FocusClassifier):
    def __init__(self):
        super().__init__(1)

    def forward(self, x):
        x = super().forward(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    predictor = FocusPredictor()
