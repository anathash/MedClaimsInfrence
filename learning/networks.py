
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self, num_classes=5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Layer:
    def __init__(self, input, output):
        self.input = input
        self.output = output


class TwoLayersNet(nn.Module):

    def __init__(self, layers):
        super(TwoLayersNet, self).__init__()
        self.fc1 = nn.Linear(18, 40)
        self.fc2 = nn.Linear(40, 10)
        #self.fc1 = nn.Linear(layers[0].input, layers[0].output)
        #self.fc2 = nn.Linear(layers[1].input, layers[1].output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
