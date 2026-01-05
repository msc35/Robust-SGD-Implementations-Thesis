"""
Neural network model architectures.

This module defines the CNN architectures used for each dataset:
- MNIST_CNN: Simple CNN for MNIST (28x28, 1 channel)
- VGG_Small: VGG-style CNN for CIFAR-100 (32x32, 3 channels)
- Cloud_ResNet18: ResNet-18 for CLOUD dataset (224x224, 3 channels)
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MNIST_CNN(nn.Module):
    """
    Simple Convolutional Neural Network for MNIST.
    
    Architecture:
    - Conv2d(1, 32) -> MaxPool2d
    - Conv2d(32, 64) -> MaxPool2d
    - Linear(64*7*7, 128)
    - Linear(128, 10)
    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VGG_Small(nn.Module):
    """
    VGG-Small architecture optimized for CIFAR-100 (32x32, 3 channels).
    
    Architecture:
    - Conv2d(3, 64) -> BatchNorm -> ReLU -> MaxPool2d
    - Conv2d(64, 128) -> BatchNorm -> ReLU -> MaxPool2d
    - Conv2d(128, 256) -> BatchNorm -> ReLU -> MaxPool2d
    - Linear(256*4*4, 512) -> ReLU -> Dropout(0.5)
    - Linear(512, num_classes)
    """
    def __init__(self, num_classes=100):
        super(VGG_Small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Cloud_ResNet18(nn.Module):
    """
    ResNet-18 architecture for CLOUD dataset (224x224, 3 channels).
    
    Uses ImageNet pretrained weights for better generalization on small datasets.
    The final fully connected layer is replaced to match the number of classes.
    
    Args:
        num_classes: Number of output classes
    """
    def __init__(self, num_classes):
        super(Cloud_ResNet18, self).__init__()
        # Use ImageNet pretrained weights for transfer learning
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Replace the final Fully Connected layer to match Cloud classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

