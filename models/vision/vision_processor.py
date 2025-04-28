import cv2
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VisionProcessor(nn.Module):
    def __init__(self, input_channels=3, output_dim=256):
        super(VisionProcessor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # TODO: Adjust the output dimension based on the input size
        self.fc = nn.Linear(64 * 7 * 7, output_dim)  # Adjust based on input size
        self.relu = nn.ReLU()