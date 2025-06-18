"""
The main CheXNet model implementation. 
link : https://arxiv.org/pdf/1711.05225.pdf
"""

import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.densenet import DenseNet121_Weights

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of the model is the same as standard DenseNet121
    
    Input: a batch of image 3-channel images with spatial dimensions of 518x518 pixels (Height x Width)
            Input shape: (batch_size, 3, 518, 518)
    Output: 
        - classes: a tensor of shape (batch_size, 3) representing the classification output
        - csi_scores: a tensor of shape (batch_size, 6) representing the CSI scores output
        - mean_csi: a tensor of shape (batch_size, 1) representing the mean CSI output

    """
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_ftrs = self.densenet121.classifier.in_features

        self.classifier = nn.Linear(num_ftrs, 3) 
        self.csi_scores = nn.Linear(num_ftrs, 6)
        self.mean_csi = nn.Linear(num_ftrs, 1)

        self.relu = nn.ReLU(inplace=True)
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.densenet121.features(x)
        x = self.relu(x)
        x = self.adaptive_avgpool(x)
        x = torch.flatten(x, 1)

        classes = self.classifier(x)
        csi_scores = self.csi_scores(x)
        mean_csi = self.mean_csi(x)

        return classes, self.relu(csi_scores), self.relu(mean_csi)

def load_pretrained_ecg(model:DenseNet121):

    checkpoint_path = 'wkdir/pretrained_chesXNet/model_chesXNet.pth.tar'

    if os.path.isfile(checkpoint_path):
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']

        # Fix the key names to match the model's state_dict
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove the "module." prefix
            new_key = key.replace("module.", "")

            # Replace ".1" in layer names with "1"
            new_key = new_key.replace(".1", "1")

            new_state_dict[new_key] = value

        # Load the corrected state_dict
        model.load_state_dict(new_state_dict, strict=False)
        print("=> Loaded checkpoint")
    else:
        print("=> No checkpoint found at '{}'".format(checkpoint_path))

    return model

# initializing the model with the number of class == 14 as the we will laod some weights
model_chest_ray = DenseNet121()

# loading pretrained model
pretrained_model = load_pretrained_ecg(model_chest_ray)

