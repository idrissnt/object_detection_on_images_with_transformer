"""
The main CheXNet model implementation.
"""
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.densenet import DenseNet121_Weights


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has 3 outputs classes.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def load_pretrained_ecg(model:DenseNet121):

    checkpoint_path = 'wkdir/pretrained/model_chesXNet.pth.tar'

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

# cudnn.benchmark = True

# initializing the model with the number of class == 14 as the we will laod some weights
model_chest_ray = DenseNet121(14)

# loading pretrained model
pretrained_model = load_pretrained_ecg(model_chest_ray)

# fine-tuning, replace 14 classes to 3 classes
nb_class = 3
pretrained_model.densenet121.classifier[0] = nn.Linear(1024, nb_class)
for param in pretrained_model.parameters():
    param.requires_grad = True
