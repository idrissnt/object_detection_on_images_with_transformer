import torch
import torch.nn as nn
from transformers import AutoModel
import torchvision
from einops import rearrange
from transformers import AutoImageProcessor


################################ ViT from microsof for chest radiography (CXR) #####################

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super(ClassificationHead, self).__init__()

        num_class = output_dim
        self.fc = nn.Linear(input_dim, num_class)

    def forward(self, x):
        logit = self.fc(x)
        return logit

class CustomDinoModel(nn.Module):
    def __init__(self):
        super(CustomDinoModel, self).__init__()

        # Initialize the model
        repo = "microsoft/rad-dino"
        self.model = AutoModel.from_pretrained(repo)

        # change the classification layer
        self.classification_head =  nn.Linear(768, 3) #ClassificationHead(input_dim=768, output_dim=3)
        # self.head_csi =  nn.Linear(768, 6) #head_csi(input_dim=768, output_dim=6), for the 6 image regions

    def forward(self, kwargs): # since the input a dic, we reference the input as kwargs (key word arguments)
        
        outputs = self.model(kwargs)  # Get the encoder outputs
        cls_embeddings = outputs.pooler_output

        classification_output = self.classification_head(cls_embeddings)
        # csi_scores = self.head_csi(cls_embeddings)
        # mean_csi = csi_scores.mean(dim=1, keepdim=True)

        # return classification_output, csi_scores, mean_csi
        return classification_output

def extract_zone_features(zones, feature_extractor, model):
    zone_features = []
    for zone in zones:
        inputs = feature_extractor(images=zone, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            cls_token = outputs.pooler_output  # Shape: (batch_size, num_channels)
            zone_features.append(cls_token)
            return zone_features

    zone_features = extract_zone_features(zones, feature_extractor, model)

########################################################################################
    

