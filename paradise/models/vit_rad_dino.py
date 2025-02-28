import torch.nn as nn
from transformers import AutoModel
from transformers import AutoImageProcessor
# import torchvision

################################ ViT from microsof for chest radiography (CXR) #####################

class CustomDinoModel(nn.Module):
    def __init__(self):
        super(CustomDinoModel, self).__init__()

        # Initialize the model
        repo = "microsoft/rad-dino"
        self.model = AutoModel.from_pretrained(repo)
        self.processor = AutoImageProcessor.from_pretrained(repo)

        # change the classification layer
        self.classification_head =  nn.Linear(768, 3) 
        self.head_csi =  nn.Linear(768, 6) #head_csi(input_dim=768, output_dim=6), for the 6 image regions
        self.mean_csi =  nn.Linear(768, 1) # 

        self.relu = nn.ReLU(inplace=True)

    def forward(self, kwargs): # since the input is a dic, we reference the input as kwargs (key word arguments)

        outputs = self.model(kwargs)  # Get the encoder outputs
        cls_embeddings = outputs.pooler_output

        classification_output = self.classification_head(cls_embeddings)
        # csi_scores = self.head_csi(cls_embeddings)
        # mean_csi = self.mean_csi(cls_embeddings)

        csi_scores = self.relu(self.head_csi(cls_embeddings))
        mean_csi = self.relu(self.mean_csi(cls_embeddings))

        return classification_output, csi_scores, mean_csi