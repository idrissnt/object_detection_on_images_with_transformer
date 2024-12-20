from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from transformers import AutoModel
# import torchvision
from transformers import AutoImageProcessor

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

    def forward(self, kwargs): # since the input a dic, we reference the input as kwargs (key word arguments)
        
        # plt.figure()
        # plt.imshow(kwargs[0, 0].cpu().detach().numpy())
        # plt.savefig('yoo-input.png')

        outputs = self.model(kwargs)  # Get the encoder outputs
        cls_embeddings = outputs.pooler_output

        # flat_patch_embeddings = outputs.last_hidden_state[:, :-1]
        flat_patch_embeddings = outputs.last_hidden_state[:, 1:]
        reshaped_patch_embeddings = reshape_patch_embeddings(flat_patch_embeddings, self.model, self.processor)

        plt.figure()
        plt.imshow(reshaped_patch_embeddings[0, 10].cpu().detach().numpy())
        plt.savefig('yoo-37_.png')

        classification_output = self.classification_head(cls_embeddings)
        csi_scores = self.head_csi(cls_embeddings)
        mean_csi = self.mean_csi(cls_embeddings)

        mean_csi_from_scores = csi_scores.mean(dim=1, keepdim=True)

        print(outputs.last_hidden_state.shape)
        print(flat_patch_embeddings.shape)
        print(reshaped_patch_embeddings.shape)
        print(cls_embeddings.shape)

        print(cls_embeddings.shapec)

        return classification_output, csi_scores, mean_csi, mean_csi_from_scores

def extract_zone_features(zones, feature_extractor, model):
    zone_features = []
    for zone in zones:
        inputs = feature_extractor(images=zone, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            cls_token = outputs.pooler_output  # Shape: (batch_size, num_channels)
            zone_features.append(cls_token)
            return zone_features

def reshape_patch_embeddings(flat_tokens: torch.Tensor, model, processor) -> torch.Tensor:
    """Reshape flat list of patch tokens into a nice grid."""
    from einops import rearrange
    image_size = processor.crop_size["height"]
    patch_size = model.config.patch_size
    embeddings_size = image_size // patch_size # embeddings_size == 37

    patches_grid = rearrange(flat_tokens, "b (h w) c -> b c h w", h=embeddings_size)
    return patches_grid

# zone_features = extract_zone_features(zones, feature_extractor, model)

# print(CustomDinoModel().model)

########################################################################################
    
#  256 => 224, 1369 => 518 
