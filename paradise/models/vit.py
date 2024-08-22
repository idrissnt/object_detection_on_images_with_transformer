
import torch.nn as nn
from transformers import AutoModel, ViTModel
from transformers import AutoImageProcessor

################################ ViT from microsof for chest radiography (CXR) #####################

# Initialize the processor and model
repo = "microsoft/rad-dino"
model = AutoModel.from_pretrained(repo)
processor = AutoImageProcessor.from_pretrained(repo)

# Access the final MLP layer in the encoder
class CustomDinov2MLP(nn.Module):
    def __init__(self):
        super(CustomDinov2MLP, self).__init__()

        self.fc1 = nn.Linear(in_features=768, out_features=3072, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(in_features=3072, out_features=6, bias=True)  # Changed to output 6 value for 6 regions

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        scores = self.fc2(x)  # No activation after this, for regression
        return scores

# Replace the original MLP with the custom one
for layer in model.encoder.layer:
    layer.mlp = CustomDinov2MLP()

ViTForScoring = model

# Model is now adapted for regression    
# print(ViTForScoring)

########################################################################################


################################ initial ViT from google ###############################

pretrained_model_name='google/vit-base-patch16-224'
vit = ViTModel.from_pretrained(pretrained_model_name)

class ViTForScoring(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(ViTForScoring, self).__init__()
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.vit.config.hidden_size, 6)  # 6 outputs for 6 regions

    def forward(self, x):
        outputs = self.vit(x).last_hidden_state[:, 0, :]
        scores = self.classifier(outputs)
        return scores
# Model is now adapted for regression    
print(ViTForScoring())
##########################################################################################


