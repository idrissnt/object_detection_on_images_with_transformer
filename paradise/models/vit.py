
import torch.nn as nn
from transformers import AutoModel

################################ ViT from microsof for chest radiography (CXR) #####################

# Initialize the model
repo = "microsoft/rad-dino"
model = AutoModel.from_pretrained(repo)

class RegressionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=1):
        super(RegressionHead, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x is [batch_size, seq_len, 768]
        # Take the mean across the sequence dimension (for example, we could also use only the CLS token output)
        x = x.mean(dim=1)  # [batch_size, 768]
        x = self.dense(x)  # [batch_size, 4]
        return x

class CustomDinoModel(nn.Module):
    def __init__(self, model):
        super(CustomDinoModel, self).__init__()
        self.model = model
        self.regression_head = RegressionHead(input_dim=768, output_dim=4)

    def forward(self, kwargs):
        outputs = self.model(kwargs)  # Get the encoder outputs
        sequence_output = outputs.last_hidden_state  # or encoder output
        regression_output = self.regression_head(sequence_output)
        return regression_output

########################################################################################

