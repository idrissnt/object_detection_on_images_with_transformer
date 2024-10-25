import torch.nn as nn
from transformers import AutoModel

################################ ViT from microsof for chest radiography (CXR) #####################

# Initialize the model
repo = "microsoft/rad-dino"
model = AutoModel.from_pretrained(repo)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super(ClassificationHead, self).__init__()

        num_class = output_dim

        self.fc1 = nn.Linear(input_dim, num_class)
        self.fc2 = nn.Linear(input_dim, num_class)
        self.fc3 = nn.Linear(input_dim, num_class)
        self.fc4 = nn.Linear(input_dim, num_class)
        # self.fc5 = nn.Linear(input_dim, num_class)
        # self.fc6 = nn.Linear(input_dim, num_class)

    def forward(self, x):
        # right_sup ,left_sup ,right_mid ,left_mid ,right_inf ,left_inf
        logit_right_sup = self.fc1(x) 
        logit_left_sup = self.fc2(x) 
        logit_right_mid = self.fc3(x)  
        logit_left_mid = self.fc4(x) 
        # logit_right_inf = self.fc5(x) 
        # logit_left_inf = self.fc6(x) 

        # return [logit_right_sup, logit_left_sup, logit_right_mid, logit_left_mid, logit_right_inf, logit_left_inf]
        return [logit_right_sup, logit_left_sup, logit_right_mid, logit_left_mid]

class CustomDinoModel(nn.Module):
    def __init__(self, model):
        super(CustomDinoModel, self).__init__()
        self.model = model
        self.classification_head = ClassificationHead(input_dim=768, output_dim=4)

    def forward(self, kwargs):

        outputs = self.model(kwargs)  # Get the encoder outputs
        cls_embeddings = outputs.pooler_output
        # sequence_output = outputs.last_hidden_state  # or encoder output
        classification_output = self.classification_head(cls_embeddings)

        return classification_output

########################################################################################
    

