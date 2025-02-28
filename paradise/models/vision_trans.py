# from typing import Optional
import timm.models.vision_transformer
from functools import partial
import torch.nn as nn
import torch


####### Turgut et al ECG encoder  
class ViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, **kwargs):
        super(ViT, self).__init__(**kwargs)

        self.csi_scores = nn.Linear(self.embed_dim, 6)
        self.classification_output = nn.Linear(self.embed_dim, 3)
        self.mean_csi = nn.Linear(self.embed_dim, 1)

        # print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward_features(self, x):

        # print(x.shape)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        x = self.blocks(x)
        x = self.norm(x)

        return x 
    
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x) # x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)

        csi_scores = self.csi_scores(x)
        classification_output = self.classification_output(x)
        mean_csi = self.mean_csi(x)
        return classification_output, csi_scores, mean_csi
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

kwargs = {'patch_size': (1,100), 'img_size': (518,518), 'in_chans': 3, 'embed_dim': 384,
          'depth':3, 'num_heads':6, 'mlp_ratio':4, 'qkv_bias':True,  'norm_layer':partial(nn.LayerNorm, eps=1e-6)}
initial_vit = ViT(**kwargs)

