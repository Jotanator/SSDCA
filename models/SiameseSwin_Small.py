import torch
import torch.nn as nn
import timm

class SiameseSwinS(nn.Module):
    def __init__(self, model_name='swin_small_patch4_window7_224', pretrained=True, feature_dim=768):
        super(SiameseSwinS, self).__init__()
        # Shared backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = feature_dim
        
        self.head = nn.Sequential(
            nn.Linear(2*feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),         # Binary output
        )

    def forward_once(self, x):
        x = self.backbone.patch_embed(x)
        x = self.backbone.layers[0](x)
        x = self.backbone.layers[1](x)
        x = self.backbone.layers[2](x)
        x = self.backbone.layers[3](x)
        x = self.backbone.norm(x)
        return self.backbone.head(x), x

    def forward(self, x1, x2):
        feat1, attn_embed1 = self.forward_once(x1)
        feat2, attn_embed2 = self.forward_once(x2)
        combined = torch.cat((feat1, feat2), dim=1)
        out = self.head(combined)
        embedding = self.head[:-1](combined)
        return out.squeeze(1), embedding#, attn_embed1, attn_embed2