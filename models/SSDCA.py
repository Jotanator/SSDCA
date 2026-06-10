
import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
import numpy as np
import os

class SiameseSwinSCross_dual(nn.Module):
    def __init__(
        self,
        model_name: str = "swin_small_patch4_window7_224",
        pretrained: bool = True,
        feature_dim: int = 768,
        num_heads: int = 8,
        mlp_hidden: int = 512,
        dropout: int = 0.3
    ):
        super().__init__()

        base = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        # self.base = base
        self.patch_embed = base.patch_embed
        self.stem = base.layers[0]
        self.stage1 = base.layers[1]
        self.stage2 = base.layers[2]
        self.stage3 = base.layers[3]
        self.norm = base.norm
        self.head_swin = base.head
        self.feature_dim = feature_dim
        self.norm_attn = nn.LayerNorm(feature_dim, eps=1e-5, elementwise_affine=True)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(2*feature_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 2),
        )
        # storage for activations & grads
        self.activations = {}
        self.gradients = {}
        self._hook_handles = []
        self._register_hooks_on_last_norm1()


    def _register_hooks_on_last_norm1(self):
        try:
            target_layer = self.stage3.blocks[-1].norm1
        except Exception as e:
            raise RuntimeError(f"Could not locate layers[-1].blocks[-1].norm1 on {type(self.stage3)}") from e

        def fwd_hook(module, inp, out):
            self.activations['feat'] = out.detach()

        def bwd_hook(module, grad_input, grad_output):
            self.gradients['feat'] = grad_output[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(bwd_hook))


    def forward_once(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.norm(x)
        return x


    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x = torch.cat([x1, x2], dim=0)
        feats = self.forward_once(x)
        B = x1.size(0)
        feat1, feat2 = feats[:B], feats[B:]

        B, H, W, C = feat1.shape
        feat1 = feat1.view(B, H * W, C)  # [8, 49, 768]
        feat2 = feat2.view(B, H * W, C)  # [8, 49, 768]

        q = feat1
        k = feat2
        v = feat2
        attn_out, attn_weights1 = self.cross_attn(q, k, v)
        attn_feat1 = attn_out.squeeze(1)
        attn_feat1 = self.norm_attn(feat1 + attn_out)

        q = feat2
        k = feat1
        v = feat1
        attn_out, attn_weights2 = self.cross_attn(q, k, v)
        attn_feat2 = attn_out.squeeze(1)
        attn_feat2 = self.norm_attn(feat2 + attn_out)

        attn_feat1_og = attn_feat1.view(B, H, W, C)
        attn_feat2_og = attn_feat2.view(B, H, W, C)

        attn_feat1 = self.head_swin(attn_feat1_og)
        attn_feat2 = self.head_swin(attn_feat2_og)

        combined = torch.cat((attn_feat1, attn_feat2), dim=1)  

        embedding = self.head[:-1](combined)
        out = self.head(combined)
        return out.squeeze(1), embedding#, attn_weights1, attn_weights2