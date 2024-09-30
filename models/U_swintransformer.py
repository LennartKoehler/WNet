 
import models.swin_transformer_v2_1d as st1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UncompatibleInputException(Exception):
    pass

# no pretrained windows

class U_swintransformer(nn.Module):
    def __init__(self,
            num_classes=100,
            embed_dim=96,
            img_size=256,
            patch_size=4,
            in_chans=1,
            depths_enc=[2, 2, 2],
            num_heads_enc=[3, 6, 12],
            depths_dec=[2, 2, 2, 2],
            num_heads_dec=[3, 6, 12, 24],
            window_size=8, mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[0, 0, 0, 0]):
        super().__init__()

        if len(depths_enc) != len(num_heads_enc):
            raise UncompatibleInputException("depths_enc and num_heads_enc must have the same length")
        if len(depths_dec) != len(num_heads_dec):
            raise UncompatibleInputException("depths_dec and num_heads_dec must have the same length")
        if len(depths_enc) != len(depths_dec)-1:
            raise UncompatibleInputException("depths_enc must be one layer shorter than depths_dec to make sure input shape equals output shape")
                                             

        self.enc = st1d.SwinTransformerV21D(img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths_enc,
            num_heads=num_heads_enc,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes)
        
        self.middle_channels = self.enc.out_channels
        self.middle_resolution = self.enc.out_resolution

        self.dec = st1d.SwinTransformerV21D_reverse(img_size=self.middle_resolution,
            patch_size=patch_size,
            in_chans=self.middle_channels,
            num_classes=num_classes,
            embed_dim=self.middle_channels,
            depths=depths_dec,
            num_heads=num_heads_dec,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            pretrained_window_sizes=pretrained_window_sizes)
        
        self.out_channels = self.dec.out_channels
        self.out_resolution = self.dec.out_resolution

    def forward(self, x, returns='both'):
        enc = self.enc(x)
        if returns == "enc":
            return enc
        dec = self.dec(F.softmax(enc, 1))
        if returns == "dec":
            return dec
        if returns == "both":
            return enc, dec
