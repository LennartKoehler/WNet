from models.U_swintransformer import U_swintransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UncompatibleInputException(Exception):
    pass

class connection(nn.Module):
    def __init__(self, middle_dim, embed_dim):
        super().__init__()

        self.match_channels = nn.Linear(middle_dim, embed_dim, bias = False)

    def forward(self, enc):
        enc_ = F.softmax(enc,1) # softmax necessary?
        enc_ = enc_.transpose(1,2)
        enc_ = self.match_channels(enc_)
        enc_ = enc_.transpose(1,2) # have to transpose back and forth to fit the dimensionality of nn.Linear
        return enc_

def init_W_swintransformer(num_classes=100,
        embed_dim=96,
        img_size=256,
        patch_size=2,
        in_chans=1,
        depths_enc=[2, 2, 2],
        num_heads_enc=[2, 6, 12],
        depths_dec=[2, 2, 2],
        num_heads_dec=[12, 6, 3],
        window_size=8, mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0]):    

    if len(depths_enc) != len(num_heads_enc):
        raise UncompatibleInputException("depths_enc and num_heads_enc must have the same length")
    if len(depths_dec) != len(num_heads_dec):
        raise UncompatibleInputException("depths_dec and num_heads_dec must have the same length")
    if len(depths_enc) != len(depths_dec):
        raise UncompatibleInputException("depths_dec and depths_enc must have the same length")
                                            

    enc = U_swintransformer(img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths_enc=depths_enc,
        num_heads_enc=num_heads_enc,
        depths_dec=depths_dec,
        num_heads_dec=num_heads_dec,
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
    
    middle_dim = enc.out_channels
    middle_resolution = enc.out_resolution



    dec = U_swintransformer(img_size=middle_resolution,
        patch_size=patch_size,
        in_chans=embed_dim,
        num_classes=in_chans, # output = input
        embed_dim=embed_dim,
        depths_enc=depths_enc,
        num_heads_enc=num_heads_enc,
        depths_dec=depths_dec,
        num_heads_dec=num_heads_dec,
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

    out_dim = dec.out_channels
    out_resolution = dec.out_resolution

    enc_to_dec = connection(middle_dim, embed_dim)

    return enc, dec, enc_to_dec

        
    