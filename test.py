import matplotlib.pyplot as plt
from data import ReadDataset
import numpy as np
import models.swin_transformer_v2_1d_unfinished as st1d
import models.swin_transformer_v2 as st2d
import torch
import models.swin_transformer_v2_1d_unfinished as t1d
import torch.nn as nn



x = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])

swin_transformer = t1d.SwinTransformerV21D(img_size=256,
                                       patcH_size=4,
                                       in_chans=1,
                                       depths=[2, 2, 6, 2],
                                       num_heads=[3, 6, 12, 24],
                                       window_size=8, mlp_ratio=4.,
                                       qkv_bias=True,
                                       drop_rate=0.,
                                       attn_drop_rate=0.,
                                       drop_path_rate=0.1,
                                       norm_layer=nn.LayerNorm,
                                       ape=False,
                                       patch_norm=True,
                                       use_checkpoint=False,
                                       pretrained_window_sizes=[0, 0, 0, 0])
tensor = torch.arange(0, 256)[None,None,:].float()

swin_transformer(tensor)

