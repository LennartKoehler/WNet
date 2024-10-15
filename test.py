import matplotlib.pyplot as plt
from data import ReadDataset
import numpy as np
import torch
import torch.nn as nn
import models.W_swintransformer as Wswin




# x = torch.arange(0,10).reshape((10,1))
# print(x)
# a = x[0::2,:]
# b = x[1::2,:]
# y = torch.cat([a,b],-1)
# y = y.view(-1, 2 * 1)
# print(y)

# z1 = y[:,0::2]
# z2 = y[:,1::2]

# z = torch.empty((10,1))
# z[0::2,:] = z1
# z[1::2,:] = z2



# print(z)




# x = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])

# swin_transformer = Wswin.W_swintransformer(img_size=256,
#                                        patch_size=4,
#                                        in_chans=1,
#                                        depths_enc=[2, 2, 6],
#                                        num_heads_enc=[3, 6, 12],
#                                        depths_dec=[2, 2, 6, 2],
#                                        num_heads_dec=[3, 6, 12, 24],
#                                        num_classes=50,
#                                        window_size=8, mlp_ratio=4.,
#                                        qkv_bias=True,
#                                        drop_rate=0.,
#                                        attn_drop_rate=0.,
#                                        drop_path_rate=0.1,
#                                        norm_layer=nn.LayerNorm,
#                                        ape=False,
#                                        patch_norm=True,
#                                        use_checkpoint=False,
#                                        pretrained_window_sizes=[0, 0, 0, 0, 0])
# tensor = torch.arange(0, 256)[None,None,:].float()
# swin_transformer(tensor)
# model_parameters = filter(lambda p: p.requires_grad, swin_transformer.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print("Number of trainable parameters: ", params)


