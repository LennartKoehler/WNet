import models.W_swintransformer as W_swintransformer
import models.WNet as WNet
import torchviz
import torch
import torch.nn as nn

# wnet = WNet.WNet(squeeze=2, in_chans=1)
wnet = W_swintransformer.W_swintransformer(num_classes=2,
        embed_dim=96,
        img_size=256,
        patch_size=2,
        in_chans=1,
        depths_enc=[1, 1],
        num_heads_enc=[2, 2],
        depths_dec=[1, 1],
        num_heads_dec=[2, 2],
        window_size=8, mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0])

x = torch.randn(2, 1, 256)
y = wnet(x, returns="enc")
dot = torchviz.make_dot(y, params = dict(wnet.named_parameters()), show_saved=True)
dot.format = 'png'
dot.render('model_arch.png')