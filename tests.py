# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:36:27 2018

@author: tao
"""

import train
import models.WNet as WNet
import models.WNet_attention as WNet_attention
import models.W_swintransformer as W_swintransformer
import torch
import numpy as np
from data import H5Dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn

def EncoderTest(verbose=True):
    shape=(2, 4, 224, 224)
    encoder=WNet.UEnc(shape[1])
    data=torch.rand((shape[0], 3, shape[2], shape[3]))
    encoded=encoder(data)
    assert tuple(encoded.shape)==shape
    var=torch.var(encoded)
    mean=torch.mean(encoded)
    if verbose:
        print('Passed Encoder Test with var=%s and mean=%s' % (var, mean))
    return encoded

def DecoderTest():
    shape=(2, 4, 224, 224)
    out_shape=(2, 3, 224, 224)
    decoder=WNet.UDec(shape[1])
    data=torch.rand(tuple(shape))
    decoded=decoder(data)
    assert tuple(decoded.shape)==out_shape
    var=torch.var(decoded)
    mean=torch.mean(decoded)
    print('Passed Decoder Test with var=%s and mean=%s' % (var, mean))

def WNetTest():
    encoded=EncoderTest(verbose=False)
    decoder=WNet.UDec(4)
    reproduced=decoder(encoded)
    var=torch.var(reproduced)
    mean=torch.mean(reproduced)
    print('Passed Decoder Test with var=%s and mean=%s' % (var, mean))

def TrainTest():
    pass

def AllTest():
    EncoderTest()
    DecoderTest()
    WNetTest()
    TrainTest()
    print('WNet Passed All Tests!')

def test():
    wnet = W_swintransformer.W_swintransformer(num_classes=20,
        embed_dim=96,
        img_size=256,
        patch_size=2,
        in_chans=1,
        depths_enc=[2, 2, 2, 2],
        num_heads_enc=[3, 6, 12, 12],
        depths_dec=[2, 2, 2, 2],
        num_heads_dec=[12, 12, 6, 3],
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
    wnet.load_state_dict(torch.load("trained_models/model_transformer_4_depths_200_it.pkl", map_location=torch.device('cpu')))

    for i in range(0,10,2):
        data1 = H5Dataset("data_segments_reduced.h5")[i][None, :]
        data2 = H5Dataset("data_segments_reduced.h5")[i+1][None, :]
        data_batch = torch.cat((data1, data2), 0)
        plot_classification(wnet, data_batch, i)

def plot_classification(model, data_batch, i):
    enc = torch.argmax(model(data_batch, returns='enc'), dim=1, keepdim=False).detach().numpy()
    dec = model(data_batch, returns="dec").detach().numpy()
    fig, ax = plt.subplots(2)
    x = np.arange(0,len(dec[0,0,:]))
    ax[0].plot(x, dec[0,0,:])
    ax[1].plot(data_batch[0, 0,:].detach().numpy())
    fig.savefig(f"test{i}.png")


if __name__ == '__main__':
    test()