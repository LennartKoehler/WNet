# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:36:27 2018

@author: tao
"""

import models.WNet as WNet
import models.WNet_attention as WNet_attention
import models.W_swintransformer as W_swintransformer
import torch
import numpy as np
from data_fast5 import H5Dataset_10
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import os
from local_variables import data_path_rna004_2048 as data_path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

def test(wnet, batch):

    enc, dec = wnet(batch, returns='both')
    enc = torch.argmax(enc, dim=1, keepdim=False).detach().cpu().numpy()
    dec = dec.detach().cpu().numpy()
    batch = batch.detach().cpu().numpy()

    for i, single_image in enumerate(batch[:10]):
        plot_segmentation(enc[i], single_image, i)
        plot_reconstruction(dec[i], single_image, i)

def plot_segmentation(enc, single_image, i):
    fig, ax = plt.subplots()
    fig.set_figwidth(15)

    x = np.arange(0,len(enc[:]))
    ax.plot(single_image[0,:])
    ax.plot(x, enc[:], color="lime", linewidth=1)


    fig.savefig(f"{save_path}test_{i}_segmentation.png", dpi=300)
    plt.close()

def plot_reconstruction(dec, single_image, i):
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    x = np.arange(0,len(dec[0,:]))
    ax.plot(single_image[0,:])
    ax.plot(x, dec[0,:], color="red", linewidth=0.7)

    fig.savefig(f"{save_path}test_{i}_reconstruction.png", dpi=300)

def run_tests(wnet, run_id, batch):
    global save_path
    save_path = f"results/{run_id}/test_images/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test(wnet, batch)

if __name__ == "__main__":
    run_id = "run_2024_11_18_T19_09_34"

    wnet = W_swintransformer.W_swintransformer(num_classes=2,
            embed_dim=96,
            img_size=2560,
            patch_size=2,
            in_chans=1,
            depths_enc=[2, 2, 2, 2],
            num_heads_enc=[3, 6, 12, 12],
            depths_dec=[2, 2, 2, 2],
            num_heads_dec=[12, 12, 6, 3],
            window_size=8, mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[0, 0, 0, 0])
    
    # wnet = WNet.WNet(squeeze=2, in_chans=1)

    wnet.load_state_dict(torch.load(f"results/{run_id}/model.pkl", map_location=torch.device('cpu')))

    dataset = H5Dataset_10(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2, pin_memory=True)
    run_tests(wnet, run_id, next(iter(dataloader)))