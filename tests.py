# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:36:27 2018

@author: tao
"""

import train
import WNet
import WNet_attention
import torch
import numpy as np
from data import ReadDataset
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
    for i in range(0,10,2):
        data1 = ReadDataset("data_segments_reduced.h5")[i][None, :]
        data2 = ReadDataset("data_segments_reduced.h5")[i+1][None, :]
        data_batch = torch.cat((data1, data2), 0)
        wnet = WNet.WNet(10)
        wnet.load_state_dict(torch.load("models/model_test_orig"))
        plot_classification(wnet, data_batch)

def plot_classification(model, data_batch):
    enc = torch.argmax(model(data_batch, returns='enc'), dim=1, keepdim=False).detach().numpy()
    fig, ax = plt.subplots(2)
    x = np.arange(0,len(enc[0,:]))
    ax[0].plot(x, enc[0,:])
    ax[1].plot(data_batch[0, 0,:].detach().numpy())
    plt.show()

def plot_loss():
    loss_original = np.load("models/rec_losses_test_orig.npy")
    loss_attention = np.load("models/rec_losses_test_att.npy")
    plt.plot(loss_original, label="original")
    plt.plot(loss_attention, label="attention")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_loss()