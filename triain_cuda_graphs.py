# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:38:02 2018
@author: Tao Lin

Implementation of the W-Net unsupervised image segmentation architecture
"""

import argparse
import torch.nn as nn
import numpy as np
import time
import datetime
import torch
from torchvision import datasets, transforms
from utils.org_soft_n_cut_loss import batch_soft_n_cut_loss
from utils.soft_n_cut_loss import soft_n_cut_loss

from data import H5Dataset
#import WNet_attention as WNet
import models.WNet as WNet
import matplotlib.pyplot as plt
import models.W_swintransformer as W_swintransformer




softmax = nn.Softmax(dim=1) # is this the right dimension? -> should be the dimension of channels (classes) where each pixel has a probability for each class
criterionIdt = torch.nn.MSELoss()

def train_op(model, optimizer, input, k, img_size, psi=0.5): # model = WNet
    enc = model(input, returns='enc')
    n_cut_loss=soft_n_cut_loss(input,  softmax(enc),  img_size)
    n_cut_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # n_cut_loss = torch.tensor(1)


    dec = model(input, returns='dec')
    rec_loss=reconstruction_loss(input, dec)
    rec_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #rec_loss = torch.tensor(1)
    return (model, n_cut_loss, rec_loss)

def reconstruction_loss(x, x_prime):
    rec_loss = criterionIdt(x_prime, x)
    return rec_loss




def main(prof):
    
    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA available: ",torch.cuda.is_available())

    # Create empty lists for average N_cut losses and reconstruction losses

    # Squeeze k
    # squeeze = args.squeeze

    #------------------Parameters-----------------
    in_channels = 1
    squeeze = 10
    img_size = 256
    wnet = W_swintransformer.W_swintransformer(num_classes=squeeze,
        embed_dim=96,
        img_size=img_size,
        patch_size=2,
        in_chans=in_channels,
        depths_enc=[2, 2, 2],
        num_heads_enc=[3, 6, 12],
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
        pretrained_window_sizes=[0, 0, 0])
    #wnet = WNet.WNet(squeeze, in_chans=1)
    learning_rate = 0.003
    optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)
    batch_size = 40
    epochs = 6
    num_workers = 4
    #---------------------------------------------
    # transform = transforms.Compose([transforms.Resize(img_size),
    #                             transforms.ToTensor()])

    # dataset = datasets.ImageFolder(args.input_folder, transform=transform)

    # # Train 1 image set batch size=1 and set shuffle to False
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    dataset = H5Dataset("data_segments_reduced.h5")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    wnet = wnet.to(device)

    # with open("worker_profiling.txt", "a") as f:
    #     f.write(f"\n batch_size={batch_size}\n")


    static_input = torch.randn(batch_size, in_channels, img_size).to(device)

    #------------warmup----------------
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            optimizer.zero_grad(set_to_none=True)
            wnet, n_cut_loss, rec_loss = train_op(wnet, optimizer, static_input, 1, img_size)
    torch.cuda.current_stream().wait_stream(s)

    #-----------capture----------------
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        wnet, n_cut_loss, rec_loss = train_op(wnet, optimizer, static_input, 1, img_size)

    #----------training---------
    for epoch in range(epochs):


        print("Epoch = " + str(epoch))


        for (idx, batch) in enumerate(dataloader):
            static_input.copy_(batch)
            g.replay()
            prof.step()



    # images, labels = next(iter(dataloader))

    # # Run wnet with cuda if enabled
    # if CUDA:
    #     images = images.to(device)

    # enc, dec = wnet(images)
    # with open("worker_profiling.txt", "a") as f:
    #     f.write(f"number_workers:{num_workers}, time: {time.time() - start_time}\n")
    # torch.save(wnet.state_dict(), "models/model_" + "test_orig")
    # np.save("models/n_cut_losses_" + "test_orig", n_cut_losses_avg)
    # np.save("models/rec_losses_" + "test_orig", rec_losses_avg)
    # print("Done")

if __name__ == '__main__':
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('profiling'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        main(prof)

