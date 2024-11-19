
import argparse
import torch.nn as nn
import numpy as np
import time
import datetime
import torch
from torchvision import datasets, transforms
from utils.org_soft_n_cut_loss import batch_soft_n_cut_loss
from utils.soft_n_cut_loss import soft_n_cut_loss

from data_fast5 import H5Dataset
#import WNet_attention as WNet
import models.WNet as WNet
import matplotlib.pyplot as plt
import models.W_swintransformer as W_swintransformer
from local_variables import data_path_rna004 as data_path

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

softmax = nn.Softmax(dim=1)
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




def train_single_image():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA available: ",torch.cuda.is_available())

    # Create empty lists for average N_cut losses and reconstruction losses

    # Squeeze k
    # squeeze = args.squeeze
    squeeze = 2
    img_size = 256
    #  wnet = WNet.WNet(squeeze=squeeze, in_chans=1)
    wnet = W_swintransformer.W_swintransformer(num_classes=squeeze,
            embed_dim=96,
            img_size=img_size,
            patch_size=2,
            in_chans=1,
            depths_enc=[2, 2, 2],
            num_heads_enc=[3, 6, 12],
            depths_dec=[2, 2, 2],
            num_heads_dec=[12, 6, 3],
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
    
    learning_rate = 0.003
    epochs = 5

    wnet = wnet.to(device)
    optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)

    n_cut_losses = []
    rec_losses = []
    start_time = time.time()

    data1 = H5Dataset(data_path)[0][None, :]
    data2 = H5Dataset(data_path)[1][None, :]
    data_batch = torch.cat((data1, data2), 0).to(device)


    for epoch in range(epochs):
        if (epoch > 0 and epoch % 100 == 0):
            learning_rate = learning_rate/10
            optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)

        # with torch.autograd.detect_anomaly():
        wnet, n_cut_loss, rec_loss = train_op(wnet, optimizer, data_batch, 1, img_size)
        n_cut_losses.append(n_cut_loss.detach().cpu())
        rec_losses.append(rec_loss.detach().cpu())


    
    np.save(f"{save_path}n_cut_losses.npy", n_cut_losses)
    np.save(f"{save_path}rec_losses.npy", rec_losses)
    torch.save(wnet.state_dict(), f"{save_path}model.pkl")

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('profiling', worker_name="default"),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True)
    #prof.start()
    run_name = "transformer_3_depths_1_batch_5_epochs_test"
    save_path = f"results/{run_name}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    train_single_image()
    #prof.stop()
    print("finished")