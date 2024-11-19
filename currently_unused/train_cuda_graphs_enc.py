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

from data_fast5 import H5Dataset
#import WNet_attention as WNet
import models.WNet as WNet
import matplotlib.pyplot as plt
import models.W_swintransformer_seperate as W_swintransformer
from local_variables import data_path



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



def main(save_name="test"):
    
    # Check if CUDA is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA available: ",torch.cuda.is_available())

    # Create empty lists for average N_cut losses and reconstruction losses
    n_cut_losses_avg = []
    rec_losses_avg = []

    # Squeeze k
    # squeeze = args.squeeze

    #------------------Parameters-----------------
    squeeze = 20
    img_size = 256


    enc, dec, enc_to_dec = W_swintransformer.init_W_swintransformer(num_classes=squeeze,
            embed_dim=96,
            img_size=img_size,
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
    # wnet = WNet.WNet(squeeze, in_chans=1)
    learning_rate = 0.003

    optimizer_enc = torch.optim.SGD(enc.parameters(), lr=learning_rate)
    dec_params = list(enc_to_dec.parameters()) + list(dec.parameters())
    optimizer_dec = torch.optim.SGD(dec_params, lr=learning_rate)
    # optimizer_dec = torch.optim.SGD(dec.parameters(), lr=learning_rate)
    # optimizer_enc_to_dec = torch.optim.SGD(enc_to_dec.parameters(), lr=learning_rate)

    batch_size = 50
    epochs = 2
    num_workers = 2  
    #---------------------------------------------
    # transform = transforms.Compose([transforms.Resize(img_size),
    #                             transforms.ToTensor()])

    # dataset = datasets.ImageFolder(args.input_folder, transform=transform)

    # # Train 1 image set batch size=1 and set shuffle to False
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    dataset = H5Dataset(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True) # drop last important for graphs
    
    enc.to(device)
    enc_to_dec.to(device)
    dec.to(device)

    single_batch = next(iter(dataloader)).to(device)


    enc = torch.cuda.make_graphed_callables(enc, (single_batch,), allow_unused_input=True)

    # enc_batch = torch.rand_like(enc(single_batch))
    # enc_to_dec = torch.cuda.make_graphed_callables(enc_to_dec, (enc_batch,), allow_unused_input=True)

    # enc_batch_ = torch.rand_like(enc_to_dec(enc_batch))
    # dec = torch.cuda.make_graphed_callables(dec, (enc_batch_,), allow_unused_input=True)


    #with open("worker_profiling.txt", "a") as f:
     #   f.write(f"\n batch_size={batch_size}\n")



    for epoch in range(epochs):

        # At 1000 epochs divide SGD learning rate by 10
        # if (epoch > 0 and epoch % 1000 == 0):
        #     learning_rate = learning_rate/10
        #     optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)

        print("Epoch = " + str(epoch))

        n_cut_losses = []
        rec_losses = []
        start_time = time.time()

        for (idx, batch) in enumerate(dataloader):
            batch = batch.to(device)

            enc_batch = enc(batch)
            n_cut_loss=soft_n_cut_loss(batch,  softmax(enc_batch),  img_size)
            n_cut_loss.backward()
            optimizer_enc.step()
            optimizer_enc.zero_grad()

            enc_batch = enc(batch)

            enc_batch_ = enc_to_dec(enc_batch)
            dec_batch = dec(enc_batch_)

            rec_loss=reconstruction_loss(batch, dec_batch)
            rec_loss.backward()

            optimizer_dec.step()
            optimizer_dec.zero_grad()
            optimizer_enc.step()
            optimizer_enc.zero_grad()
   
            n_cut_losses.append(n_cut_loss.detach())
            rec_losses.append(rec_loss.detach())
 
            # prof.step()


        n_cut_losses_avg.append(torch.mean(torch.FloatTensor(n_cut_losses)))
        rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))
        print("--- %s seconds ---" % (time.time() - start_time))


    # images, labels = next(iter(dataloader))

    # # Run wnet with cuda if enabled
    # if CUDA:
    #     images = images.to(device)

    # enc, dec = wnet(images)
    # with open("worker_profiling.txt", "a") as f:
    #     f.write(f"number_workers:{num_workers}, time: {time.time() - start_time}\n")
    torch.save(enc.state_dict(), "trained_models/model__enc" + save_name)
    torch.save(enc_to_dec.state_dict(), "trained_models/model_enc_to_dec" + save_name)
    torch.save(dec.state_dict(), "trained_models/model_dec_" + save_name)

    # np.save("loss_results/n_cut_losses_" + save_name + ".npy", n_cut_losses_avg)
    # np.save("loss_results/rec_losses_" + save_name + ".npy", rec_losses_avg)

if __name__ == '__main__':
    # prof = torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('profiling', "cuda_graphs_partial_enc_dec_seperate_no_pools"),
    #     record_shapes=True,
    #     profile_memory=True)
    # prof.start()
    main("transformer_4_depths_enc_graph")
    # prof.stop()
    print("finished")


# python .\train.py --e 100 --input_folder="data/images/" --output_folder="/output/"
