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

from data import ReadDataset
#import WNet_attention as WNet
import models.WNet as WNet
import matplotlib.pyplot as plt
import models.W_swintransformer as W_swintransformer

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--name', metavar='name', default=str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')), type=str,
                    help='Name of model')
parser.add_argument('--in_Chans', metavar='C', default=3, type=int, 
                    help='number of input channels')
parser.add_argument('--squeeze', metavar='K', default=4, type=int, 
                    help='Depth of squeeze layer')
parser.add_argument('--out_Chans', metavar='O', default=3, type=int, 
                    help='Output Channels')
parser.add_argument('--epochs', metavar='e', default=100, type=int, 
                    help='epochs')
parser.add_argument('--input_folder', metavar='f', default=None, type=str, 
                    help='Folder of input images')
parser.add_argument('--output_folder', metavar='of', default=None, type=str, 
                    help='folder of output images')



softmax = nn.Softmax(dim=1) # is this the right dimension? -> should be the dimension of channels (classes) where each pixel has a probability for each class
criterionIdt = torch.nn.MSELoss()

def train_op(model, optimizer, input, k, img_size, psi=0.5): # model = WNet
    enc = model(input, returns='enc')
    d = enc.clone().detach()
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

def test():
    wnet=WNet.WNet(4)
    synthetic_data=torch.rand((1, 3, 128, 128))
    optimizer=torch.optim.SGD(wnet.parameters(), 0.001) #.cuda()
    train_op(wnet, optimizer, synthetic_data)

def show_image(image):
    img = image.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()

def train_single_image():
    CUDA = torch.cuda.is_available()

    # Create empty lists for average N_cut losses and reconstruction losses
    n_cut_losses_avg = []
    rec_losses_avg = []

    # Squeeze k
    # squeeze = args.squeeze
    squeeze = 2
    img_size = 256
    # wnet = WNet.WNet(squeeze=squeeze, in_chans=1)
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
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[0, 0, 0])
    if(CUDA):
        wnet = wnet.cuda()
    # learning_rate = 0.003
    learning_rate = 0.003
    optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)

    n_cut_losses = []
    rec_losses = []
    start_time = time.time()

    data1 = ReadDataset("data_segments_reduced.h5")[0][None, :]
    data2 = ReadDataset("data_segments_reduced.h5")[1][None, :]
    data_batch = torch.cat((data1, data2), 0)


    for epoch in range(500):
        if (epoch > 0 and epoch % 1000 == 0):
            learning_rate = learning_rate/10
            optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)




        wnet, n_cut_loss, rec_loss = train_op(wnet, optimizer, data_batch, 1, img_size)
        n_cut_losses.append(n_cut_loss.detach())
        rec_losses.append(rec_loss.detach())
        if epoch%10 == 0:
            print("Epoch = " + str(epoch))
            print("n_cut_loss", n_cut_loss.item())
            print("rec_loss", rec_loss.item())

    n_cut_losses_avg.append(torch.mean(torch.FloatTensor(n_cut_losses)))
    rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))
    print("--- %s seconds ---" % (time.time() - start_time))
    torch.save(wnet.state_dict(), "w_swin_state_dict_with_rec_loss_500_epoch.pkl")




def main():
    # Check if CUDA is available
    CUDA = torch.cuda.is_available()

    # Create empty lists for average N_cut losses and reconstruction losses
    n_cut_losses_avg = []
    rec_losses_avg = []

    # Squeeze k
    # squeeze = args.squeeze

    #------------------Parameters-----------------
    squeeze = 10
    img_size = 256
    wnet = WNet.WNet(squeeze, in_chans=1)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)
    batch_size = 10
    epochs = 10
    num_workers = 0
    #---------------------------------------------
    # transform = transforms.Compose([transforms.Resize(img_size),
    #                             transforms.ToTensor()])

    # dataset = datasets.ImageFolder(args.input_folder, transform=transform)

    # # Train 1 image set batch size=1 and set shuffle to False
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    dataset = ReadDataset("data_segments_reduced.h5")
    if(CUDA):
        wnet = wnet.cuda()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


    for epoch in range(epochs):

        # At 1000 epochs divide SGD learning rate by 10
        if (epoch > 0 and epoch % 1000 == 0):
            learning_rate = learning_rate/10
            optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)

        print("Epoch = " + str(epoch))

        n_cut_losses = []
        rec_losses = []
        start_time = time.time()

        for (idx, batch) in enumerate(dataloader):
            if CUDA:
                batch = batch.cuda()
            wnet, n_cut_loss, rec_loss = train_op(wnet, optimizer, batch, 1, img_size)
            n_cut_losses.append(n_cut_loss.detach())
            rec_losses.append(rec_loss.detach())
            if idx%10==0:
                print(f"n_cut_loss: {n_cut_loss.item()}")
                print(f"rec_loss: {rec_loss.item()} \n")




        n_cut_losses_avg.append(torch.mean(torch.FloatTensor(n_cut_losses)))
        rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))
        print("--- %s seconds ---" % (time.time() - start_time))


    # images, labels = next(iter(dataloader))

    # # Run wnet with cuda if enabled
    # if CUDA:
    #     images = images.cuda()

    # enc, dec = wnet(images)

    torch.save(wnet.state_dict(), "models/model_" + "test_orig")
    np.save("models/n_cut_losses_" + "test_orig", n_cut_losses_avg)
    np.save("models/rec_losses_" + "test_orig", rec_losses_avg)
    print("Done")

if __name__ == '__main__':
    train_single_image()



# python .\train.py --e 100 --input_folder="data/images/" --output_folder="/output/"