# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:38:02 2018
@author: Tao Lin

Implementation of the W-Net unsupervised image segmentation architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, seperable=True):
        super(Block, self).__init__()
        
        if seperable:
            
            self.spatial1=nn.Conv1d(in_filters, in_filters, kernel_size=3, groups=in_filters, padding=1)
            self.depth1=nn.Conv1d(in_filters, out_filters, kernel_size=1)
            
            self.conv1=lambda x: self.depth1(self.spatial1(x))
            
            self.spatial2=nn.Conv1d(out_filters, out_filters, kernel_size=3, padding=1, groups=out_filters)
            self.depth2=nn.Conv1d(out_filters, out_filters, kernel_size=1)
            
            self.conv2=lambda x: self.depth2(self.spatial2(x))
            
        else:
            
            self.conv1=nn.Conv1d(in_filters, out_filters, kernel_size=3, padding=1)
            self.conv2=nn.Conv1d(out_filters, out_filters, kernel_size=3, padding=1)
        
        
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1) # TODO was 0.65 before
        self.batchnorm1=nn.BatchNorm1d(out_filters)
        
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1) 
        self.batchnorm2=nn.BatchNorm1d(out_filters)

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x)).clamp(0)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.batchnorm2(self.conv2(x)).clamp(0)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        return x

class UEnc(nn.Module):
    def __init__(self, squeeze, ch_mul=64, in_chans=1):
        super(UEnc, self).__init__()
        
        self.enc1=Block(in_chans, ch_mul, seperable=False)
        self.enc2=Block(ch_mul, 2*ch_mul)
        self.enc3=Block(2*ch_mul, 4*ch_mul)
        self.enc4=Block(4*ch_mul, 8*ch_mul)
        
        self.middle=Block(8*ch_mul, 16*ch_mul)
        
        self.up1=nn.ConvTranspose1d(16*ch_mul, 8*ch_mul, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.dec1=Block(16*ch_mul, 8*ch_mul)
        self.up2=nn.ConvTranspose1d(8*ch_mul, 4*ch_mul, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.dec2=Block(8*ch_mul, 4*ch_mul)
        self.up3=nn.ConvTranspose1d(4*ch_mul, 2*ch_mul, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.dec3=Block(4*ch_mul, 2*ch_mul)
        self.up4=nn.ConvTranspose1d(2*ch_mul, ch_mul, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.dec4=Block(2*ch_mul, ch_mul, seperable=False)
        
        self.final=nn.Conv1d(ch_mul, squeeze, kernel_size=1)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        
        enc1=self.enc1(x)
        
        enc2=self.enc2(F.max_pool1d(enc1, (4)))
        
        enc3=self.enc3(F.max_pool1d(enc2, (4)))
        
        enc4=self.enc4(F.max_pool1d(enc3, (4)))
        
        
        middle=self.middle(F.max_pool1d(enc4, (4)))
        
        m2 = self.up1(middle)
        up1=torch.cat([enc4, m2], 1)
        dec1=self.dec1(up1)
        
        up2=torch.cat([enc3, self.up2(dec1)], 1)
        dec2=self.dec2(up2)
        
        up3=torch.cat([enc2, self.up3(dec2)], 1)
        dec3=self.dec3(up3)
        
        up4=torch.cat([enc1, self.up4(dec3)], 1)
        dec4=self.dec4(up4)
        
        
        final=self.final(dec4)
            
        return final

class UDec(nn.Module):
    def __init__(self, squeeze, ch_mul=64, in_chans=1):
        super(UDec, self).__init__()
        
        self.enc1=Block(squeeze, ch_mul, seperable=False)
        self.enc2=Block(ch_mul, 2*ch_mul)
        self.enc3=Block(2*ch_mul, 4*ch_mul)
        self.enc4=Block(4*ch_mul, 8*ch_mul)
        
        self.middle=Block(8*ch_mul, 16*ch_mul)
        
        self.up1=nn.ConvTranspose1d(16*ch_mul, 8*ch_mul, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.dec1=Block(16*ch_mul, 8*ch_mul)
        self.up2=nn.ConvTranspose1d(8*ch_mul, 4*ch_mul, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.dec2=Block(8*ch_mul, 4*ch_mul)
        self.up3=nn.ConvTranspose1d(4*ch_mul, 2*ch_mul, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.dec3=Block(4*ch_mul, 2*ch_mul)
        self.up4=nn.ConvTranspose1d(2*ch_mul, ch_mul, kernel_size=5, stride=4, padding=1, output_padding=1)
        self.dec4=Block(2*ch_mul, ch_mul, seperable=False)
        
        self.final=nn.Conv1d(ch_mul, in_chans, kernel_size=1)
        
    def forward(self, x):
        
        enc1 = self.enc1(x)
        
        enc2 = self.enc2(F.max_pool1d(enc1, (4)))
        
        enc3 = self.enc3(F.max_pool1d(enc2, (4)))
        
        enc4 = self.enc4(F.max_pool1d(enc3, (4)))
        
        
        middle = self.middle(F.max_pool1d(enc4, (4)))
        
        
        up1 = torch.cat([enc4, self.up1(middle)], 1)
        dec1 = self.dec1(up1)
        
        up2 = torch.cat([enc3, self.up2(dec1)], 1)
        dec2 = self.dec2(up2)
        
        up3 = torch.cat([enc2, self.up3(dec2)], 1)
        dec3 =self.dec3(up3)
        
        up4 = torch.cat([enc1, self.up4(dec3)], 1)
        dec4 = self.dec4(up4)
        
        
        final=self.final(dec4)
        
        return final

class WNet(nn.Module):
    def __init__(self, squeeze, ch_mul=64, in_chans=1, out_chans=1000):
        super(WNet, self).__init__()
        if out_chans==1000:
            out_chans=in_chans
        self.UEnc=UEnc(squeeze, ch_mul, in_chans)
        self.UDec=UDec(squeeze, ch_mul, out_chans)
    def forward(self, x, returns='both'):
         # doesnt this mean that it first runs the encoder part and then runs the encoder+decoder
         # couldn't the encoder be run, then ncutloss, then calculate the decoder with the input of the encoder and then run the reconstruction loss?
         # or would it be problematic if the decoder is run with the output of the encoder of one iteration before
         # currently, how train_op is set up:
         #1. run encoder and calc ncutloss and backpropegate
         #2. run encoder and decoder and then calc reconstruction loss and backpropegate
        enc = self.UEnc(x)
        
        if returns=='enc':
            return enc
        
        dec=self.UDec(F.softmax(enc, 1)) # FIXME softmax necessary here?
        
        if returns=='dec':
            return dec
        
        if returns=='both':
            return enc, dec
        
        else:
            raise ValueError('Invalid returns, returns must be in [enc dec both]')
        
if __name__ == "__main__":
    model = WNet(100)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters: ", params)