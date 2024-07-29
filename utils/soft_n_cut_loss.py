# This file has been written by @AsWali and @ErwinRussel

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair, _quadruple

def calculate_weights(input, batch_size, img_size=256, ox=4, radius=5 ,oi=10):
    channels = 1
    image = torch.mean(input, dim=1, keepdim=True) # 
    l = img_size
    p = radius

    image = F.pad(input=image, pad=(p, p), mode='constant', value=0) # pad around image to not reduce size
    # Use this to generate random values for the padding.
    # randomized_inputs = (0 - 255) * torch.rand(image.shape).cuda() + 255
    # mask = image.eq(0)
    # image = image + (mask *randomized_inputs)

    # kh, kw = radius*2 + 1, radius*2 + 1
    # dh, dw = 1, 1
    kh = radius*2 + 1
    dh = 1
    patches = image.unfold(2, kh, dh) # sliding window over dimension 2, with size kh and step dh written into new dimension

    patches = patches.contiguous().view(batch_size, channels, -1, kh)
    patches = patches.permute(0, 2, 1, 3)
    patches = patches.view(-1, channels, kh)
    # patches: sliding window of size kh at each point written into new dimension
    # e.g.
    # [1,2,3,4,5]
    # [2,3,4,5,6]
    # [3,4,5,6,7]

    center_values = patches[:, :, radius]
    center_values = center_values[:, :, None]
    center_values = center_values.expand(-1, -1, kh) # expand tensor in dimension 2 by kh: batches x channels x 1 [10 1 1] -> [10 1 kh]
    # center_values: patches but each sliding window is filled with the value which is in the center position
    # e.g. for patches example
    # [3,3,3,3,3]
    # [4,4,4,4,4]
    # [5,5,5,5,5]

    k_row = torch.arange(1, kh + 1) - radius - 1

    if torch.cuda.is_available():
        k_row = k_row.cuda()

    distance_weights = (k_row ** 2 + k_row.T**2) # not sure if this needs to be squared

    mask = distance_weights.le(radius) # distance weights less than radius, is this even necessary?
    distance_weights = torch.exp(torch.div(-1*(distance_weights), ox**2)) # exp(-||X(i)-X(j)||^2_2  /  sigma^2_X)
    distance_weights = torch.mul(mask, distance_weights) # weights where distance > r are set to 0


    patches = torch.exp(torch.div(-1*((patches - center_values)**2), oi**2)) # exp(-||F(i)-F(j)||^2_2  /  sigma^2_I)
    # patches - center_values: distance of each pixel value to its center pixel value
    # e.g.
    # [-2,-1,0,1,2]
    # [-2,-1,0,1,2]

    return torch.mul(patches, distance_weights)

def soft_n_cut_loss_single_k(weights, enc, batch_size, img_size, radius=5):
    channels = 1
    h, w = img_size
    p = radius

    kh, kw = radius*2 + 1, radius*2 + 1
    dh, dw = 1, 1
    encoding = F.pad(input=enc, pad=(p, p, p, p), mode='constant', value=0)

    seg = encoding.unfold(2, kh, dh).unfold(3, kw, dw)
    seg = seg.contiguous().view(batch_size, channels, -1, kh, kw)
    seg = seg.permute(0, 2, 1, 3, 4)
    seg = seg.view(-1, channels, kh, kw)

    nom = weights * seg

    nominator = torch.sum(enc * torch.sum(nom, dim=(1,2,3)).reshape(batch_size, h, w), dim=(1,2,3))
    denominator = torch.sum(enc * torch.sum(weights, dim=(1,2,3)).reshape(batch_size, h, w), dim=(1,2,3))

    return torch.div(nominator, denominator)

def soft_n_cut_loss(image, enc, img_size):
    loss = []
    batch_size = image.shape[0]
    k = enc.shape[1]
    weights = calculate_weights(image, batch_size, img_size)
    for i in range(0, k):
        loss.append(soft_n_cut_loss_single_k(weights, enc[:, (i,), :, :], batch_size, img_size))
    da = torch.stack(loss)
    return torch.mean(k - torch.sum(da, dim=0))
