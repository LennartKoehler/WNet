# This file has been written by @AsWali and @ErwinRussel

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair, _quadruple


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator



def calculate_weights(batch, batch_size, img_size=256, ox=4, radius=5 ,oi=1): # maybe save the weights for images to then reuse?
    image = torch.mean(batch, dim=1, keepdim=True) # mean over channels which is 1? -> does nothing?

    image = F.pad(input=image, pad=(radius, radius), mode='constant', value=0) # pad around image to not reduce size
    # Use this to generate random values for the padding.
    # randomized_inputs = (0 - 255) * torch.rand(image.shape).cuda() + 255
    # mask = image.eq(0)
    # image = image + (mask *randomized_inputs)

    # kh, kw = radius*2 + 1, radius*2 + 1
    # dh, dw = 1, 1
    channels = 1
    kh = radius*2 + 1
    dh = 1



    distances = torch.abs(torch.arange(1, kh + 1) - radius - 1)
    # distances: tensor([5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5])
    distance_weights = torch.exp(torch.div(-1*(distances), ox**2)) # exp(-||X(i)-X(j)||^2_2  /  sigma^2_X)
    if torch.cuda.is_available():
        distance_weights = distance_weights.cuda()
    # distance_weights: tensor([0.7316, 0.7788, 0.8290, 0.8825, 0.9394, 1.0000, 0.9394, 0.8825, 0.8290, 0.7788, 0.7316])


    patches = image.unfold(2, kh, dh) # sliding window over dimension 2, with size kh and step dh written into new dimension
    # patches = patches.contiguous().view(batch_size, channels, -1, kh)
    # patches = patches.permute(0, 2, 1, 3)
    # patches = patches.view(-1, channels, kh) # remove batch dimension by concatenating all batches
    # patches: sliding window of size kh at each point written into new dimension
    # e.g.
    # [2,3,4,6,1,4,5,6,3,2,3]
    # [5,6,3,4,2,3,4,5,3,1,3]
    # [2,3,4,6,1,4,5,6,3,2,3]

    center_values = patches[:, :, :, radius] # get the center value of each sliding window
    center_values = center_values[:, :, :, None] # make new empty dimension
    center_values = center_values.expand(-1, -1, -1, kh) # expand tensor in dimension 3 by kh: batches x channels x 1 [10 1 1] -> [10 1 kh]
    # center_values: patches but each sliding window is filled with the value which is in the center position
    # e.g. for patches example
    # [4,4,4,4,4,4,4,4,4,4,4]
    # [3,3,3,3,3,3,3,3,3,3,3]
    # [4,4,4,4,4,4,4,4,4,4,4]


    helper_matrix = patches - center_values
    # THIS IS BASICALLY ALWAYS [1,1,1,1,1,1,1,1] BECAUSE THE DISTANCES IN PIXEL VALUE ARE VERY SMALL
    patches = torch.exp(torch.div(-1*((helper_matrix*10)**2), oi**2)) # exp(-||F(i)-F(j)||^2_2  /  sigma^2_I)
    # patches - center_values: distance of each pixel value to its center pixel value
    # if a center value is the same as its neighbors, then this will return [0,0,0,0,0,0,0,0,0,0,0]
    # e.g.
    # [-2,-1,0,2,-3,...-1]

    # patches * distance_weights -> how similar are the pixels to each other, closer pixels have more influence




    return torch.mul(patches, distance_weights)

# def test():
#     batch = torch.tensor([[[1.0, 55.0, 13.0 , 25.0, 500.0]]])
#     print(calculate_weights(batch, 1, 5, radius=1))

#   returns: tensor([[[9.3007e-01, 1.0000e+00, 2.0362e-13]],

#         [[2.0362e-13, 1.0000e+00, 2.0507e-08]],

#         [[2.0507e-08, 1.0000e+00, 2.2257e-01]],

#         [[2.2257e-01, 1.0000e+00, 0.0000e+00]],

#         [[0.0000e+00, 1.0000e+00, 0.0000e+00]]])

def soft_n_cut_loss_single_k(weights, enc, batch_size, img_size, radius=5):
     # only one channel ("class") is given to this function
     # for this class the loss is calculated
    channels = 1
    kh, kw = radius*2 + 1, radius*2 + 1
    dh, dw = 1, 1
    encoding = F.pad(input=enc, pad=(radius, radius), mode='constant', value=0)

    enc_windows = encoding.unfold(2, kh, dh)

    # print("enc\n",enc[0,0,30],"\n")
    # print("enc_windows\n",enc_windows[0,0,30,:],"\n")
    # print("weights\n",weights[0,0,30,:],"\n")

    # seg = seg.contiguous().view(batch_size, channels, -1, kh) # does this do anything?
    # seg = seg.permute(0, 2, 1, 3)
    # seg = seg.view(-1,channels, kh) # first dimension becomes batches x values (e.g. 4 x 256)
    # seg now has the same shape as weights, but the values of seg are the encoded image for one class
    # while weights is calculated with the actual input image pixel values
    # seg is basically just reshaped and padded input encoding

    # seg = p(v=A_k)
    # enc = p(u=A_k)

    nom = weights * enc_windows
    # print("nom\n",nom[0,0,30,:],"\n")

    # nom: "for each pixel, compare class prediction to weights"
    # if two neighboring pixels recieved a "one" for this class and in the original image they also have the same value
    # then nom for this pixel is high. Therefore the nominator is high and therefore the return of this function is high
    # there is a K - sum(this function) so this in the end reduces the loss

    # if weights and nom have the same values for pixels, then nom is maximized


    # sum_batches = torch.sum(nom, dim=(3))
    # sum_weights = torch.sum(weights, dim=(3))

    # nominator = torch.sum(enc * sum_batches, dim=(2)) # multiply each pixel (enc[i]) by each weighted pixel in the surrouding of the pixel (i) (sum_batches[i]) and sum them up
    # denominator = torch.sum(enc * sum_weights, dim=(2))


    enc = enc[:,:,:,None]

    help_matrix_nom = enc * nom
    # print("help_matrix_nom\n",help_matrix_nom[0,0,30,:],"\n")

    nominator = torch.sum(help_matrix_nom, dim = (2,3)) # shape: [batch_size, channels] = [batch_size, 1]

    help_matrix_denom = enc * weights
    # print("help_matrix_denom\n",help_matrix_denom[0,0,30,:],"\n")

    denominator = torch.sum(help_matrix_denom, dim = (2,3))

    result = torch.div(nominator, denominator)
    # result = result.detach().numpy()
    # plt.plot(np.arange(0, len(result[0,0,:])),result[0,0,:])
    # plt.show()
    return result

def plot_function(tensor):
    tensor = tensor.detach().numpy()
    tensor = tensor[:,:,90:120,:] # show only part of signal
    radius = tensor.shape[3]
    n_pixels = tensor.shape[2] + radius

    pixels = np.arange(5, n_pixels - 6, 1)
    padded_pixels = np.arange(0, n_pixels, 1)
    a,b = np.meshgrid(padded_pixels, padded_pixels)
    D = np.zeros((n_pixels, n_pixels))
    for i in pixels:
        D[i, i-5 : i+6] = tensor[0,0,i-6]
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(a,b, D, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def soft_n_cut_loss(batch, enc, img_size):
    loss = []
    batch_size = batch.shape[0]
    k = enc.shape[1]
    weights = calculate_weights(batch, batch_size, img_size)
    for i in range(0, k):
        loss.append(soft_n_cut_loss_single_k(weights, enc[:, (i,), :], batch_size, img_size))
    losses_of_all_classes = torch.stack(loss)
    sum_over_classes = torch.sum(losses_of_all_classes, dim=0)
    mean_over_batches = torch.mean(k - sum_over_classes)
    return mean_over_batches


def test2():
    original = torch.tensor([[[255.0,255.0,255.0,255.0,255.0,0.0,0.0,0.0,0.0,0.0]]]).float()
    channel1 = torch.tensor([[[10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0]]]).float()
    enc = torch.nn.Softmax(dim=1)(torch.cat((channel1, -1*channel1), dim = 1))
    plt.plot(enc[0,0,:], label="raw class")
    plt.plot(enc[0,1,:], label="raw class")
    print(soft_n_cut_loss(original, enc, 10))

# test2()