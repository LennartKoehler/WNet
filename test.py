import numpy as np
import os
import data_pod5
import matplotlib.pyplot as plt
import torch
from local_variables import data_path_rna004_2048


data = data_pod5.H5Dataset(data_path_rna004_2048)


plt.plot(list(range(len(data[0][0]))),data[0].numpy()[0])
plt.savefig("testt.png")


# radius=5
# kh=radius*2 + 1
# ox=2
# distances = torch.abs(torch.arange(1, kh + 1) - radius - 1) # IMPORTANT creating new tensor here! watch out for CPU/GPU
# # distances: tensor([5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5])
# distance_weights = torch.exp(torch.div(-1*(distances), ox**2)) # exp(-||X(i)-X(j)||^2_2  /  sigma^2_X)
# plt.plot(distance_weights)
# plt.savefig("test.png")