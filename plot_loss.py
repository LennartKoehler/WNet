import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import models.W_swintransformer as Wswin

data_rec_path = "loss_results/rec_losses_transformer_4_depths_200_it.npy"
data_cut_path = "loss_results/n_cut_losses_transformer_4_depths_200_it.npy"

data_rec = np.load(data_rec_path)
data_cut = np.load(data_cut_path)

plt.plot(data_cut)
plt.savefig(data_cut_path[:-4])
plt.close()
plt.plot(data_rec)
plt.savefig(data_rec_path[:-4])
