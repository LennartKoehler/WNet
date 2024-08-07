import matplotlib.pyplot as plt
import torch
from data import ReadDataset
import numpy as np


dataset = ReadDataset("data_segments_reduced.h5")
print(dataset[0])
plt.plot(np.arange(0,len(dataset[0])),dataset[0][:])
plt.show()
