import data
import h5py
import time
import torch

start_time = time.time()
path = "data_segments.h5"
dataset = data.H5Dataset_old(path)

batch_size = 20
num_workers = 5

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

for x in dataloader:
    print(x)


print(time.time()-start_time)


