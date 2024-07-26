import torch
from torch.utils.data import Dataset
import numpy as np


import h5py
import os

class SquiggleDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files_list = os.listdir(root_dir)


    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        f5file = h5py.File(self.root_dir + self.files_list[idx], 'r')
        read_number = list(f5file["Raw"]["Reads"].keys())[0]
        return f5file["Raw"]["Reads"][read_number]["Signal"][:]


def save_squiggle_segments(read: h5py.File, read_name, segment_length, segment_overlap, output_dir):
    read_number = list(read["Raw"]["Reads"].keys())[0]
    signal = read["Raw"]["Reads"][read_number]["Signal"][:]
    signal_length = len(signal)
    for i in range(0, signal_length, segment_length - segment_overlap):
        if i + segment_length > signal_length:
            break
        segment = signal[i:i + segment_length]
        segment_name = f"{output_dir}/{read_name}_{i}_{i + segment_length}.npy"
        np.save(segment_name, segment)

#dataset = ReadsDataset(f"/home/lennart/Projektmodul/ecoli/tombo/fast5_files_gzip/")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

for (idx, batch) in enumerate(dataloader):
    print(batch[0])
