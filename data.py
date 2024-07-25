import torch
from torch.utils.data import Dataset

import h5py
import os

class ReadsDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files_list = os.listdir(root_dir)


    def __len__(self):
        return len(self.files_list)


    def __getitem__(self, idx):
        return h5py.File(self.root_dir + self.files_list[idx], 'r')["Raw"]["Signal"][:]

dataset = ReadsDataset(f"/home/lennart/Projektmodul/ecoli/tombo/fast5_files_gzip")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

for (idx, batch) in enumerate(dataloader):
    print(batch[0])
