import torch
import numpy as np
import pickle




import h5py
import os

class ReadData:
    def __init__(self, file_name):
        # metadata
        self.file_name = file_name
        self.read_name = file_name.split("/")[-1][:-6]
        self.h5_file = h5py.File(file_name, "r")
        read_number = list(self.h5_file["Raw"]["Reads"].keys())[0]

        self.offset = self.h5_file["UniqueGlobalKey"]["channel_id"].attrs["offset"]
        self.digitisation = self.h5_file["UniqueGlobalKey"]["channel_id"].attrs["digitisation"]
        self.range = self.h5_file["UniqueGlobalKey"]["channel_id"].attrs["range"]

        #signal
        self.signal = self.h5_file["Raw"]["Reads"][read_number]["Signal"][:]
        self.signal = self.scale_to_pA(self.signal)
        self.signal = self.z_norm(self.signal)
        self.signal_length = len(self.signal)


    def __len__(self):
        return len(self.segments)
    
    def z_norm(self, signal):
        return (signal - np.mean(signal))/np.std(signal)
    
    def split_signal(self, segment_length, segment_overlap):
        segments = []
        for i in range(0, self.signal_length, segment_length - segment_overlap):
            if i + segment_length > self.signal_length:
                break
            segment = self.signal[i:i + segment_length]
            segments.append(segment)
        return segments
    
    def __getitem__(self, idx):
        return self.segments[idx]
    
    def get_segments(self):
        return self.segments
    
    def save_segments(self):
        for i,segment in enumerate(self.segments):
            np.savez_compressed(f"squiggle_data/{self.read_name}_{i}.npz", segment)

    def scale_to_pA(self, value):
        return (value + self.offset) * self.range/self.digitisation



def write_data_file(root_dir, segment_length, segment_overlap, number_segments, out_file):
    h5_file = h5py.File(out_file, "w")

    for file_name in os.listdir(root_dir)[:number_segments]:
        try:
            file = os.path.join(root_dir, file_name)
            read = ReadData(file)
            segments = read.split_signal(segment_length, segment_overlap)
            for i,segment in enumerate(segments):
                h5_file.create_dataset(f"{read.read_name}_{i}", data=segment)
        except Exception as e:
            print("Error: ", e, "\nCan't read file: ", file_name)
    h5_file.close()
    return


#---------------------------------------------------------#

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.file_path = path
        self.dataset = None

        with h5py.File(self.file_path, 'r') as file:
            self.segment_names = list(file.keys())

    def __getitem__(self, index):

        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r', swmr=True)
        signal = self.dataset[self.segment_names[index]][:]

        return torch.from_numpy(signal).float().unsqueeze(0)# unsqueeze: add channel axis

    def __len__(self):
        return len(self.segment_names)
    

# used for the pytorch dataloader
# class H5Dataset_old(torch.utils.data.Dataset):

#     def __init__(self, h5_file_name, transform=None):
#         self.h5_file = h5py.File(h5_file_name, "r")
#         self.transform = transform
#         self.segment_names = list(self.h5_file.keys())


#     def __len__(self):
#         return len(self.segment_names)
    
#     def __getitem__(self, idx):
#         signal = self.h5_file[self.segment_names[idx]][:]
#         return torch.from_numpy(signal).float().unsqueeze(0) # unsqueeze: add channel axis
    

    

if __name__ == "__main__":
    write_data_file("/home/lennart/Projektmodul/ecoli/tombo/fast5_files_gzip", 256, 20, -1, "data_segments_all_4000_reads.h5")
    print(len(h5py.File("data_segments_all_4000_reads.h5", "r")))
