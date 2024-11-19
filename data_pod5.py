import torch
import numpy as np
import pickle
import pod5
from tqdm import tqdm
import pysam

import h5py
import os

class Basecalled_Read:
    def __init__(self, basecalled_read):
        self.basecalled_read = basecalled_read
        self.readid = basecalled_read.query_name

        self.signalid = basecalled_read.get_tag("pi") if basecalled_read.has_tag("pi") else self.readid
        self.basecall_start = basecalled_read.get_tag("ts") # ts:i:     the number of samples trimmed from the start of the signal
        self.basecall_stop = basecalled_read.get_tag("ns") # ns:i:     the number of samples in the signal after to trimming
        self.split_read = basecalled_read.get_tag("sp") if basecalled_read.has_tag("sp") else 0 # if split read get start offset of the signal
        self.quality_score = basecalled_read.get_tag("qs")
        self.shift = basecalled_read.get_tag("sm")
        self.scale = basecalled_read.get_tag("sd")

    def is_viable_read(self, quality_score_threshold):
        return True if self.quality_score > quality_score_threshold and self.split_read == 0 else False

class ReadData:
    def __init__(self, read_record):
        # metadata

        self.read_name = str(read_record.read_id)
        self.read_number = read_record.read_number

        self.read_id = read_record.read_id
        self.signal = read_record.signal

        self.signal_start = read_record.start_sample
        self.duration = len(self.signal) - self.signal_start

        self.segments = []

        #signal
        self.signal_length = len(self.signal)


    def __len__(self):
        return len(self.segments)
    
    def z_norm(self, signal):
        return (signal - np.mean(signal))/np.std(signal)
    
    def split_signal(self, segment_length, segment_overlap):
        for i in range(0, self.signal_length, segment_length - segment_overlap):
            if i + segment_length > self.signal_length:
                break
            segment = self.signal[i:i + segment_length]
            self.segments.append(segment)
        return self.segments
    
    def __getitem__(self, idx):
        return self.segments[idx]
    
    def get_segments(self):
        return self.segments
    
    def save_segments(self, filename):
        for i,segment in enumerate(self.segments):
            np.savez_compressed(filename, segment)

    def filter(self, basecalled_read: Basecalled_Read):
        self.signal = (self.signal - basecalled_read.shift) / basecalled_read.scale
        self.signal = self.signal[basecalled_read.basecall_start : basecalled_read.basecall_stop]
        self.signal_length = len(self.signal)





def write_data_file(in_file, segment_length, segment_overlap, number_segments, out_file_name):
    out_file = h5py.File(out_file_name, "w")
    p5_file_reader = pod5.DatasetReader(in_file)
    #segmentscounter is used as readcounter
    segments_counter = 0
    for read in tqdm(p5_file_reader):
        read = ReadData(read)
        segments = read.split_signal(segment_length, segment_overlap)
        for i,segment in enumerate(segments):
 
            out_file.create_dataset(f"{read.read_name}_{i}", data=segment)
        segments_counter+=1
        if segments_counter > number_segments:
            break

    out_file.close()
    return

def write_data_file_filtered(pod5_file, basecall_file, out_file_name, segment_length, segment_overlap, max_number_reads, quality_score_threshold):
    out_file = h5py.File(out_file_name, "w")
    p5_file_reader = pod5.DatasetReader(pod5_file)
    #segmentscounter is used as readcounter
    reads_counter = 0
    unused_reads_counter = 0
    with pysam.AlignmentFile(basecall_file, "r" if basecall_file.endswith('.sam') else "rb", check_sq=False) as samfile:
        for i, basecalled_read_raw in tqdm(enumerate(samfile.fetch(until_eof=True))):

            basecalled_read = Basecalled_Read(basecalled_read_raw)
            if basecalled_read.is_viable_read(quality_score_threshold):
                p5_read = next(p5_file_reader.reads([basecalled_read.signalid]))
                read = ReadData(p5_read)
                

                read.filter(basecalled_read)
                segments = read.split_signal(segment_length, segment_overlap)
                for i,segment in enumerate(segments):
                    out_file.create_dataset(f"{read.read_name}_{i}", data=segment)


                reads_counter+=1
                if reads_counter > max_number_reads:
                    break
            else:
                unused_reads_counter+=1
    print(f"Number of reads: {reads_counter}\nNumber of filtered out reads: {unused_reads_counter}")

    out_file.close()
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

        return torch.from_numpy(signal).float().unsqueeze(0) # unsqueeze: add channel axis

    def __len__(self):
        return len(self.segment_names)
    



if __name__ == "__main__":
    write_data_file_filtered("/work/zo48kij/data_masters/PNXRXX240011_10k_random.pod5", "/work/zo48kij/data_masters/PNXRXX240011_10k_random.bam", "/work/zo48kij/data_masters/data_segments_10000_length_2048_filtered_rna004.h5", 2048, 40, 10000, 20)
    print(len(h5py.File("/work/zo48kij/data_masters/data_segments_10000_length_2048_filtered_rna004.h5", "r")))
