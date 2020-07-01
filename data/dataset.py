import os
import torch
import numpy as np
import soundfile as sf
from scipy import signal
import pandas as pd
import cmath
import re

from torch.utils import data

class SoundSegmentationDataset(data.Dataset):
    def __init__(self, root, split="train", task="segmentation", spatial_type=None, mic_num=1, angular_resolution=1, input_dim=1):
        self.split = split
        self.task = task

        self.spatial_type = spatial_type # None, ipd, complex are supported
        self.mic_num = mic_num
        self.angular_resolution = angular_resolution
        self.input_dim = input_dim

        self.duration = 256
        self.freq_bins = 256
        self.n_classes = 75

        self.label_csv = pd.read_csv(filepath_or_buffer=os.path.join(root, "label.csv"), sep=",", index_col=0)
        #print(self.label_csv)

        if split == "train":
            mode_dir = os.path.join(root, "train")
        elif split == "val":
            mode_dir = os.path.join(root, "val")
        else:
            raise ValueError("undefined dataset")

        self.data_pair_folders = []
        
        datapair_dirs = os.listdir(mode_dir)
        datapair_dirs.remove('make_toy_dataset4_multi.py')
        datapair_dirs.sort(key=int)                                 # sort by folder number
        for datapair_dir in datapair_dirs:
            datapair_dir = os.path.join(mode_dir, datapair_dir)
            if os.path.isdir(datapair_dir):  
                self.data_pair_folders.append(datapair_dir)
        """
        for i in range(200): # for debug 
            datapair_dir = os.path.join(mode_dir, str(i))
            if os.path.isdir(datapair_dir):  
                self.data_pair_folders.append(datapair_dir)
        """
        
    def __len__(self):
        return len(self.data_pair_folders)


    def __getitem__(self, index):
        with open(os.path.join(self.data_pair_folders[index], "sound_direction.txt"), "r") as f:
            direction = f.read().split("\n")[:-1]

        mixture = np.zeros((self.input_dim, self.freq_bins, self.duration), dtype=np.float32)
        mixture_phase = np.zeros((self.freq_bins * 2, self.duration), dtype=np.float32)
        
        if self.task == "sed":
            label = np.zeros((self.n_classes, self.duration), dtype=np.float32)
        elif self.task == "segmentation":
            label = np.zeros((self.n_classes, self.freq_bins, self.duration), dtype=np.float32)
        elif self.task == "ssl":
            label = np.zeros((self.angular_resolution, self.duration), dtype=np.float32)
        elif self.task == "ssls":
            label = np.zeros((self.angular_resolution, self.freq_bins, self.duration), dtype=np.float32)    
        elif self.task == "seld":
            label = np.zeros((self.n_classes, self.angular_resolution, self.duration), dtype=np.float32)
        elif self.task == "cube":
            label = np.zeros((self.n_classes, self.angular_resolution, self.freq_bins, self.duration), dtype=np.float32)            


        direction_index = 0
        filelist = os.listdir(self.data_pair_folders[index])
        for filename in filelist:
            if filename[-4:] == ".wav" and not filename[:-4] == "BGM":
                waveform, fs = sf.read(os.path.join(self.data_pair_folders[index], filename))

                _, _, stft = signal.stft(x=waveform.T, fs=fs, nperseg=512, return_onesided=False)
                
                if filename[0:3] == "0__":
                    if self.mic_num == 1:
                        stft = stft[:, 1:len(stft.T) - 1]
                        mixture_phase = np.angle(stft)
                        mixture = abs(stft[:256])
                        mixture = mixture[np.newaxis,:,:]

                elif filename[:7] == "0_multi": 
                    if self.mic_num == 8:
                        stft = stft[:, :, 1:len(stft.T) - 1]
                        mixture_phase = np.angle(stft[0])
                        for nchan in range(self.mic_num):
                            if self.spatial_type == "ipd":
                                if nchan == 0:
                                    mixture[nchan] = abs(stft[nchan][:256])
                                else:
                                    mixture[nchan*2-1] = np.cos(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                                    mixture[nchan*2] = np.sin(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                                    
                            elif self.spatial_type == "complex":
                                mixture[nchan * 3] = abs(stft[nchan][:256])
                                mixture[nchan*3 + 1] = stft[nchan][:256].real
                                mixture[nchan*3 + 2] = stft[nchan][:256].imag
                            else:
                                raise ValueError("Please use spatial feature when you use multi channel microphone array") 

                else:
                    stft = stft[:, 1:len(stft.T) - 1]
                    if self.angular_resolution == 1:
                        if self.task == "sed":
                            label[self.label_csv.T[filename[:-4]][0]] += abs(stft[:256]).max(0)
                            label[:,np.newaxis,:]
                        
                        elif self.task == "segmentation":
                            label[self.label_csv.T[filename[:-4]][0]] += abs(stft[:256])                        
                    
                    else:
                        angle = int(re.sub("\\D", "", direction[direction_index].split("_")[1])) // (360 // self.angular_resolution)
                        if self.task == "ssl":
                            label[angle] += abs(stft[:256]).max(0)
                            label = ((label > 0.1) * 1)
                            label = label[:,np.newaxis,:]
                        
                        elif self.task == "ssls":
                            label[angle] += abs(stft[:256])          
                        
                        elif self.task == "seld":
                            label[self.label_csv.T[filename[:-4]][0]][angle] += abs(stft[:256]).max(0)
                            label = ((label > 0.1) * 1)
                        
                        elif self.task == "cube":
                            label[self.label_csv.T[filename[:-4]][0]][angle] += abs(stft[:256])
                        direction_index += 1
                        
        if self.task == "cube":
            label = label.reshape((self.n_classes * self.angular_resolution, self.freq_bins, self.duration))
                        
        mixture, label = self.normalize(mixture, label)

        mixture = torch.from_numpy(mixture).float()
        label = torch.from_numpy(label).float()
        
        return mixture, label, mixture_phase


    def normalize(self, mixture, label):
        if self.spatial_type == "complex":
            sign = (mixture > 0) * 2 - 1
            sign = sign.astype(np.float32)
            mixture = abs(mixture) 

        elif self.spatial_type == "ipd":
            mixture[0] += 10**-8
            mixture[0] = 20 * np.log10(mixture[0])
            mixture[0] = np.nan_to_num(mixture[0])
            mixture[0] = (mixture[0] + 120) / 120
        
        else:
            mixture += 10**-8
            mixture = 20 * np.log10(mixture)
            mixture = np.nan_to_num(mixture)
            mixture = (mixture + 120) / 120
            
        label += 10**-8
        label = 20 * np.log10(label)
        label = np.nan_to_num(label) 
        label = (label + 120) / 120

        mixture = np.clip(mixture, 0.0, 1.0)
        label = np.clip(label, 0.0, 1.0)

        if self.spatial_type == "complex":
            inputs = inputs * sign

        return mixture, label
