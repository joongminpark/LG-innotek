from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import shutil
from torch import nn
import collections
import pickle
import random


class LGDatasets(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        directory, filename = os.path.split(file_path)
        self.directory = directory

        # load preprocessed files
        with open(file_path, "rb") as f:
            self.normal_data, self.abnormal_data = pickle.load(f)
        
        # aggregate data
        self.aggregate_data = self.normal_data + self.abnormal_data

        # allocate labels (when training, do weighted sampling)
        self.labels = [data[-1] for data in self.aggregate_data]
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((214,214)),
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, item):
        raw_dir = os.path.join(self.directory, 'raw_data')
        x_y_z_file_path = list(
            map(lambda x: raw_dir + '/' + x, self.aggregate_data[item][:3])
        )
        
        # Loading the images
        x_y_z_images = list(map(Image.open, x_y_z_file_path))
        # Apply inage transformations
        img_x, img_y, img_z = map(self.transform, x_y_z_images)

        # loading labels
        label = torch.tensor(self.labels[item], dtype=torch.long)

        return img_x, img_y, img_z, label


    def __len__(self):
        return len(self.aggregate_data)