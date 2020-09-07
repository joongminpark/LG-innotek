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

def preprocessing(file_dir="./DB/raw_data"):
    files = os.listdir(file_dir)

    filter_files = collections.defaultdict(list)
    for f in files:
        f_split = f.split("_")
        if f.endswith('.jpg') and ('w' in f_split):
            # " . "   eliminated
            f_split[3] = f_split[3][:-2]
            
            event = ('_').join(f_split[:4])
            filter_files[event].append(f)
    
    # sorted events
    for event in filter_files.keys():
        filter_files[event] = sorted(filter_files[event])

    # check that there is 3 data in each events  ->  event have 3 data
    check_number_data = [len(data) for data in filter_files.values()]
    if len(set(check_number_data)) != 1:
        raise(ValueError, 'each event must have 3 data: x, y, z graphs')
    
    # give labels (0: normal, 1: abnormal)
    split_normal_abnormal = collections.defaultdict(list)

    for event in filter_files.keys():
        # abnormal
        if event[:2] == 'NG':
            split_normal_abnormal[1].append(filter_files[event] + [1])
        # normal
        else:
            split_normal_abnormal[0].append(filter_files[event] + [0])
    
    # length of normal, abnormal data
    num_normal, num_abnormal = len(split_normal_abnormal[0]), len(split_normal_abnormal[1])
    
    # As there are few abnormal data, split train & test data with proportion of abnormal data
    # shuffle
    random.shuffle(split_normal_abnormal[0])
    random.shuffle(split_normal_abnormal[1])
    
    # test data: same proportion (normal:abnormal = 1:1)
    num_test = int(num_abnormal * 0.2)

    test_normal = split_normal_abnormal[0][:num_test]
    test_abnormal = split_normal_abnormal[1][:num_test]
    
    train_normal = split_normal_abnormal[0][num_test:]
    train_abnormal = split_normal_abnormal[1][num_test:]

    # save pickle files
    pre_directory, _ = os.path.split(file_dir)

    train_file = os.path.join(pre_directory, 'train_lg.pkl')
    test_file = os.path.join(pre_directory, 'test_lg.pkl')
    
    with open(train_file, "wb") as f:
        pickle.dump(
            (train_normal, train_abnormal),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    
    with open(test_file, "wb") as f:
        pickle.dump(
            (test_normal, test_abnormal),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


preprocessing()