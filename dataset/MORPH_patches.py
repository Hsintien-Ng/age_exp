import os
import json
import numpy as np
import math
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as trans


class MORPHPATCHES(data.Dataset):

    @staticmethod
    def read_data(filePath):
        fp = open(filePath, 'r')
        data_dict = json.load(fp)
        fp.close()

        return data_dict


    @staticmethod
    def get_alias():

        return 'MORPHPATCHES'


    def __init__(self, index_dir, split):
        assert split in ['train', 'valid', 'test']
        self.data_names = os.listdir(os.path.join(index_dir, split))
        self.data_paths = []
        self.label = []
        for data_name in self.data_names:
            if os.path.splitext(data_name)[1] == '.json':
                data_dict = self.read_data(os.path.join(index_dir, split, data_name))
                self.data_paths.append(data_dict)
                self.label.append(int(data_name[-7:-5]))

        self.split = split
        self.index_dir = index_dir
        self.mode = ['face', 'left_eye', 'right_eye',
                     'left_cheek', 'right_cheek', 'mouth']


    def __getitem__(self, index):
        data_dict = self.data_paths[index]
        label = self.label[index]

        imgs, labels = self.format_data(data_dict, label)

        return imgs, labels


    def __len__(self):
        length = self.data_paths.__len__()

        return length


    def augment_trans_func(self, mode):
        if self.split == 'train':
            if mode == 'face':
                transform = trans.Compose([
                    trans.Resize(96),
                    trans.CenterCrop(96),
                    trans.RandomRotation(10),
                    trans.RandomHorizontalFlip(0.5),
                    trans.ToTensor(),
                    trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:
                transform = trans.Compose([
                    trans.Resize(24),
                    trans.CenterCrop(24),
                    trans.ToTensor(),
                    trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        else:
            if mode == 'face':
                transform = trans.Compose([
                    trans.Resize(96),
                    trans.CenterCrop(96),
                    trans.ToTensor(),
                    trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            else:
                transform = trans.Compose([
                    trans.Resize(24),
                    trans.CenterCrop(24),
                    trans.ToTensor(),
                    trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

        return transform


    def normal_distribute(self, mean, std=0.7, num_dim=100):
        vec = []
        for i in range(num_dim):
            exponent = math.exp(-(i - 1. - mean) ** 2 / (2. * (std ** 2)))
            multiplier = 1. / (math.sqrt(2. * math.pi) * std)
            vec.append(exponent * multiplier)
        vec = np.asarray(vec)
        vec = vec / np.sum(vec)

        return vec


    def format_data(self, data_dict, label):
        norm_dtb_label = self.normal_distribute(label)
        labels = [label, norm_dtb_label]
        imgs = []

        for m in self.mode:
            img = Image.open(data_dict[m])
            self.transform = self.augment_trans_func(m)
            img = self.transform(img)
            imgs.append(img)

        return imgs, labels