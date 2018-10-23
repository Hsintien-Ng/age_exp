import os
import numpy as np
import math
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as trans


Stage_Name = {1:'age_10_19', 2:'age_20_29', 3:'age_30_39',
              4:'age_40_49', 5:'age_50_59', 6:'age_60_69',
              7:'age_70_79'}
Stage_Num = 7


class MORPH(data.Dataset):

    @staticmethod
    def read_age_data(data_dir, mode, stage_id):
        fileName = 'age_{}_{}.txt'.format(stage_id*10, stage_id*10+9)
        filePath = os.path.join(data_dir, mode, fileName)
        fp = open(filePath, 'r')
        img_path = fp.read().splitlines()
        fp.close()

        return img_path


    @staticmethod
    def get_alias():
        return 'MORPH'


    def __init__(self, data_dir, mode, balance):
        assert mode in ['train', 'valid', 'test']
        self.age_paths = []
        self.age_labels = []
        # for 7 stages
        for s in range(0, Stage_Num):
            stage_id = s + 1
            agePath = self.read_age_data(data_dir, mode, stage_id)
            ageLabel = []
            for aP in agePath:
                imgName = os.path.split(aP)[1]
                ageLabel.append(int(imgName[-6:-4]))
            if balance:
                self.age_paths.append(agePath)
                self.age_labels.append(ageLabel)
            else:
                self.age_paths += agePath
                self.age_labels += ageLabel

        if balance:
            self.next_stage = 0

        self.mode = mode
        self.balance = balance
        self.data_dir = data_dir
        self.augment_trans_func()


    def __getitem__(self, index):
        if not self.balance:
            path = self.age_paths[index]
            label = self.age_labels[index]
        else:
            next_age_num = self.age_paths[self.next_stage].__len__()
            path = self.age_paths[self.next_stage][index % next_age_num]
            label = self.age_labels[self.next_stage][index % next_age_num]
            self.next_stage = (self.next_stage + 1) % Stage_Num

        img, label, norm_dtb_label = self.format_data(path, label)

        return img, label, norm_dtb_label


    def __len__(self):
        if not self.balance:
            return self.age_paths.__len__()
        else:
            length = max(*[age_data.__len__() for age_data in self.age_paths])
            return length


    def augment_trans_func(self):
        if self.mode == 'train':
            self.transform = trans.Compose([
                trans.Resize(200),
                trans.CenterCrop(200),
                trans.RandomRotation(10),
                trans.RandomHorizontalFlip(0.5),
                trans.ToTensor(),
                trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = trans.Compose([
                trans.Resize(200),
                trans.CenterCrop(200),
                trans.ToTensor(),
                trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def normal_distribute(self, mean, std=0.7, num_dim=100):
            vec = []
            for i in range(num_dim):
                exponent = math.exp(-(i - 1. - mean) ** 2 / (2. * (std ** 2)))
                multiplier = 1. / (math.sqrt(2. * math.pi) * std)
                vec.append(exponent * multiplier)
            vec = np.asarray(vec)
            vec = vec / np.sum(vec)

            return vec


    def format_data(self, path, label):
        img = Image.open(path)
        img = self.transform(img)
        # one_hot_label = \
        #     torch.zeros(100).scatter_(0, torch.LongTensor([label]), 1)
        norm_dtb_label = self.normal_distribute(label)

        return img, label, norm_dtb_label