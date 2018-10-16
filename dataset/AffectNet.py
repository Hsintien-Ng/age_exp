import os
import torch as t
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as trans

EXP_NAME = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
            4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt',
            8: 'None', 9: 'uncertain', 10: 'Non-Face',
            -1: 'Error'}

EXP_NUM = 8


class AffectNet(data.Dataset):

    @staticmethod
    def read_exp_data(index_dir, exp_id):
        index_path = os.path.join(index_dir, '%s_index.txt' % EXP_NAME[exp_id])
        index_file = open(index_path, 'r')
        exp_data = []

        for line in index_file:
            exp_data.append(line.strip())

        return exp_data

    def __init__(self, index_dir, data_dir, mode, balance):
        assert mode in ['train', 'val']
        self.exps_paths = []
        self.exps_labels = []
        # for first 8 expressions only
        for exp in range(0, EXP_NUM):
            exp_paths = self.read_exp_data(index_dir, exp)
            exp_labels = [exp for i in range(exp_paths.__len__())]
            if balance:
                self.exps_paths.append(exp_paths)
                self.exps_labels.append(exp_labels)
            else:
                self.exps_paths += exp_paths
                self.exps_labels += exp_labels

        if balance:
            self.next_exp = 0

        self.mode = mode
        self.balance = balance
        self.data_dir = data_dir
        self.def_augment_trans_function()

    def __getitem__(self, index):
        if not self.balance:
            path = self.exps_paths[index]
            label = self.exps_labels[index]
        else:
            next_exp_num = self.exps_paths[self.next_exp].__len__()
            path = self.exps_paths[self.next_exp][index % next_exp_num]
            label = self.exps_labels[self.next_exp][index % next_exp_num]
            self.next_exp = (self.next_exp+1) % EXP_NUM

        img, one_hot_label = self.format_data(path, label)
        return img, one_hot_label

    def __len__(self):
        if not self.balance:
            return self.exps_paths.__len__()
        else:
            length = max(*[exp_paths.__len__() for exp_paths in self.exps_paths])
            return length


    def def_augment_trans_function(self):
        """
        Define transform function.
        Note that if one wants to resize image by padding rather than
        crop, he may transform this at data pre-process stage.
        :return:
        """
        if self.mode == 'train':
            self.transform = trans.Compose([
                trans.Resize(224),
                trans.CenterCrop(224),
                trans.RandomRotation(10),
                trans.RandomHorizontalFlip(0.5),
                trans.ToTensor(),
                trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif self.mode == 'val':
            self.transform = trans.Compose([
                trans.Resize(224),
                trans.CenterCrop(224),
                trans.ToTensor(),
                trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def format_data(self, path, label):
        img = Image.open(os.path.join(self.data_dir, path))
        img = self.transform(img)
        # for 8 exp only
        # in the end I decided to do this work in loss calculating
        # one_hot = t.Tensor(1, EXP_NUM).zero_()
        # one_hot.scatter_(dim=1, index=label, src=1.)
        return img, label




