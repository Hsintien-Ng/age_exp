from train.loss import loss
import torch.nn as nn
import torch as t


class MSELoss(loss.Loss):

    def __init__(self, class_num, batch_size):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.class_num = class_num
        self.batch_size = batch_size
        self.one_hot = t.Tensor(self.batch_size, self.class_num).zero_()

    def get_alias(self):
        return 'MSE'

    def calculate_loss(self, output, label):
        self.one_hot.scatter_(dim=1, index=label, src=1.)

        return self.criterion(output, self.one_hot)
