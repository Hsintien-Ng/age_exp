from train.loss import loss
import torch.nn as nn
import torch as t


class CrossEntropyLoss(loss.Loss):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        # do not use nn.CrossEntropy because it will do softmax automatically,
        # which had been done during inferring
        self.criterion = nn.NLLLoss()

    def get_alias(self):
        return 'CrossEntropy'

    def calculate_loss(self, output, label):
        log_softmax = t.log(output)

        return self.criterion(log_softmax, label)

