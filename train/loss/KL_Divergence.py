from train.loss import loss
import torch.nn as nn
import torch as t


class KLDivLoss(loss.Loss):

    def __init__(self):
        super(KLDivLoss, self).__init__()
        # do not use nn.KLDiv because it will do softmax automatically,
        # which had been done during inferring
        self.criterion = nn.KLDivLoss()

    def get_alias(self):
        return 'KLDiv'

    def calculate_loss(self, output, label):
        if isinstance(output, list):
            output = output[-1]
        if isinstance(label, list):
            label = label[-1]
        log_softmax = t.log(output)

        return self.criterion(log_softmax, label) * 10