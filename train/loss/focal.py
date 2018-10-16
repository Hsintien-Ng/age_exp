from train.loss import loss
import torch.nn as nn
import torch as t


class FocalLoss(loss.Loss):

    def __init__(self, pow):
        super(FocalLoss, self).__init__()
        self.pow = pow
        self.criterion = nn.NLLLoss(reduce=False)

    def get_alias(self):
        return 'Focal({})'.format(self.pow)

    def calculate_loss(self, output, label):
        log_softmax = t.log(output)
        cross_entropy = self.criterion(log_softmax, label)

        focal_loss = t.pow(1 - output.gather(1, label.view(-1, 1)), self.pow) * cross_entropy
        #focal_loss = t.pow(1-t.max(output, 1)[0], self.pow)*cross_entropy

        return t.mean(focal_loss)

