from train.predictor import predictor
import torch as t
import numpy as np


class ExpectionPredictor(predictor.Predictor):

    def __init__(self):
        super(ExpectionPredictor, self).__init__()
        num = 100
        self.multiplier = np.expand_dims(np.linspace(1, num, num), 1)
        self.multiplier = t.Tensor(self.multiplier)
        self.multiplier = self.multiplier.cuda()

    def get_alias(self):
        return 'Expection'

    def predict(self, output):
        """
        calculate the expection of output as predicting result.
        Note that output must be normalized before, e.g. using softmax
        :param output: output of net, of shape [n, c], where n is batch_size and c
                is class_num
        :return:
        """
        # output = output.float()
        if isinstance(output, list):
            output = output[-1]
        res = t.squeeze(t.mm(output, self.multiplier), dim=1)

        return t.round(res)