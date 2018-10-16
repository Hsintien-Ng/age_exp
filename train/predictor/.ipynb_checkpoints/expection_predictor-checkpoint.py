import predictor
import torch as t


class ExpectionPredictor(predictor.Predictor):

    def __init__(self):
        super(ExpectionPredictor, self).__init__()

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
        pass
