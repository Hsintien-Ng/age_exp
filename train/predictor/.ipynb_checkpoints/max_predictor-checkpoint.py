import predictor
import torch as t


class MaxPredictor(predictor.Predictor):

    def __init__(self):
        super(MaxPredictor, self).__init__()

    def get_alias(self):
        return 'Max'

    def predict(self, output):
        """
        choose the class of highest probability(or scores, if without softmax) as
        predicting result
        :param output: output of net, of shape [n, c], where n is batch_size and c
                is class_num
        :return:
        """
        return t.argmax(output, dim=1)
