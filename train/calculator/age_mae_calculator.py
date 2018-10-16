from train.calculator import calculator
import torch as t


class AgeMAECalculator(calculator.Calculator):

    def __init__(self):
        super(AgeMAECalculator, self).__init__()
        self.sample_ae = 0
        self.sample_num = 0

    def get_alias(self):
        return 'Age'

    def record(self, prediction, label):
        """
        Calculate the MAE between the prediction and label.
        :param prediction: shape [n], where n is batch_size  
        :param label: shape [n], where n is batch_size
        :return: 
        """
        assert isinstance(prediction, t.Tensor)
        assert isinstance(label, t.Tensor)
        abs = (prediction.float() - label.float()).abs()
        AE = abs.sum()
        self.sample_ae += AE
        self.sample_num += prediction.size()[0]

        return AE.float() / prediction.size()[0]

    def current_cal(self):
        if self.sample_num == 0:
            return -1
        else:
            return self.sample_ae * 1.0 / self.sample_num