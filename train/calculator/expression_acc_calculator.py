from train.calculator import calculator
import torch as t


class ExpressionAccCalculator(calculator.Calculator):

    def __init__(self):
        super(ExpressionAccCalculator, self).__init__()
        self.correct_num = 0
        self.sample_num = 0

    def get_alias(self):
        return 'Exp'

    def record(self, prediction, label):
        assert isinstance(prediction, t.Tensor)
        assert isinstance(label, t.Tensor)
        correct_n = t.eq(prediction.int(), label.int()).sum().int()
        self.correct_num += correct_n
        self.sample_num += prediction.size()[0]

        return correct_n.float()/prediction.size()[0]

    def current_cal(self):
        if self.sample_num == 0:
            return 0
        else:
            return self.correct_num*1.0/self.sample_num
