from abc import ABCMeta, abstractmethod


class Predictor:

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def get_alias(self):
        return

    @abstractmethod
    def predict(self, output):
        return