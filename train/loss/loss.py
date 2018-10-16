from abc import ABCMeta, abstractmethod


class Loss:

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def get_alias(self):
        return

    @abstractmethod
    def calculate_loss(self, output, label):
        return
