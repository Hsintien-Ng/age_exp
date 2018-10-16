from abc import ABCMeta, abstractmethod


class Calculator:

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def get_alias(self):
        return

    @abstractmethod
    def record(self, prediction, label):
        return

    @abstractmethod
    def current_cal(self):
        return
