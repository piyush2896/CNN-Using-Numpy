import numpy as np
from abc import ABCMeta, abstractmethod

class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_params(self):
        pass

    @abstractmethod
    def set_input_shape(self, in_shape):
        pass

    @abstractmethod
    def get_out_shape(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dZ):
        pass

    @abstractmethod
    def update_params(self, lr):
        pass

    @abstractmethod
    def get_nparams(self):
        pass

class Input(object):

    LAYER_TYPE = 'input'

    def __init__(self, in_shape):
        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(in_shape)
        self.nparams = 0

