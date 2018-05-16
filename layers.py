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

class Conv2D(Layer):

    LAYER_TYPE = 'conv'

    def __init__(self,
                 kernels,
                 k_size=3,
                 strides=2,
                 padding=1):
        self.kernels = kernels
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def _zero_pad(self, X, pad):
        return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                      'constant', constant_values=(0, 0))

    def _conv_single_step(self, X_slice, W, b):
        return np.sum(X_slice * W) + float(b)

    def set_input_shape(self, in_shape):
        self.in_shape = in_shape

    def get_out_shape(self):
        (n_H_prev, n_W_prev, n_C_prev) = self.in_shape
        n_H = int((n_H_prev - self.k_size + 2 * self.padding) / self.strides) + 1
        n_W = int((n_W_prev - self.k_size + 2 * self.padding) / self.strides) + 1
        return (n_H, n_W, self.kernels)

    def init_params(self):
        self.W = np.random.randn(self.k_size,
                                 self.k_size,
                                 self.in_shape[2],
                                 self.kernels)
        self.b = np.random.randn(1, 1, 1, self.kernels)

    def forward(self, X):
        self.X = X
        (m, n_H_prev, n_W_prev, n_C_prev) = X.shape
        (f, f, n_C_prev, n_C) = self.W.shape

        n_H = int((n_H_prev - f + 2 * self.padding) / self.strides) + 1
        n_W = int((n_W_prev - f + 2 * self.padding) / self.strides) + 1

        Z = np.zeros((m, n_H, n_W, n_C))
        X_pad = self._zero_pad(self.X, self.padding)

        for i in range(m):
            x_pad = X_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h + h * (self.strides - 1)
                        vert_end = vert_start + f
                        horiz_start = w + w * (self.strides - 1)
                        horiz_end = horiz_start + f

                        x_slice = x_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        Z[i, h, w, c] = self._conv_single_step(x_slice,
                                                               self.W[:, :, :, c],
                                                               self.b[:, :, :, c])
        return Z

    def backward(self, dZ):
        (m, n_H_prev, n_W_prev, n_C_prev) = self.X.shape
        (f, f, n_C_prev, n_C) = self.W.shape
        (m, n_H, n_W, n_C) = dZ.shape

        self.dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        self.dW = np.zeros((f, f, n_C_prev, n_C))
        self.db = np.zeros((1, 1, 1, n_C))

        X_pad = self._zero_pad(self.X, self.padding)
        dA_prev_pad = self._zero_pad(self.dA_prev, self.padding)

        for i in range(m):
            x_pad = X_pad[i]
            da_prev_pad = dA_prev_pad[i]
            
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h + h * (self.strides - 1)
                        vert_end = vert_start + f
                        horiz_start = w + w * (self.strides - 1)
                        horiz_end = horiz_start + f

                        x_slice = x_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:, :, :, c] * dZ[i, h, w, c]
                        self.dW[:,:,:,c] += x_slice * dZ[i, h, w, c]
                        self.db[:,:,:,c] += dZ[i, h, w, c]
                    
        self.dA_prev[i, :, :, :] = da_prev_pad[self.padding:-self.padding, self.padding:-self.padding, :]
        return self.dA_prev

    def update_params(self, lr):
        self.W  = self.W - lr * self.dW
        self.b  = self.b - lr * self.db

    def get_nparams(self):
        return np.product(self.W.shape) + np.product(self.b.shape)