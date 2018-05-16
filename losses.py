import numpy as np

def crossentropy_loss(Y, Y_hat, deriv=False):
    m = Y.shape[0]
    log_likelihood = -np.multiply(Y, np.log(Y_hat + 1e-9))
    return np.sum(np.array(log_likelihood)) / m