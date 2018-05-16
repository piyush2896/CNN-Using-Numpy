import numpy as np

def _shuffle(X, Y):
    m = Y.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)

    return X[indices], Y[indices]

def batches_generator(X_train, Y_train=None, batch_size=32, shuffle=False):
    m = X_train.shape[0]
    
    if Y_train is None:
        for j in range(0, m, batch_size):
            yield X_train[j:j+batch_size]
    
    else:
        if shuffle:
            X_train, Y_train = _shuffle(X_train, Y_train)

        for j in range(0, m, batch_size):
            yield X_train[j:j+batch_size], Y_train[j:j+batch_size]