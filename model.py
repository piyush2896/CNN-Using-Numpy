from losses import *
from utils import batches_generator
from tqdm import tqdm

class Model(object):

    LOSS_TO_METHOD = {
        'crossentropy': crossentropy_loss
    }

    def __init__(self):
        self.compiled = False
        self.layers = []

    def add_layers(self, layer):
        self.layers.append(layer)

    def compile(self, loss='crossentropy', lr=0.01):
        assert loss in list(Model.LOSS_TO_METHOD.keys())
        in_shape = self.layers[0].in_shape
        for i in range(1, len(self.layers)):
            self.layers[i].set_input_shape(in_shape)
            in_shape = self.layers[i].get_out_shape()
            self.layers[i].init_params()
        self.loss = Model.LOSS_TO_METHOD[loss]
        self.lr = lr
        self.compiled = True

    def summary(self):
        assert self.compiled
        base_str = '|{:10s}|{:^20s}|{:^20s}|{:^20s}|'
        print('\n' + base_str.format('Layer Type',
                                     'in shape',
                                     'out shape',
                                     'Params'))
        print('-' * 75)
        print(base_str.format(self.layers[0].LAYER_TYPE,
                              str(self.layers[0].in_shape),
                              str(self.layers[0].out_shape),
                              str(self.layers[0].nparams)))
        params = []
        for i in range(1, len(self.layers)):
            params.append(self.layers[i].get_nparams())
            print(base_str.format(self.layers[i].LAYER_TYPE,
                                  str(self.layers[i].in_shape),
                                  str(self.layers[i].get_out_shape()),
                                  str(params[-1])))
        print('-'* 75)
        print('Total Number of Params: ' + str(np.sum(params)))
        print('-'* 75 + '\n')

    def fit(self, X_train, Y_train, n_epochs=1, batch_size=32, shuffle=False):
        assert self.compiled

        for i in range(n_epochs):
            print('Epoch {}/{}'.format(i, n_epochs))
            gen = batches_generator(X_train, Y_train, batch_size, shuffle)
            l = []
            for X_, y in tqdm(gen, total=X_train.shape[0] // batch_size + 1):
                Y_hat = X_
                for k in range(1, len(self.layers)):
                    Y_hat = self.layers[k].forward(Y_hat)

                l.append(self.loss(y, Y_hat))
                dZ = self.layers[-1].backward(y)
                for k in range(len(self.layers)-1, 0, -1):
                    dZ = self.layers[k].backward(dZ)
                    self.layers[k].update_params(self.lr)
            print('\tloss: {}'.format(np.mean(l)))

    def predict(self, X, batch_size=32):
        Y_hat = []
        gen = batches_generator(X, batch_size=batch_size)
        for X_ in tqdm(gen, total=X.shape[0] // batch_size + 1):
            y_hat = X_
            for k in range(1, len(self.layers)):
                y_hat = self.layers[k].forward(y_hat)
            Y_hat.append(y_hat)
        return np.vstack(Y_hat)
