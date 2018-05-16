from layers import *
from model import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

model = Model()
model.add_layers(Input([32, 32, 3]))
model.add_layers(Conv2D(4))
model.add_layers(Activation('relu'))
model.add_layers(Flatten())
model.add_layers(Dense(10))
model.add_layers(Activation('softmax'))
model.compile()
model.summary()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)

print('\n' + '-'*33 + 'Training ' + '-' * 33)
model.fit(x_train, y_train, shuffle=True, n_epochs=3)

print('\n' + '-'*34 + 'Predict ' + '-' * 34)
Y_hat = model.predict(x_test)

Y_hat = np.argmax(Y_hat, axis=1)
print('Accuracy:', np.mean(y_test == Y_hat))