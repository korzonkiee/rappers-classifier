"""
Convolutional neural network model.

by Maciej Korzeniewski
14.04.2018
"""

from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Activation
from keras.models import Model, Sequential


class CNNModel(Sequential):

    """ CNN model class"""

    def __init__(self, input_shape, num_classes):
        Sequential.__init__(self)

        self.add(Conv2D(32, (3, 3), padding='same',
                        activation='relu', input_shape=input_shape))
        self.add(Conv2D(32, (3, 3), padding='same',
                        activation='relu', input_shape=input_shape))
        self.add(MaxPool2D((2, 2)))

        self.add(Conv2D(32, (3, 3), padding='same',
                        activation='relu', input_shape=input_shape))
        self.add(Conv2D(32, (3, 3), padding='same',
                        activation='relu', input_shape=input_shape))
        self.add(MaxPool2D((2, 2)))

        self.add(Flatten())
        self.add(Dense(128, activation='relu'))
        self.add(Dense(128, activation='relu'))

        self.add(Dense(num_classes, activation='softmax'))
