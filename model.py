"""
Convolutional neural network model.

by Maciej Korzeniewski
14.04.2018
"""

from keras.layers import Flatten, Dense
from keras.models import Model
from keras.applications import ResNet50


class CNNModelBuilder(object):
    """ CNN model class"""

    def build(self, input_shape, num_classes):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        x = Flatten()(base_model.output)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        return Model(inputs=base_model.input, outputs=predictions)

