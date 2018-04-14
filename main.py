"""
Convolutional Neural Network designed to
recognize 4 famous polish rappers.

by Maciej Korzeniewski
14.04.2018
"""

import os
import math

from model import CNNModel
from keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = "./data/train"
VALID_DIR = "./data/valid"

# Number of output classes.
NUM_CLASSES = 4

# Number of batch size (size of group
# of tranining samples). `Mini-batch learning`.
BATCH_SIZE = 32

# Number of epochs
NUM_EPOCHS = 1


def main():
    """Main method"""

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    model = CNNModel(
        input_shape=(224, 224, 3),
        num_classes=NUM_CLASSES
    )
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_steps,
        epochs=NUM_EPOCHS,
        validation_data=valid_generator
    )

    model.save('model.h5')


if __name__ == "__main__":
    main()
