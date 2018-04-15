"""
Convolutional Neural Network designed to
recognize 4 famous polish rappers.

by Maciej Korzeniewski
14.04.2018
"""

import os
import math

from model import CNNModelBuilder
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

TRAIN_DIR = "/data/train"
VALID_DIR = "/data/validation"

# Number of output classes.
NUM_CLASSES = 4

# Number of batch size (size of group
# of tranining samples). `Mini-batch learning`.
BATCH_SIZE = 32

# Number of epochs
NUM_EPOCHS = 20

IMG_WIDTH = 197
IMG_HEIGHT = 197


def main():
    """Main method"""

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples / BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

    model = CNNModelBuilder().build((IMG_WIDTH, IMG_HEIGHT, 3), NUM_CLASSES)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        zoom_range=[1, 1.15],
        rotation_range=15,
        width_shift_range=0.1,
        fill_mode='wrap',
        brightness_range=(0.5, 1)
    )
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        save_to_dir="./augmented"
    )

    earlyStopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.1,
        patience=10,
        verbose=1,
        mode='auto')

    csvLogger = CSVLogger(
        filename='training.log'
    )

    checkpointer = ModelCheckpoint('model_best.h5', verbose=1, save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_steps,
        epochs=NUM_EPOCHS,
        validation_data=valid_generator,
        validation_steps=num_valid_steps,
        callbacks=[earlyStopping, csvLogger, checkpointer]
    )

    model.save('model_resnet.h5')


if __name__ == "__main__":
    main()
