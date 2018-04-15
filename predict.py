"""
Convolutional Neural Network designed to
recognize 4 famous polish rappers.

by Maciej Korzeniewski
14.04.2018
"""
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing import image

from PIL import Image

from keras import backend as K

def main():
    """Main method"""

    # data = np.genfromtxt('./training.log', delimiter=",",
    #
    #                      skip_header=2, names="epoch,acc,loss,val_acc,val_loss")
    #
    # plt.plot(data['epoch'], data['val_loss'], color='r', label='valid')
    # plt.plot(data['epoch'], data['loss'], color='g', label='train')
    # plt.legend(loc='best')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.title('Training process - loss')
    # plt.grid(True)
    # plt.show()
    #
    # plt.plot(data['epoch'], data['val_acc'], color='r', label='valid')
    # plt.plot(data['epoch'], data['acc'], color='g', label='train')
    # plt.legend(loc='best')
    # plt.xlabel('epochs')
    # plt.ylabel('acc')
    # plt.title('Training process - acc')
    # plt.grid(True)
    # plt.show()

    img = image.load_img("./data/valid/keke/0a20a094-308b-11e8-9998-0242ac110002.png", target_size=(96, 96))
    x = image.img_to_array(img)
    x /= 255.
    x = np.expand_dims(x, axis=0)

    model = load_model("/output/model_new_new.h5")
    res = model.predict(x)

    print(res)

if __name__ == "__main__":
    main()