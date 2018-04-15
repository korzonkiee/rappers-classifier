"""
Convolutional Neural Network designed to
recognize 4 famous polish rappers.

by Maciej Korzeniewski
14.04.2018
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing import image

from PIL import Image

from keras import backend as K

def main():
    """Main method"""

    img = image.load_img("./taco.png", target_size=(197, 197))
    x = image.img_to_array(img)
    x /= 255.
    x = np.expand_dims(x, axis=0)

    model = load_model("./model_resnet.h5")
    res = model.predict(x)

    print(res)

if __name__ == "__main__":
    main()