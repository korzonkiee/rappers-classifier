# """
# Preprocessor module.
# It transform input image so that it can be consumed
# and corectly interpreted by convolutional neural network.
#
# by Maciej Korzeniewski
# 10.04.2018
# """
#
# import dlib
# import numpy as np
#
# from PIL import Image
# from openface import AlignDlib
#
# SHAPE_PREDICATOR = "shape_predictor_68_face_landmarks.dat"
#
#
# class Preprocessor(object):
#     """Preprocessor class."""
#
#     def __init__(self):
#         self.detector = dlib.get_frontal_face_detector()
#         self.win = dlib.image_window()
#
#     def process(self, img):
#         """
#         Process image.
#         :param Image img: Image to be processed
#
#         Returns image containg centered face.
#         """
#
#         img = np.array(img)
#         alignment = AlignDlib(SHAPE_PREDICATOR)
#         face_bounding_box = alignment.getLargestFaceBoundingBox(img)
#         aligned_face = alignment.align(
#             96, img, face_bounding_box,
#             landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
#
#         return Image.fromarray(aligned_face)
