from keras.callbacks import Callback
import numpy as np


def dice_coef_3D(y_true, y_pred, smooth=0.0001):
    intersection = np.sum(y_true[...,1:] * y_pred[...,1:])
    union = np.sum(y_true[...,1:]) + np.sum(y_pred[...,1:])
    return (2. * intersection + smooth) / (union + smooth)
