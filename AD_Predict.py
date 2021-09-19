# Rudimentary testing code to see whether the trained weights from the AD model can be used to make an accurate prediction
import tensorflow as tf
import numpy as np
from scipy import ndimage
from tensorflow import keras
from AD_Train import validation_preprocessing

test = np.load('testing.npz')
x_test = test['a']
y_test = test['b']