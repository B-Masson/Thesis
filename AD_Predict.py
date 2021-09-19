# Rudimentary testing code to see whether the trained weights from the AD model can be used to make an accurate prediction
import tensorflow as tf
import numpy as np
from scipy import ndimage
from tensorflow import keras
from AD_Train import validation_preprocessing
from sklearn.utils import shuffle

# Fetch the saved testing data
test = np.load('testing.npz')
x_test = test['a']
y_test = test['b']

# Shuffle the test sets (for more varied testing)
x_test_shuff, y_test_shuff = shuffle(np.array(x_test), np.array(y_test))

# Quick test
print(y_test)
'''
# Load up the model
model = model.load_weights('ADModel')
for i in range(0,4):
    prediction = model.predict(np.expand_dims(x_test_shuff[i], axis=0))[0]
    print("The model predicts that this scan is type:", prediction[0], "| Actual cdr value:", y_test_shuff[i])
print("Done.")
'''