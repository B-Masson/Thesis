# Rudimentary testing code to see whether the trained weights from the AD model can be used to make an accurate prediction
import tensorflow as tf
import numpy as np
from scipy import ndimage
from tensorflow import keras
from sklearn.utils import shuffle

# Fetch the saved testing data
test = np.load('testing.npz')
x_test = test['a']
y_test = test['b']
print("Data loaded.")

# Shuffle the test sets (for more varied testing)
x_test_shuff, y_test_shuff = shuffle(np.array(x_test), np.array(y_test))

# Quick test
print(y_test_shuff)

# Load up the model
model = keras.models.load_model('ADModel')
classes = ['normal', 'mild impairment', 'dementia']
for i in range(0, min(4, len(y_test_shuff))):
    prediction = model.predict(np.expand_dims(x_test_shuff[i], axis=0))[0]
    print("The model predicts the following:")
    for j in range(0, len(prediction)):
        print(round(prediction[j],2), "\% ", classes[j], ", ", sep='', end='')
    print("\nActual:", y_test_shuff[i])
print("Done.")
