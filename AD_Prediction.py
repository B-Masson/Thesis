# Testing class for running predictions based the trained model
import tensorflow as tf
import numpy as np
from scipy import ndimage
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

# Fetch the saved testing data
testloc = "testing.npz"
test = np.load(testloc)
x_test = test['a']
y_test = test['b']
print("Data loaded.")

# Shuffle the test sets (for more varied testing)
x_test, y_test = shuffle(np.array(x_test), np.array(y_test))
print("Data shuffled.")

# Load up the model
modelname = "ADModel_Overweight"
model = keras.models.load_model(modelname)
print("Model loaded.")

print("Test input shapes are:", x_test[0].shape)
print("Expanded:", np.expand_dims(x_test[0], axis=0).shape)
sample = model.predict(np.expand_dims(x_test[0], axis=0))[0]

classes = ['normal', 'mild impairment', 'dementia']
for i in range(0, min(10, len(y_test))):
    prediction = model.predict(np.expand_dims(x_test[i], axis=0))[0]
    print("Prediction", i+1, ":")
    for j in range(0, len(prediction)):
        print(round(prediction[j],2), "\% ", classes[j], ", ", sep='', end='')
    print("\nActual:", y_test[i])
'''
# Generate a classification matrix
print("Generating classification report...\n")
Y_test = np.argmax(y_test, axis=1)
y_pred = model.predict(x_test, batch_size=1)
Y_pred = np.argmax(y_pred, axis=1)
print("Testing before argmax:", y_test[0])
print("After:", Y_test[0])
print("Predictions before argmax:", y_pred[0])
print("After:", Y_pred[0])
print(classification_report(Y_test, y_pred))
'''

Y_test = np.argmax(y_test, axis=1)
classes = model.predict(x_test, batch_size=8, verbose=1)
print(classes)

print("Done.")
