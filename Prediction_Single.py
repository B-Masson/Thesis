# Testing running a prediction on a single element
import NIFTI_Engine as ne
import tensorflow as tf
import numpy as np
from scipy import ndimage
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

# Params
scale = 4
w = int(208/scale)
h = int(240/scale)
d = int(256/scale)

# Obtain working sample
root = "/scratch/mssric004/ADNI_Data_NIFTI/MCI/023_S_6356"
x, y, = ne.extractSingle(w, h,d, root, "MCI")
x = np.asarray(x)
y = np.asarray(y)

# Load up the model
modelname = "ADModel_True"
model = keras.models.load_model(modelname)
print("Model loaded.")
print("X shape:", x.shape)

y_pred = model.predict(x)
pred_class = np.argmax(y_pred, axis=1)
print("Model predicted a ", pred_class, " vs. actual class ", y, sep='')

print("Done.")