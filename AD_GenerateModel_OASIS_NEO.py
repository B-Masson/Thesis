# Sleek, refined, no chaff. This is AD training at its perfect form.
# Richard Masson
# Info: Going back to basics and trying a version of the code with less stuff going on.
# Last use in 2021: October 29th
print("IMPLEMENTATION: NEO OASIS")
print("CURRENT TEST: OASIS Dataset, can it save us?")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from nibabel import test
import NIFTI_Engine as ne
import LabelReader as lr
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
print("TF Version:", tf.version.VERSION)
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
import random
import datetime
from collections import Counter

# Attempt to better allocate memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Are we in testing mode?
testing_mode = False
modelname = "ADModel_OASIS"

# Model hyperparameters
if testing_mode:
    epochs = 2 #Small for testing purposes
    batches = 1
else:
    epochs = 20
    batches = 1 # Going to need to fiddle with this over time (balance time save vs. running out of memory)

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 1
else:
    scale = 1 # Quarter the size of optimal
w = int(128/scale)
h = int(128/scale)
d = int(64/scale)

# Fetch all our seperated data
rootloc = "/scratch/mssric004/Data" # HPC world
if testing_mode:
    rootloc = "/scratch/mssric004/Data_Small"
    print("TEST MODE ENABLED.")
x_arr, scan_meta = ne.extractArrays('all', w, h, d, root=rootloc)
clinic_sessions, cdr_meta = lr.loadCDR()

def genClassLabels(scan_meta):
    # Categorical approach
    # Set cdr score for a given image based on the closest medical analysis
    y_arr = []
    for scan in scan_meta:
        scan_day = scan['day']
        scan_cdr = -1
        cdr_day = -1
        min = 100000
        try:
            for x in cdr_meta[scan['ID']]:
                diff = abs(scan_day-x[0])
                if diff < min:
                    min = diff
                    scan_cdr = x[1]
                    cdr_day = x[0]
            scaled_val = int(scan_cdr*2) #0 = 0, 0.5 = 1, 1 = 2 (elimate decimals)
            #if scaled_val > 2:
            #    scaled_val = 2 # Cap out at 2 since we can classify anything at 1 or above as AD
            if scaled_val > 1: # TEMP FOR NOW
                scaled_val = 1
            y_arr.append(scaled_val) 
        except KeyError as k:
            print(k, "| Seems like the entry for that patient doesn't exist.")
    classNo = len(np.unique(y_arr))
    print("There are", classNo, "unique classes. ->", np.unique(y_arr), "in the dataset.")
    classCount = Counter(y_arr)
    print("Class count:", classCount)
    y_arr = tf.keras.utils.to_categorical(y_arr)
    return y_arr, classNo

#y_arr, classNo = genClassLabels(scan_meta)
#unique, counts = np.unique(y_arr, return_counts=True)
#print(dict(zip(unique, counts)))

# Split data
x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, stratify=y_arr, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
if (testing_mode):
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5) # Don't stratify test data, and just split 50/50.
else:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.2) # 70/30 val/test

if not testing_mode:
    np.savez_compressed('testing_sub', a=x_test, b=y_test)

x_train = np.asarray(x_train)
print("Shape of training data:", np.shape(x_train))
y_train = np.asarray(y_train)
print("Training labels:", y_train)
x_val = np.asarray(x_val)
print("Shape of validation labels:", np.shape(x_val))
y_val = np.asarray(y_val)
print("Validation labels:", y_val)

# Model architecture go here
def gen_model(width=208, height=240, depth=256, classes=3): # Make sure defaults are equal to image resizing defaults
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    # Contruction seems to be pretty much the same as if this was 2D. Kernal should default to 5,5,5
    x = layers.Conv3D(filters=32, kernel_size=5, activation="relu")(inputs) # Layer 1: Simple 32 node start
    x = layers.MaxPool3D(pool_size=2)(x) # Usually max pool after the conv layer
    x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    #x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)
    # NOTE: RECOMMENTED LOL

    x = layers.Conv3D(filters=64, kernel_size=5, activation="relu")(x) # Double the filters
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x) # Double filters one more time
    #x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)
    # NOTE: Also commented this one for - we MINIMAL rn

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dropout(0.2)(x) # Start low, and work up if overfitting seems to be present

    outputs = layers.Dense(units=classes, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")

    return model

# Build model.
model = gen_model(width=w, height=h, depth=d, classes=classNo)
model.summary()
optim = keras.optimizers.Adam(learning_rate=0.001) # LR chosen based on principle but double-check this later
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy']) # Temp binary for only two classes
# NOTE: LOOK AT THIS AGAIN WHEN DOING 3-WAY CLASS

# Run the model
print("Fitting model...")

if testing_mode:
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0)
else:
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0, shuffle=True)
if testing_mode:
    modelname = "ADModel_OASIS_Testing"
#model.save(modelname)
model.save("oasis_weights.h5")
print(history.history)

# Final evaluation
print("\nEvaluating using test data...")
print("Testing data shape:", np.shape(x_test))
print("Testing labels:", y_test)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
scores = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
acc = scores[1]*100
loss = scores[0]
print("Evaluated scores - Acc:", acc, "Loss:", loss)

from sklearn.metrics import classification_report

print("\nGenerating classification report...")
y_pred = model.predict(x_test, batch_size=1)
y_pred = np.argmax(y_pred, axis=1)
rep = classification_report(y_test, y_pred)
print(rep)
print("\nActual test set:")
print(y_test)
print("Predictions are  as follows:")
print(y_pred)

print("Done.")
