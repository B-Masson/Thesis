# Sleek, refined, no chaff. This is AD training at its perfect form.
# Richard Masson
# Info: Going back to basics and trying a version of the code with less stuff going on.
# Last use in 2021: October 29th
print("IMPLEMENTATION: NEO")
print("CURRENT TEST: NEO model, returning to our basics.")
import os
import subprocess as sp # Memory shit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from nibabel import test
import NIFTI_Engine as ne
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
import sys

# Memory shit
def gpu_memory_usage(gpu_id):
    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.used --format=csv"
    output_cmd = sp.check_output(command.split())
    
    memory_used = output_cmd.decode("ascii").split("\n")[1]
    # Get only the memory part as the result comes as '10 MiB'
    memory_used = int(memory_used.split()[0])

    return memory_used
# The gpu you want to check
gpu_id = 0
initial_memory_usage = gpu_memory_usage(gpu_id)

# Attempt to better allocate memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from datetime import date
print("Today's date:", date.today())

# Are we in testing mode?
testing_mode = False
modelname = "ADModel_NEO"

# Model hyperparameters
if testing_mode:
    epochs = 2 #Small for testing purposes
    batches = 1
else:
    epochs = 25
    batches = 1 # Going to need to fiddle with this over time (balance time save vs. running out of memory)

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 1
else:
    scale = 1 #
w = int(208/scale)
h = int(240/scale)
d = int(256/scale)

# Fetch all our seperated data
adniloc = "/scratch/mssric004/ADNI_Data_NIFTI" # HPC world
if testing_mode:
    adniloc = "/scratch/mssric004/ADNI_Test"
    print("TEST MODE ENABLED.")
modo = 1 # 1 for CN/MCI, 2 for CN/AD, 3 for CN/MCI/AD
if modo == 3:
    classNo = 3 # Expected value
else:
    classNo = 2 # Expected value


# Grab the data
#dataset = ne.extractADNILoader(w, h, d, root=adniloc, mode=modo)
x_arr, y_arr = ne.extractADNI(w, h, d, root=adniloc, mode=modo)
#unique, counts = np.unique(y_arr, return_counts=True)
#print(dict(zip(unique, counts)))

print("OBTAINED DATA.")

# Split data
x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, stratify=y_arr, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
if (testing_mode):
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5) # Don't stratify test data, and just split 50/50.
else:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.2) # 70/30 val/test

if not testing_mode:
    np.savez_compressed('testing_sub', a=x_test, b=y_test)
'''
x_train = np.asarray(x_train)
print("Shape of training data:", np.shape(x_train))
y_train = np.asarray(y_train)
print("Training labels:", y_train)
x_val = np.asarray(x_val)
print("Shape of validation labels:", np.shape(x_val))
y_val = np.asarray(y_val)
print("Validation labels:", y_val)
'''

# Data loaders?
#train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# Data augmentation functions
@tf.function
def rotate(image):
    def scipy_rotate(image): # Rotate by random angular amount
        # define some rotation angles
        angles = [-5, -3, -2, -1, 0, 0, 1, 2, 3, 5]
        # pick angles at random
        angle = random.choice(angles)
        # rotate image
        image = ndimage.rotate(image, angle, reshape=False)
        image[image < 0] = 0
        image[image > 1] = 1
        '''
        # define some rotation angles
        angles = [-20, -10, -5, 0, 0, 5, 10, 20]
        # Pick angel at random
        angle = random.choice(angles)
        # Rotate on x axis
        image2 = ndimage.interpolation.rotate(image, angle, mode='nearest', axes=(0, 1), reshape=False)
        # Generate new angle
        angle = random.choice(angles)
        # Roate on y axis
        image3 = ndimage.interpolation.rotate(image2, angle, mode='nearest', axes=(0, 2), reshape=False)
        angle = random.choice(angles)
        # Rotate on z axis
        image_final = ndimage.interpolation.rotate(image3, angle, mode='nearest', axes=(1, 2), reshape=False)
        image_final[image_final < 0] = 0
        image_final[image_final > 1] = 1
        return image_final
        '''
        return image

    augmented_image = tf.numpy_function(scipy_rotate, [image], tf.float32)
    return augmented_image

def shift(image):
    def scipy_shift(image):
        # generate random x shift pixel value
        x = (int)(random.uniform(-15, 15))
        # generate random y shift pixel value
        y = (int)(random.uniform(-15, 15))
        image = ndimage.interpolation.shift(image, (x, y, 0), mode='nearest')
        image[image < 0] = 0
        image[image > 1] = 1
        return image

    augmented_image = tf.numpy_function(scipy_shift, [image], tf.float32)
    return augmented_image

def train_preprocessing(image, label): # Only use for training, as it includes rotation augmentation
    # Rotate image
    #image = rotate(image) # Currently not rotating to test if this is causing the bug
    # Now shift it
    #image = shift(image)
    #image = tf.expand_dims(image, axis=3) # Only needed for OASIS data
    return image, label

def validation_preprocessing(image, label): # Can be used for val or test data (just ensures the dimensions are ok for the model)
    """Process validation data by only adding a channel."""
    #image = tf.expand_dims(image, axis=3) # Only needed for OASIS data
    return image, label

print("Ready to apply augmentations.")
# Augment data, as well expand dimensions to make the training model accept it (by adding a 4th dimension)
print("Setting up data augmenters...")
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
batch_size = batches
# Augment the on the fly during training.
train_set = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(batch_size)
)
# Only rescale.
validation_set = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(batch_size)
)

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

# Memory
latest_gpu_memory = gpu_memory_usage(gpu_id)
print(f"(GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

# Run the model
print("Fitting model...")

if testing_mode:
    #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0)
    history = model.fit(train_set, validation_data=validation_set, batch_size=batches, epochs=epochs, verbose=0)
else:
    #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0, shuffle=True)
    history = model.fit(train_set, validation_data=validation_set, batch_size=batches, epochs=epochs, verbose=0, shuffle=True)
if testing_mode:
    modelname = "ADModel_NEO_Testing"
model.save(modelname)
print(history.history)

# Memory
latest_gpu_memory = gpu_memory_usage(gpu_id)
print(f"(GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

# Final evaluation
print("\nEvaluating using test data...")
#print("Testing data shape:", np.shape(x_test))
#print("Testing labels:", y_test)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
scores = model.evaluate(x_test, y_test, verbose=0, batch_size=1)
acc = scores[1]*100
loss = scores[0]
print("Evaluated scores - Acc:", acc, "Loss:", loss)

from sklearn.metrics import classification_report

print("\nGenerating classification report...")
y_pred = model.predict(x_test, batch_size=1)
y_pred = np.argmax(y_pred, axis=1)
rep = classification_report(y_test, y_pred)
print(rep)
if testing_mode:
    print("\nActual test set:")
    print(y_test)
    print("Predictions are  as follows:")
    print(y_pred)

# Memory
latest_gpu_memory = gpu_memory_usage(gpu_id)
print(f"(GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

print("Done.")
