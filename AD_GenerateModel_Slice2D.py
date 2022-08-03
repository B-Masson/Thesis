# Implementing 2D model using slices and majority voting
# Richard Masson
# Last use in 2021: October 29th
print("\nIMPLEMENTATION: 2D Slices")
print("CURRENT TEST: Attempting to train on every slice.")
#print("CURRENT TEST: First official test on everything.")
# TO DO: Model2
import os
from pyexpat import model
import subprocess as sp # Memory shit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import nibabel as nib
import NIFTI_Engine as ne
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from mlxtend.classifier import EnsembleVoteClassifier
import tensorflow as tf
#print("TF Version:", tf.version.VERSION)
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import random
import datetime
from collections import Counter
from volumentations import * # OI, WE NEED TO CITE VOLUMENTATIONS NOW
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
print("Imports working.")
# Attempt to better allocate memory.

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

from datetime import date
print("Today's date:", date.today())

# Are we in testing mode?
testing_mode = False
memory_mode = False
#limiter = False
#pure_mode = False
strip_mode = False
norm_mode = False
trimming = True
maxxing = False
modelname = "ADModel_2DSlice_V1.4-entire"
logname = "2DSlice_V1.4-entire"
if not testing_mode:
    print("MODELNAME:", modelname)
    print("LOGS CAN BE FOUND UNDER", logname)

# Model hyperparameters
if testing_mode:
    epochs = 1 #Small for testing purposes
    batches = 1
else:
    epochs = 15 # JUST FOR NOW
    batches = 1 # Going to need to fiddle with this over time (balance time save vs. running out of memory)

# Set which slices to use, based on previous findings
# To-do: I really need to automate this by saving the best values
priority_slices = [56, 57, 58, 64, 75, 85, 88, 89, 96]

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 2
else:
    scale = 1 # For now
w = int(169/scale)
h = int(208/scale)
d = int(179/scale)

# Slice params
#n = -1 # If it ever comes up as -1, we know future assignments aren't working.

# Prepare parameters for fetching the data
modo = 2 # 1 for CN/MCI, 2 for CN/AD, 3 for CN/MCI/AD, 4 for weird AD-only, 5 for MCI-only
if modo == 3 or modo == 4:
    print("Setting for 3 classes")
    classNo = 3 # Expected value
else:
    print("Setting for 2 classes")
    classNo = 2 # Expected value
if testing_mode: # CHANGIN THINGS UP
	filename = ("Directories/test_adni_" + str(modo)) # CURRENTLY AIMING AT TINY ZONE
else:
    filename = ("Directories/adni_" + str(modo))
if testing_mode:
    print("TEST MODE ENABLED.")
if trimming:
    print("TRIMMING DOWN CLASSES TO PREVENT IMBALANCE")

if trimming:
    imgname = filename+"_trimmed_images.txt"
    labname = filename+"_trimmed_labels.txt"
else:
    imgname = filename + "_images.txt"
    labname = filename + "_labels.txt"

# Grab the data
print("Reading from", imgname, "and", labname)
path_file = open(imgname, "r")
path = path_file.read()
path = path.split("\n")
path_file.close()
label_file = open(labname, 'r')
labels = label_file.read()
labels = labels.split("\n")
labels = [ int(i) for i in labels]
label_file.close()
print("Data distribution:", Counter(labels))
print("ClassNo:", classNo)
#print(labels)
labels = to_categorical(labels, num_classes=classNo, dtype='float32')
#print("Categorical shape:", labels[0].shape)
print("\nOBTAINED DATA. (Scaling by a factor of ", scale, ")", sep='')

# Split data
if testing_mode:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, test_size=0.5, stratify=labels, shuffle=True) # 50/50 (for eventual 50/25/25)
else:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, stratify=labels, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
if testing_mode:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.5) # Don't stratify test data, and just split 50/50.
else:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.2) # 70/30 val/test

if not testing_mode:
    np.savez_compressed('testing_sub', a=x_test, b=y_test)

# To observe data distribution
def countClasses(categors, name):
    temp = np.argmax(categors, axis=1)
    print(name, "distribution:", Counter(temp))

print("Number of training images:", len(x_train))
countClasses(y_train, "Training")
print("Number of validation images:", len(x_val))
countClasses(y_val, "Validation")
print("Number of testing images:", len(x_test), "\n")

# Data augmentation functions
def get_augmentation(patch_size):
    return Compose([
        Rotate((-3, 3), (-3, 3), (-3, 3), p=0.6), #0.5
        #Flip(2, p=1)
        ElasticTransform((0, 0.05), interpolation=2, p=0.3), #0.1
        #GaussianNoise(var_limit=(1, 1), p=1), #0.1
        RandomGamma(gamma_limit=(0.6, 1), p=0) #0.4
    ], p=1) #0.9 #NOTE: Temp not doing augmentation. Want to take time to observe the effects of this stuff
aug = get_augmentation((w,h,d)) # For augmentations

def load_image(file, label):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    if norm_mode:
        nifti = ne.resizeADNI(nifti, w, h, d, stripped=True)
    else:
        nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)
    data = {'image': nifti}
    aug_data = aug(**data)
    nifti = aug_data['image']
    nifti = tf.convert_to_tensor(nifti, np.float32)
    return nifti, label

def load_test(file): # NO AUG, NO LABEL
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    if norm_mode:
        nifti = ne.resizeADNI(nifti, w, h, d, stripped=True)
    else:
        nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)
    nifti = tf.convert_to_tensor(nifti, np.float32)
    return nifti

def load_slice(file, label):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    #print("using slice", n)
    slice = nifti[:,:,n]
    slice = ne.organiseSlice(slice, w, h, strip=strip_mode)
    # Augmentation
    # TO DO
    slice = tf.convert_to_tensor(slice, np.float32)
    return slice, label

def load_testslice(file):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    #print("using slice", n)
    slice = nifti[:,:,n]
    slice = ne.organiseSlice(slice, w, h, strip=strip_mode)
    # Augmentation
    # TO DO
    slice = tf.convert_to_tensor(slice, np.float32)
    return slice

def load_image_wrapper(file, labels):
    return tf.py_function(load_image, [file, labels], [np.float32, np.float32])

def load_test_wrapper(file):
    return tf.py_function(load_test, [file], [np.float32])

def load_slice_wrapper(file, labels):
    return tf.py_function(load_slice, [file, labels], [np.float32, np.float32])

def load_testslice_wrapper(file):
    return tf.py_function(load_testslice, [file], [np.float32])

# This needs to exist in order to allow for us to use an accuracy metric without getting weird errors
def fix_shape(images, labels):
    images.set_shape([None, w, h, 1])
    labels.set_shape([1, classNo])
    return images, labels

def fix_dims(image):
    image.set_shape([None, w, h, d, 1])
    return image

print("Setting up dataloaders...")
# TO-DO: Augmentation stuff
batch_size = batches
# Data loaders
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_set = (
    #train.shuffle(len(train))
    train.map(load_slice_wrapper)
    .batch(batch_size)
    .map(fix_shape)
    .prefetch(batch_size)
)

# Only rescale.
validation_set = (
    #val.shuffle(len(x_val))
    val.map(load_slice_wrapper)
    .batch(batch_size)
    .map(fix_shape)
    .prefetch(batch_size)
)

# Model architecture go here
def gen_basic_model(width, height, channels, classes=3): # Baby mode
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, channels))

    x = layers.Conv2D(filters=32, kernel_size=3, padding='valid', activation="relu", kernel_regularizer =tf.keras.regularizers.l2( l=0.01), data_format="channels_last")(inputs) # Layer 1: Simple 32 node start
    x = layers.MaxPool2D(pool_size=5, strides=5)(x) # Usually max pool after the conv layer

    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units

    outputs = layers.Dense(units=classes, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN_Basic")

    return model

# Build model.
optim = keras.optimizers.Adam(learning_rate=0.001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later

# Custom callbacks (aka make keras actually report stuff during training)
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        #print("End of training epoch {} of training; got log keys: {}".format(epoch, keys))
        print("Epoch {}/{} > ".format(epoch+1, epochs))
        #if (epoch+1) == epochs:
        #    print('')

# Setting class weights
from sklearn.utils import class_weight

y_org = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_org), y=y_org)
class_weight_dict = dict()
for index,value in enumerate(class_weights):
    class_weight_dict[index] = value
#class_weight_dict = {i:w for i,w in enumerate(class_weights)}
print("Class weight distribution will be:", class_weight_dict)

# Run the model
print("\nGenerating models...")

# Each image needs to become a bunch of slices. Slices are the 2D extractions that then need to be replicated into 3 channels
# Start of useful data: 50 , end of useful data: 156 (Alt)
# For every slice, create a model and fit it onto the 2D image array
# Plan: Loop n from 50 to 156. Each loop, set some global var, which then tells the loader to only extract slices at that point
# Loader then needs to load image, turn into array, take slice, replicate into 3 channels, then pass that into the 2D model
# Record the validation accuracy of each model in a 1D array
# Array of models are then entered into the scikit ensemble class, 'soft' voting, with weights set to an array of validation accuracies
# Can then evaluate on test data

channels = 1 # Replicating into 3 channels is proving annoying

def generateModels(start, stop):
    models = []
    weights = []
    for i in range(start, stop):
        global n
        n = i
        print("Fitting for slice", n, "out of", stop-1)

        # Set up a model
        model = gen_basic_model(w, h, channels, classes=classNo)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy']) #metrics=[tf.keras.metrics.BinaryAccuracy()]
        
        # Checkpointing & Early Stopping
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Temp at 30 to circumvent issue with first epoch behaving weirdly
        checkpointname = "/2d_V1_checkpoints/slice" +str(n) +".h5"
        if testing_mode:
            checkpointname = "2d_V1_checkpoints_testing.h5"
        mc = ModelCheckpoint(checkpointname, monitor='val_loss', mode='min', verbose=1, save_best_only=True) #Maybe change to true so we can more easily access the "best" epoch
        if testing_mode:
            log_dir = "/scratch/mssric004/test_logs/fit/2dslice/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            log_dir = "/scratch/mssric004/logs/fit/2dsliceNEO/" + logname + "/slice" +str(n) +"_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        # Fitting time
        if testing_mode:
            history = model.fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0, callbacks=[CustomCallback()]) # DON'T SPECIFY BATCH SIZE, CAUSE INPUT IS ALREADY A BATCHED DATASET
        else:
            history = model.fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0, shuffle=True)
            # Note: I have TEMPORARILY REMOVED CALLBACKS
            # DEAR ME, I HAVE REMOVED THE CLASS WEIGHTING STUFF. FOR NOW.
        modelname = "model-"+str(n)
        models.append(tuple((modelname, model)))
        print(history.history)
        weight = history.history['val_accuracy'][-1]
        weights.append(weight)
    return models, weights

def generatePriorityModels(slices):
    models = []
    weights = []
    for i in range(len(slices)):
        global n
        n = slices[i]
        print("Fitting for slice", n, ".")
        display = [str(x) for x in slices]
        display.insert(i, "->")
        print(display)
        # Set up a model
        model = gen_basic_model(w, h, channels, classes=classNo)
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy']) #metrics=[tf.keras.metrics.BinaryAccuracy()]
        
        # Checkpointing & Early Stopping
        es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Temp at 30 to circumvent issue with first epoch behaving weirdly
        checkpointname = "/2d_V1_checkpoints/slice" +str(n) +".h5"
        if testing_mode:
            checkpointname = "2d_V1_checkpoints_testing.h5"
        mc = ModelCheckpoint(checkpointname, monitor='val_loss', mode='min', verbose=1, save_best_only=True) #Maybe change to true so we can more easily access the "best" epoch
        if testing_mode:
            log_dir = "/scratch/mssric004/test_logs/fit/2dslice/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            log_dir = "/scratch/mssric004/logs/fit/2dsliceNEO/" + logname + "/slice" +str(n) +"_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        # Fitting time
        if testing_mode:
            history = model.fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0, callbacks=[CustomCallback()]) # DON'T SPECIFY BATCH SIZE, CAUSE INPUT IS ALREADY A BATCHED DATASET
        else:
            history = model.fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0, callbacks=[es, tb], shuffle=True)
            # Note: I have TEMPORARILY REMOVED CHECKPOINTS
        modelname = "model-"+str(n)
        models.append(tuple((modelname, model)))
        print(history.history)
        weight = history.history['val_accuracy'][-1]
        weights.append(weight)
    return models, weights

startpoint = 50 # According to SelectSlices findings
endpoint = 100 # # Remember, it will end at n-1
# and the after, do 100-157
#if testing_mode:
if testing_mode:
    startpoint = 95
    endpoint = 100

if maxxing:
    model_list, weights = generatePriorityModels(priority_slices)
else:
    model_list, weights = generateModels(startpoint, endpoint)
#print("Validation accuracies:")
wcount = []
wcount = startpoint
for weight in weights:
    #print("Slice", wcount, "-", round(weight,2)*100)
    wcount += 1
thresh = 60
print("********\nSlices with > ", thresh, "% validation acc:", sep='')
try:
    for i in range(len(weights)):
        rounded = round(weights[i]*100, 2)
        if rounded >= thresh:
            print("Slice", startpoint+i, "-", rounded, "%")
except Exception as e:
    print(e)
print("********\nAssigning to voting classifier...")
'''
# Sklearn strat
voter = EnsembleVoteClassifier(clfs=model_list, voting='hard', weights=weights, verbose=2, fit_base_estimators=False)
voter.fit(None,np.array([0,1]))

predx, predy = next(iter(validation_set))
pred = voter.predict(predx)
print("Predicted:", pred, "\nActual:", predy)
'''
# Manual strat

test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_x = tf.data.Dataset.from_tensor_slices((x_test))
print("Test data prepared.")

test_set = (
    val.map(load_slice_wrapper)
    .batch(batch_size)
    #.map(fix_shape)
    .prefetch(batch_size)
)

try:
    test_set_x = (
        test_x.map(load_testslice_wrapper)
        .batch(batch_size)
        #.map(fix_dims)
        .prefetch(batch_size)
    )
except Exception as e:
    print("Couldn't fit test_set_x for some reason. Error:\n", e)
#if not testing_mode: # NEED TO REWORK THIS

preds=[]
predi=[]
evals=[]
n = startpoint
print("Evaluating...")
for j in range(len(model_list)):
    scores = model_list[j][1].evaluate(test_set, verbose=0)
    acc = scores[1]*100
    loss = scores[0]
    evals.append(acc)
    try:
        pred = model_list[j][1].predict(test_set_x)
        preds.append(pred)
        predi.append(np.argmax(pred, axis=1))
    except:
        preds.append[[-1,-1]]
        predi.append(-1)
    n += 1

from sklearn.metrics import accuracy_score
from statistics import mode

# Soft = combine the probabilities of all slices together (also using weights)
def soft_voting(predicted_probas : list, weights : list) -> np.array:

    #sv_predicted_proba = np.mean(predicted_probas, axis=0)
    sv_predicted_proba = np.average(predicted_probas, axis=0, weights=weights)
    sv_predicted_proba[:,-1] = 1 - np.sum(sv_predicted_proba[:,:-1], axis=1)    

    return sv_predicted_proba, sv_predicted_proba.argmax(axis=1)

# Hard pick the modal prediction value
def hard_voting(predictions : list) -> np.array:
    return [mode(v) for v in np.transpose(np.array(predictions))]

#sv_predicted_proba, sv_predictions = soft_voting(preds)
sv_predicted_proba, sv_predictions = soft_voting(preds, weights)
hv_predictions = hard_voting(predi)

Y_test=np.argmax(y_test, axis=1)

for k in range(len(model_list)):
    #print("COMPARE THESE:\n", y_test, "\n", preds[k])
    print(f"Accuracy of {model_list[k][0]}: {accuracy_score(Y_test, predi[k])}")
    
print(f"\nAccuracy of Soft Voting: {accuracy_score(Y_test, sv_predictions)}")
print(f"Accuracy of Hard Voting: {accuracy_score(Y_test, hv_predictions)}")

print("All done!")