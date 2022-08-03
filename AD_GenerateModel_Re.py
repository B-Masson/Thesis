# Messing around with stuff without breaking the original version of the code.
# Richard Masson
# Info: Trying to fix the model since I'm convinced it's scuffed.
# Last use in 2021: October 29th
print("\nIMPLEMENTATION: RESCALE")
print("CURRENT TEST: Testing model 2. Also now printing out loss values.")
# TO DO: Check all four norm strategies
import os
import subprocess as sp
from time import perf_counter # Memory shit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import nibabel as nib
import NIFTI_Engine as ne
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
#print("TF Version:", tf.version.VERSION)
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import sys
import random
import datetime
from collections import Counter
from volumentations import * # OI, WE NEED TO CITE VOLUMENTATIONS NOW
from sklearn.metrics import classification_report
print("Imports working.")
'''
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
'''
# Attempt to better allocate memory.

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
'''
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
'''
from datetime import date
print("Today's date:", date.today())

# Are we in testing mode?
testing_mode = False
memory_mode = False
limiter = False
pure_mode = False
strip_mode = False
norm_mode = False
curated = False
trimming = True
modelname = "ADModel_RESCALE_V1" #Next in line: ADMODEL_NEO_v1.3
logname = "Rescale_V1" #Neo_V1.3
if not testing_mode:
    if not pure_mode:
        print("MODELNAME:", modelname)
        print("LOGS CAN BE FOUND UNDER", logname)

# Model hyperparameters
if testing_mode or pure_mode:
    epochs = 1 #Small for testing purposes
    batch_size = 1
else:
    epochs = 30 # JUST FOR NOW
    batch_size = 1 # Going to need to fiddle with this over time (balance time save vs. running out of memory)

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 2
elif pure_mode:
    scale = 2
else:
    scale = 1 # For now
w = int(169/scale) # 208 # 169
h = int(208/scale) # 240 # 208
d = int(179/scale) # 256 # 179

# Prepare parameters for fetching the data
norm_mode = 0
modo = 2 # 1 for CN/MCI, 2 for CN/AD, 3 for CN/MCI/AD, 4 for weird AD-only, 5 for MCI-only
if modo == 3 or modo == 4:
    #print("Setting for 3 classes")
    classNo = 3 # Expected value
else:
    #print("Setting for 2 classes")
    classNo = 2 # Expected value
if testing_mode: # CHANGIN THINGS UP
	filename = ("Directories/test_adni_" + str(modo)) # CURRENTLY AIMING AT TINY ZONE
elif pure_mode:
    filename = ("Directories/test_tiny_adni_" + str(modo)) # CURRENTLY AIMING AT TINY ZONE
else:
    filename = ("Directories/adni_" + str(modo))
if testing_mode:
    print("TEST MODE ENABLED.")
elif limiter:
    print("LIMITERS ENGAGED.")
if curated:
    print("USING CURATED DATA.")
if pure_mode:
    print("PURE MODE ENABLED.")
if norm_mode:
    print("USING NORMALIZED, STRIPPED IMAGES.")
if trimming:
    print("TRIMMING DOWN CLASSES TO PREVENT IMBALANCE")
elif strip_mode:
    print("USING STRIPPED IMAGES.")
#print("Filepath is", filename)
if curated:
    imgname = "Directories/curated_images.txt"
    labname = "Directories/curated_labels.txt"
elif norm_mode:
    imgname = filename+"_images_normed.txt"
    labname = filename+"_labels_normed.txt"
elif strip_mode:
    imgname = filename+"_images_stripped.txt"
    labname = filename+"_labels_stripped.txt"
elif trimming:
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
#print("ClassNo:", classNo)
if testing_mode:
    print("Training labels:\n", labels)
labels = to_categorical(labels, num_classes=classNo, dtype='float32')
#print("Categorical shape:", labels[0].shape)
#print("Ensuring everything is equal length:", len(path), "&", len(labels))
print("\nOBTAINED DATA. (Scaling by a factor of ", scale, ")", sep='')

# Split data
#path = random.shuffle(path)
rar = 0 # Random state seed
if pure_mode:    
    x_train, y_train = shuffle(path, labels, random_state=0)
    x_test = x_train
    y_test = y_train
else:
    if testing_mode:
        x_train, x_val, y_train, y_val = train_test_split(path, labels, test_size=0.5, stratify=labels, random_state=rar, shuffle=True) # 50/50 (for eventual 50/25/25)
    else:
        if limiter:
            path, path_discard, labels, labels_discard = train_test_split(path, labels, stratify=labels, test_size=0.9)
            del path_discard
            del labels_discard
            epochs = min(epochs,2)
        x_train, x_val, y_train, y_val = train_test_split(path, labels, stratify=labels, random_state=rar, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
    if testing_mode:
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.5, random_state=rar, shuffle=True) # Just split 50/50.
    else:
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, random_state=rar, shuffle=True) # 70/30 val/test

if not testing_mode or pure_mode:
    np.savez_compressed('testing_sub', a=x_test, b=y_test)

# To observe data distribution
def countClasses(categors, name):
    temp = np.argmax(categors, axis=1)
    print(name, "distribution:", Counter(temp))

print("Number of training images:", len(x_train))
countClasses(y_train, "Training")
#y_train = np.asarray(y_train)
if not pure_mode:
    print("Number of validation images:", len(x_val))
    countClasses(y_val, "Validation")
print("Number of testing images:", len(x_test), "\n")
print("Label type:", y_train[0].dtype)

# Data augmentation functions
def get_augmentation(patch_size):
    return Compose([
        Rotate((-3, 3), (-3, 3), (-3, 3), p=0), #0.5
        #Flip(2, p=1)
        ElasticTransform((0, 0.06), interpolation=2, p=0), #0.1
        #GaussianNoise(var_limit=(1, 1), p=1), #0.1
        RandomGamma(gamma_limit=(0.6, 1), p=0) #0.4
    ], p=1) #0.9 #NOTE: Temp not doing augmentation. Want to take time to observe the effects of this stuff
aug = get_augmentation((w,h,d)) # For augmentations

# NOTE: DEAR ME, I HAVE CHANGED ALL INSTANCES OF FLOAT64 TO FLOAT32
# THE REASON FOR THIS IS BECAUSE MY NORMALIZE FUNCTION OUTPUTS FLOAT32

def load_image(file, label):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    if norm_mode:
        nifti = ne.resizeADNI(nifti, w, h, d, stripped=True)
    else:
        nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)

    # Augmentation
    data = {'image': nifti}
    aug_data = aug(**data)
    nifti = aug_data['image']

    nifti = tf.convert_to_tensor(nifti, np.float32)
    #label.set_shape([1]) # For the you-know-what
    #nifti.set_shape([None, w, h, d, 1])
    #label.set_shape([1, classNo])
    #print("map")
    #nifti = np.expand_dims(nifti, axis=0)
    if nifti.shape == (5,23,8,1):
        print("Why the hell is the shape", nifti.shape, "??")
        print("File in question:", loc)
        sys.exit()
    return nifti, label

def load_norm(file, label):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    
    #print("norm_mode is set to", norm_mode)

    nifti = ne.organiseOptimise(nifti, w, h, d, norm_mode)
    nifti = tf.convert_to_tensor(nifti, np.float32)

    return nifti, label

def load_val(file, label): # NO AUG
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    if norm_mode:
        nifti = ne.resizeADNI(nifti, w, h, d, stripped=True)
    else:
        nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)
    nifti = tf.convert_to_tensor(nifti, np.float32)
    #nifti.set_shape([None, w, h, d, 1])
    #label.set_shape([1, classNo])
    #nifti = np.expand_dims(nifti, axis=0)
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

def load_normtest(file): # NO AUG, NO LABEL
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    #print("norm_mode is set to", norm_mode)

    nifti = ne.organiseOptimise(nifti, w, h, d, norm_mode)
    nifti = tf.convert_to_tensor(nifti, np.float32)

    return nifti

def load_image_wrapper(file, labels):
    return tf.py_function(load_image, [file, labels], [np.float32, np.float32])

def load_val_wrapper(file, labels):
    return tf.py_function(load_val, [file, labels], [np.float32, np.float32])

def load_test_wrapper(file):
    return tf.py_function(load_test, [file], [np.float32])

def load_norm_wrapper(file, labels):
    return tf.py_function(load_norm, [file, labels], [np.float32, np.float32])

def load_normtest_wrapper(file):
    return tf.py_function(load_normtest, [file], [np.float32])

# This needs to exist in order to allow for us to use an accuracy metric without getting weird errors
def fix_shape(images, labels):
    #print("Going in:", images.shape, "|", labels.shape)
    images.set_shape([None, w, h, d, 1])
    labels.set_shape([1, classNo])
    #print("Going out:", images.shape, "|", labels.shape)
    return images, labels

def fix_wrapper(file, labels):
    return tf.py_function(fix_shape, [file, labels], [np.float32, np.float32])

# Model architecture go here
# For consideration: https://www.frontiersin.org/articles/10.3389/fbioe.2020.534592/full#B22
# Current inspiration: https://ieeexplore.ieee.org/document/7780459 (VGG19)
def gen_model(width=208, height=240, depth=256, classes=3): # Make sure defaults are equal to image resizing defaults
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    x = layers.Conv3D(filters=8, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x) # Paper conv and BN go together, then pooling
    #x = layers.Dropout(0.1)(x) # Apparently there's merit to very light dropout after each conv layer
    
    x = layers.Conv3D(filters=16, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)
    # NOTE: RECOMMENTED LOL

    # kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
    x = layers.Conv3D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)
    # NOTE: Also commented this one for - we MINIMAL rn
    
    x = layers.Conv3D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)

    x = layers.Dropout(0.5)(x)
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dense(units=1300)(x)
    x = layers.Dense(units=50)(x)
    #x = layers.Dropout(0.3)(x) # Start low, and work up if overfitting seems to be present

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")


    return model

def gen_model_2(width=208, height=240, depth=256, classes=3): # Make sure defaults are equal to image resizing defaults
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    x = layers.Conv3D(filters=8, kernel_size=3, padding="same", activation="relu")(inputs)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x) # Paper conv and BN go together, then pooling
    #x = layers.Dropout(0.1)(x) # Apparently there's merit to very light dropout after each conv layer
    
    # kernel_regularizer=l2(0.01)
    x = layers.Conv3D(filters=16, kernel_size=3, padding="same", activation="relu")(x) #(x)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.Dropout(0.1)(x)
    # NOTE: RECOMMENTED LOL

    x = layers.Conv3D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.Dropout(0.1)(x)
    # NOTE: Also commented this one for - we MINIMAL rn
    
    x = layers.Conv3D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    #x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.Dropout(0.1)(x)

    #x = layers.Conv3D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    #x = layers.Dropout(0.1)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    #x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dense(units=128)(x)
    x = layers.Dense(units=64)(x)
    #x = layers.Dropout(0.3)(x) # Start low, and work up if overfitting seems to be present

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")


    return model

def gen_basic_model(width=208, height=240, depth=256, classes=3): # Baby mode
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    x = layers.Conv3D(filters=32, kernel_size=3, padding='valid', activation="relu")(inputs) # Layer 1: Simple 32 node start
    x = layers.MaxPool3D(pool_size=10, strides=10)(x) # Usually max pool after the conv layer
    
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x) # Here or below?
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN_Basic")

    return model

# Checkpointing & Early Stopping
mon = 'val_accuracy'
es = EarlyStopping(monitor=mon, patience=15, restore_best_weights=True) # Temporarily turning this off because I want to observe the full scope
checkpointname = "norm_checkpoints.h5"
if testing_mode:
    checkpointname = "norm_checkpoints_testing.h5"
mc = ModelCheckpoint(checkpointname, monitor=mon, mode='auto', verbose=2, save_best_only=True) #Maybe change to true so we can more easily access the "best" epoch
if testing_mode:
    log_dir = "/scratch/mssric004/test_logs/fit/norm_mode" +str(norm_mode) +"/" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")
else:
    if logname != "na":
        log_dir = "/scratch/mssric004/logs/fit/norm_mode" +str(norm_mode) +"/" + logname + "_" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")
    else:
        log_dir = "/scratch/mssric004/logs/fit/norm_mode" +str(norm_mode) +"/" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")#.strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Custom callbacks (aka make keras actually report stuff during training)
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        #print("End of training epoch {} of training; got log keys: {}".format(epoch, keys))
        print("Epoch {}/{} > ".format(epoch+1, epochs))
        #if (epoch+1) == epochs:
        #    print('')
    '''
    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))
    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))
    
    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))
    '''

def make_unique(file_name, extension):
        if os.path.isfile(file_name):
            #print("I have determined that", file_name, "already exists.")
            expand = 1
            while True:
                new_file_name = file_name.split(extension)[0] + str(expand) + extension
                #print("What about ", new_file_name, "?", sep='')
                if os.path.isfile(new_file_name):
                    #print("It ALSO exists.")
                    expand += 1
                    continue
                else:
                    #print("It does not exist. Excellent.")
                    file_name = new_file_name
                    break
        else:
            print("", end='')
        print("Saving to", file_name)
        return file_name

# Params
print("\nParams:", epochs, "epochs &", batch_size, "batches.")

normscores = []
normloss = []
for i in range(4):
    # Display stuff
    norm_mode = i
    if i == 0:
        print("\nCURRENT TEST: NORM WITH TRIMMING")
        modelname += "norm_trim"
    elif i == 1:
        print("\nCURRENT TEST: NORM WITH NO TRIMMING")
        modelname += "norm_no_trim"
    elif i == 2:
        print("\nCURRENT TEST: NORM PER IMAGE")
        modelname += "norm_per"
    elif i == 3:
        print("\nCURRENT TEST: STANDARDIZE")
        modelname += "norm_standard"

    # Set up data
    print("Setting up dataloaders...")
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    
    train_set = (
        train.map(load_norm_wrapper)
        #.map(fix_shape)
        .batch(batch_size)
        .map(fix_shape)
        .prefetch(batch_size)
    )

    # Only rescale.
    validation_set = (
        val.map(load_norm_wrapper)
        #.map(fix_shape)
        .batch(batch_size)
        .map(fix_shape)
        .prefetch(batch_size)
    )

    # Build model
    print("Building model")
    model = gen_model_2(width=w, height=h, depth=d, classes=classNo)
    if norm_mode == 0:
        model.summary()
    optim = keras.optimizers.Adam(learning_rate=0.0001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
    #model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy']) # Temp binary for only two classes
    metric = 'accuracy'
    if metric == 'binary_accuracy':
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()]) #metrics=['accuracy']) #metrics=[tf.keras.metrics.BinaryAccuracy()]
    else:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy']) #metrics=[tf.keras.metrics.BinaryAccuracy()]

    print("Model built. Now fitting...")
    if testing_mode:
        #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=0)
        tic = perf_counter()
        history = model.fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0, callbacks=[CustomCallback()]) # DON'T SPECIFY BATCH SIZE, CAUSE INPUT IS ALREADY A BATCHED DATASET
        print("Time taken to fit: ", round(perf_counter()-tic, 2), "s", sep='')
        # Note: Class weight stuff has temporarily been removed
    else:
        #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=0, shuffle=True)
        history = model.fit(train_set, validation_data=validation_set, epochs=epochs, callbacks=[mc, tb, es], verbose=0, shuffle=True)
        # DEAR ME, I HAVE REMOVED THE CLASS WEIGHTING STUFF. FOR NOW.
    print(history.history)

    # Plotting
    plotting = True
    if plotting:
        try:
            print("Importing matplotlib.")
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            plotname = "Plots/norm/" +logname + "_mode" +str(norm_mode)
            if testing_mode:
                plotname = plotname + "_testing"
            # Plot stuff
            plt.plot(history.history[metric])
            plt.plot(history.history[('val_'+metric)])
            plt.legend(['train', 'val'], loc='upper left')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            name = plotname + "_acc.png"
            name = make_unique(name, ".png")
            plt.savefig(name)
            plt.clf()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.legend(['train', 'val'], loc='upper left')
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            name = plotname + "_loss.png"
            name = make_unique(name, ".png")
            plt.savefig(name)
            plt.clf()
            #plt.savefig(plotname + "_val" + ".png")
            print("Saved plot, btw.")
        except Exception as e:
            print("Plotting didn't work out. Error:", e)

    # Readings
    try:
        print("\nAccuracy max:", round(max(history.history[metric])*100,2), "% (epoch", history.history[metric].index(max(history.history[metric])), ")")
        print("Loss min:", round(min(history.history['loss']),2), "(epoch", history.history['loss'].index(max(history.history['loss'])), ")")
        print("Validation accuracy max:", round(max(history.history['val_'+metric])*100,2), "% (epoch", history.history['val_'+metric].index(max(history.history['val_'+metric])), ")")
        print("Val loss min:", round(min(history.history['val_loss']),2), "(epoch", history.history['val_loss'].index(max(history.history['val_loss'])), ")")
    except Exception as e:
        print("Cannot print out summary data. Reason:", e)

    # Final evaluation
    print("\nEvaluating using test data...")

    # First prepare the test data
    #print("Testing length check:", len(x_test), "&", len(y_test))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_x = tf.data.Dataset.from_tensor_slices((x_test))
    print("Test data prepared.")
    countClasses(y_test, "Test")

    try:
        test_set = (
            test.map(load_norm_wrapper)
            .batch(batch_size)
            .prefetch(batch_size)
        ) # Later we may need to use a different wrapper function? Not sure.
        try:
            scores = model.evaluate(test_set, verbose=0) # Should not need to specify batch size, because of set
            acc = scores[1]*100
            loss = scores[0]
            print("Evaluated scores - Acc:", acc, "Loss:", loss)
            normscores.append(acc)
            normloss.append(loss)
        except:
            print("Error occured during evaluation. Isn't this weird?\nTest set labels are:", y_test)
    except:
        print("Couldn't assign test_set to a wrapper (for evaluation).")

    try:
        test_set_x = (
            test_x.map(load_normtest_wrapper)
            .batch(batch_size)
            .prefetch(batch_size)
        )
        #if not testing_mode: # NEED TO REWORK THIS
        
        print("\nGenerating classification report...")
        try:
            y_pred = model.predict(test_set_x, verbose=0)
            #print("Before argmax:", y_pred)
            y_pred = np.argmax(y_pred, axis=1)
            #print("test:", y_test)
            #print("Argmax pred:", y_pred)
            #print("\nand now we crash")
            Y_test = np.argmax(y_test, axis=1)
            rep = classification_report(Y_test, y_pred)
            print(rep)
            limit = min(20, len(Y_test))
            print("\nActual test set (first ", (limit+1), "):", sep='')
            print(Y_test[:limit])
            print("Predictions are  as follows (first ", (limit+1), "):", sep='')
            print(y_pred[:limit])
        except Exception as e:
            print("Error occured in classification report (ie. predict). Error:", e)
    except:
        print("Couldn't assign test_set_x to a wrapper (for matrix).")

print("\nSummarised scores:")
#print(normscores)
print("Trimming:", normscores[0], "- Loss:", normloss[0], "\nNo trim:", normscores[1], "- Loss:", normloss[1], "\nPer:", normscores[2], "- Loss:", normloss[2], "\nStandard:", normscores[3], "- Loss:", normloss[3])
print("Done.")