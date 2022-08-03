# Messing around with stuff without breaking the original version of the code.
# Richard Masson
# Info: Trying to fix the model since I'm convinced it's scuffed.
# Last use in 2021: October 29th
print("\nIMPLEMENTATION: K-Fold")
print("CURRENT TEST: Pre-emptively testing K-fold on the slightly more complex model.")
# TO DO: Model, without normed files, but looking at the stripped files
import os
import subprocess as sp # Memory shit
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
#print("Importing matplotlib.")
#mport matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
#import matplotlib as plt
import random
import datetime
from collections import Counter
from volumentations import * # OI, WE NEED TO CITE VOLUMENTATIONS NOW
from sklearn.model_selection import KFold, StratifiedKFold
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
strip_mode = False
norm_mode = False
modelname = "ADModel_K_v2" #Next in line: ADMODEL_NEO_v1.3
logname = "K_V2" #Neo_V1.3
if not testing_mode:
    print("MODELNAME:", modelname)
    print("LOGS CAN BE FOUND UNDER", logname)

# Model hyperparameters
if testing_mode:
    epochs = 2 #Small for testing purposes
    batches = 1
else:
    epochs = 15 # JUST FOR NOW
    batches = 1 # Going to need to fiddle with this over time (balance time save vs. running out of memory)

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 2
else:
    scale = 1 # For now
w = int(208/scale)
h = int(240/scale)
d = int(256/scale)

# Prepare parameters for fetching the data
modo = 1 # 1 for CN/MCI, 2 for CN/AD, 3 for CN/MCI/AD, 4 for weird AD-only, 5 for MCI-only
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
if norm_mode:
    print("USING NORMALIZED, STRIPPED IMAGES.")
elif strip_mode:
    print("USING STRIPPED IMAGES.")
#print("Filepath is", filename)
if norm_mode:
    imgname = filename+"_images_normed.txt"
    labname = filename+"_labels_normed.txt"
elif strip_mode:
    imgname = filename+"_images_stripped.txt"
    labname = filename+"_labels_stripped.txt"
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
#labels = to_categorical(labels, num_classes=classNo, dtype='float32')
#print("Categorical shape:", labels[0].shape)
print("\nOBTAINED DATA. (Scaling by a factor of ", scale, ")", sep='')

# Split data
if testing_mode:
    x, x_test, y, y_test = train_test_split(path, labels, test_size=0.25, stratify=labels, shuffle=True) # 50/50 (for eventual 50/25/25)
else:
    x, x_test, y, y_test = train_test_split(path, labels, test_size=0.2, stratify=labels, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
x = np.array(x)
y = np.array(y)

# Need to make sure y_test is already prepared
y_test = to_categorical(y_test, num_classes=classNo, dtype='float32')

# To observe data distribution
def countClasses(categors, name):
    #temp = np.argmax(categors, axis=1)
    print(name, "distribution:", Counter(categors))

print("Number of training/validation images:", len(x))
countClasses(y, "Training/validation")
#y_train = np.asarray(y_train)
#print("Validation distribution:", Counter(y_val))
print("Number of testing images:", len(x_test), "\n")

# Memory
if memory_mode:
    latest_gpu_memory = gpu_memory_usage(gpu_id)
    print(f"Post data aquisition (GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

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

def get_augmentation(patch_size):
    return Compose([
        Rotate((-3, 3), (-3, 3), (-3, 3), p=1), #0.5
        #Flip(2, p=1)
        ElasticTransform((0, 0.06), interpolation=2, p=0.4), #0.1
        #GaussianNoise(var_limit=(1, 1), p=1), #0.1
        RandomGamma(gamma_limit=(0.6, 1), p=0.4) #0.4
    ], p=0) #0.9 #NOTE: Temp not doing augmentation. Want to take time to observe the effects of this stuff
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
    return nifti, label

def load_val(file, label): # NO AUG
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    if norm_mode:
        nifti = ne.resizeADNI(nifti, w, h, d, stripped=True)
    else:
        nifti = ne.organiseADNI(nifti, w, h, d, strip=strip_mode)
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

def load_image_wrapper(file, labels):
    return tf.py_function(load_image, [file, labels], [np.float32, np.float32])

def load_val_wrapper(file, labels):
    return tf.py_function(load_val, [file, labels], [np.float32, np.float32])

def load_test_wrapper(file):
    return tf.py_function(load_test, [file], [np.float32])

# This needs to exist in order to allow for us to use an accuracy metric without getting weird errors
def fix_shape(images, labels):
    images.set_shape([None, w, h, d, 1])
    labels.set_shape([1, classNo])
    return images, labels

print("Quickly preparing test data...")
batch_size = batches

# Prepare the test data over here first
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_x = tf.data.Dataset.from_tensor_slices((x_test))
test_set_x = (
    test_x.map(load_test_wrapper)
    .batch(batch_size)
    .prefetch(batch_size)
)
#if not testing_mode: # NEED TO REWORK THIS

test_set = (
    test.map(load_val_wrapper)
    .batch(batch_size)
    .prefetch(batch_size)
) # Later we may need to use a different wrapper function? Not sure.
print("Test data prepared.")

# Model architecture go here
# For consideration: https://www.frontiersin.org/articles/10.3389/fbioe.2020.534592/full#B22
# Current inspiration: https://ieeexplore.ieee.org/document/7780459 (VGG19)
def gen_model(width=208, height=240, depth=256, classes=3): # Make sure defaults are equal to image resizing defaults
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    x = layers.Conv3D(filters=32, kernel_size=7, strides=2, activation="relu")(inputs)
    #x = layers.MaxPool3D(pool_size=2)(x)

    # Contruction seems to be pretty much the same as if this was 2D. Kernal should default to 5,5,5
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x) # Layer 1: Simple 32 node start
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x) # Usually max pool after the conv layer
    #x = layers.BatchNormalization()(x)

    # Doing this for now to test some things
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)
    # NOTE: Using it again, let's see how this goes

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x) # Double the filters
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x) # Double filters one more time
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=3)(x) # Pool size 3 cause I dunno, it makes the Flatten layer less intense
    #x = layers.BatchNormalization()(x)
    # NOTE: Also commented this one for - we MINIMAL rn

    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units
    #x = layers.Dropout(0.5)(x) # Start low, and work up if overfitting seems to be present

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")

    return model

def gen_basic_model(width=208, height=240, depth=256, classes=3): # Baby mode
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    x = layers.Conv3D(filters=32, kernel_size=3, padding='valid', activation="relu")(inputs) # Layer 1: Simple 32 node start
    x = layers.MaxPool3D(pool_size=5, strides=5)(x) # Usually max pool after the conv layer
    #x = layers.BatchNormalization()(x) # Do we bother with this?
    #x = layers.Dropout(0.1)(x) # Apparently there's merit to very light dropout after each conv layer
    
    x = layers.Conv3D(filters=64, kernel_size=3, padding='valid', activation="relu")(x)
    x = layers.MaxPool3D(pool_size=5, strides=5)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.1)(x)
    # NOTE: RECOMMENTED LOL

    # kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),
    #x = layers.Conv3D(filters=128, kernel_size=3, padding='valid',  activation="relu")(x) # Double filters one more time
    #x = layers.MaxPool3D(pool_size=3, strides=3)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.Dropout(0.1)(x)
    # NOTE: Also commented this one for - we MINIMAL rn
    
    #x = layers.Dropout(0.1)(x) # Here or below?
    #x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units
    #x = layers.Dense(units=506, activation="relu")(x)
    #x = layers.Dense(units=20, activation="relu")(x)
    x = layers.Dropout(0.2)(x) # Start low, and work up if overfitting seems to be present

    outputs = layers.Dense(units=classNo, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")

    return model

# Checkpointing & Early Stopping
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=False) # Temp at 30 to circumvent issue with first epoch behaving weirdly
checkpointname = "k_fold_checkpoints.h5"
if testing_mode:
    checkpointname = "k_fold_checkpoints_testing.h5"
mc = ModelCheckpoint(checkpointname, monitor='val_loss', mode='min', verbose=2, save_best_only=False) #Maybe change to true so we can more easily access the "best" epoch
if testing_mode:
    log_dir = "/scratch/mssric004/test_logs/fit/k_fold/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
else:
    if logname != "na":
        log_dir = "/scratch/mssric004/logs/fit/k_fold/" + logname + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        log_dir = "/scratch/mssric004/logs/fit/k_fold/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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

# Setting class weights
from sklearn.utils import class_weight

y_org = y
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_org), y=y_org)
class_weight_dict = dict()
for index,value in enumerate(class_weights):
    class_weight_dict[index] = value
#class_weight_dict = {i:w for i,w in enumerate(class_weights)}
print("Class weight dsitribution will be:", class_weight_dict)

# K-Fold setup
n_folds = 5
if testing_mode:
    n_folds = 2
acc_per_fold = []
loss_per_fold = []
#if testing_mode:
    #skf = KFold(n_splits=n_folds, shuffle=True)
#else:
skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

print("\nStarting cross-fold validation process...")
fold = 0
for train_index, val_index in skf.split(x, y):
    fold = fold + 1
    print("***************\nNow on Fold", fold, "out of", n_folds)

    # Set up train-val split per fold
    x_train = x[train_index]
    x_val = x[val_index]
    y_train = y[train_index]
    y_val = y[val_index]

    # Have to convert the labels to categorical here since KFolds doesn't like that
    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)
    
    print("Training iteration on " + str(len(x_train)) + " training samples, " + str(len(x_val)) + " validation samples")

    print("Setting up dataloaders...")
    # TO-DO: Augmentation stuff
    # Data loaders
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_set = (
    train.shuffle(len(train))
    .map(load_image_wrapper)
    .batch(batch_size)
    .map(fix_shape)
    .prefetch(batch_size)
    )

    # Only rescale.
    validation_set = (
    val.shuffle(len(x_val))
    .map(load_val_wrapper)
    .batch(batch_size)
    .map(fix_shape)
    .prefetch(batch_size)
    )

    # Build model.
    model = gen_basic_model(width=w, height=h, depth=d, classes=classNo)
    model.summary()
    optim = keras.optimizers.Adam(learning_rate=0.001)# , epsilon=1e-3) # LR chosen based on principle but double-check this later
    #model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy']) # Temp binary for only two classes
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy']) #metrics=[tf.keras.metrics.BinaryAccuracy()]
    # ^^^^ Temp solution for the ol' "as_list() is not defined on an unknown TensorShape issue"
    # NOTE: LOOK AT THIS AGAIN WHEN DOING 3-WAY CLASS

    # Memory
    if memory_mode:
        latest_gpu_memory = gpu_memory_usage(gpu_id)
        print(f"Pre train (GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

    # Run the model
    print("Fitting model...")

    if testing_mode:
    #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0)
        history = model.fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0, class_weight=class_weight_dict, callbacks=[CustomCallback()]) # DON'T SPECIFY BATCH SIZE, CAUSE INPUT IS ALREADY A BATCHED DATASET
    else:
    #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0, shuffle=True)
        history = model.fit(train_set, validation_data=validation_set, epochs=epochs, class_weight=class_weight_dict, callbacks=[mc, tb, es], verbose=0, shuffle=True)
    print("RESULTS FOR FOLD", fold, ":")
    print(history.history)
    best_epoch = np.argmin(history.history['val_loss']) + 1
    print("Epoch with lowest validation loss: Epoch", best_epoch, ": Val_Loss[", history.history['loss'][best_epoch-1], "] Val_Acc[", history.history['val_accuracy'][best_epoch-1], "]")
    '''
    try:
    plotname = "model"
    if testing_mode:
        plotname = plotname + "_testing"
    # Plot stuff
    plt.plot(history.history['accuracy'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
        plt.legend(['train', 'val'], loc='upper left')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig(plotname + "_acc.png")
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
        plt.legend(['train', 'val'], loc='upper left')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(plotname +"_val.png")
    print("Saved plot, btw.")
    except Exception as e:
    print("Couldn't save plots - rip.\nError:", e)
    '''
    # Memory
    if memory_mode:
        latest_gpu_memory = gpu_memory_usage(gpu_id)
        print(f"Post train (GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")
    
    # Classification report
    from sklearn.metrics import classification_report

    print("\nGenerating classification report...")
    y_pred = model.predict(test_set_x, verbose=0)
    #print("Before argmax:", y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    #print("test:", y_test)
    #print("Argmax pred:", y_pred)
    #print("\nand now we crash")
    y_test_arged = np.argmax(y_test, axis=1)
    rep = classification_report(y_test_arged, y_pred)
    print(rep)
    limit = min(10, len(y_test_arged))
    print("\nActual test set (first ", (limit+1), "):", sep='')
    print(y_test_arged[:limit])
    print("Predictions are  as follows (first ", (limit+1), "):", sep='')
    print(y_pred[:limit])

    # Final evaluation
    print("\nEvaluating using test data...")
    scores = model.evaluate(test_set, verbose=0) # Should not need to specify batch size, because of set
    acc = scores[1]*100
    loss = scores[0]
    print("Fold", fold, "evaluated scores - Acc:", acc, "Loss:", loss)
    acc_per_fold.append(acc)
    loss_per_fold.append(loss)

    # Memory
    if memory_mode:
        latest_gpu_memory = gpu_memory_usage(gpu_id)
        print(f"Post evaluation (GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

# Save outside the loop my sire
if testing_mode:
    modelname = "ADModel_K_Testing"
modelname = modelname +".h5"
model.save(modelname)

# Average scores
print("------------------------------------------------------------------------")
print("Score per fold")
for i in range(0, len(acc_per_fold)):
  print("------------------------------------------------------------------------")
  print("Fold", i+1, "- Loss:", loss_per_fold[i], "- Accuracy:", acc_per_fold[i])
print("------------------------------------------------------------------------")
print("Average scores for all folds:")
print("Accuracy:", np.mean(acc_per_fold), "+-", np.std(acc_per_fold))
print("Loss:",  np.mean(loss_per_fold))
print("------------------------------------------------------------------------")

print("Done.")