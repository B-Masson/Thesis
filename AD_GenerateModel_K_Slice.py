# Messing around with stuff without breaking the original version of the code.
# Richard Masson
print("\nIMPLEMENTATION: K-Fold 2D Slices")
desc = "KFold slices - let's go."
print(desc)
import os
import subprocess as sp
from time import perf_counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys
import nibabel as nib
import NIFTI_Engine as ne
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
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
from volumentations import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from statistics import mode, mean
import glob
print("Imports working.")

# Attempt to better allocate memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from datetime import date
print("Today's date:", date.today())

# Are we in testing mode?
testing_mode = False
memory_mode = False
strip_mode = False
norm_mode = False
trimming = True
bad_data = False
nosplit = False
logname = "SliceK_V6-prio-advanced"
modelname = "ADModel_"+logname
if not testing_mode:
    print("MODELNAME:", modelname)
    print("LOGS CAN BE FOUND UNDER", logname)

# Model hyperparameters
if testing_mode:
    epochs = 1 #Small for testing purposes
    batch_size = 3
else:
    epochs = 25
    batch_size = 3

# Set which slices to use, based on previous findings
priority_slices = [56, 57, 58, 64, 75, 85, 88, 89, 96]

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 2
else:
    scale = 1 # For now
w = int(169/scale)
h = int(208/scale)
d = int(179/scale)
tic = perf_counter()

# Prepare parameters for fetching the data
modo = 2 # 1 for CN/MCI, 2 for CN/AD, 3 for CN/MCI/AD, 4 for AD-only, 5 for MCI-only
if modo == 3 or modo == 4:
    classNo = 3 # Expected value
else:
    classNo = 2 # Expected value
if testing_mode:
	filename = ("Directories/test_adni_" + str(modo))
else:
    filename = ("Directories/adni_" + str(modo))
if testing_mode:
    print("TEST MODE ENABLED.")
if trimming:
    print("TRIMMING DOWN CLASSES TO PREVENT IMBALANCE")
if bad_data:
    filename = "Directories/baddata_adni_" + str(modo)
print("Filepath is", filename)
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
labels = to_categorical(labels, num_classes=classNo, dtype='float32')
print("\nOBTAINED DATA. (Scaling by a factor of ", scale, ")", sep='')

# Split data
rar = 0
if testing_mode:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, test_size=0.5, stratify=labels, random_state=rar, shuffle=True) # 50/50 (for eventual 50/25/25)
else:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, stratify=labels, random_state=rar, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
if testing_mode:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.5, random_state=rar, shuffle=True) # Just split 50/50.
else:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, random_state=rar, test_size=0.4, shuffle=True) # 60/40 val/test
# Now stitch them together
x = x_train + x_val
traintemp = np.argmax(y_train, axis=1).tolist()
valtemp = np.argmax(y_val, axis=1).tolist()
y = traintemp+valtemp
x = np.array(x)
y = np.array(y)

# To observe data distribution
def countClasses(categors, name):
    temp = np.argmax(categors, axis=1)
    print(name, "distribution:", Counter(temp))

print("Number of training/validation images:", len(x))
print("Number of testing images:", len(x_test), "\n")
if testing_mode:
    print("Training labels:", y)

# Data augmentation functions
def get_augmentation(patch_size):
    return Compose([
        Rotate((-3, 3), (-3, 3), (-3, 3), p=0.6),
        ElasticTransform((0, 0.05), interpolation=2, p=0.3),
        RandomGamma(gamma_limit=(0.6, 1), p=0)
    ], p=1)
aug = get_augmentation((w,h,d)) # For augmentations

# 2D Augmentation stuff
import imgaug as ia
import imgaug.augmenters as iaa
rotaterand = lambda aug: iaa.Sometimes(0.6, aug)
elastrand = lambda aug: iaa.Sometimes(0.3, aug)
seq = iaa.Sequential([
    rotaterand(iaa.Rotate((-3, 3))),
    elastrand(iaa.ElasticTransformation(alpha=(0, 0.05), sigma=0.1))
])

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
    slice = nifti[:,:,n]
    slice = ne.organiseSlice(slice, w, h, strip=strip_mode)
    # Augmentation
    slice = seq(image=slice)
    slice = tf.convert_to_tensor(slice, np.float32)
    return slice, label

def load_val_slice(file, label):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    slice = nifti[:,:,n]
    slice = ne.organiseSlice(slice, w, h, strip=strip_mode)
    # No augmentation for val
    slice = tf.convert_to_tensor(slice, np.float32)
    return slice, label

def load_testslice(file):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    slice = nifti[:,:,n]
    slice = ne.organiseSlice(slice, w, h, strip=strip_mode)
    # Augmentation
    slice = tf.convert_to_tensor(slice, np.float32)
    return slice

def load_image_wrapper(file, labels):
    return tf.py_function(load_image, [file, labels], [np.float32, np.float32])

def load_test_wrapper(file):
    return tf.py_function(load_test, [file], [np.float32])

def load_slice_wrapper(file, labels):
    return tf.py_function(load_slice, [file, labels], [np.float32, np.float32])

def load_sliceval_wrapper(file, labels):
    return tf.py_function(load_val_slice, [file, labels], [np.float32, np.float32])

def load_testslice_wrapper(file):
    return tf.py_function(load_testslice, [file], [np.float32])

def fix_shape(images, labels):
    images.set_shape([None, w, h, 1])
    labels.set_shape([images.shape[0], classNo])
    return images, labels

def fix_dims(image):
    image.set_shape([None, w, h, d, 1])
    return image

print("Quickly preparing test data...")

# Prepare the test data over here first
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_x = tf.data.Dataset.from_tensor_slices((x_test))

test_set_x = (
        test_x.map(load_testslice_wrapper)
        .batch(batch_size)
        .prefetch(batch_size)
)
test_set = (
    test.map(load_sliceval_wrapper)
    .batch(batch_size)
    .prefetch(batch_size)
)

print("Test data prepared.")

# Model architecture go here
def gen_advanced_2d_model(width=169, height=208, depth=179, classes=2):
    modelname = "Advanced-2D-CNN"
    #print(modelname)
    inputs = keras.Input((width, height, depth))
    
    x = layers.Conv2D(filters=8, kernel_size=5, padding='valid', activation='relu', data_format="channels_last")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(filters=16, kernel_size=5, padding='valid', activation='relu', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(filters=32, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(filters=64, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    
    outputs = layers.Dense(units=classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name=modelname)
    
    return model

# Metrics
if batch_size > 1:
    metric = 'binary_accuracy'
else:
    metric = 'accuracy'

# Checkpointing & Early Stopping
mon = 'val_' +metric
print("Metric will be:", mon)
es = EarlyStopping(monitor=mon, patience=10, restore_best_weights=True)
checkpointname = "/scratch/mssric004/Checkpoints/kfoldslice-advanced-{epoch:02d}.ckpt"
if testing_mode:
    print("Setting checkpoint")
    checkpointname = "/TestCheckpoints/kslice_checkpoint-{epoch:02d}.ckpt"
mc = ModelCheckpoint(checkpointname, monitor=mon, mode='auto', verbose=2, save_weights_only=True, save_best_only=False)
if testing_mode:
    log_dir = "/scratch/mssric004/test_logs/fit/neo/" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")
else:
    if logname != "na":
        log_dir = "/scratch/mssric004/logs/fit/neo/" + logname + "_" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")
    else:
        log_dir = "/scratch/mssric004/logs/fit/neo/" + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M")#.strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Custom callbacks (aka make keras actually report stuff during training)
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        #print("End of training epoch {} of training; got log keys: {}".format(epoch, keys))
        print("Epoch {}/{} > ".format(epoch+1, epochs))
        #if (epoch+1) == epochs:
        #    print('')

class DebugCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        #keys = list(logs.keys())
        #print("End of training epoch {} of training; got log keys: {}".format(epoch, keys))
        print("Epoch {}/{} > ".format(epoch+1, epochs))
        #if (epoch+1) == epochs:
        #    print('')
    def on_train_batch_begin(self, batch, logs=None):
        print("...Training: start of batch {}".format(batch))

# Slice generation stuff
optim = keras.optimizers.Adam(learning_rate=0.0001)
channels = 1

def generatePriorityModels(slices, metric):
    models = []
    weights = []
    for i in range(len(slices)):
        global n
        n = slices[i]
        # Set up a model
        model = gen_advanced_2d_model(w, h, channels, classes=classNo)
        if metric == 'binary_accuracy':
            model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
        else:
            model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
        models.append(model)
        weight = model.get_weights()
        weights.append(weight)
    return models, weights

# Build model list
model_list, initials = generatePriorityModels(priority_slices, metric)

def reset_weights(reused_model, init_weights):
    reused_model.set_weights(init_weights)

# Define voting
def soft_voting(predicted_probas : list, weights : list) -> np.array:

    sv_predicted_proba = np.average(predicted_probas, axis=0, weights=weights)
    sv_predicted_proba[:,-1] = 1 - np.sum(sv_predicted_proba[:,:-1], axis=1)    

    return sv_predicted_proba, sv_predicted_proba.argmax(axis=1)

# K-Fold setup
n_folds = 5
if testing_mode:
    n_folds = 2
acc_per_fold = []
loss_per_fold = []
rar = 0
skf = StratifiedKFold(n_splits=n_folds, random_state=rar, shuffle=True)
mis_classes = []

fold = 0
# Start training

print("\nStarting cross-fold validation process...")
print("Params:", epochs, "epochs &", batch_size, "batches.")
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
    # Data loaders
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    train_set = (
        train.shuffle(len(train))
        .map(load_slice_wrapper)
        .batch(batch_size)
        .map(fix_shape)
        .prefetch(batch_size)
    )

    # Only rescale.
    validation_set = (
        val.shuffle(len(x_val))
        .map(load_sliceval_wrapper)
        .batch(batch_size)
        .map(fix_shape)
        .prefetch(batch_size)
    )

    # Reset the weights
    print("Resetting weights...")
    for i in range(len(model_list)):
        reset_weights(model_list[i], initials[i])

    print("Fitting model...")
    voting_weights = []
    names = []
    slices = priority_slices
    for i in range(len(slices)):
        global n
        n = slices[i]
        print("Fitting for slice", n, ".")
        display = [str(x) for x in slices]
        display.insert(i, "->")
        print(display)
        # Checkpoint stuff PER slice model
        # Give each fold a different local checkpoint
        localcheck = "/scratch/mssric004/TrueChecks/slicemodel" + str(slices[i]) +"_fold" +str(fold) +".ckpt"
        be = ModelCheckpoint(localcheck, monitor=mon, mode='auto', verbose=2, save_weights_only=True, save_best_only=True)
        
        if testing_mode:
            history = model_list[i].fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0, callbacks=[be, CustomCallback()])
        else:
            history = model_list[i].fit(train_set, validation_data=validation_set, epochs=epochs, callbacks=[es, be, CustomCallback()], verbose=0, shuffle=True)
        print(history.history)
        
        model_list[i].load_weights(localcheck)
        val_weight = history.history['val_'+metric][-1]
        names.append("model"+str(n))
        voting_weights.append(val_weight)
        # Clean up checkpoints
        print("Cleaning up...")
        found = glob.glob(localcheck+"*")
        if len(found) == 0:
            print("The system cannot find", localcheck)
        else:
            removecount = 0
            for checkfile in found:
                removecount += 1
                os.remove(checkfile)
            print("Successfully cleaned up", removecount, "checkpoint files.")

    print("RESULTS FOR FOLD", fold, ":")
    
    # Final evaluation
    preds=[]
    predi=[]
    evals=[]
    model_loss=[]
    print("Evaluating...")
    for j in range(len(model_list)):
        n = priority_slices[j]
        scores = model_list[j].evaluate(test_set, verbose=0)
        acc = scores[1]*100
        loss = scores[0]
        model_loss.append(loss)
        evals.append(acc)
        pred = model_list[j].predict(test_set_x)
        preds.append(pred)
        predi.append(np.argmax(pred, axis=1))
    sv_predicted_proba, sv_predictions = soft_voting(preds, voting_weights)
    Y_test=np.argmax(y_test, axis=1)
    for k in range(len(model_list)):
        print(f"Accuracy of {names[k]}: {accuracy_score(Y_test, predi[k])}")
    acc = accuracy_score(Y_test, sv_predictions)*100
    print("Fold", fold, "evaluated scores - Soft Voting Acc:", acc, "Loss:", loss)
    acc_per_fold.append(acc)
    loss_per_fold.append(mean(model_loss))
    print("Average so far:" , np.mean(acc_per_fold), "+-", np.std(acc_per_fold))
    # Test mechanism
    if nosplit:
        print("ENDING AFTER ONE FOLD.")
        break

# Not saving this model since it takes up insane space and isn't needed for future access

# Average scores
print("------------------------------------------------------------------------")
print("Score per fold")
for i in range(0, len(acc_per_fold)):
  print("------------------------------------------------------------------------")
  print("Fold", i+1, "- Loss:", loss_per_fold[i], "- Accuracy:", acc_per_fold[i])
print("------------------------------------------------------------------------")
print("Average scores for all folds:")
print("Accuracy:", np.mean(acc_per_fold), "+-", np.std(acc_per_fold))
print("Loss:",  np.mean(loss_per_fold), "+-", np.std(loss_per_fold))
print("------------------------------------------------------------------------")

# Save stuff so I can make a box and whisker plot later
if testing_mode:
    loc = "Means/KTesting"
else:
    loc = "Means/" + logname
print("Saving means to:", loc)
np.savez(loc, acc_per_fold, loss_per_fold)

print("Done.")