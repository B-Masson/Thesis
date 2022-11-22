# Core 3D model implementation
# Richard Masson
print("\nIMPLEMENTATION: NEO")
desc = "Advanced model. 3D Model."
print(desc)
import os
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
from volumentations import *
print("Imports working.")
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("GPU device not detected.")

tic_total = perf_counter()

# Attempt to better allocate memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from datetime import date
print("Today's date:", date.today())

# Are we in testing mode?
testing_mode = False
# Please set a normalization mode
norm_setting = 2
memory_mode = False
limiter = False
pure_mode = False
strip_mode = False
norm_mode = False
curated = False
trimming = True
bad_data = False
aug = True
reg = True
logname = "NEO_V6-AD-ultimate"
modelname = "ADModel_"+logname
if not testing_mode:
    if not pure_mode:
        print("MODELNAME:", modelname)
        print("LOGS CAN BE FOUND UNDER", logname)

# Model hyperparameters
if testing_mode or pure_mode:
    epochs = 1 #Small for testing purposes
    batch_size = 3
else:
    epochs = 40
    batch_size = 3

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 2
elif pure_mode:
    scale = 2
else:
    scale = 1
w = int(169/scale)
h = int(208/scale)
d = int(179/scale)

# Prepare parameters for fetching the data
modo = 2 # 1 for CN/MCI, 2 for CN/AD, 3 for CN/MCI/AD, 4 for AD-only, 5 for MCI-only
if modo == 3 or modo == 4:
    classNo = 3 # Expected value
else:
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
if not reg:
    print("NOT USING REGULARIZATION")
if curated:
    print("USING CURATED DATA.")
if pure_mode:
    print("PURE MODE ENABLED.")
if trimming:
    print("TRIMMING DOWN CLASSES TO PREVENT IMBALANCE")
if norm_mode:
    print("USING NORMALIZED, STRIPPED IMAGES.")
elif strip_mode:
    print("USING STRIPPED IMAGES.")
if bad_data:
    filename = "Directories/baddata_adni_" + str(modo)
print("Filepath is", filename)
print("Norm mode:", norm_setting)
if curated:
    imgname = "Directories/curated_images.txt"
    labname = "Directories/curated_labels.txt"
elif norm_mode:
    imgname = filename+"_images_normed.txt"
    labname = filename+"_labels_normed.txt"
elif strip_mode:
    if trimming:
        imgname = filename+"_trimmed_images_stripped.txt"
        labname = filename+"_trimmed_labels_stripped.txt"
    else:
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
labels = to_categorical(labels, num_classes=classNo, dtype='float32')
print("\nOBTAINED DATA. (Scaling by a factor of ", scale, ")", sep='')

# Split data
rar = 0 # Random state seed
if testing_mode:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, test_size=0.5, stratify=labels, random_state=rar, shuffle=True) # 50/50 (for eventual 50/25/25)
else:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, stratify=labels, random_state=rar, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
if testing_mode:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.5, random_state=rar, shuffle=True) # Just split 50/50.
else:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, random_state=rar, test_size=0.4, shuffle=True) # 60/40 val/test

if not testing_mode or pure_mode:
    np.savez_compressed('testing_saved', a=x_test, b=y_test)

# To observe data distribution
def countClasses(categors, name):
    temp = np.argmax(categors, axis=1)
    print(name, "distribution:", Counter(temp))

print("Number of training images:", len(x_train))
countClasses(y_train, "Training")
print("Number of validation images:", len(x_val))
countClasses(y_val, "Validation")
print("Number of testing images:", len(x_test), "\n")
if testing_mode:
    print("Training labels:", y_train)
print("Label type:", y_train[0].dtype)

# Memory
if memory_mode:
    latest_gpu_memory = gpu_memory_usage(gpu_id)
    print(f"Post data aquisition (GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

# Data augmentation functions
aug_rate = 0
if aug:
    aug_rate = 1 # Activate augmentation
else:
    print("NO AUGMENTATION.")
def get_augmentation(patch_size):
    return Compose([
        Rotate((-3, 3), (-3, 3), (-3, 3), p=0.6),
        ElasticTransform((0, 0.05), interpolation=2, p=0.3),
        RandomGamma(gamma_limit=(0.6, 1), p=0)
    ], p=aug_rate)
aug = get_augmentation((w,h,d)) # For augmentations

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

# This needs to exist in order to allow for us to use an accuracy metric
def fix_shape(images, labels):
    images.set_shape([None, w, h, d, 1])
    labels.set_shape([images.shape[0], classNo])
    return images, labels

def fix_wrapper(file, labels):
    return tf.py_function(fix_shape, [file, labels], [np.float32, np.float32])

print("Setting up dataloaders...")
# Data loaders
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
if not pure_mode:
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_set = (
    train.shuffle(len(train))
    .map(load_image_wrapper)
    .batch(batch_size)
    .map(fix_shape)
    .prefetch(batch_size)
)

if not pure_mode:
    # Only rescale.
    validation_set = (
        val.shuffle(len(x_val))
        .map(load_val_wrapper)
        .batch(batch_size)
        .map(fix_shape)
        .prefetch(batch_size)
    )

# Model architecture go here
def gen_advanced_model(width=169, height=208, depth=179, classes=2):
    modelname = "Advanced-3D-CNN"
    print(modelname)
    inputs = keras.Input((width, height, depth, 1))
    
    x = layers.Conv3D(filters=8, kernel_size=5, padding='valid', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv3D(filters=16, kernel_size=5, padding='valid', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv3D(filters=32, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv3D(filters=64, kernel_size=5, padding='valid', kernel_regularizer =tf.keras.regularizers.l2( l=0.01), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    
    outputs = layers.Dense(units=classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name=modelname)
    
    return model

def gen_advanced_noreg(width=169, height=208, depth=179, classes=2):
    modelname = "Advanced-3D-CNN-NOREGUL"
    print(modelname)
    inputs = keras.Input((width, height, depth, 1))
    
    x = layers.Conv3D(filters=8, kernel_size=5, padding='valid', activation='relu')(inputs)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    
    x = layers.Conv3D(filters=16, kernel_size=5, padding='valid', activation='relu')(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    
    x = layers.Conv3D(filters=32, kernel_size=5, padding='valid', activation='relu')(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    
    x = layers.Conv3D(filters=64, kernel_size=5, padding='valid', activation='relu')(x)
    x = layers.MaxPool3D(pool_size=2, strides=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation='relu')(x)
    x = layers.Dense(units=64, activation='relu')(x)
    
    outputs = layers.Dense(units=classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name=modelname)
    
    return model

# Build model.

tic = perf_counter()

# Build model.
if reg:
    print("USING ADVANCED MODEL.")
    model = gen_advanced_model(width=w, height=h, depth=d, classes=classNo)
else:
    print("USING NO REGULARIZATION MODEL")
    model = gen_advanced_noreg(width=w, height=h, depth=d, classes=classNo)
model.summary()
optim = keras.optimizers.Adam(learning_rate=0.0001)
if batch_size > 1:
    metric = 'binary_accuracy'
else:
    metric = 'accuracy'
if metric == 'binary_accuracy':
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
else:
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
print("Metric being used:", metric)

# Checkpointing & Early Stopping
mon = 'val_' +metric
es = EarlyStopping(monitor=mon, patience=10, restore_best_weights=True)
checkpointname = "/scratch/mssric004/Checkpoints/testing-{epoch:02d}.ckpt"
localcheck = "/scratch/mssric004/TrueChecks/" + modelname +".ckpt"
be = ModelCheckpoint(localcheck, monitor=mon, mode='auto', verbose=2, save_weights_only=True, save_best_only=True)
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

# Run the model
print("---------------------------\nFITTING MODEL")
print("Params:", epochs, "epochs & batch size [", batch_size, "].")

if testing_mode:
    history = model.fit(train_set, validation_data=validation_set, epochs=epochs, callbacks=[be], shuffle=True)
else:
    history = model.fit(train_set, validation_data=validation_set, epochs=epochs, callbacks=[es, be, CustomCallback()], verbose=0, shuffle=True)

toc = perf_counter()

if not testing_mode:
    if not pure_mode:
        modelname = modelname +".h5"
        print("Saving model to", modelname)
        try:
            model.save("/scratch/mssric004/Saved Models/"+modelname)
        except Exception as e:
            print("Couldn't save model. Reason:", e)
print(history.history)

def make_unique(file_name, extension):
    if os.path.isfile(file_name):
        expand = 1
        while True:
            new_file_name = file_name.split(extension)[0] + str(expand) + extension
            if os.path.isfile(new_file_name):
                expand += 1
                continue
            else:
                file_name = new_file_name
                break
    else:
        print("")
    print("Saving to", file_name)
    return file_name

plotting = not testing_mode
if plotting:
    try:
        print("Importing matplotlib.")
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt
        plotname = "Plots/Single/model"
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
        #plt.savefig(plotname + "_val" + ".png")
        print("Saved plot, btw.")
    except Exception as e:
        print("Plotting didn't work out. Error:", e)

# Readings
try:
    print("\nAccuracy max:", round(max(history.history[metric])*100,2), "% (epoch", history.history[metric].index(max(history.history[metric]))+1, ")")
    print("Loss min:", round(min(history.history['loss']),2), "(epoch", history.history['loss'].index(min(history.history['loss']))+1, ")")
    print("Validation accuracy max:", round(max(history.history['val_'+metric])*100,2), "% (epoch", history.history['val_'+metric].index(max(history.history['val_'+metric]))+1, ")")
    print("Val loss min:", round(min(history.history['val_loss']),2), "(epoch", history.history['val_loss'].index(min(history.history['val_loss']))+1, ")")
except Exception as e:
    print("Cannot print out summary data. Reason:", e)
    
# Number of epochs trained for
epochcount = len(history.history['val_loss'])

# Final evaluation
print("\nEvaluating using test data...")

# First prepare the test data
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_x = tf.data.Dataset.from_tensor_slices((x_test))
print("Test data prepared.")
countClasses(y_test, "Test")

try:
    test_set = (
        test.map(load_val_wrapper)
        .batch(batch_size)
        .prefetch(batch_size)
    )
except:
    print("Couldn't assign test_set to a wrapper (for evaluation).")

try:
    test_set_x = (
        test_x.map(load_test_wrapper)
        .batch(batch_size)
        .prefetch(batch_size)
    )
except Exception as e:
    print("Couldn't assign test_set_x to a wrapper (for matrix). Reason:", e)


model.load_weights(localcheck)

try:
    scores = model.evaluate(test_set, verbose=0) # Should not need to specify batch size, because of set
    acc = scores[1]*100
    loss = scores[0]
    print("Evaluated scores - Acc:", acc, "Loss:", loss)
except:
    print("Error occured during evaluation. Isn't this weird?\nTest set labels are:", y_test)

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
print("\nGenerating classification report...")
try:
    y_pred = model.predict(test_set_x, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    y_hat = np.argmax(y_test, axis=1)
    rep = classification_report(y_hat, y_pred)
    conf = confusion_matrix(y_hat, y_pred)
    coh = cohen_kappa_score(y_hat, y_pred)
    print(rep)
    print("\nConfusion matrix:")
    print(conf)
    print("Cohen Kappa Score (0 = chance, 1 = perfect):", coh)
    limit = min(30, len(y_hat))
    print("\nActual test set (first ", (limit+1), "):", sep='')
    print(y_hat[:limit])
    print("Predictions are  as follows (first ", (limit+1), "):", sep='')
    print(y_pred[:limit])
except:
    print("Error occured in classification report (ie. predict). Test set labels are:\n", y_test)

# Clean up checkpoints
print("Cleaning up...")
import glob
found = glob.glob(localcheck+"*")
if len(found) == 0:
    print("The system cannot find", localcheck)
else:
    removecount = 0
    for checkfile in found:
        removecount += 1
        os.remove(checkfile)
    print("Successfully cleaned up", removecount, "checkpoint files.")

# Writing some test results down
if not testing_mode:
    loc = "/home/mssric004/ComplexityTests.txt"
    f = open(loc, 'a')
    f.write(logname + " | Mode: " + str(modo) + " | Acc: " + str(acc) + " | Loss: " + str(loss) + " | Desc: " + desc + " Time: " + datetime.datetime.now().strftime("%d/%m/%Y-%H:%M") + "\n")
    f.close()

toc_total = perf_counter()
total_seconds = (int) (toc_total-tic_total)
train_seconds = (int) (toc-tic)
total_time = datetime.timedelta(seconds=(total_seconds))
train_time = datetime.timedelta(seconds=train_seconds)
percen = (int)(train_seconds/total_seconds*100)

print("Done. Trained for", epochcount, "epochs.\nTotal time:", total_time, "- Training time:", train_time, ">", percen)