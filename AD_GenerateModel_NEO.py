# Messing around with stuff without breaking the original version of the code.
# Richard Masson
# Info: Trying to fix the model since I'm convinced it's scuffed.
# Last use in 2021: October 29th
print("\nIMPLEMENTATION: NEO")
print("CURRENT TEST: Augmentation, slightly better model, etc.")
# TO DO: Implement augmentations. Elastic deformation and intensity control are top contenders.
import os
import subprocess as sp # Memory shit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import nibabel as nib
import NIFTI_Engine as ne
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
#print("TF Version:", tf.version.VERSION)
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
import random
import datetime
from collections import Counter
import sys
from volumentations import * # OI, WE NEED TO CITE VOLUMENTATIONS NOW
print("Imports working.")

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
modelname = "ADModel_NEO_v1.2"
logname = "Neo_V1.2"

# Model hyperparameters
if testing_mode:
    epochs = 1 #Small for testing purposes
    batches = 2
else:
    epochs = 15 # JUST FOR NOW
    batches = 1 # Going to need to fiddle with this over time (balance time save vs. running out of memory)

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 1.5
else:
    scale = 1 # For now
w = int(208/scale)
h = int(240/scale)
d = int(256/scale)

# Prepare parameters for fetching the data
modo = 1 # 1 for CN/MCI, 2 for CN/AD, 3 for CN/MCI/AD
if modo == 3:
    classNo = 3 # Expected value
else:
    classNo = 2 # Expected value
if testing_mode:
	filename = ("Directories/test_adni_" + str(modo))
else:
	filename = ("Directories/adni_" + str(modo))
if testing_mode:
    print("TEST MODE ENABLED.")
print("Filepath is", filename)
imgname = filename + "_images.txt"
labname = filename + "_labels.txt"

# Grab the data
path_file = open(imgname, "r")
path = path_file.read()
path = path.split("\n")
path_file.close()
label_file = open(labname, 'r')
labels = label_file.read()
labels = labels.split("\n")
labels = [ int(i) for i in labels]
label_file.close()

print("\nOBTAINED DATA. (Scaling by a factor of ", scale, ")", sep='')

# Split data
if testing_mode:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, test_size=0.5, stratify=labels, shuffle=True) # 50/50 (for eventual 50/25/25)
else:
    x_train, x_val, y_train, y_val = train_test_split(path, labels, stratify=labels, shuffle=True) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
if testing_mode:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5) # Don't stratify test data, and just split 50/50.
else:
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.2) # 70/30 val/test

if not testing_mode:
    np.savez_compressed('testing_sub', a=x_test, b=y_test)

print("Number of training images:", len(x_train))
print("Training distribution:", Counter(y_train))
#y_train = np.asarray(y_train)
if testing_mode:
    print("Training labels:", y_train)
print("Number of validation images:", len(x_val))
print("Validation distribution:", Counter(y_val))
print("Number of testing images:", len(x_test))
print("Testing distribution:", Counter(y_test), "\n")

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
        Rotate((-3, 3), (-3, 3), (-3, 3), p=0.5),
        #Flip(2, p=1)
        ElasticTransform((0, 0.15), interpolation=2, p=0.1),
        #GaussianNoise(var_limit=(0, 5), p=0.5),
        RandomGamma(gamma_limit=(0.2, 1), p=0.4)
    ], p=0.7)
aug = get_augmentation((w,h,d)) # For augmentations



def load_image(file, label):
    
    #print("Doing map stuff.")
    loc = file.numpy().decode('utf-8')
    #label = label.numpy().decode('utf-8')
    #label = int(label)
    nifti = np.asarray(nib.load(loc).get_fdata())
    nifti = ne.organiseADNI(nifti, w, h, d)
    '''
    znum = 0 # Random variable to assist in saving augmentation test images
    while os.path.exists("zaug_before_%s.jpg" % znum):
        znum += 1
    plt.imshow(nifti[:,:,(int)(d/2)], cmap='bone')
    plt.savefig("zaug_before_%s.jpg" % znum)
    '''
    # Augmentation
    data = {'image': nifti}
    aug_data = aug(**data)
    nifti = aug_data['image']
    '''
    plt.imshow(nifti[:,:,(int)(d/2)], cmap='bone')
    plt.savefig("zaug_after_%s.jpg" % znum)
    znum += 1
    '''
    nifti = tf.convert_to_tensor(nifti, np.float64)
    #label.set_shape([1]) # For the you-know-what
    return nifti, label

def load_test(file):
    loc = file.numpy().decode('utf-8')
    nifti = np.asarray(nib.load(loc).get_fdata())
    nifti = ne.organiseADNI(nifti, w, h, d)
    nifti = tf.convert_to_tensor(nifti, np.float64)
    return nifti

def load_image_wrapper(file, labels):
    return tf.py_function(load_image, [file, labels], [np.float64, tf.int32])

def load_image_testing(file):
    return tf.py_function(load_test, [file], [np.float64])

# This needs to exist in order to allow for us to use an accuracy metric without getting weird errors
def fix_shape(images, labels):
    images.set_shape([None, w, h, d, 1])
    labels.set_shape([1])
    return images, labels

print("Setting up dataloaders...")
# TO-DO: Augmentation stuff
batch_size = batches
# Data loaders
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val = tf.data.Dataset.from_tensor_slices((x_val, y_val))

train_set = (
    train.shuffle(len(train))
    .map(load_image_wrapper)
    .batch(batch_size)
    #.map(fix_shape)
    .prefetch(batch_size)
)
# Only rescale.
validation_set = (
    val.shuffle(len(x_val))
    .map(load_image_wrapper)
    .batch(batch_size)
    #.map(fix_shape)
    .prefetch(batch_size)
)

# Model architecture go here
# For consideration: https://www.frontiersin.org/articles/10.3389/fbioe.2020.534592/full#B22
def gen_model(width=208, height=240, depth=256, classes=3): # Make sure defaults are equal to image resizing defaults
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    # Contruction seems to be pretty much the same as if this was 2D. Kernal should default to 5,5,5
    x = layers.Conv3D(filters=32, kernel_size=5, activation="relu")(inputs) # Layer 1: Simple 32 node start
    x = layers.MaxPool3D(pool_size=2)(x) # Usually max pool after the conv layer
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # NOTE: RECOMMENTED LOL

    x = layers.Conv3D(filters=64, kernel_size=5, activation="relu")(x) # Double the filters
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x) # Double filters one more time
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # NOTE: Also commented this one for - we MINIMAL rn

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dropout(0.5)(x) # Start low, and work up if overfitting seems to be present

    outputs = layers.Dense(units=1, activation="sigmoid")(x) # Units = no of classes. Also softmax because we want that probability output

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")

    return model

# Build model.
model = gen_model(width=w, height=h, depth=d, classes=classNo)
model.summary()
optim = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-3) # LR chosen based on principle but double-check this later
#model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy']) # Temp binary for only two classes
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
# ^^^^ Temp solution for the ol' "as_list() is not defined on an unknown TensorShape issue"
# NOTE: LOOK AT THIS AGAIN WHEN DOING 3-WAY CLASS

# Memory
if memory_mode:
    latest_gpu_memory = gpu_memory_usage(gpu_id)
    print(f"Pre train (GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

# Checkpointing & Early Stopping
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=False) # Temp at 30 to circumvent issue with first epoch behaving weirdly
checkpointname = "neo_checkpoints.h5"
if testing_mode:
    checkpointname = "neo_checkpoints_testing.h5"
mc = ModelCheckpoint(checkpointname, monitor='val_loss', mode='min', verbose=2, save_best_only=False) #Maybe change to true so we can more easily access the "best" epoch
if testing_mode:
    log_dir = "/scratch/mssric004/test_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
else:
    if logname != "na":
        log_dir = "/scratch/mssric004/logs/fit/" + logname + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        log_dir = "/scratch/mssric004/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Run the model
print("Fitting model...")

if testing_mode:
    #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0)
    history = model.fit(train_set, validation_data=validation_set, epochs=epochs, verbose=0) # DON'T SPECIFY BATCH SIZE, CAUSE INPUT IS ALREADY A BATCHED DATASET
else:
    #history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batches, epochs=epochs, verbose=0, shuffle=True)
    history = model.fit(train_set, validation_data=validation_set, epochs=epochs, callbacks=[mc, tb, es], verbose=0, shuffle=True)
if testing_mode:
    modelname = "ADModel_NEO_Testing"
modelname = modelname +".h5"
model.save(modelname)
print(history.history)

# Memory
if memory_mode:
    latest_gpu_memory = gpu_memory_usage(gpu_id)
    print(f"Post train (GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

# Final evaluation
print("\nEvaluating using test data...")

# First prepare the test data
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_x = tf.data.Dataset.from_tensor_slices((x_test))
print("Test data prepared.")
test_set = (
    test.map(load_image_wrapper)
    .batch(batch_size)
    .prefetch(batch_size)
) # Later we may need to use a different wrapper function? Not sure.

test_set_x = (
    test_x.map(load_image_testing)
    .batch(batch_size)
    .prefetch(batch_size)
)

try:
    scores = model.evaluate(test_set, verbose=0) # Should not need to specify batch size, because of set
    acc = scores[1]*100
    loss = scores[0]
    print("Evaluated scores - Acc:", acc, "Loss:", loss)
except:
    print("Error occured during evaluation. Isn't this weird?\nTest set labels are:", y_test)

#if not testing_mode: # NEED TO REWORK THIS
from sklearn.metrics import classification_report

print("\nGenerating classification report...")
try:
    y_pred = model.predict(test_set_x, verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    #print("test:", y_test)
    print("pred:", y_pred)
    #print("\nand now we crash")
    #y_test = np.argmax(y_test, axis=1)
    rep = classification_report(y_test, y_pred)
    print(rep)
    limit = min(20, len(y_test))
    print("\nActual test set (first ", (limit+1), "):", sep='')
    print(y_test[:limit])
    print("Predictions are  as follows (first ", (limit+1), "):", sep='')
    print(y_pred[:limit])
except:
    print("Error occured in classification report (ie. predict).\nTest set labels are:", y_test)

# Memory
if memory_mode:
    latest_gpu_memory = gpu_memory_usage(gpu_id)
    print(f"Post evaluation (GPU) Memory used: {latest_gpu_memory - initial_memory_usage} MiB")

print("Done.")

