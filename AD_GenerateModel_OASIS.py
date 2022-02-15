# Combined form of the AD_Process and AD_Train classes, to be fed into the HPC cluster at max sample size
# Richard Masson
# Info: Temporary version of the HPC class where I run tests on the old OASIS vers.
# Last use in 2021: Novemeber 16th
print("IMPLEMENTATION: STANDARD")
print("CURRENT TEST: Standard run minus k-fold.")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#from nibabel import test
import LabelReader as lr
import NIFTI_Engine as ne
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
import random
import datetime
from collections import Counter
import pandas as pd
from sklearn.model_selection import KFold
import sys

# Are we in testing mode?
testing_mode = True
logname = "ShinModelV1.2"

# Class or regression, that is the question
classmode = True
classNo = 2 # Expected value

# Attempt to better allocate memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Define image size (lower image resolution in order to speed up for broad testing)
if testing_mode:
    scale = 1
else:
    scale = 1
# Default dimensions
w = 128/scale
h = 128/scale
d = 64/scale
# ADNI dimensions (need to verify this at some point)
#w = 208
#h = 240
#d = 256

# Fetch all our seperated data
#x_arr, scan_meta = ne.extractArrays('all', root="/home/rmasson/Documents/Data") # Linux workstation world
rootloc = "/scratch/mssric004/Data" # HPC world
adniloc = "/scratch/mssric004/ADNI_Data"
if testing_mode:
    rootloc = "/scratch/mssric004/Data_Tiny"
    adniloc = "/scratch/mssric004/ADNI_Test"
    print("TEST MODE ENABLED.")
x_arr, scan_meta = ne.extractArrays('all', w, h, d, root=rootloc)
clinic_sessions, cdr_meta = lr.loadCDR()
#x_arr, y_arr = ne.extractADNI(w, h, d, root=adniloc)

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

def genRegLabels(scan_meta):
    # Regression approach
    # If a scan lies between two different cdr values, then set the value based on the time frame
    y_arr = []
    for scan in scan_meta:
        scan_day = scan['day']
        scan_cdr_a = -1
        scan_cdr_b = -1
        cdr_day_a = -1
        cdr_day_b = -1
        min_a = 100000
        min_b = 100000
        try:
            for x in cdr_meta[scan['ID']]:
                if (x[0] > scan_day):
                    diff = x[0]-scan_day
                    if diff < min_a:
                        min_a = diff
                        scan_cdr_a = x[1]
                        cdr_day_a = x[0]
                else:
                    diff = scan_day-x[0]
                    if diff < min_b:
                        min_b = diff
                        scan_cdr_b = x[1]
                        cdr_day_b = x[0]
            if scan_cdr_a == scan_cdr_b or scan_cdr_b == -1:
                val = scan_cdr_a
            elif scan_cdr_a == -1:
                val = scan_cdr_b
            else:
                print("Scan", scan['ID'], "has cdr", scan_cdr_b, "at", cdr_day_b, " (", min_b, " days before ) and cdr", scan_cdr_a, "at", cdr_day_a, " (", min_a, "days after ).")
                total_span = min_a+min_b
                val = round(scan_cdr_a*(min_b/total_span + scan_cdr_b*(min_a/total_span)), 2)
                print("Produces a scaled cdr of", val)
            y_arr.append(val) 
        except KeyError as k:
            print(k, "| Seems like the entry for that patient doesn't exist.")
    return y_arr

#print(len(x_arr), "x array elements.")
#print(y_arr)
print("Label generation is currently commented out since we're working with ADNI.")

# Generate some cdr y labels for each scan
if (classmode):
    print("Model type: Classification")
    y_arr, classNo = genClassLabels(scan_meta)
else:
    print("Model type: Regression [Experimental]")
    y_arr = genRegLabels(scan_meta)


print("Data successfully loaded in.")
force_diversity = True # Quick code to force-introduce class 1 to tiny test sets
if testing_mode:
    if force_diversity:
        print("ACTIVATING CLASS DIVERSITY MODE")
        mid = int(len(y_arr)/2)
        new_arr = []
        for i in range (len(y_arr)):
            if i < mid:
                new_arr.append(0)
            else:
                new_arr.append(1)
        random.shuffle(new_arr)
        y_arr = new_arr
        print("Diversified set:", y_arr)

# Split data
if testing_mode:
    if force_diversity:
        x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, stratify=y_arr) # ONLY USING WHILE THE SET IS TOO SMALL FOR STRATIFICATION
        print("y train:", y_train)
        print("y val:", y_val)
        if len(y_val) >= 5:
            x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.2)
        else:
            print("Dataset too small to fully stratify - temporarily bypassing that...")
            x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.2)
    else:
        x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr) # ONLY USING WHILE THE SET IS TOO SMALL FOR STRATIFICATION
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.2) # ALSO TESTING BRANCH NO STRATIFY LINE
    #x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.2)
else:
    x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, stratify=y_arr) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.2) # 80/20 val/test, therefore 75/20/5 train/val/test.
    #x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, stratify=y_arr ,test_size=0.2)
'''
if testing_mode:
    np.savez_compressed('testing_sub', a=x_test, b=y_test)
else:
    np.savez_compressed('testing', a=x_test, b=y_test)
'''
'''
# Ascertain what the class breakdown is
print("Class breakdowns:")
print("Training:", collections.Counter(y_train))
print("Validation:", collections.Counter(y_val))
print("Testing:", collections.Counter(y_test))
print("Data has been preprocessed. Moving on to model...")
'''
# Model architecture go here
def gen_model(width=128, height=128, depth=64, classes=3): # Make sure defaults are equal to image resizing defaults
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    # Contruction seems to be pretty much the same as if this was 2D. Kernal should default to 3,3,3
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs) # Layer 1: Usual 64 filter start
    x = layers.MaxPool3D(pool_size=2)(x) # Usually max pool after the conv layer
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # NOTE: UNCOMMENTED THIS LINE AS WE CAN HOPEFULLY UP THE COMPLEXITY NOW

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x) # Double the filters
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x) # Double filters one more time
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=518, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dropout(0.5)(x) # 50% seems like a good tried and true value

    outputs = layers.Dense(units=classes, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output

    '''
    # Baby mode version (TO-DO)
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    # Contruction seems to be pretty much the same as if this was 2D. Kernal should default to 3,3,3
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu")(inputs) # Layer 1: Usual 64 filter start
    x = layers.MaxPool3D(pool_size=2)(x) # Usually max pool after the conv layer
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dropout(0.5)(x) # 50% seems like a good tried and true value

    outputs = layers.Dense(units=classes, activation="softmax")(x) # Units = no of classes. Also softmax because we want that probability output
    '''

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")

    return model

def gen_model_reg(width=128, height=128, depth=64): # Make sure defaults are equal to image resizing defaults
    # Initial build version - no explicit Sequential definition
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    # Contruction seems to be pretty much the same as if this was 2D. Kernal should default to 3,3,3
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs) # Layer 1: Usual 64 filter start
    x = layers.MaxPool3D(pool_size=2)(x) # Usually max pool after the conv layer
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # NOTE: UNCOMMENTED THIS LINE AS WE CAN HOPEFULLY UP THE COMPLEXITY NOW

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x) # Double the filters
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x) # Double filters one more time
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dropout(0.5)(x) # 50% seems like a good tried and true value

    outputs = layers.Dense(units=1)(x) # Regression needs no activation function?

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN_Reg")

    return model










# buffer












# Model hyperparameters
if testing_mode:
    epochs = 2 #Small for testing purposes
    batches = 3
else:
    epochs = 30
    batches = 8 # Going to need to fiddle with this over time (balance time save vs. running out of memory)
'''
# Data augmentation functions
@tf.function
def rotate(image):
    def scipy_rotate(image): # Rotate by random angular amount
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate image
        image = ndimage.rotate(image, angle, reshape=False)
        image[image < 0] = 0
        image[image > 1] = 1
        return image

    augmented_image = tf.numpy_function(scipy_rotate, [image], tf.float32)
    return augmented_image
'''
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
    image = rotate(image)
    # Now shift it
    #image = shift(image)
    image = tf.expand_dims(image, axis=3)
    return image, label

def validation_preprocessing(image, label): # Can be used for val or test data (just ensures the dimensions are ok for the model)
    """Process validation data by only adding a channel."""
    image = tf.expand_dims(image, axis=3)
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
print("Set up.")

# Build model.
if classmode:
    model = gen_model(width=128, height=128, depth=64, classes=classNo)
else:
    model = gen_model_reg(width=128, height=128, depth=64)
model.summary()
optim = keras.optimizers.Adam(learning_rate=0.001) # LR chosen based on principle but double-check this later
# Note: These things will have to change if this is changed into a regression model
#model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy']) # Categorical loss since there are move than 2 classes.
if classmode:
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy']) # Temp binary for only two classes
else:
    model.compile(optimizer=optim, loss= "mean_squared_error", metrics=["mean_squared_error"]) # Regression compiler

# Ensure that the test data is prepped and ready. (This might be for the ADNI stuff?)
#Y_test = np.argmax(y_test, axis=1)
#X_test = np.expand_dims(x_test, axis=-1)

# Class weighting
# Data distribution is {0: 2625, 1: 569, 2: 194}
#class_weight = {0: 1., 1: 3., 2: 8.}
class_weight = {0: 1., 1: 2.}

# Checkpointing & Early Stopping
es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True) # Temp at 30 to circumvent issue with first epoch behaving weirdly
checkpointname = "weight_history.h5"
if testing_mode:
    checkpointname = "weight_history_testing.h5"
mc = ModelCheckpoint(checkpointname, monitor='val_loss', mode='min', verbose=2, save_best_only=False) #Maybe change to true so we can more easily access the "best" epoch
if testing_mode:
    log_dir = "/scratch/mssric004/test_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
else:
    if logname != "na":
        log_dir = "/scratch/mssric004/logs/fit/" + logname + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        log_dir = "/scratch/mssric004/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# K-Fold setup
n_folds = 10
if testing_mode:
    n_folds = 2
acc_per_fold = []
loss_per_fold = []
kfold = KFold(n_splits=n_folds, shuffle=True)

# Run the model
print("Fitting model...")

print("Shape of input set:", np.shape(x_train))
sys.exit()
if testing_mode:
    history = model.fit(train_set, validation_data=validation_set, batch_size=batches, epochs=epochs, verbose=0)
else:
    history = model.fit(train_set, validation_data=validation_set, batch_size=batches, epochs=epochs, verbose=0, shuffle=True, callbacks=[mc, tb, es], class_weight=class_weight)
modelname = "ADModel_True"
if testing_mode:
    modelname = "ADModel_Testing"
model.save(modelname)
#print(history.history)
actual_epochs = len(history.history['val_loss'])
print("Complete. Ran for ", actual_epochs, "/", epochs, " epochs.\nParameters saved to", modelname)
for i in range(actual_epochs):
    print("Epoch", i+1, ": Loss [", history.history['loss'][i], "] Val Loss [", history.history['val_loss'][i], "]")
best_epoch = np.argmin(history.history['val_loss']) + 1
print("Epoch with lowest validation loss: Epoch", best_epoch, "[", history.history['loss'][best_epoch-1], "]")



'''
# Generate a classification matrix
def classification_report_csv(report, filename):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index = False)
'''
if classmode and not testing_mode:
    from sklearn.metrics import classification_report

    print("Generating classification report...\n")
    y_pred = model.predict(X_test, batch_size=2)
    y_pred = np.argmax(y_pred, axis=1)
    print("Actual test set:")
    print(Y_test)
    print("Predictions are  as follows:")
    print(y_pred)
    rep = classification_report(Y_test, y_pred)
    print(rep)

    # Final evaluation
    score = model.evaluate(X_test, Y_test, verbose=0, batch_size=2)
    print("Evaluated scored:", score)

print("Done.")
