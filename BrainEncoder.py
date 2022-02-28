# Sparse encoder generator. Saves the model so that it can be used in the AD classes.
# Richard Masson
# Created on 17th Feb 2022.
# https://ida.loni.usc.edu/pages/access/search/advancedDownload.jsp?loginKey=1857340483452350622&downloadKey=null&totalImageSelectionSize=null&collectionName=CN_3D_SAGI&userEmail=mssric004%40myuct.ac.za&collectionId=246318&format=AS_ARCHIVED&project=ADNI&ts=1645710603346
print("GENERATING SPARSE ENCODER FOR THE BRAIN SCANS")
from cgi import test
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
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
#from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
import random
from random import randrange
import datetime
from collections import Counter
import pandas as pd
from sklearn.model_selection import KFold
import sys
import matplotlib.pyplot as plt

testing_mode = True

def buildEncoder(w, h, d, compression=2):
    #bottleneck = int(200000/compression)
    # hard coded for now
    bottleneck = 1500
    inputs = keras.Input((w, h, d, 1))
    
    enc = layers.Conv3D(filters=64, kernel_size=5, strides=2, activation="relu")(inputs)
    #enc = layers.MaxPool3D(pool_size=2)(enc)
    enc = layers.Conv3D(filters=128, kernel_size=5, strides=2, activation="relu")(enc)
    #enc = layers.MaxPool3D(pool_size=2)(enc)
    enc = layers.Conv3D(filters=128, kernel_size=5, strides=2, activation="relu")(enc)
    enc = layers.Flatten()(enc)
    enc = layers.Dense(units=bottleneck, activation="sigmoid", activity_regularizer=regularizers.l1(10e-5))(enc)
    model_enc = keras.Model(inputs, enc, name="Original Encoder")
    
    inputs_dec = keras.Input(shape=(None, bottleneck))
    #model_dec = "temp" '''
    dec = layers.Dense(units=(1536))(inputs_dec)
    dec = layers.Reshape((44, 52, 56, 128))(dec)
    #dec = layers.UpSampling3D(size=2)(dec)
    dec = layers.Conv3DTranspose(filters=128, kernel_size=5, strides=2, activation="relu")(dec)
    #dec = layers.UpSampling3D(size=2)(dec)
    dec = layers.Conv3DTranspose(filters=64, kernel_size=5, strides=2, activation="relu")(dec)
    dec = layers.Conv3DTranspose(filters=1, kernel_size=5, strides=2, activation="relu")(dec)
    model_dec = keras.Model(inputs_dec, dec, name="Simple Decoder")
    #'''
    return model_enc, model_dec

def buildEncoderTest(width, height, depth):
    enc_dim = 200
    
    inputs = keras.Input((width, height, depth, 1))
    
    enc = layers.Conv3D(filters=64, kernel_size=5, strides=2, activation="relu", padding="same")(inputs)
    enc = layers.Conv3D(filters=128, kernel_size=5, strides=2, activation="relu", padding="same")(enc)
    enc = layers.Conv3D(filters=256, kernel_size=5, strides=2, activation="relu", padding="same")(enc)
    enc = layers.Conv3D(filters=256, kernel_size=5, strides=2, activation="relu", padding="same")(enc)
    enc = layers.Flatten()(enc)
    enc = layers.Dense(units=enc_dim, activation="sigmoid", activity_regularizer=regularizers.l1(10e-5))(enc)
    model_enc = keras.Model(inputs, enc, name="Tested Encoder")
    
    inputs_dec = keras.Input(shape=(None, enc_dim))
    dec = layers.Dense(units=(40960))(inputs_dec)
    dec = layers.Reshape((256, 4, 4, 10))(dec)
    dec = layers.Conv3D(filters=256, kernel_size=3, strides=2, activation="relu")(dec)
    dec = layers.Conv3D(filters=128, kernel_size=3, strides=2, activation="relu")(dec)
    dec = layers.Conv3D(filters=64, kernel_size=3, strides=2, activation="relu")(dec)
    dec = layers.Conv3D(filters=1, kernel_size=3, strides=2, activation="sigmoid")(dec)
    model_dec = keras.Model(inputs_dec, dec, name="Tested Decoder")
    
    return model_enc, model_dec

def buildConvEncoder(w, h, d, compression=2):
    #bottleneck = int(200000/compression)
    # hard coded for now
    bottleneck = 1500
    inputs = keras.Input((w, h, d, 1))
    
    enc = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same", name="Enc_Conv_1")(inputs)
    enc = layers.MaxPool3D(pool_size=2)(enc)
    enc = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", name="Enc_Conv_2")(enc)
    enc = layers.MaxPool3D(pool_size=2)(enc)
    enc = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", name="Enc_Conv_3")(enc)
    #enc = layers.MaxPool3D(pool_size=2)(enc)
    #enc = layers.Flatten()(enc)
    #enc = layers.Dense(units=bottleneck, activation="sigmoid", activity_regularizer=regularizers.l1(10e-5), name="Enc_Flattened")(enc)
    model_enc = keras.Model(inputs, enc, name="Conv_Encoder")
    
    #dec = layers.Dense(units=(1536), name="Dec_Flattened")(enc)
    #dec = layers.Reshape((13, 15, 16, 8))(dec)
    dec = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", name="Dec_Conv_1")(enc)
    #dec = layers.UpSampling3D(size=2)(enc)
    dec = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", name="Dec_Conv_2")(dec)
    dec = layers.UpSampling3D(size=2)(dec)
    dec = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same", name="Dec_Conv_3")(dec)
    dec = layers.UpSampling3D(size=2)(dec)
    dec = layers.Conv3D(filters=1, kernel_size=3, activation="sigmoid", padding="same", name="Dec_Conv_Final")(dec)
    auto = keras.Model(inputs, dec, name="Conv_Autoencoder")

    return inputs, model_enc, auto

def buildSimpleEncoder(w, h, d):
    inputs = keras.Input(shape=(w, h, d))
    enc = layers.Flatten()(inputs)
    enc = layers.Dense(units=200, activation="sigmoid")(enc)
    model_enc = keras.Model(inputs, enc, name="Simple Encoder")
    
    inputs_dec = keras.Input(shape=(None, 200))
    dec = layers.Dense(units=(w*h*d))(inputs_dec)
    dec = layers.Reshape((w, h, d))(dec)
    model_dec = keras.Model(inputs_dec, dec, name="Simple Decoder")
    return model_enc, model_dec

print("Powering up the Brain Encoder...")

if testing_mode:
    scale = 4 # while testing, scale down the image size by a factor of X
else:
    scale = 1 # while training, do we use the full size or not?

# ADNI dimensions (need to verify this at some point)
w = int(208/scale)
h = int(240/scale)
d = int(256/scale)
wt = 160 # Force it to test that one page
ht = 64
dt = 64

# Fetch all our seperated data
adniloc = "/scratch/mssric004/ADNI_Data"
if testing_mode:
    adniloc = "/scratch/mssric004/ADNI_Test"
    print("TEST MODE ENABLED.")
x_arr, y_arr = ne.extractADNI(w, h, d, root=adniloc)
print("Data successfully loaded in.")
print("Raw data shape:", np.array(x_arr).shape)

force_diversity = False
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
        x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, stratify=y_arr)
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
else:
    x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, stratify=y_arr) # Defaulting to 75 train, 25 val/test. Also shuffle=true and stratifytrue.
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, stratify=y_val, test_size=0.2) # 80/20 val/test, therefore 75/20/5 train/val/test.
print("Data is sorted and ready.")

# Generate the model
''' # Using a function (not 100% sure if this works for updating the encoder section)
print("Attempting to generate encoder and decoder...")
#enc, dec = buildEncoderTest(wt, ht, dt)
enc, auto = buildConvEncoder(w, h, d)
#enc, dec = buildSimpleEncoder(w, h, d)
#enc, dec = buildEncoder(w, h, d)
#enc.summary()
#dec.summary()
'''
# Just put it all here for the world to see
#bottleneck = int(200000/compression)
bottleneck = 1500
inputs = keras.Input((w, h, d, 1))

enc = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same", name="Enc_Conv_1")(inputs)
enc = layers.MaxPool3D(pool_size=2)(enc)
enc = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", name="Enc_Conv_2")(enc)
enc = layers.MaxPool3D(pool_size=2)(enc)
enc = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", name="Enc_Conv_3")(enc)
#enc = layers.MaxPool3D(pool_size=2)(enc)
#enc = layers.Flatten()(enc)
#enc = layers.Dense(units=bottleneck, activation="sigmoid", activity_regularizer=regularizers.l1(10e-5), name="Enc_Flattened")(enc)

#dec = layers.Dense(units=(1536), name="Dec_Flattened")(enc)
#dec = layers.Reshape((13, 15, 16, 8))(dec)
dec = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", name="Dec_Conv_1")(enc)
#dec = layers.UpSampling3D(size=2)(enc)
dec = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", name="Dec_Conv_2")(dec)
dec = layers.UpSampling3D(size=2)(dec)
dec = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same", name="Dec_Conv_3")(dec)
dec = layers.UpSampling3D(size=2)(dec)
dec = layers.Conv3D(filters=1, kernel_size=3, activation="sigmoid", padding="same", name="Dec_Conv_Final")(dec)
auto = keras.Model(inputs, dec, name="Conv_Autoencoder")
auto.summary()
#plot_model(auto, "autoencoder.png", show_shapes=True)
print("Done.")
auto.compile(optimizer='adam', loss='mse')
'''

# Test image outputs
plt.imshow(x_test[0][:,:,30], cmap='bone')
plt.savefig("ADNI_slice.png")
'''
# Hyper-params
epochs = 2
if not testing_mode:
    epochs = epochs*10
if testing_mode:
    batches = 3
else:
    batches = 8

# Fit the data (autoencoder)
print("Attempting to fit data to autoencoder...")
print("Training set shape:", np.array(x_train).shape)
x_train = np.array(x_train)
x_val = np.array(x_val)
x_test = np.array(x_test)
hist = auto.fit(x_train, x_train, epochs=epochs, batch_size=batches, shuffle=True, validation_data=(x_val, x_val))
print("Model trained!")
modelname = "Autoencoder"
if testing_mode:
    modelname = "Autoencoder_Test"
auto.save(modelname)

# Test it out
print("Attempting to test the data...")
index = randrange(len(x_test))
predictions = auto.predict(x_test)
test_image = x_test[index]
test_pred = predictions[index]
plt.imshow(test_image[:,:,30], cmap='bone')
plt.savefig("test_original.png")
plt.imshow(test_pred[:,:,30], cmap='bone')
plt.savefig("test_prediction.png")

# Use trained autoencoder to produce encoder model
print("Saving an encoder based on the previous results...")
enc = keras.Model(inputs, enc, name="Conv_Encoder")
enc.summary()
encname = "Encoder"
if testing_mode:
    encname = "Encoder_Test"
enc.save(encname)
print("Done.")