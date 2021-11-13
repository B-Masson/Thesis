# Class used to extract labelled data from the ADNI data-set - currently to be used as a test set
# Richard Masson
import numpy as np
import skimage, os
import os.path as op
#import scipy.misc
from skimage.io import imread
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from time import sleep

def extractADNI(w, h, d, root="C:\\Users\\richa\\Documents\\Uni\\Thesis\\ADNI"):
    scan_array = []
    label_array = []
    print("Loading in scan data...")
    for classes in os.listdir(root):
        for samples in os.listdir(op.join(root, classes)):
            image_root = op.join(root, classes, samples)
            #image_file = os.listdir(image_root)[0]
            #image_dir = op.join(image_root, image_file)
            image_data_raw = nib.load(image_root).get_fdata()
            image_data = organiseImage(image_data_raw, w, h, d)
            scan_array.append(image_data)
            label = "-1"
            if classes == "CN":
                label = 0
            elif classes == "MCI" or classes == "AD":
                label = 1
            label_array.append(label)
    return scan_array, label_array

'''
# And then copy this over into the application class later
import ADNI_Engine as ae
x_test_full, y_test_labels = ae.extractADNI(w, h, d, root=rootadni)
y_test_full = tf.keras.utils.to_categorical(y_test_raw)
subset_no = 400
sub_ind = np.random.choice(np.arange(len(x_test)), subset_no, replace=False)
x_test = x_test_full[sub_ind]
y_test = y_test_full[sub_ind]
# etc
'''

# Normalise pixel values to range from 0 to 1:
# Need to run AD_MinMax to get min and max values.
def normalize(image_data, min=0, max=4095):
    image_data = (image_data - min) / (max - min)
    image_data = image_data.astype("float32")
    return image_data

# Resize the data to some uniform amount so it actually fits into a training model
def resize(image_data, w=128, h=128, d=64):
    # Get current dimensions
    width = image_data.shape[0]
    height = image_data.shape[1]
    depth = image_data.shape[-1]
    # Compute all the factors that we need to scale the dimensions by
    width_factor = 1/(width/w)
    height_factor = 1/(height/h)
    depth_factor = 1/(depth/d)
    # Resize using factors
    image_data = ndimage.zoom(image_data, (width_factor, height_factor, depth_factor), order=1)
    return image_data

def organiseImage(data, w, h, d):
    data = normalize(data)
    data = resize(data, w, h, d)
    return data