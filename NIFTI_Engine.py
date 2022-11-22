# Host of functions for use in data collection and pre-processing for the brain scans
# Richard Masson
print("Importing NIFTI engine stuff...")
import numpy as np
import skimage, os
import os.path as op
from skimage.io import imread
import nibabel as nib
from scipy import ndimage
from time import sleep
import tensorflow as tf
print("Engine loaded.")

# Function: Run through an entire data folder and extract all data of a certain scan type. Graps NIFTI data and returns numpy arrays for the x-array
def extractADNI(w, h, d, orientation=0, root="C:\\Users\\richa\\Documents\\Uni\\Thesis\\central.xnat.org", mode=3, strip=False):
    scan_array = []
    label_array = []
    class_array = []
    class_dirs = os.listdir(root)
    exclusions = "na"
    if mode == 1:
        exclusions = "AD"
        print("Mode: Classify CN vs. MCI")
    elif mode == 2:
        exclusions = "MCI"
        print("Mode: Classify CN vs. AD")
    else:
        print("Mode: Classify from all 3 categories.")
    print("\nLoading in scan data...\n")
    for classes in class_dirs:
        if classes != exclusions and classes != "Zips":
            for scan in os.listdir(op.join(root, classes)):
                for type in os.listdir(op.join(root, classes, scan)):
                    for date in os.listdir(op.join(root, classes, scan, type)):
                        for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
                            image_root = op.join(root, classes, scan, type, date, image_folder)
                            if strip:
                                image_file = os.listdir(image_root)[1] # Strip
                            else:
                                image_file = os.listdir(image_root)[0] # No strip
                            image_dir = op.join(image_root, image_file)
                            image_data_raw = nib.load(image_dir).get_fdata()
                            image_data = nib.load(image_dir).get_data().astype('float32')
                            image_data = organiseADNI(image_data_raw, w, h, d)
                            scan_array.append(image_data)
                            if classes == "CN":
                                label_array.append(0)
                            elif classes == "MCI":
                                label_array.append(1)
                            elif classes == "AD":
                                label_array.append(2)
                            else:
                                print("One of the folders does not match any of the expected forms.")
                            class_array.append(classes)
    return scan_array, label_array

# Extracts a curated list used for earlier tests
def extractPure(root):
    scan_array = []
    exclusions = 0
    class_dirs = os.listdir(root)
    print("\nLoading in scan data...\n")
    for classes in class_dirs:
        if classes != "Zips":
            for scan in os.listdir(op.join(root, classes)):
                for type in os.listdir(op.join(root, classes, scan)):
                    if type == "AAHead_Scout_MPR_sag" or type == "AAHead_Scout" or type == "3D_MPRAGE":
                        exclusions += 1
                    else:
                        for date in os.listdir(op.join(root, classes, scan, type)):
                            for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
                                image_root = op.join(root, classes, scan, type, date, image_folder)
                                image_options = os.listdir(image_root)
                                image_file = '[none]'
                                for file in image_options:
                                    if "NORMED" not in file:
                                        if "STRIPPED" not in file:
                                            image_file = file
                                image_dir = op.join(image_root, image_file)
                                image_data_raw = nib.load(image_dir).get_data().astype('float32')
                                image_data = cropADNI(image_data_raw)
                                scan_array.append(image_data)
    print("Discarded", exclusions, "excluded types.")
    return scan_array

# Extracts directories themselves
def extractDirs(root, mode=3):
    dir_array = []
    label_array = []
    class_dirs = os.listdir(root)
    exclusions = "na"
    if mode == 1:
        exclusions = "AD"
        print("Mode: Classify CN vs. MCI")
    elif mode == 2:
        exclusions = "MCI"
        print("Mode: Classify CN vs. AD")
    else:
        print("Mode: Classify from all 3 categories.")
    print("\nLoading in scan data...\n")
    for classes in class_dirs:
        if classes != exclusions and classes != "Zips":
            for scan in os.listdir(op.join(root, classes)):
                for type in os.listdir(op.join(root, classes, scan)):
                    for date in os.listdir(op.join(root, classes, scan, type)):
                        for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
                            image_root = op.join(root, classes, scan, type, date, image_folder)
                            image_file = os.listdir(image_root)[0]
                            image_dir = op.join(image_root, image_file)
                            dir_array.append(image_dir)
                            if classes == "CN":
                                label_array.append(0)
                            elif classes == "MCI":
                                label_array.append(1)
                            elif classes == "AD":
                                label_array.append(2)
                            else:
                                print("One of the folders does not match any of the expected forms.")
                            #class_array.append(classes)
    dir_array = np.asarray(dir_array)
    label_array = np.asarray(label_array)
    return dir_array, label_array

# Extract a single raw image, for testing other functions
def extractSingle(w, h, d, root, type):
    types = os.listdir(root)
    dates = os.listdir(op.join(root, types[0]))
    folders = os.listdir(op.join(root, types[0], dates[0]))
    images = os.listdir(op.join(root, types[0], dates[0], folders[0]))
    true_root = op.join(root, types[0], dates[0], folders[0], images[0])
    scan_array = []
    label_array = []
    image_data_raw = nib.load(true_root).get_fdata()
    image_data = organiseADNI(image_data_raw, w, h, d)
    scan_array.append(image_data)
    if type == "CN":
        label_array.append(0)
    elif type == "MCI":
        label_array.append(1)
    elif type == "AD":
        label_array.append(2)
    else:
        print("Incorrect class inputted.")
    return scan_array, label_array

# Keeps the data as a Nibabel object
def extractSingleNib(root, type):
    classdirs = os.listdir(root)
    for classes in classdirs:
        if classes == type:
            print("Extracting from", classes, "folder.")
            patients = os.listdir(op.join(root, classes))
            print("Patients:", patients)
            folders = os.listdir(op.join(root, classes, patients[0]))
            print("Folders:", folders)
            dates = os.listdir(op.join(root, classes, patients[0], folders[0]))
            print("Dates:", dates)
            images = os.listdir(op.join(root, classes, patients[0], folders[0], dates[0]))
            print("Images:", images)
            nifti = os.listdir(op.join(root, classes, patients[0], folders[0], dates[0], images[0]))
            print("Nifti:", nifti)
            true_root = op.join(root, classes, patients[0], folders[0], dates[0], images[0], nifti[0])
            #print("Target:", true_root_more_true)
            image_data_raw = nib.load(true_root)
            return image_data_raw

# Values found through earlier experimentation
#Mean: 231.33
#Standard deviation: 457.82
#98th percentile: 1691.0
#2nd percentile: 0.0

# Normalize pixel values to range from 0 to 1 based on 2nd-98th percentile values:
def normalize(image_data, min=0, max=1691):
    image_data = image_data.astype("float32") # Changed from 32
    image_data[image_data < min] = min
    image_data[image_data > max] = max
    image_data = (image_data - min) / (max - min)
    return image_data

# Same, but do not trim the outliers
def normalize_notrim(image_data, min=0, max=1691):
    image_data = image_data.astype("float32")
    image_data = (image_data - min) / (max - min)
    return image_data

# Normalize using local min and max values per image
def normalize_per(image_data):
    max = image_data.max()
    min = 0
    image_data = image_data.astype("float32")
    image_data = (image_data - min) / (max - min)
    return image_data

# Standardize images using mean and std deviation values
def normalize_std(image_data, mean=231, std=458):
    image_data = (image_data - mean) / std
    return image_data

# Resize the data to some uniform amount so it actually fits into a training model
def resizeADNI(image_data, w = 169, h = 208, d = 179, stripped=False):
    # Crop it first (currently just always defaults to the usual recipe)
    image_data = cropADNI(image_data, stripped=stripped)
    # Get current dimensions
    width = image_data.shape[0]
    height = image_data.shape[1]
    depth = image_data.shape[2]
    #print("Assessing shape: [", w, h, d, "]")
    if (width, height, depth) != (w, h, d):
        # Compute all the factors that we need to scale the dimensions by
        width_factor = 1/(width/w)
        height_factor = 1/(height/h)
        depth_factor = 1/(depth/d)
        length_factor = 1
        # Resize using factors
        if len(image_data.shape) == 3:
            image_data = ndimage.zoom(image_data, (width_factor, height_factor, depth_factor), order=1)
            image_data = np.expand_dims(image_data, axis=-1)
        else:
            image_data = ndimage.zoom(image_data, (width_factor, height_factor, depth_factor, length_factor), order=1)
        #image_data = ndimage.zoom(image_data, (width_factor, height_factor, depth_factor), order=1)
    #print("Data shape coming out is:", image_data.shape)
    return image_data

def resizeSlice(image_data, w = 169, h = 208, stripped=False):
    image_data = cropSlice(image_data, stripped=stripped)
    width = image_data.shape[0]
    height = image_data.shape[1]
    if (width, height) != (w, h):
        width_factor = 1/(width/w)
        height_factor = 1/(height/h)
        length_factor = 1
        # Resize using factors
        if len(image_data.shape) == 2:
            image_data = ndimage.zoom(image_data, (width_factor, height_factor), order=1)
            image_data = np.expand_dims(image_data, axis=-1)
        else:
            image_data = ndimage.zoom(image_data, (width_factor, height_factor, length_factor), order=1)
    return image_data

# Crop an image
def cropADNI(data, cropw=169, croph=208, cropd=179, stripped=False):
    if stripped:
        w,h,d = data.shape
        startw = w//2-(cropw//2)
        starth = h//2-(croph//2)
        startd = d//2-(cropd//2)        
        data = data[startw:startw+cropw,starth:starth+croph,startd:startd+cropd]
        data = np.expand_dims(data, axis=-1)
    else:
        w,h,d,_ = data.shape
        startw = w//2-(cropw//2)
        starth = h//2-(croph//2)
        startd = d//2-(cropd//2)        
        data = data[startw:startw+cropw,starth:starth+croph,startd:startd+cropd,:]
    if data.shape != (cropw, croph, cropd, 1):
        print("Anomalous image detected. Shape:", data.shape)
    return data

# Crop a 2D slice of an image
def cropSlice(data, cropw=169, croph=208, stripped=False):
    if stripped:
        w,h = data.shape
        startw = w//2-(cropw//2)
        starth = h//2-(croph//2)     
        data = data[startw:startw+cropw,starth:starth+croph]
        data = np.expand_dims(data, axis=-1)
    else:
        w,h,_ = data.shape
        startw = w//2-(cropw//2)
        starth = h//2-(croph//2)     
        data = data[startw:startw+cropw,starth:starth+croph,:]
    return data


def organiseADNI(data, w, h, d, strip=False):
    data = normalize_per(data) # Current best performing: per
    data = resizeADNI(data, w, h, d, stripped=strip)
    return data

def organiseOptimise(data, w, h, d, mode):
    if mode == 0:
        data = normalize(data)
    elif mode == 1:
        data = normalize_notrim(data)
    elif mode == 2:
        data = normalize_per(data)
    elif mode == 3:
        data = normalize_std(data)
    elif mode == 4:
        print("NO NORM")
    data = resizeADNI(data, w, h, d, stripped=False)
    return data

def organiseSlice(data, w, h, strip=False):
    data = normalize(data)
    data = resizeSlice(data, w, h, stripped=strip)
    return data
