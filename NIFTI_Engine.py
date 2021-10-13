# Host of functions for use in data collection and pre-processing for the brain scans
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
print("Imported engine packages.")

# Function: Run through an entire data folder and extract all data of a certain scan type. Graps NIFTI data and returns numpy arrays for the x-array
# Also returns an an array alongside with the patient ID and the day of the MRI
# Orientation currently sets the choice orientation in which 2D reference images are generates (not: the returned data is still 3D)
def extractArrays(scantype, w, h, d, orientation=0, root="C:\\Users\\richa\\Documents\\Uni\\Thesis\\central.xnat.org"):
    scan_array = []
    meta_array = []
    sample_dirs = os.listdir(root)
    print("Loading in scan data...")
    for sample in tqdm(sample_dirs):
        sleep(0.25)
        if scantype == "all":
            for scan in os.listdir(op.join(root, sample)):
                if ("anat" in scan) or ("func" in scan): #Mostly to sift out BIDS folders/random unwanted stuff
                    image_root = op.join(root, sample, scan, 'NIFTI')
                    #print("Reading file at", image_root, ": ", os.listdir(image_root)[0])
                    image_file = os.listdir(image_root)[0]
                    image_dir = op.join(image_root, image_file)
                    image_data_raw = nib.load(image_dir).get_fdata()
                    image_data = organiseImage(image_data_raw, w, h, d)
                    scan_array.append(image_data)
                    meta_segments = sample.split("_")
                    meta_array.append({'ID': meta_segments[0], 'day': int(meta_segments[2][1:])})
                    try:
                        if orientation == 1:
                            plt.imshow(image_data[50,:,:], cmap='bone')
                            plt.show()
                        elif orientation == 2:
                            plt.imshow(image_data[:,50,:], cmap='bone')
                            plt.show()
                        elif orientation == 3:
                            plt.imshow(image_data[:,:,30], cmap='bone')
                            plt.show()
                    except TypeError as e:
                        print("Cannot display example slices:", e)
        elif scantype in os.listdir(op.join(root, sample)): # This needs to be cleaned up and incorporated into the above at some point, to reduce redundant code
            root_niftis = op.join(root, sample, scantype)
            check_niftis = os.listdir(root_niftis)
            if "NIFTI" in check_niftis:
                image_root = op.join(root_niftis, 'NIFTI')
                image_file = os.listdir(image_root)[0]
                image_dir = op.join(image_root, image_file)
                image_data_raw = nib.load(image_dir).get_fdata()
                image_data = organiseImage(image_data_raw, w, h, d)
                scan_array.append(image_data)
                meta_segments = sample.split("_")
                meta_array.append({'ID': meta_segments[0], 'day': int(meta_segments[2][1:])})
                try:
                    if orientation == 1:
                        plt.imshow(image_data[50,:,:], cmap='bone')
                        plt.show()
                    elif orientation == 2:
                        plt.imshow(image_data[:,50,:], cmap='bone')
                        plt.show()
                    elif orientation == 3:
                        plt.imshow(image_data[:,:,30], cmap='bone')
                        plt.show()
                except TypeError as e:
                    print("Cannot display example slices:", e)
            else: print("No NIFTI file found. This should not occur if the dataset is half-decent.")
        else:
            print("Warning:", sample, "does not possess data of type", scantype)
    #print("Returned array has size", len(scan_array))
    return scan_array, meta_array

# Remove unecessary ranges from the NIFTI data, then normalise to range from 0 to 1
def normalize(image_data, min=-1000, max=400):
    image_data[image_data < min] = min
    image_data[image_data > max] = max
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

#if __name__ == "__main__":
    #Why does this trigger even when its not main?
    #scans, meta = extractArrays('anat3', 0) #Orientation 0 to not display anything
    #print(meta)
