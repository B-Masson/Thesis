# Host of functions for use in data collection and pre-processing for the brain scans
# Richard Masson
import numpy as np
import skimage, os
import os.path as op
#import scipy.misc
from skimage.io import imread
import nibabel as nib
import matplotlib.pyplot as plt
print("Imported.")

# Function: Run through an entire data folder and extract all data of a certain scan type. Graps NIFTI data and returns numpy arrays for the x-array
# Also returns an an array alongside with the patient ID and the day of the MRI
# Orientation currently sets the choice orientation in which 2D reference images are generates (not: the returned data is still 3D)
def extractArrays(scantype, orientation=0, root="C:\\Users\\richa\\Documents\\Uni\\Thesis\\central.xnat.org"):
    scan_array = []
    meta_array = []
    sample_dirs = os.listdir(root)
    for sample in sample_dirs:
        if scantype in os.listdir(op.join(root, sample)):
            root_niftis = op.join(root, sample, scantype)
            check_niftis = os.listdir(root_niftis)
            if "NIFTI" in check_niftis:
                image_root = op.join(root_niftis, 'NIFTI')
                image_file = os.listdir(image_root)[0]
                image_dir = op.join(image_root, image_file)
                image_data=nib.load(image_dir).get_fdata()
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

#scans, meta = extractArrays('anat3', 0) #Orientation 0 to not display anything
#print(meta)