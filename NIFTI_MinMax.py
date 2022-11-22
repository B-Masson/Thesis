# Get min max values for NIFTI_Engine
# Richard Masson
print("Retrieving voxel min-max values...")

import numpy as np
import skimage, os
import os.path as op
from skimage.io import imread
import nibabel as nib
from tqdm import tqdm
from time import sleep

root = "/scratch/mssric004/ADNI_Data_NIFTI"
print("ROOT:", root)
testval = 2000
count = 0
exceptions = {}
total = 0
min_intensity = 5000
max_intensity = -1
maxsum = 0
outlier = "na"
class_dirs = os.listdir(root)

def append_vals(dict, key, value):
    if key not in dict:
        dict[key] = list()
    dict[key].append(value)
    return dict

print("Loading in scan data...")
for classes in class_dirs:
    if classes != "Zips":
        print("At folder:", classes)
        for scan in os.listdir(op.join(root, classes)):
            for type in os.listdir(op.join(root, classes, scan)):
                for date in os.listdir(op.join(root, classes, scan, type)):
                    for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
                        image_root = op.join(root, classes, scan, type, date, image_folder)
                        image_file = os.listdir(image_root)[0]
                        image_dir = op.join(image_root, image_file)
                        image_data_raw = nib.load(image_dir).get_fdata(dtype='float32')
                        min = image_data_raw.min()
                        max = image_data_raw.max()
                        maxsum += max
                        if min_intensity > min:
                            min_intensity = min
                        if max_intensity < max:
                            max_intensity = max
                            outlier = image_dir
                        if max > testval:
                            count += 1
                            exceptions = append_vals(exceptions, image_dir, max)
                        total += 1
                        '''
                        for x in image_data_raw:
                            for y in x:
                                for z in y:
                                    voxels.append(z)
                        '''
print("MIN VALUE:", min_intensity)
print("MAX VALUE:", max_intensity, "- found at", outlier)
print(count, "out of", total, "cases exceed", testval)
print("Average max:", round(maxsum/total, 2))