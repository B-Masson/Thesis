# Get min max values for NIFTI_Engine
# Richard Masson
import numpy as np
import skimage, os
import os.path as op
from skimage.io import imread
import nibabel as nib
from tqdm import tqdm
from time import sleep

root="/scratch/mssric004/Data_Small"
testval = 1800
count = 0
total = 0
min_intensity = 5000
max_intensity = -1
maxsum = 0
sample_dirs = os.listdir(root)
print("Loading in scan data...")
for sample in tqdm(sample_dirs):
    sleep(0.25)
    for scan in os.listdir(op.join(root, sample)):
        if ("anat" in scan) or ("func" in scan): #Mostly to sift out BIDS folders/random unwanted stuff
            image_root = op.join(root, sample, scan, 'NIFTI')
            #print("Reading file at", image_root, ": ", os.listdir(image_root)[0])
            image_file = os.listdir(image_root)[0]
            image_dir = op.join(image_root, image_file)
            image_data_raw = nib.load(image_dir).get_fdata()
            min = image_data_raw.min()
            max = image_data_raw.max()
            maxsum += max
            if min_intensity > min:
                min_intensity = min
            if max_intensity < max:
                max_intensity = max
            if max > testval:
                count += 1
            total += 1
print("MIN VALUE:", min_intensity)
print("MAX VALUE:", max_intensity)
print(count, "out of", total, "cases exceed", testval)
print("Average max:", round(maxsum/total, 2))