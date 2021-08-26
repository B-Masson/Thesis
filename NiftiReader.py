# Basic code to display the MRI image data found within a given NIFTI file
# Richard Masson
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
'''
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
'''

root = "C:\\Users\\richa\\Documents\\Uni\\Thesis\\central.xnat.org"
print("Hello world.")
folder = os.listdir(root)
count = 1
for file in folder:
    print(count, ":", file)
    count+=1
#choice = input("Select which sample you wish to try to read:\n")
choice = "5" # we out here hard coding for now
image_folder = folder[(int)(choice)-1]
scan_folders = os.listdir(os.path.join(root, image_folder))

# Import packages necessary for processing NIFTI files + displaying
print("Importing...")
import scipy.misc
import numpy as np
from skimage.io import imread
import nibabel as nib
import matplotlib.pyplot as plt

scans = []
# Observe every folder
for folder in scan_folders:
    print("Now at:", folder, end=" | ")
    root_niftis = os.path.join(root, image_folder, folder)
    check_niftis = os.listdir(root_niftis)
    if "NIFTI" in check_niftis:
        image_root = os.path.join(root_niftis, 'NIFTI')
        image_file = os.listdir(image_root)[0]
        image_dir = os.path.join(image_root, image_file)
        print("OK")
        test_image=nib.load(image_dir).get_fdata()
        #from skimage.util import montage as montage2d
        #fig, ax1 = plt.subplots(1, 1, figsize = (10, 10))
        #ax1.imshow(montage2d(test_image), cmap ='bone')
        #plt.show()
        content = folder + " | " + str(test_image.shape)
        scans.append(content)
    else: print("SKIPPED")

# Let's quickly write a reference doc
f = open("scan_shapes.txt", "w")
for scan in scans:
    f.write(scan+"\n")
f.close()
print("DONE")