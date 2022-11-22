# Check the dimensions of contained images
# Richard Masson
import nibabel as nib
import numpy as np
from scipy import stats
import os.path as op

dirs = []
dirs.append("Directories/adni_1_images.txt")
dirs.append("Directories/adni_1_images_normed.txt")
dirs.append("Directories/adni_3_images.txt")
dirs.append("Directories/adni_3_images_normed.txt")

for dir_file in dirs:
    # Grab the data
    print("READING FROM:", dir_file)
    if op.exists(dir_file):
        path_file = open(dir_file, "r")
        path = path_file.read()
        path = path.split("\n")
        path_file.close()

        # Dimension counts
        hcount = []
        wcount = []
        dcount = []
        counter = 0

        for file in path:
            nifti = np.asarray(nib.load(file).get_fdata())
            hcount.append(nifti.shape[0])
            wcount.append(nifti.shape[1])
            dcount.append(nifti.shape[2])
            counter += 1

        print("Analysed a total of", counter, "images.")
        have = (int)(np.sum(hcount)/counter)
        wave = (int)(np.sum(wcount)/counter)
        dave = (int)(np.sum(dcount)/counter)
        print("Average dimensions: ( ", have, ", ", wave, ", ", dave, " )", sep='')
        hmode = stats.mode(hcount)
        wmode = stats.mode(wcount)
        dmode = stats.mode(dcount)
        print("Modal dimensions:\n", hmode, "\n", wmode, "\n", dmode, "\n")
