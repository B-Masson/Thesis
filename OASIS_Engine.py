# Host of functions specifically for generating x and y arrays of OASIS data
# Richard Masson
import LabelReader as lr
import NIFTI_Engine as ne
from collections import Counter
import numpy as np

def genClassLabels(scan_meta, cdr_meta):
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
    #y_arr = tf.keras.utils.to_categorical(y_arr) # <---------- CHANGE MADE HERE
    return y_arr, classNo

def gen_OASIS(testing_mode, w, h, d):
    rootloc = "/scratch/mssric004/Data_Small" # FOR NOW, ONLY LOOK AT THIS ONE
    if testing_mode:
        rootloc = "/scratch/mssric004/Data_Small"
    x_arr, scan_meta = ne.extractArrays('all', w, h, d, root=rootloc)
    clinic_sessions, cdr_meta = lr.loadCDR()
    y_arr, classNo = genClassLabels(scan_meta, cdr_meta)
    print("Testing data distribution:", Counter(y_arr))
    return x_arr, y_arr