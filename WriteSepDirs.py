# Code to generate two text files, one list of image paths for (currently) the ADNI dataset, and one list of their corresponding labels
import os
import os.path as op
import re
from cv2 import threshold
import random
from numpy import not_equal

# https://stackoverflow.com/questions/71045309/how-to-create-a-dataset-for-tensorflow-from-a-txt-file-containing-paths-and-labe

gen_tests = True
tiny_mode = False
strip = False
norm = False
no_dupes = False
trimming = True
if gen_tests:
	root = "/scratch/mssric004/ADNI_Test"
elif tiny_mode:
    root = "/scratch/mssric004/ADNI_Test_Tiny"
else:
	root = "/scratch/mssric004/ADNI_Data_NIFTI"

# Set this to generate the different sets
for mode in range(1, 4):
    print("MODE:", mode, "\n")
    if gen_tests:
        filename = "Directories/test_adni_" + str(mode)
    elif tiny_mode:
        filename = "Directories/test_tiny_adni_" +str(mode)
    else:
        filename = "Directories/adni_" + str(mode)

    if no_dupes:
        filename = filename + "_NODUPE"
    if trimming:
        filename = filename + "_trimmed"

    if norm:
        imgfile = filename+"_images_normed.txt"
        labelfile = filename+"_labels_normed.txt"
    elif strip:
        imgfile = filename+"_images_stripped.txt"
        labelfile = filename+"_labels_stripped.txt"
    else:
        imgfile = filename + "_images.txt"
        labelfile = filename + "_labels.txt"
    print("Writing to", filename)
    newline = '' # Little trick to avoid having a newline at the end of file
    countCN = 0
    countMCI = 0
    countAD = 0
    write_count = 0
    label_count = 0
    blanks = 0
    arr0 = []
    arr1 = []
    arr2 = []

    with open(imgfile, 'w') as i:
        with open(labelfile, 'w') as l:
            class_dirs = os.listdir(root)
            exclusions = []
            if mode == 1:
                exclusions.append("AD")
                print("Mode: Classify CN vs. MCI")
            elif mode == 2:
                exclusions.append("MCI")
                print("Mode: Classify CN vs. AD")
            elif mode == 4:
                exclusions.append("CN")
                exclusions.append("MCI")
                print("Mode: ONLY AD. Only use for testing purposes.")
            elif mode == 5:
                exclusions.append("CN")
                exclusions.append("AD")
            else:
                print("Mode: Classify from all 3 categories.")
            print("Exclusions are:", exclusions)
            print("\nReading through image directory...\n")
            for classes in class_dirs:
                if classes not in exclusions and classes != "Zips":
                    if no_dupes:
                        for scan in os.listdir(op.join(root, classes)):
                            type = os.listdir(op.join(root, classes, scan))[0]
                            if type == "AAHead_Scout_MPR_sag":
                                print("AAHead discarded.")
                            elif type == "AAHead_Scout":
                                print("AAHead Scout discarded.")
                            elif type == "3D_MPRAGE":
                                print("3D MPRAGE discarded.")
                            else:
                                date = os.listdir(op.join(root, classes, scan, type))[0]
                                image_folder = os.listdir(op.join(root, classes, scan, type, date))[0]
                                image_root = op.join(root, classes, scan, type, date, image_folder)
                                image_options = os.listdir(image_root)
                                image_file = '[none]'
                                if norm:
                                    for file in image_options:
                                        if "NORMED" in file:
                                            image_file = file
                                elif strip:
                                    for file in image_options:
                                        if "STRIPPED" in file:
                                            image_file = file
                                else:
                                    for file in image_options:
                                        if "NORMED" not in file:
                                            if "STRIPPED" not in file:
                                                image_file = file
                                if gen_tests or tiny_mode:
                                    print("Code chose:", image_file)
                                if image_file == '[none]':
                                    blanks += 1
                                else:
                                    image_dir = op.join(image_root, image_file)
                                    #i.write(newline+image_dir)
                                    #img_arr.append(image_dir)
                                    #write_count += 1
                                    if classes == "CN":
                                        #l.write(newline+"0")
                                        arr0.append(image_dir)
                                        #label_count += 1
                                        #countCN += 1
                                    elif classes == "MCI":
                                        #l.write(newline+"1")
                                        arr1.append(image_dir)
                                        #label_count += 1
                                        #countMCI += 1
                                    elif classes == "AD":
                                        if mode == 2:
                                            #l.write(newline+"1")
                                            arr1.append(image_dir)
                                        else:
                                            #l.write(newline+"2")
                                            arr2.append(image_dir)
                                        #label_count += 1
                                        #countAD += 1
                                    else:
                                        print("One of the folders does not match any of the expected forms.")
                                    #newline = '\n'
                    else:
                        for scan in os.listdir(op.join(root, classes)):
                            for type in os.listdir(op.join(root, classes, scan)):
                                if type == "AAHead_Scout_MPR_sag":
                                    print("AAHead discarded.")
                                elif type == "AAHead_Scout":
                                    print("AAHead Scout discarded.")
                                elif type == "3D_MPRAGE":
                                    print("3D MPRAGE discarded.")
                                else:
                                    for date in os.listdir(op.join(root, classes, scan, type)):
                                        for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
                                            image_root = op.join(root, classes, scan, type, date, image_folder)
                                            image_options = os.listdir(image_root)
                                            image_file = '[none]'
                                            if norm:
                                                for file in image_options:
                                                    if "NORMED" in file:
                                                        image_file = file
                                            elif strip:
                                                for file in image_options:
                                                    if "STRIPPED" in file:
                                                        image_file = file
                                            else:
                                                for file in image_options:
                                                    if "NORMED" not in file:
                                                        if "STRIPPED" not in file:
                                                            image_file = file
                                            if gen_tests or tiny_mode:
                                                print("Code chose:", image_file)
                                            if image_file == '[none]':
                                                blanks += 1
                                            else:
                                                image_dir = op.join(image_root, image_file)
                                                #i.write(newline+image_dir)
                                                #img_arr.append(image_dir)
                                                #write_count += 1
                                                if classes == "CN":
                                                    #l.write(newline+"0")
                                                    arr0.append(image_dir)
                                                    #label_count += 1
                                                    #countCN += 1
                                                elif classes == "MCI":
                                                    #l.write(newline+"1")
                                                    arr1.append(image_dir)
                                                    #label_count += 1
                                                    #countMCI += 1
                                                elif classes == "AD":
                                                    if mode == 2:
                                                        #l.write(newline+"1")
                                                        arr1.append(image_dir)
                                                    else:
                                                        #l.write(newline+"2")
                                                        arr2.append(image_dir)
                                                    #label_count += 1
                                                    #countAD += 1
                                                else:
                                                    print("One of the folders does not match any of the expected forms.")
                                                #newline = '\n'
            # write here
            if mode == 1 or mode == 2:     
                if trimming:
                    threshold = min(len(arr0), len(arr1))
                    print("Minimum threshold point is:", threshold)
                else:
                    threshold = 10000
                random.shuffle(arr0)
                random.shuffle(arr1)
                lim = 0
                for x in arr0:
                    lim += 1
                    #print("lim is", lim, "and threshold is", threshold)
                    if lim <= threshold:
                        i.write(newline+x)
                        l.write(newline+"0")
                        newline = '\n'
                        write_count += 1
                        countCN += 1
                lim = 0
                for y in arr1:
                    lim += 1
                    #print("lim is", lim, "and threshold is", threshold)
                    if lim <= threshold:
                        i.write(newline+y)
                        l.write(newline+"1")
                        newline = '\n'
                        write_count += 1
                        countMCI += 1
            elif mode == 3:
                if trimming:
                    threshold = min(len(arr0), len(arr1), len(arr2))
                    print("Minimum threshold point is:", threshold)
                else:
                    threshold = 10000
                random.shuffle(arr0)
                random.shuffle(arr1)
                random.shuffle(arr2)
                lim = 0
                for x in arr0:
                    lim += 1
                    #print("lim is", lim, "and threshold is", threshold)
                    if lim <= threshold:
                        i.write(newline+x)
                        l.write(newline+"0")
                        newline = '\n'
                        write_count += 1
                        countCN += 1
                lim = 0
                for y in arr1:
                    lim += 1
                    #print("lim is", lim, "and threshold is", threshold)
                    if lim <= threshold:
                        i.write(newline+y)
                        l.write(newline+"1")
                        newline = '\n'
                        write_count += 1
                        countMCI += 1
                lim = 0
                for z in arr2:
                    lim += 1
                    if lim <= threshold:
                        i.write(newline+z)
                        l.write(newline+"2")
                        newline = '\n'
                        write_count += 1
                        countAD += 1
            else:
                print("Not yet implemented.")

    print(write_count, "images written!")
    #print(label_count, "labels written!")
    counts = {"CN": countCN, "MCI": countMCI, "AD": countAD}
    for cata in counts:
        if (counts[cata] != 0):
            print(cata, ": ", counts[cata], " images.", sep='')
    if blanks != 0:
        print("Skipped over", blanks, "entries that did not match specifications.")
print("All done!")