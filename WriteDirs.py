# Code to generate a big ol' text file of image paths for (currently) the ADNI dataset
import os
import os.path as op

# https://stackoverflow.com/questions/71045309/how-to-create-a-dataset-for-tensorflow-from-a-txt-file-containing-paths-and-labe

gen_tests = False
if gen_tests:
	root = "/scratch/mssric004/ADNI_Test"
else:
	root = "/scratch/mssric004/ADNI_Data_NIFTI"

# Set this to generate the different sets
mode = 1
if gen_tests:
	filename = "Directories/test_adni_" + str(mode) +".txt"
else:
	filename = "Directories/adni_" + str(mode) +".txt"

print("Writing to", filename)

with open(filename, 'w') as f:
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
	print("\nReading through image directory...\n")
	for classes in class_dirs:
		if classes != exclusions and classes != "Zips":
			for scan in os.listdir(op.join(root, classes)):
				for type in os.listdir(op.join(root, classes, scan)):
					for date in os.listdir(op.join(root, classes, scan, type)):
						for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
							image_root = op.join(root, classes, scan, type, date, image_folder)
							image_file = os.listdir(image_root)[0]
							image_dir = op.join(image_root, image_file)
							#image_data_raw = nib.load(image_dir)
							#image_data = organiseADNI(image_data_raw, w, h, d)
							#scan_array.append(image_data_raw)
							if classes == "CN":
								f.write(image_dir +" 0\n")
							elif classes == "MCI":
								f.write(image_dir +" 1\n")
							elif classes == "AD":
								f.write(image_dir +" 2\n")
							else:
								print("One of the folders does not match any of the expected forms.")
							#class_array.append(classes)

print("File written!")