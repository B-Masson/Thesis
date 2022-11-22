# Check what's up with the skull stripping process
# Richard Masson
import os
import os.path as op
import re

root = "/scratch/mssric004/ADNI_Data_NIFTI"
stripped_images = 0
total_images = 0
red_flag = 0
double1 = 0
double2 = 0


class_dirs = os.listdir(root)
for classes in class_dirs:
	if classes != "Zips":
		for scan in os.listdir(op.join(root, classes)):
			for type in os.listdir(op.join(root, classes, scan)):
				for date in os.listdir(op.join(root, classes, scan, type)):
					for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
						image_root = op.join(root, classes, scan, type, date, image_folder)
						image_files = os.listdir(image_root)
						if len(image_files) != 0:
							total_images += 1
						red_flag = 0
						for file in image_files:
							if "STRIPPED" in file:
								stripped_images += 1
								red_flag += 1
								if (red_flag > 1):
									double1 += 1
							if "STRIPPED_STRIPPED" in file:
								double2 += 1
						if (red_flag == 0):
							print("No strips found in", image_root, "\n", image_files)
									
print("There are a total of", stripped_images, "stripped out of", total_images)
print("Can confirm there are", double1, "doubled entries.", double2, "of which are stripped_stripped.")
print("Factoring this in puts as at:", (stripped_images-double1), "/", total_images, "correct strips.")
print("All done.")