# Remove thsoe doubled-up entries.
import os
import os.path as op
import sys

root = "/scratch/mssric004/ADNI_Data_NIFTI"
target_count = 0

class_dirs = os.listdir(root)
for classes in class_dirs:
	if classes != "Zips":
		for scan in os.listdir(op.join(root, classes)):
			for type in os.listdir(op.join(root, classes, scan)):
				for date in os.listdir(op.join(root, classes, scan, type)):
					for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
						image_root = op.join(root, classes, scan, type, date, image_folder)
						image_files = os.listdir(image_root)
						for file in image_files:
							if "STRIPPED_STRIPPED" in file:
								rem_file = op.join(image_root, file)
								print(rem_file)
								#os.remove(rem_file)
								target_count += 1
								#sys.exit()

print("\nTotal targets for deletion:", target_count)