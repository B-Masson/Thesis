# Engine for converting all nifti-files into skull-stripped versions
# https://github.com/jcreinhold/intensity-normalization
# https://github.com/jcreinhold/intensity-normalization/blob/master/tutorials/5min_tutorial.rst
# https://www.nitrc.org/projects/robex
# Richard Masson
import nibabel as nib
from pyrobex.robex import robex
import os
import os.path as op
print("Imports done.")

def genStrips(root, test=False):
	class_dirs = os.listdir(root)
	for classes in class_dirs:
		if classes != "Zips":
			for scan in os.listdir(op.join(root, classes)):
				for type in os.listdir(op.join(root, classes, scan)):
					for date in os.listdir(op.join(root, classes, scan, type)):
						for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
							image_root = op.join(root, classes, scan, type, date, image_folder)
							image_file = os.listdir(image_root)[0]
							image_dir = op.join(image_root, image_file)
							image_data = nib.load(image_dir)
							if test:
								print("Image loaded in. Now stripping...")
							try:
								stripped, mask = robex(image_data)
								outfile = "STRIPPED_" + image_file
								strip_path = op.join(image_root, outfile)
								nib.save(stripped, str(strip_path))
							except Exception as e:
								print("Could not strip image. Error:", e, "\nLocation:", image_root)
	print("All done.")

root = "/scratch/mssric004/ADNI_Data_NIFTI"
genStrips(root, test=False)