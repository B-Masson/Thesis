# Testing Intensity Normalization stuff
# Richard Masson
import nibabel as nib
import NIFTI_Engine as ne
from time import perf_counter
import os
import os.path as op
print("Most imports working.")
from intensity_normalization.normalize.fcm import FCMNormalize
print("Intensity normalization imported.")

def normalize(image):
	tic = perf_counter()
	fcm_norm = FCMNormalize()
	normalized = fcm_norm(image)
	toc = perf_counter()
	print("Time taken: ", round(toc-tic, 2), "s", sep='')
	return normalized

def RunTest():
	print("Running test.")
	target = "/scratch/mssric004/ADNI_Test_Tiny/MCI/002_S_6864/Accelerated_Sagittal_MPRAGE/2020-03-11_08_50_25.0/S934629/STRIPPED_ADNI_002_S_6864_MR_Accelerated_Sagittal_MPRAGE__br_raw_20200313151614103_26_S934629_I1303613.nii"
	image = nib.load(target)
	data = image.get_fdata()
	print("Original data loaded in. Beginning norm process.")

	#fcm_norm = FCMNormalize()
	for i in range(3):
		normalized = normalize(data)
	print("Done.")
	
def genNorms(root):
	count = 0
	fcm_norm = FCMNormalize()
	class_dirs = os.listdir(root) 
	for classes in class_dirs:
		if classes != "Zips":
			for scan in os.listdir(op.join(root, classes)):
				for type in os.listdir(op.join(root, classes, scan)):
					for date in os.listdir(op.join(root, classes, scan, type)):
						for image_folder in os.listdir(op.join(root, classes, scan, type, date)):
							image_root = op.join(root, classes, scan, type, date, image_folder)
							image_options = os.listdir(image_root)
							for file in image_options:
								if "STRIPPED" in file:
									norm_present = False
									for filen in image_options:
										if "NORMED" in filen:
											norm_present = True
									if norm_present == False:
										image = nib.load(op.join(image_root, file))
										data = image.get_fdata(dtype='float32')
										normalized = fcm_norm(data)
										outfile = op.join(image_root, ("NORMED_"+file[9:]))
										nib.Nifti1Image(normalized, image.affine).to_filename(outfile)
										count += 1
	print("All done.")
	print("Saved", count, "new normalized images.")

root = "/scratch/mssric004/ADNI_Data_NIFTI"
genNorms(root)
