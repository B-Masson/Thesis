import nibabel as nib
from deepbrain import Extractor
import sys
import matplotlib.pyplot as plt
print("Imports working.")

# Load a nifti as 3d numpy image [H, W, D]
img_path = "/scratch/mssric004/ADNI_Data_NIFTI/MCI/003_S_0908/Accelerated_Sagittal_MPRAGE/2017-10-20_13_22_01.0/S663469/ADNI_003_S_0908_MR_Accelerated_Sagittal_MPRAGE__br_raw_20180302130514878_150_S663469_I969412.nii"
img = nib.load(img_path).get_fdata()

plt.imshow(img[:,:,100], cmap='bone')
plt.savefig("skull_before.png")

ext = Extractor()

# `prob` will be a 3d numpy image containing probability 
# of being brain tissue for each of the voxels in `img`
prob = ext.run(img) 

# mask can be obtained as:
mask = prob > 0.5

plt.imshow(img[:,:,100], cmap='bone')
plt.savefig("skull_after.png")

#TO-DO CONVERT OVER TO UPDATED DEEPBRAIN