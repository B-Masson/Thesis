# Can we trim these brain images?
import numpy as np
import nibabel as nib
print("Imports working.")

target = "TestRobex/saved_stripped.nii"
image = nib.load(target).get_fdata()
print(np.where(image)[0].max())

print("All done.")