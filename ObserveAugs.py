# Saving augmented nifti files so we can observe the effects
# Richard Masson

from volumentations import *
import nibabel as nib
import os
import os.path as op
print("Imports done.")

target = "/home/mssric004/TestRobex/ADNI_002_S_6864_MR_Accelerated_Sagittal_MPRAGE__br_raw_20200313151614103_26_S934629_I1303613.nii"
image = nib.load(target)
nifti = image.get_fdata(dtype='float32')

def get_augmentation(patch_size):
    return Compose([
        Rotate((-3, 3), (-3, 3), (-3, 3), p=0.7), #0.5
        #Flip(2, p=1)
        ElasticTransform((0, 0.06), interpolation=2, p=0.3), #0.1
        #GaussianNoise(var_limit=(1, 1), p=1), #0.1
        RandomGamma(gamma_limit=(0.6, 1), p=0.3) #0.4
    ], p=1) #0.7? #NOTE: Temp not doing augmentation. Want to take time to observe the effects of this stuff
aug = get_augmentation(nifti.shape) # For augmentations

# Augmentation
print("Applying augmentation...")
data = {'image': nifti}
aug_data = aug(**data)
nifti = aug_data['image']

name = "Aug_Noise.nii"
outfile = "/home/mssric004/TestRobex/" +name
print("Saving to", name)

nib.Nifti1Image(nifti, image.affine).to_filename(outfile)