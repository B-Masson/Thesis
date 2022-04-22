# Testing Intensity Normalization stuff
import nibabel as nib
#import intensity_normalization
#from intensity_normalization.typing import Modality, TissueType
from intensity_normalization.normalize.fcm import FCMNormalize
import matplotlib.pyplot as plt
import NIFTI_Engine as ne
print("Imports working.")

target = "/scratch/mssric004/ADNI_Test_Tiny/MCI/002_S_6864/Accelerated_Sagittal_MPRAGE/2020-03-11_08_50_25.0/S934629/STRIPPED_ADNI_002_S_6864_MR_Accelerated_Sagittal_MPRAGE__br_raw_20200313151614103_26_S934629_I1303613.nii"
image = nib.load(target)
data = image.get_fdata()

fcm_norm = FCMNormalize()
print("Normalizing...")
normalized = fcm_norm(data)
'''
new_norm = nib.Nifti1Image(normalized, image.affine, image.header)
loc = "TestRobex/normed_image.nii"
print("Successfully normalized image. Saving to", loc)
nib.save(new_norm, loc)
'''
thre = 80
inc = 3
dims = data.shape
cutoff = (int)(dims[1]/2)
print(data[thre:thre+inc,thre:thre+inc,thre:thre+inc])
print("Original elements are:", data[0][0][0].dtype)
print("Shape:", dims)
plt.imshow(data[:,cutoff,:], cmap='bone')
plt.savefig("norm_pre.png")
print("\n", normalized[thre:thre+inc,thre:thre+inc,thre:thre+inc])
print("Normalized elements are:", normalized[0][0][0].dtype)
print("Shape:", normalized.shape)
plt.imshow(normalized[:,cutoff,:], cmap='bone')
plt.savefig("norm_post.png")
norm_self = ne.normalize(data)
print("\n", norm_self[thre:thre+inc,thre:thre+inc,thre:thre+inc])
print("Norm self elements are:", norm_self[0][0][0].dtype)
print("Shape:", norm_self.shape)
print("All done.")