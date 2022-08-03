# Iterate through brain slices to determine what we need to include in the 2D implementation
# Richard Masson
import nibabel as nib
import numpy as np
import random
import NIFTI_Engine as ne
print("Most imports done.")
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
print("Matplotlib plot done.")

stripped = False
if stripped:
    imgname = "/home/mssric004/Directories/adni_3_images_stripped.txt"
else:
    imgname = "/home/mssric004/Directories/adni_3_images.txt"

# Fetch file locations
path_file = open(imgname, "r")
path = path_file.read()
path = path.split("\n")
path_file.close()

# Shuffle it up
random.seed(11)
random.shuffle(path)

# Extract AN image (for now)
print("Loading image...")
img = path[0]
nifti = nib.load(img)
data = np.asarray(nifti.get_fdata(dtype='float32'))

# Resize
image = ne.cropADNI(data, stripped=stripped)
dims = image.shape
print(dims)

# Save it if you're feeling it
#nib.Nifti1Image(image, nifti.affine).to_filename("resized.nii")

# Display
print("Generating slices...")
for n in range (dims[2]):
    plt.imshow(image[:,:,n], cmap='bone')
    plt.savefig("SlicePicsAlt/slice"+str(n)+".png")

print("All done.")

# Start of useful data: 50 (Alt)
# End of useful data: 156 (Alt)
