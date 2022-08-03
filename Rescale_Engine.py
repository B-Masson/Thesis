print("RESCALE ENGINE.")
from matplotlib import testing
import NIFTI_Engine as ne
import numpy as np
import statistics

testing_mode = False

if testing_mode:
    print("TEST MODE")
roota = "/scratch/mssric004/ADNI_Data_NIFTI"
if testing_mode:
    roota = "/scratch/mssric004/ADNI_Test"

images = ne.extractPure(roota)
images = np.asarray(images)
m = images.mean()
s = images.std()
#m = statistics.mean(images)
print("Mean:", round(m,2))
print("Standard deviation:", round(s, 2))

flat = images.flatten()
print("\nShape:", images.shape)
print("Flattened shape:", flat.shape)

print("\n98th percentile:", np.percentile(images, 98))
print("2nd percentile:", np.percentile(images, 2))

print("All done.")