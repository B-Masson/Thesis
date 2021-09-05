# Application class for pre-processing all the data needed to train a model for Alzheimer's Disease prediction
# Richard Masson
import LabelReader as lr
import NIFTI_Engine as ne
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from scipy import ndimage
#import h5py
#import hickle

# Fetch all our seperated data
print("Starting up")
x_arr, scan_meta = ne.extractArrays('anat3')
clinic_sessions, cdr_meta = lr.loadCDR(size=50)
print('*'*10)

# Ascertain lengths and shapes
print("Scan array length:", len(x_arr), "| Scan meta length:", len(scan_meta))
print("Clinical sessions length:", len(clinic_sessions), "| Patient CDR meta length:", len(cdr_meta))
print(scan_meta)
print('*'*10)

# Generate some cdr y labels for each scan
# Current plan: Use the cdr from the closest clinical entry by time (won't suffice in the longterm but it will do for now)
y_arr = []
for scan in scan_meta:
    scan_day = scan['day']
    print("Patient", scan['ID'], "scan on day:", scan_day)
    scan_cdr = -1
    cdr_day = -1
    min = 1000
    for x in cdr_meta[scan['ID']]:
        diff = abs(scan_day-x[0])
        if diff < min:
            min = diff
            scan_cdr = x[1]
            cdr_day = x[0]
    print("Assigned cdr", scan_cdr, "- from day", cdr_day)
    y_arr.append(scan_cdr)
print('*'*10)
print(y_arr)

# Split data
x_train, x_val, y_train, y_val = train_test_split(x_arr, y_arr, stratify=y_arr) #Defaulting to 75 train, 25 val. Also shuffle=true and stratifytrue.
print("Data successfully split. Train [", len(x_train), "] | Validate [", len(x_val), "]", sep='')

# Save processed data so we can actually split up all this effort
#f = h5py.File("train_data.hdf5", "w")

with open('training', 'wb') as f:
    np.save(f, [x_train, y_train])
    
with open('validation', 'wb') as f:
    np.save(f, [x_val, y_val])
'''
# Data augmentation functions
@tf.function
def rotate(image):
    """Rotate the image by a few degrees"""
    def scipy_rotate(image):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate image
        image = ndimage.rotate(image, angle, reshape=False)
        image[image < 0] = 0
        image[image > 1] = 1
        return image

    augmented_image = tf.numpy_function(scipy_rotate, [image], tf.float32)
    return augmented_image


def train_preprocessing(image, label):
    """Process training data by rotating and adding a channel."""
    # Rotate image
    image = rotate(image)
    image = tf.expand_dims(image, axis=3)
    return image, label


def validation_preprocessing(image, label):
    """Process validation data by only adding a channel."""
    image = tf.expand_dims(image, axis=3)
    return image, label


# Augment data, as well expand dimensions to make the training model accept it (by adding a 4th dimension)
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
batch_size = 1
# Augment the on the fly during training.
train_set = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(batch_size)
)
# Only rescale.
validation_set = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(batch_size)
)
print("Done")
'''