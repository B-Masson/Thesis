import tensorflow as tf
import numpy as np
from scipy import ndimage

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

# Load in that data
train = np.load('training', allow_pickle=True)
val = np.load('validation', allow_pickle=True)
print(len(train))
print(len(val))
x_train = np.asarray(train[0]).astype('float32')
y_train = np.asarray(train[1]).astype('float32')
x_val = np.asarray(val[0]).astype('float32')
y_val = np.asarray(val[1]).astype('float32')

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
print("Data is ready for training.")