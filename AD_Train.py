import tensorflow as tf
import numpy as np
from scipy import ndimage
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
import random

# Model architecture go here
def gen_model(width=128, height=128, depth=64): # Make sure defaults are equal to image resizing defaults
    inputs = keras.Input((width, height, depth, 1)) # Added extra dimension in preprocessing to accomodate that 4th dim

    # Contruction seems to be pretty much the same as if this was 2D. Kernal should default to 3,3,3
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs) # Layer 1: Usual 64 filter start
    x = layers.MaxPool3D(pool_size=2)(x) # Always max pool after the conv layer
    x = layers.BatchNormalization()(x)

    #x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x) # Might not hurt to remove this layer for simplicity
    #x = layers.MaxPool3D(pool_size=2)(x)
    #x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x) # Double filters
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x) # Double filters one more time
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x) # Implement a simple dense layer with double units
    x = layers.Dropout(0.3)(x) # 30% dropout rate for now (this differs from original paper which used 60% so might get changed later)

    outputs = layers.Dense(units=1, activation="softmax")(x) # Units = no of classes. Also softmax because classes are mutually exclusive

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")
    return model

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
train = np.load('training.npz')
val = np.load('validation.npz')
#x_train = np.asarray(train[0]).astype('float32')
#y_train = np.asarray(train[1]).astype('float32')
#x_val = np.asarray(val[0]).astype('float32')
#y_val = np.asarray(val[1]).astype('float32')
x_train = train['a']
y_train = train['b']
x_val = val['a']
y_val = val['b']

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

# Build model.
model = gen_model(width=128, height=128, depth=64)
model.summary()
optim = keras.optimizers.Adam(learning_rate=0.001) # LR chosen based on principle but double-check this later
# Note: These things will have to change if this is changed into a regression model
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy']) # Binary cross is usual for single-class 0-1 stuff. Accuracy is straightfoward

# Model hyperparameters
epochs = 5 # Small for testing purposes
# CHECKPOINT CODE GO HERE
# POTENTIAL EARLY STOPPING GO HERE

# Run the model
print("Ready to train.")
print('*'*10)
history = model.fit(train_set, validation_data=validation_set, batch_size=1, epochs=epochs, shuffle=True, verbose=1)
print(history.history)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
model.save('ADModel')
