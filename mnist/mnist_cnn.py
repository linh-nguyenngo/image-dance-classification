import numpy as np
import mnist
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout

# Load dataset
train_imgs = mnist.train_images()
train_labels = mnist.train_labels()

test_imgs = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the dataset
train_imgs = (train_imgs/255) - .5
test_imgs = (test_imgs/255) - .5

# Expand dimension to get the correct shapes for training
train_imgs = np.expand_dims(train_imgs, axis=3)
test_imgs = np.expand_dims(test_imgs, axis=3)

# Check shapes of the dataset
# print(train_imgs.shape)
# print(test_imgs.shape)

# Define variables to use in the model
num_filters = 8
filter_size = 3
pool_size = 2

# Create model
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
  Conv2D(num_filters, filter_size),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax')
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Start training
model.fit(
  train_imgs,
  to_categorical(train_labels),
  epochs=10,
  validation_data=(test_imgs, to_categorical(test_labels)),
)

model.save_weights('cnn.h5')
