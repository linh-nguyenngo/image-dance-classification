import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

test_data_set = keras.preprocessing.image_dataset_from_directory('./testing', batch_size=64, image_size=(28, 28))
train_data_set = keras.preprocessing.image_dataset_from_directory('./training', batch_size=64, image_size=(28, 28))

scaling = Rescaling(scale=1./255)
num_filters =32
filter_size = 3
pool_size = 2
num_classes = 10

model = Sequential([
  scaling,
  Conv2D(num_filters, filter_size),
  MaxPooling2D(),
  Conv2D(num_filters, filter_size),
  MaxPooling2D(),
  Conv2D(num_filters, filter_size),
  MaxPooling2D(),
  Flatten(),
  Dense(128, activation='relu'),
  Dense(num_classes, activation='softmax')
])

model.compile(
  'adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'],
)

model.fit(
  train_data_set,
  epochs=3,
  validation_data=test_data_set
)
