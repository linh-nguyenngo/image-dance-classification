


!pip install tf-nightly

#Import libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

#Load dataset
import pathlib
from google.colab import drive
drive.mount('/content/gdrive')

dataset_dir = "/content/gdrive/My Drive/Database/images"
data_dir = pathlib.Path(dataset_dir)
print(dataset_dir)

# Count to make sure dataset is ready
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Define some variables to be used inside model
batch_size = 64
img_height = 250
img_width = 250

# Define traning dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Define validating dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  shuffle=True,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Check to make sure dataset is loaded
class_names = train_ds.class_names
print(class_names)

# import matplotlib to provide ploting/drawing functions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Tuning for better performance by enabling caching
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(5000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Standardize the data from RGB [0, 255] range to [0, 1] for neural network using a rescaling layer
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Define augmentation variable to use in the model
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal",
                                                 input_shape=(img_height,
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

# Check to make sure images can be augmentated
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

# Define callback to stop training when neeeded
# Only use when absolutely needed
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# Define model
num_classes = 5
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Complie model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# View a summary of the model to check
model.summary()

# Start training
epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Get training results and put into graphs for better visulization
epochs = len(history.history['loss'])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.ylim([0.8,max(plt.ylim())])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.ylim([0,0.5])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save model to use later
saved_models_path = "/content/gdrive/My Drive/saved_models/sep09"
model.save(saved_models_path)

# Test the models by passing in a dataset from a given directoy
# Each dataset should belong to one type of dance only
# This is to support the usa case when a video is splited into images

prediction_path = "/content/gdrive/My Drive/mixed_predictions/"
predictions_dir = pathlib.Path(prediction_path)

image_count = len(list(predictions_dir.glob('*.jpg')))
print(image_count)

prediction_ds = tf.keras.preprocessing.image_dataset_from_directory(
  predictions_dir,
  shuffle=True,
  image_size=(250, 250)
)

predictions = model.predict(prediction_ds, batch_size=64)

scores = []
class_name_indexes = []
unique_class_name_indexes = []
unique_class_name_indexes_counts = []
unique_class_name_indexes_percentages = []

# Loop through prediction results to get the scores, and class_name_indexes
for prediction in predictions:
  score = tf.nn.softmax(prediction)
  class_name_index = np.argmax(score)
  scores.append(score)
  class_name_indexes.append(class_name_index)

# Remove all duplicated name indexes
unique_class_name_indexes = list(set(class_name_indexes))

# Count the number of images for each unique index
for unique_index in unique_class_name_indexes:
  count = 0
  for index in class_name_indexes:
    if unique_index == index:
      count+=1
  unique_class_name_indexes_counts.append(count)

# Calculate the percentage for each unique index
for count in unique_class_name_indexes_counts:
  unique_class_name_indexes_percentages.append(100*count/sum(unique_class_name_indexes_counts))

# Print out the results in a more human friendly way
for idx, count in enumerate(unique_class_name_indexes_counts):
  print(f"{count} image(s) around {format(unique_class_name_indexes_percentages[idx], '.2f')}% seems to belong to {class_names[unique_class_name_indexes[idx]]}")

print(f"* The images most likely to be '{class_names[unique_class_name_indexes[unique_class_name_indexes_counts.index(max(unique_class_name_indexes_counts))]]}'")

# This is to check the accuracy of the first image
# score = tf.nn.softmax(predictions[0])
# print(
#     "This image most likely belongs to {} with a {:.2f} percent confidence."
#     .format(class_names[np.argmax(score)], 100 * np.max(score))
# )
