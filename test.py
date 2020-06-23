from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

AUTOTUNE = tf.data.experimental.AUTOTUNE
import os
import matplotlib.pyplot as plt

def preprocess_image(image):
  image = tf.image.decode_png(image, channels = 1)
  image = tf.image.resize(image, [64, 64])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(image, label):
  image = tf.io.read_file(image)
  return preprocess_image(image), label

import pathlib
data_root_orig = 'data_middle/'
test_root_orig = 'test_middle/'
data_root = pathlib.Path(data_root_orig)
test_root = pathlib.Path(test_root_orig)
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

def restore_label():
    with open('label.pickle', 'rb') as f:
        label_to_index = pickle.load(f)
#     print(label_to_index)
    return label_to_index

def init():
  label_to_index = dict((name, index) for index, name in enumerate(label_names))
  with open('label.pickle', 'wb') as f:
        pickle.dump(label_to_index, f, pickle.HIGHEST_PROTOCOL)
  return label_to_index

label_to_index=restore_label()
# label_to_index = init()
print(label_to_index)

import random

batch_size = 32

train_image_paths = list(data_root.glob('*/*'))
train_image_paths = [str(path) for path in train_image_paths]
random.shuffle(train_image_paths)
train_image_count = len(train_image_paths)
print(train_image_count)
train_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in train_image_paths]
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths,train_image_labels))
train_dataset = train_dataset.shuffle(buffer_size=6000).repeat()
train_dataset = train_dataset.map(load_and_preprocess_image).batch(batch_size)
train_dateset = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()



test_image_paths = list(test_root.glob('*/*'))
test_image_paths = [str(path) for path in test_image_paths]
random.shuffle(test_image_paths)
test_image_count = len(test_image_paths)
print(test_image_count)
test_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in test_image_paths]
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths,test_image_labels))
test_dataset = test_dataset.shuffle(buffer_size=6000).repeat()
test_dataset = test_dataset.map(load_and_preprocess_image).batch(batch_size)
test_dateset = tf.compat.v1.data.make_one_shot_iterator(test_dataset).get_next()

def create_model():
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(len(label_to_index), activation = 'softmax')
    ])
    return model

# model.compile(optimizer='adam',
#             #   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#             loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy'])
model_name = 'my_model_2.h5'
def load_model(mode):
  if mode =="create":
    return create_model()
  if mode == "read":
    return tf.keras.models.load_model(model_name)

model = load_model('read')
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

model.summary()
print('testing test data:')
test_loss, test_acc = model.evaluate(test_dataset, steps=100)
print('\nTest accuracy:', test_acc)