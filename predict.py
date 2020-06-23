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
import pathlib

def preprocess_image(image):
  image = tf.image.decode_png(image, channels = 1)
  image = tf.image.resize(image, [64, 64])
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(image, label):
  image = tf.io.read_file(image)
  return preprocess_image(image), label

def restore_label():
    with open('label.pickle', 'rb') as f:
        label_to_index = pickle.load(f)
#     print(label_to_index)
    return label_to_index

label_to_index=restore_label()

model_name = 'my_model_2.h5'
model = tf.keras.models.load_model(model_name)

###
import sys
prediction_root_orig = sys.argv[1]
print('Path: ', prediction_root_orig)
###
# prediction_root_orig=str(r"C:\Users\hang\Desktop\middle\prediction")

batch_size = 32
# prediction_root_orig = 'prediction/'
prediction_root = pathlib.Path(prediction_root_orig)
prediction_image_paths = list(prediction_root.glob('*/'))
prediction_image_paths = [str(path) for path in prediction_image_paths]
prediction_image_labels = len(prediction_image_paths)*[-1]
prediction_dataset = tf.data.Dataset.from_tensor_slices((prediction_image_paths,prediction_image_labels))
# prediction_dataset = prediction_dataset.shuffle(buffer_size=6000).repeat()
prediction_dataset = prediction_dataset.map(load_and_preprocess_image).batch(batch_size)
prediction_dateset = tf.compat.v1.data.make_one_shot_iterator(prediction_dataset).get_next()

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(prediction_dateset, steps = 1)
print(predictions.shape)
predictions = [np.argmax(item)
              for item in predictions]
print(predictions)
index_to_label = {v : k for k,v in label_to_index.items()}
# print(prediction_image_paths)
# for item in predictions:
#   print(index_to_label[item])

for i in range(len(prediction_image_paths)):
  print(prediction_image_paths[i], "\t", index_to_label[predictions[i]])

import os
os.system('pause')