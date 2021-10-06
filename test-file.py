import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os
import numpy as np
from keras.models import load_model
labels = ['Semi-Matang/1','Matang', 'Mentah']
classes = ['Semi-Matang/1', 'Matang', 'Mentah']
from keras.models import load_model
model = load_model('support/model-new.h5')

data_test = []
img_size = 224
image = cv2.imread('/home/kecilin/Downloads/kematangan buah tomat/Semi-Matang/2/1 (15).jpg')[...,::-1]
resized_arr = cv2.resize(image, (img_size, img_size))
# print(resized_arr)
data_test.append([resized_arr, 2])
data_test = np.array(data_test)
# print(data_test)
x_test, y_test = [], []
for feature, label in data_test:
  x_test.append(feature)
  y_test.append(label)
x_test = np.array(x_test) / 255

x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)
probs = model.predict([x_test])
probs.shape

prediction = probs.argmax(axis=1)
# print(probs[prediction] * 100)
print(prediction[0])
print(classes[prediction[0]])