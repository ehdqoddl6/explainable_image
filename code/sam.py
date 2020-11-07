import numpy as np
import pandas as pd

# List all files under the input directory
# import os
# for dirname, _, filenames in os.walk('../dataSet'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2 #install opencv-python
import os
import tensorflow as tf
from tensorflow.keras import Model, layers

# 데이터셋은 train / test / val 3개 폴더로 이루어져 있고, 각 폴더는 정상 / 폐렴 폴더로 나누어져있다.
# X-ray 이미지는 광저우 Women and Children's Medical Center 에서  1~5세 아동 환자로부터 수집되었다.
# 이미지는 저품질, 판독 불가 수준의 스캔을 배제하고, 전문가의 진단을 받았다.

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


train = get_training_data('dataSet/chest_xray/chest_xray/train')
test = get_training_data('dataSet/chest_xray/chest_xray/test')
val = get_training_data('dataSet/chest_xray/chest_xray/val')

x = []
for i in train:
    if i[1] == 0:
        x.append("Pneumonia")
    else:
        x.append("Normal")
sns.set_style('darkgrid')
sns.countplot(x)
# plt.show()

plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])
# plt.show()

x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)



# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

print(x_val)
print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print('\n')

# resize data for deep learning
x_train = x_train.reshape(1, img_size, img_size, 3)
y_train = np.array(y_train)

x_val = x_val.reshape(1, img_size, img_size, 3)
y_val = np.array(y_val)

x_test = x_test.reshape(1, img_size, img_size, 3)
y_test = np.array(y_test)

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)
print('\n')