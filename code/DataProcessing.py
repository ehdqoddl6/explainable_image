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


train = get_training_data('../dataSet/chest_xray/train')
test = get_training_data('../dataSet/chest_xray/test')
val = get_training_data('../dataSet/chest_xray/val')

x = []
for i in train:
    if i[1] == 0:
        x.append("Pneumonia")
    else:
        x.append("Normal")
sns.set_style('darkgrid')
sns.countplot(x)
plt.show()

plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])
plt.show()