import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import warnings
import cv2
warnings.filterwarnings(action='ignore') 

print('Notebook run using keras:', keras.__version__)


inet_model = inc_net.InceptionV3()
#inet_model.summary()

from keras.models import load_model
model = load_model('my_model')


#model.build([None, 150, 150, 3])
model.summary()
labels = ['PNEUMONIA', 'NORMAL']



#img_arr = cv2.imread(os.path.join('dataSet/chest_xray/test/PNEUMONIA/', 'person1_virus_11.jpeg'), cv2.IMREAD_GRAYSCALE)
img = image.load_img('dataSet/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg', target_size=(150, 150))
x = image.img_to_array(img)
print(x.shape)
images = np.expand_dims(x, axis=0)
#x = inc_net.preprocess_input(x)
#print('\n')
print(x.shape)

"""
images = cv2.resize(img_arr, (150,150,3)) # Reshaping images to preferred size
print(images.shape)
images = np.array(images) / 255
images = np.expand_dims(images, axis=0)
print(images.shape)
images = images.reshape(-1, 150, 150, 1)
print(images.shape)
"""


#images = transform_img_fn([os.path.join('dataSet/chest_xray/test/PNEUMONIA/','person1_virus_11.jpeg')])
#print(images)

# I'm dividing by 2 and adding 0.5 because of how this Inception represents images
#preds = model.predict(images)
preds = model.predict_classes(images)
#preds = np.argmax(model.predict(images), axis=-1)
print(preds)
#preds = preds.reshape(1,-1)
#print(preds)


import os,sys
import lime
from lime import lime_image

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images[0].astype('double'), model.predict_classes, top_labels=2, hide_color=0, num_samples=100)



from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=1000, hide_rest=False, min_weight=0.1)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()


ind =  explanation.top_labels[0]

#Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

#Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
plt.show()


temp, mask = explanation.get_image_and_mask(106, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()


temp, mask = explanation.get_image_and_mask(106, positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()


