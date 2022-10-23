import keras
from keras import Model
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from scipy import stats
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
import cv2



class DataGenerator(keras.utils.Sequence):
	
    def __init__(self, images, labels, batch_size=32, n_classes = 2, shuffle=True):
	
	self.batch_size = batch_size
	self.shuffle = shuffle
	self.images = images
	self.labels = labels
	self.n_classes = n_classes
	self.on_epoch_end()

    def __len__(self):
	return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
		
	indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

	x_list = []
	y_list = []
	for i in indexes:
	  imd = image.load_img(images[i])
	  imarr = np.array(imd)
	  x_list.append(imarr)
	  y_list.append(labels[i])

	x = np.stack(x_list)
	a,yi = np.unique(ylist,return_inverse=True)
	y = np_utils.to_categorical(yi, self.n_classes)

	return x, y

    def on_epoch_end(self):
	
	self.indexes = np.arange(len(self.images))
	if self.shuffle == True:
	    np.random.shuffle(self.indexes)





class PredGenerator(keras.utils.Sequence):
	
    def __init__(self, images, batch_size=32):
	
	self.batch_size = batch_size
	self.images = images
	self.indexes = np.arange(len(self.images))


    def __len__(self):
	return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
		
	indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

	x_list = []

	for i in indexes:
	  imd = image.load_img(images[i])
	  imarr = np.array(imd)
	  x_list.append(imarr)


	x = np.stack(x_list)

	return x

	
	
	
	
	
	
