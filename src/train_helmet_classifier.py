import keras
from keras import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Lambda
from keras.layers import Conv1D, MaxPooling1D, Concatenate,UpSampling2D,GlobalAveragePooling2D
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
from keras.preprocessing import image
import cv2
import opts
import classifier_dataloader




def train(opt):


	data_path = './classifier_annotations.txt'
	
	images = []
	labels = []
	with open(data_path,'r') as f:
		for line in f:
			line_split = line.strip().split(',')
			(filename,attr_1,attr_2) = line_split
			images.append(filename)
			labels.append(attr_1)
	
	
	split_point = 0.8*(len(images))
	
	img_train = images[:split_point]
	labels_train = labels[:split_point]
	
	img_val = images[split_point:]
	labels_val = labels[split_point:]
	
	
	training_generator = DataGenerator(img_train, labels_train, 32, 2, True)
	validation_generator = DataGenerator(img_val, labels_val, 32, 2, True)



	#helmet classifier model
	inputim = Input(shape =(224,224,3))
	resout = ResNet50(weights='imagenet', include_top=False)(inputim)
	pool_global_average = GlobalAveragePooling2D(data_format='channels_last')(resout)
	drop = Dropout(0.5)(pool_global_average)
	dense = Dense(5, activation='softmax')(drop)
	drop1 = Dropout(0.5)(dense)
	dense1 = Dense(2, activation='softmax')(drop1)
	maskclassifier = Model(inputs = inputim, outputs = dense1)


	print(maskclassifier.summary())

	maskclassifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	#train
	maskclassifier.fit(training_generator, validation_data = validation_generator, epochs = 100, verbose =1)

	#save the model
	maskclassifier.save('helmetclassifier.h5')




if __name__ == '__main__':


	opt = opts.parse_opt()
	opt = vars(opt)
	
	train(opt)
	
