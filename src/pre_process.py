from scipy import stats
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from keras.preprocessing import image
import cv2
import opts


def pre_process(opt):
	
	dirName = opt['test_images']
	imagelist = []
	for (dirpath, dirnames, filenames) in os.walk(dirName):

		for file in filenames:
			imagelist.append(file)
			filepath = dirName + '/' + file 
			imf = image.load_img(filepath)
			imarr = np.array(imf)
			resized_im = cv2.resize(imarr, (640,480), interpolation=cv2.INTER_CUBIC)

			writepath =  './keras-frcnn-master/test_images/' + file
			image.save_img(writepath,resized_im)
	



if __name__ == '__main__':


	opt = opts.parse_opt()
	opt = vars(opt)
	
	pre_process(opt)
  
  
  
