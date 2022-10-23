import keras
from keras import Model
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



def parse_detector_output(opt):
	
	obj_pred_path = opt['object_preds']
	dataset = pd.read_csv(obj_pred_path)
	data = dataset.values
	dirName = opt['test_images']
	
	
	output = []
	impaths = []
	
	count = 0
	for bb in range(0,data.shape[0]):
		
		#read image
		img = data[bb][0]
		filepath = dirName + str(img)
		imd = image.load_img(filepath)
		imarr = np.array(imd)
		
		
		#calibrate predictions
		a = data[bb][2]
		b = data[bb][4]
		c = data[bb][1]
		d = data[bb][3]
		rl = int(a*224/640)
		ru = int(b*224/640)
		cl = int(c*224/480)
		cu = int(d*224/480)


		#crop and resize the desired predicted objects
		if (bb[5] == "head"):
			
			cropped_im = imarr[rl:ru,cl:cu,:]
			
			if cropped_im.shape[0]!=0 and cropped_im.shape[1]!=0:
				
				count = count + 1
				resized_im = cv2.resize(cropped_im, (224,224), interpolation=cv2.INTER_CUBIC)
		
				#save resized ands cropped object predictions
				writedir =  './pred_objects' 
				if not os.path.exists(writedir):
					os.makedirs(writedir)
				writepath = writedir + '/' + str(count) + '_' + img
				image.save_img(writepath,resized_im)
				impaths.append(writepath)


				o = str(img) + ',' + 'xtl=' + str(cl) + ',' + 'ytl=' + str(rl) + ',' + 'xbr=' + str(cu) + ',' + 'ybr=' + str(ru) 
				output.append(o)

				
	return output, impaths



def gen_attribute_preds(opt, img_paths, pred_objs):
	
	#load the trained classifier networks
	maskclassifier = keras.models.load_model('maskclassifier.h5')
	helmetclassifier = keras.models.load_model('helmetclassifier.h5')
	
	#get predictions
	test_generator = PredGenerator(img_paths, 32)
	mask = maskclassifier.predict(test_generator)
	helmet = helmetclassifier.predict(test_generator)
	mpred = np.argmax(mask, axis = 1)
	hpred = np.argmax(helmet, axis = 1)

	#label map
	maskdict = {0:'invisible',1:'no',2:'wrong',3:'yes'}
	helmetdict = {0:'no',1:'yes'}



	for m in range(0,mask.shape[0]):
		output[m] =  output[m]  + ',' +'mask=' + maskdict[mpred[m]] + ','+ 'helmet=' + maskdict[hpred[m]]

	#save final output : img,x1,y1,x2,y2,mask?,helmet?
	data = pd.DataFrame(output)
	fpath = opt['output']
	data.to_csv(fpath, header=None, index=None, sep=' ')
	
		






if __name__ == '__main__':


	opt = opts.parse_opt()
	opt = vars(opt)
	
	
	predicted_objs, obj_imgs = parse_detector_output(opt)
	
	gen_attribute_preds(opt, obj_imgs, predicted_objs)
	
	
	
	
	
