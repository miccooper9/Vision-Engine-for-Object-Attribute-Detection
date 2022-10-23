from scipy import stats
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from keras.preprocessing import image
import cv2
import opts



def parse_xml(opt):
	
	
	#get images from annotations
	annotations = opt['train_annotations']	
	tree = ET.parse(annotations)
	root = tree.getroot()
	
	
	im2bbs = dict()
	
	for im in root.findall("image"):
		im_id = im.get('id')
		im_file = str(im_id) + '.' + opt['format']
		#print(im_id)
		bxs = []
		for box in im.findall("box"):

			label = box.get('label')
			xt = box.get('xtl')
			yt = box.get('ytl')
			xb = box.get('xbr')
			yb = box.get('ybr')
			
						
			blist = []
			
			blist.append(float(xt))
			blist.append(float(yt))
			blist.append(float(xb))
			blist.append(float(yb))
			blist.append(label)
			
			for attr in box.findall(".//*[@name='has_safety_helmet']"):
				atb  = attr.get('name')
				val = attr.text
				blist.append(val)
				
			for attr in box.findall(".//*[@name='mask']"):
				atb  = attr.get('name')
				val = attr.text
				blist.append(val)
			

			bxs.append(blist)
		im2bbs[im_file] = bxs
				
	print("number of images in annotation file: ", len(im2bbs))
	
	
	
	#get images from directory
	imagelist = set() 
	dirName = opt['train_images']

	for (dirpath, dirnames, filenames) in os.walk(dirName):
		for file in filenames:
			imagelist.add(file)
			#print(file)

	print("number of train images : ", len(imagelist))
	
	
	
	#select images with annotations available 
	if (len(imagelist) != len(im2bbs)) :
		
		print("taking largest common subset...")
		for ii in im2bbs:
			if not (ii in imagelist):
				im2bbs.pop(ii)
				
	print("number of images with labels : ", len(im2bbs))
	
	
	return im2bbs


def gen_obj_detector_samples(im2boxlabels, opt):
	
	
	obj_annotations =[]
	
	for img, labels in im2boxlabels.items():


		filepath = opt['train_images'] + img 
		imf = image.load_img(filepath)
		imarr = np.array(imf)
		
		#resize images to 640 x 480 for faster rcnn
		resized_im = cv2.resize(imarr, (640,480), interpolation=cv2.INTER_CUBIC)
		
		#save resized image
		writedir =  './keras-frcnn-master/'+'train_images' 
		if not os.path.exists(writedir):
			os.makedirs(writedir)
		writepath = writedir + '/' + img
		image.save_img(writepath,resized_im)
		
		
		#generate labels
		impath = 'train_images/' + img
		for bb in labels:
			y1 = int(bb[1])
			y2 = int(bb[3])
			x1 = int(bb[0])
			x2 = int(bb[2])
			
			x1_r = str(int((x1*(480))/(imarr.shape[1])))
			x2_r = str(int((x2*(480))/(imarr.shape[1]))) 
			y1_r = str(int((y1*(640))/(imarr.shape[0])))
			y2_r= str(int((y2*(640))/(imarr.shape[0])))
			obj = bb[4]
			if(obj!='head'):
				obj='nothead'

			f = impath + ',' + x1_r + ','+ y1_r + ','+ x2_r + ','+ y2_r + ',' + obj
			obj_annotations.append(f)
		
	#save annotation file
	data = pd.DataFrame(obj_annotations)
	dirName = r'keras-frcnn-master/';
	fpath = dirName + 'annotate.txt'
	data.to_csv(fpath, header=None, index=None, sep=' ')
	
	
def gen_classifier_samples(im2boxlabels, opt):
	
	
	attr_annotations = []
	
	for img, labels in im2boxlabels.items():
		
		filepath = opt['train_images'] + img 
		imf = image.load_img(filepath)
		imarr = np.array(imf)
		
		count = 0
		
		for bb in labels:
			if(bb[4] == "head"):
				
				count = count + 1
				
				#crop and resize image
				rl = int(bb[1])
				ru = int(bb[3]) + 1
				cl = int(bb[0])
				cu = int(bb[2]) + 1
				cropped_im = imarr[rl:ru,cl:cu,:]
				resized_im = cv2.resize(cropped_im, (224,224), interpolation=cv2.INTER_CUBIC)
				
				#save resized and cropped image
				writedir =  './classifier_train_images' 
				if not os.path.exists(writedir):
					os.makedirs(writedir)
				writepath = writedir + '/' + str(count) + '_' + img
				image.save_img(writepath,resized_im)
				
				
				#classifier labels
				f = writepath + ',' bb[5] + ',' + bb[6] 
				attr_annotations.append(f)
				
	
	
	
				
	#save annotation file
	data = pd.DataFrame(attr_annotations)
	fpath = 'classifier_annotations.txt'
	data.to_csv(fpath, header=None, index=None, sep=' ')
				
	
				
				







if __name__ == '__main__':


	opt = opts.parse_opt()
	opt = vars(opt)



	im2boxlabels = parse_xml(opt)

	
	#generate training samples for frcnn
	gen_obj_detector_samples(im2boxlabels, opt)
	
	#generate training samples for classifier
	gen_classifier_samples(im2boxlabels, opt)
  	
  
  
  
  
  
  
  
  
  
