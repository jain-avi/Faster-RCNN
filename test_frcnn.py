"""
Author : Sravan Patchala
Code for testing the model, with images taken from the test directory
"""

from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import keras_frcnn.rpn as rpn_network  
import keras_frcnn.faster_rcnn_classifier as classifier_frcnn  
import keras_frcnn.image_helpers as image_helpers

import train

from keras.applications.resnet50 import ResNet50
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Required while running on a GPU
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth=True
set_session(tf.Session(config=config_gpu))
sys.setrecursionlimit(40000)


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

# Function to get the bounding boxes and probabilities from the ROIs
# Parts from https://github.com/yhenon/keras-frcnn
def get_bbox_prob(R,F):
	
	# Spatial pyramid pooling on the proposed regions
	bboxes = {}
	probs = {}

	bbox_threshold = 0.9

	for index in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*index:C.num_rois*(index+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if index == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[predicted_class, predicted_regr] = classifier_model.predict([F, ROIs])

		for class_index in range(predicted_class.shape[1]):

			if np.max(predicted_class[0, class_index, :]) < bbox_threshold or np.argmax(predicted_class[0, class_index, :]) == (predicted_class.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(predicted_class[0, class_index, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, class_index, :]

			cls_num = np.argmax(predicted_class[0, class_index, :])
			try:
				(tx, ty, tw, th) = predicted_regr[0, class_index, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = image_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(predicted_class[0, class_index, :]))
	return bboxes,probs

# Funtion to label images 
def labelled_img(img,x1,y1,x2,y2,key,textLabel):

	cv2.rectangle(img,(x1, y1), (x2, y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
	(shiftVal,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.7,1)
	textOrg = (x1, y1-0)
	cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+shiftVal[0] + 5, textOrg[1]-shiftVal[1] - 5), (0, 0, 0), 2)
	cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+shiftVal[0] + 5, textOrg[1]-shiftVal[1] - 5), (255, 255, 255), -1)
	cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)
	return img


# Initialisations 
num_rois = 32
num_features = 1024
num_anchors = 9

# Load the config file, with parsed data
with open("config_final.pickle", 'rb') as config_file:
	C = pickle.load(config_file)

class_mapping = C.class_mapping
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

shared_model = ResNet50(include_top=False,input_tensor=img_input)
print("Resnet50 Loaded")
shared_layers = shared_model.get_layer('activation_40').output
rpn = rpn_network.network(shared_layers, num_anchors)
classifier = classifier_frcnn.network(feature_map_input, roi_input, num_rois, nb_classes=21)
rpn_model = Model(inputs=img_input,outputs=rpn)
classifier_model = Model(inputs=[feature_map_input,roi_input],outputs=classifier)

print('Loading weights from my_frcnn_try.hdf5')
rpn_model.load_weights("my_frcnn_try.hdf5", by_name=True)
classifier_model.load_weights("my_frcnn_try.hdf5", by_name=True)

rpn_model.compile(optimizer='sgd', loss='mse')
classifier_model.compile(optimizer='sgd', loss='mse')

img_path = "test"

for img_idx, img_name in enumerate(sorted(os.listdir(img_path))):
	
	# Do not check all other files
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue

	print(img_name)

	filepath = os.path.join(img_path,img_name)
	img = cv2.imread(filepath)

	# Resizes image so that min_dim is as in Config
	# Permutes the channels from BGR to RBG
	X, ratio = image_helpers.format_img(img, C)
	X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = rpn_model.predict(X)

	# Finds the ROIs and gives coordinates of the 
	R = rpn_network.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# Get the bounding boxes and the probabilities
	(bboxes,probs) = get_bbox_prob(R,F)

	objects_in_img = []

	for key in bboxes:
		bbox = np.array(bboxes[key])
		
		# Find the best bounding box and adjusted probabilities after non-max suppression
		adjusted_boxes, adjusted_probs = image_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		
		for index in range(adjusted_boxes.shape[0]):
			objects_in_img.append((key,100*adjusted_probs[index]))
			(x1, y1, x2, y2) = adjusted_boxes[index,:]
			label_box = '{}: {}'.format(key,int(100*adjusted_probs[index]))
			(x1_new, y1_new, x2_new, y2_new) = get_real_coordinates(ratio, x1, y1, x2, y2)
			# Put the labels and boxes on the image
			img = labelled_img(img,x1_new,y1_new,x2_new,y2_new,key,label_box)

	print(objects_in_img)
	cv2.imwrite('test_results/{}.bmp'.format(img_idx),img)
