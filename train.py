"""
Author : Avineil Jain 
This code is the main training code for training our model of Faster RCNN 
Some of the code in train_faster_rcnn() has been inspired from https://github.com/yhenon/keras-frcnn
"""

from __future__ import division
import random
import time
import numpy as np
import pickle

from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config
from keras.utils import generic_utils

import tensorflow as tf
from keras.applications.resnet50 import ResNet50
import keras_frcnn.rpn as rpn_network  
import keras_frcnn.faster_rcnn_classifier as classifier_frcnn  

#Will be using the VOC training data here
from keras_frcnn.pascal_voc_parser import get_data
import keras_frcnn.image_helpers as image_helpers

from keras.backend.tensorflow_backend import set_session
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth=True
set_session(tf.Session(config=config_gpu))

num_anchors=9
num_rois = 32

#--------------------------------------------------FUNCTIONS--------------------------------------------------

def create_faster_rcnn_model(img_input,roi_input):
	#Pre-Trained ResNet50 model used with ImageNet weights
	shared_model = ResNet50(include_top=False,input_tensor=img_input)
	print("Resnet50 Loaded")
	shared_layers = shared_model.get_layer('activation_40').output
	rpn = rpn_network.network(shared_layers, num_anchors)
	classifier = classifier_frcnn.network(shared_layers, roi_input, num_rois, nb_classes=21)

	#Note that RPN and the classifier model share the ResNet50 weights
	#They will be updated while training and hence, we need to define the model taking the image as input
	rpn_model = Model(inputs=img_input,outputs=rpn[:2])
	print("Regional Proposal Network Created")
	classifier_model = Model(inputs=[img_input,roi_input],outputs=classifier)
	print("Classifier Created")
	#This is a model that holds both the RPN and the classifier, used to load/save weights for the models
	combined_model = Model([img_input, roi_input], rpn[:2] + classifier)

	return rpn_model,classifier_model,combined_model


def compile_models(rpn_model,classifier_model,combined_model):
	print("Compiling Models...")
	#Models use custom loss functions which are written and standard Adam Optimizer
	rpn_model.compile(optimizer=Adam(lr=1e-5), loss=[rpn_network.loss_cls, rpn_network.loss_regr])
	classifier_model.compile(optimizer=Adam(lr=1e-5), loss=[classifier_frcnn.loss_cls, classifier_frcnn.loss_regr], metrics={'dense_class_21': 'accuracy'})
	combined_model.compile(optimizer='sgd', loss='mae')
	print("Models Compiled...")


def set_config(augment,num_rois):
	C = config.Config()
	if augment==True:
		C.use_horizontal_flips = True
		C.use_vertical_flips = True
		C.rot_90 = True

	C.num_rois = num_rois

	with open("config.pickle", 'wb') as config_f:
		pickle.dump(C,config_f)
		print('Config has been written to config.pickle')

	return C


def get_image_data(train_path):
	#The pascal_voc_parser is a great script to extract out the data and the information!
	all_imgs, classes_count, class_mapping = get_data(train_path)
	if 'bg' not in classes_count:
		classes_count['bg'] = 0
		class_mapping['bg'] = len(class_mapping)

	random.shuffle(all_imgs)
	train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
	val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

	return train_imgs,val_imgs,classes_count,class_mapping


"""
The training done in the project is something called Joint Training. It reduces the actual training time by a factor of 2 
There are 4 methods suggested in the paper, I recommend looking at them 
However, we have implemented the Joint Training, which jointly trains the RPN and the classifier network 
First trains the RPN, then uses its prediction and trains the classifier! 
Thus, as the predictions of RPN starts becoming more and more accurate, the classifier will perform better as well!
"""
def train_faster_rcnn(rpn_model,classifier_model,combined_model,num_epochs,C,train_imgs,val_imgs,classes_count,class_mapping):
	epoch_length = 1000
	#Ideally should be trained for 1000 epochs atleast, however it would take a week of GPU training! :O 
	num_epochs = num_epochs
	iter_num = 0

	losses = np.zeros((epoch_length, 5))
	rpn_accuracy_rpn_monitor = []
	rpn_accuracy_for_epoch = []
	start_time = time.time()

	best_loss = np.Inf

	#The data generator generates the required random image and its RPN output 
	data_generator = rpn_network.generate_train_data(train_imgs, classes_count, C, image_helpers.get_img_output_length, K.image_dim_ordering(), mode='train')

	print("Starting Training.....")

	for epoch_num in range(num_epochs):

		progbar = generic_utils.Progbar(epoch_length)
		print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
		while(True):
			try:
				if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
					mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
					rpn_accuracy_rpn_monitor = []
					print('Mean overlapping boxes : {}'.format(mean_overlapping_bboxes))
					if mean_overlapping_bboxes == 0:
						print('Something is wrong with RPN, check the settings')

				#Getting one random image
				X, Y, img_data = next(data_generator)
				#Training the RPN on the image
				loss_rpn = rpn_model.train_on_batch(X, Y)
				#Using the predictions of RPN for the classifier
				rpn_cls, rpn_regr = rpn_model.predict_on_batch(X)

				#Converting the RPN predictions to ROI's
				R = rpn_network.rpn_to_roi(rpn_cls, rpn_regr, C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
				
				#Using the predicted ROI's to calculate the ground truth labels for the classifier
				X2, Y1, Y2, IouS = classifier_frcnn.calc_classifier_ground_truth(R, img_data, C, class_mapping)

				if X2 is None:
					rpn_accuracy_rpn_monitor.append(0)
					rpn_accuracy_for_epoch.append(0)
					continue

				#We look at the last column of Y1 to find the samples with class "bg"
				neg_samples = np.where(Y1[0, :, -1] == 1)
				pos_samples = np.where(Y1[0, :, -1] == 0)

				if len(neg_samples) > 0:
					neg_samples = neg_samples[0]
				else:
					neg_samples = []

				if len(pos_samples) > 0:
					pos_samples = pos_samples[0]
				else:
					pos_samples = []
				
				rpn_accuracy_rpn_monitor.append(len(pos_samples))
				rpn_accuracy_for_epoch.append((len(pos_samples)))

				#We do it for 32 ROI's max for training
				# Its possible that we might have very few positive samples early on, when the RPN still has to train a lot
				if len(pos_samples) < C.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
				#Filling the rest with negative samples
				try:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
				except:
					selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

				sel_samples = selected_pos_samples + selected_neg_samples

				#Now training the classifier
				loss_class = classifier_model.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

				losses[iter_num, 0] = loss_rpn[1]
				losses[iter_num, 1] = loss_rpn[2]

				losses[iter_num, 2] = loss_class[1]
				losses[iter_num, 3] = loss_class[2]
				losses[iter_num, 4] = loss_class[3]

				iter_num += 1

				progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
										  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])

				if iter_num == epoch_length:
					loss_rpn_cls = np.mean(losses[:, 0])
					loss_rpn_regr = np.mean(losses[:, 1])
					loss_class_cls = np.mean(losses[:, 2])
					loss_class_regr = np.mean(losses[:, 3])
					class_acc = np.mean(losses[:, 4])

					mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
					rpn_accuracy_for_epoch = []

					if C.verbose:
						print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
						print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
						print('Loss RPN classifier: {}'.format(loss_rpn_cls))
						print('Loss RPN regression: {}'.format(loss_rpn_regr))
						print('Loss Detector classifier: {}'.format(loss_class_cls))
						print('Loss Detector regression: {}'.format(loss_class_regr))
						print('Elapsed time: {}'.format(time.time() - start_time))

					curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
					iter_num = 0
					start_time = time.time()

					if curr_loss < best_loss:
						if C.verbose:
							print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
						best_loss = curr_loss
						combined_model.save_weights("my_frcnn_try.hdf5")

					break

			except Exception as e:
				print('Exception: {}'.format(e))
				continue

	print('Training complete, exiting.')



#--------------------------------------------------------MAIN-----------------------------------------------------------
def main():

	C = set_config(False,32)
	print("Fetching Data")
	train_imgs,val_imgs,classes_count,class_mapping = get_image_data("VOCdevkit")
	print("Data Fetched...")
	#---------------------------------------
	#We want it to work for variable sized images
	input_shape_img = (None, None, 3)
	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(None, 4))

	rpn_model,classifier_model,combined_model = create_faster_rcnn_model(img_input,roi_input)
	compile_models(rpn_model,classifier_model,combined_model)
	
	#----------------------------------------
	train_faster_rcnn(rpn_model,classifier_model,combined_model,200,C,train_imgs,val_imgs,classes_count,class_mapping)
	

if __name__ == '__main__':
	main()
