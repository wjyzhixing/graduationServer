#13/Dec/2017
#In this function we are going to use the trained models to test other genomes



from __future__ import division
from __future__ import print_function

from numpy import *
import numpy as np

#--------------- Fixing the seed --------------

seed = 14
np.random.seed(seed)

import sys
import os

#If you would like to use GPU un-comment the following line
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu1,floatX=float32,exception_verbosity=high, dnn.conv.algo_bwd_filter=deterministic, dnn.conv.algo_bwd_data=deterministic"

#If you don't have GPU un-comment the following line
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,floatX=float32,exception_verbosity=high, dnn.conv.algo_bwd_filter=deterministic, dnn.conv.algo_bwd_data=deterministic"


#--------------- Import libraries --------------

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D, Convolution2D
from keras.utils import np_utils

from attention import Attention,myFlatten

from decimal import Decimal

from keras import backend as k
#k.set_image_dim_ordering('th')
#version change
k.set_image_data_format('channels_first')


from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


import scipy.io as spio
import os.path

import DataRep

import matplotlib.pyplot as plt

#--------------- Arguments Parser --------------

import argparse

parser = argparse.ArgumentParser(description='Parameters to be used for models testing.')
parser.add_argument('--inputFile', help='Input file', required=True)
parser.add_argument('--inputModel', help='Input model', required=True)
#parser.add_argument('--DataName', help='Data file name', required=True)
#parser.add_argument('--FileName', help='File names tag', required=True)
parser.add_argument('--Shuffle', help='Shuffle samples', required=False, default=False)

args = parser.parse_args()

# 0--->正例
# 1--->负例

#----------------------------- functions Definitions -------------------------------------


def load_data(FileName):

	FileName2='Data/Inputs.npy' 
	FileName3='Data/Targets.npy' 

	#if os.path.isfile(FileName2) & os.path.isfile(FileName3):
	#	X=np.load(FileName2)
	#	Y=np.load(FileName3)
	#else:
	[X, x, Y, y]=DataRep.Data2Image(args.inputFile)

	
	#-----------Data further processing------------
	
	X=X.astype('float32')	
	Y=Y.astype(int)
		
	

	channel= 1
	samples=X.shape[0]
	rows=X.shape[1]
	cols=X.shape[2]
	indices = np.arange(X.shape[0])
	#In case of all variant data we have to shuffle so no consequence variants
	if args.Shuffle==True:
		np.random.shuffle(indices)
		X = X[indices]
		Y = Y[indices]
	
	print('samples: ', samples)
	print('rows: ', rows)
	print('columns: ', cols)
	print('channels: ', channel)
	print('data original shape: '+str(X.shape)+'\n')
	
	return [X, Y,samples,channel,rows,cols]
	
	
	
#-------------------------------- main function ------------------------------------------

if __name__ == "__main__":
	#--------------- Define some parameters --------------

	nb_classes=2
	nb_epoch = 100
	val_split = 0.2
	pool_size = (1,2) 
	batch_size = 16
	optimizer = 'Adadelta'
	init_mode = 'zero'
	activation = 'tanh'
	filter = 50
	filterL = 30
	filterW = 32
	neurons = 32
	dropout_rate = 0.1



	#------------- data Processing-------------
	
	[X, Y,samples,channel,rows,cols] = load_data(args.inputFile)

	
	#------------- start classification -------------

#	X = X.reshape(X.shape[0], channel, rows, cols)
	Y = np_utils.to_categorical(Y, nb_classes)
	test_label_rep = Y
	print('data re-shape: '+str(X.shape)+'\n')
	

	#--------------- Creating the model --------------
	print("\nLoading the pre-trained model ...\n")
	
	model=load_model(args.inputModel, custom_objects={'Attention':Attention,'myFlatten':myFlatten})
	
	model.summary()
	model.get_config()

	# evaluate the model	  
	scores = model.evaluate(X, Y, verbose=1)
	pred_dataAll = model.predict(X)
	pred_data = argmax(pred_dataAll,axis=1)
	print(pred_data)
	
	FileName2 = 'TestingFiles/test_label_rep.txt'
	np.savetxt(FileName2,test_label_rep)
	FileName3 = 'TestingFiles/pred_data.txt' 
	np.savetxt(FileName3,pred_data)

#	读取文件
	lines = []
	with open('../public/sequences/SequenceTemp.txt','r') as file_to_read:
	  while True:
	    line = file_to_read.readline()
	    if not line:
	      break
	    line = line.strip('\n')
	    lines.append(line)

	lines = list(map(list,zip(lines)))
	print(lines)

#	数值大小
	pred_value = []
	j = 0
	for i in pred_data:
	  pred_value.append(str(format(pred_dataAll[j,i],'.7f')))
	  j = j + 1
	
	pred_value = list(map(list,zip(pred_value)))
	print(pred_value)
	
#	是否存在
	result = []
	for i in pred_data:
	  if(i == 0):
	    result.append("Yes")
	  else:
	    result.append("No")
	
	result = list(map(list,zip(result)))
	print(result)
	
#	合并
	out = np.hstack((lines,pred_value,result))
	print(out)
#	into document
	
	
	#------------- Results -------------

	cm1 = confusion_matrix(np.argmax(test_label_rep,axis=1), pred_data)
	average_precision = average_precision_score(np.argmax(test_label_rep,axis=1), pred_data)
	precision, recall, _ = precision_recall_curve(np.argmax(test_label_rep,axis=1), pred_data)
	total1=sum(sum(cm1))
	#accuracy1=(cm1[0,0]+cm1[1,1])/total1
	#sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
	#specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])

	#print('Confusion Matrix : \n', cm1)
	#print ('Accuracy : ', accuracy1)
	#print('Sensitivity : ', sensitivity1)
	#print('Specificity : ', specificity1)
	#print ('AUC: '+str(roc_auc_score(np.argmax(test_label_rep,axis=1), pred_data)))
	#print ('AUPR: '+str(average_precision))
	#print ('Clasification report:\n', classification_report(np.argmax(test_label_rep,axis=1), pred_data))
	#print ('\nprecision score:',precision_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
	#print ('\nrecall score:',recall_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
	#print ('\nf1 score:',f1_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
	#print ('\ncohen kappa score:',cohen_kappa_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
	#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	FileNameR = 'Results/Results.txt' 
	Result_file=open(FileNameR,'w')
	for i in out:
  	  Result_file.write("%s\n" %(i))
	  
	FileNameR2 = 'Results/Results2.txt' 
	Result_file2=open(FileNameR2,'w')
	for i in out:
	  Result_file2.write("%s,%s,%s\n" %(i[0],i[1],i[2]))
	  
#	Result_file.write("%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))
#	Result_file.write("Sensitivity: %.2f%%\n" %( sensitivity1*100))
#	Result_file.write("Specificity: %.2f%%\n" %(specificity1*100))
#	Result_file.write('AUPR: '+str(average_precision))
#	Result_file.write('\nAUC: '+str(roc_auc_score(np.argmax(test_label_rep,axis=1), pred_data)))
	Result_file.close()	

	#------------- Plotting -------------
'''
	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
	PlotFileName = 'Plots/AUC.pdf' 
	plt.savefig(PlotFileName)
'''