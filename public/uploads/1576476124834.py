#6/Dec/2017
#In this function we are going to train the genome data of variant species using 1D-CNN

from __future__ import division
from __future__ import print_function

from numpy import *
import numpy as np

#--------------- Fixing the seed --------------

#For reproducibility 
seed = 14
np.random.seed(seed)

import sys
import os

#If you would like to use GPU un-comment the following line
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high, dnn.conv.algo_bwd_filter=deterministic, dnn.conv.algo_bwd_data=deterministic"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"	
#If you don't have GPU un-comment the following line
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,floatX=float32,exception_verbosity=high, dnn.conv.algo_bwd_filter=deterministic, dnn.conv.algo_bwd_data=deterministic"

#--------------- Import libraries --------------

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import MaxPooling1D, Convolution1D
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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

parser = argparse.ArgumentParser(description='Parameters to be used for 1D-CNN training.')
parser.add_argument('--PosInputFile', help='Positive input file', required=True)
parser.add_argument('--NegInputFile', help='Negative input file', required=True)
#parser.add_argument('--DataName', help='Data file name', required=True)
#parser.add_argument('--FileName', help='File names tag', required=True)

args = parser.parse_args()

#--------------- Define some parameters --------------


nb_classes=2
nb_epoch = 5
val_split = 0.2
pool_size = 2
embedding_dim = 20
WordLength=3
batch_size = 32
optimizer = 'Adadelta'
init_mode = 'zero'
activation = 'tanh'
filter = 50
filterW = 32
dropout_rate = 0.1


#------------- data definition-------------

[posFile, posCount, negFile, negCount, data, labels]=DataRep.Data2Text(args.PosInputFile,args.NegInputFile,WordLength)

labels=labels[:,0]

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

word_index = tokenizer.word_index
data = pad_sequences(sequences)

# mixing the data
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
X = data[indices]
Y = labels[indices]

sequence_length = np.shape(X)[1]

#------------- data preparation -------------

Y=Y.astype(int)
X=X.astype('float32')

print('data original shapes: '+str(X.shape)+'\n')
Y = np_utils.to_categorical(Y, nb_classes)
	
X_train, X_test, y_train, y_test, i_train, i_test  = train_test_split(X, Y, indices, test_size=0.25, random_state=seed)
	
test_label_rep = y_test
print('training samples: ', X_train.shape[0])
print('testing samples: ', X_test.shape[0])

	
#--------------- Creating the model --------------

print("\nCreating the model ...\n")
model=Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, input_length=sequence_length))

model.add(Convolution1D(nb_filter=filter, filter_length=filterW, init=init_mode, border_mode='same', bias='True', subsample_length=1)) 
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=pool_size)) 

model.add(Convolution1D(100, 8)) 
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_length=pool_size)) 
model.add(Dropout(dropout_rate))

model.add(Flatten())

model.add(Dense(256)) 
model.add(Activation(activation))
model.add(Dropout(dropout_rate))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()
model.get_config()

# Compile model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	
# Fit the model
filepath="Models/bestModel.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
hist=model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, validation_split=val_split, callbacks=[early_stopping,checkpoint])

FileName2 = 'Models/Model.h5'
model.save(FileName2) 
FileName3 = 'Models/Weights.h5'
model.save_weights(FileName3)

# evaluate the model	  
scores = model.evaluate(X_test, y_test, verbose=1)
scoreTr = model.evaluate(X_train, y_train, verbose=1)
pred_data = model.predict_classes(X_test)

FileName2 = 'TestingFiles/test_label_rep.txt'
np.savetxt(FileName2,test_label_rep)
FileName3 = 'TestingFiles/pred_data.txt'
np.savetxt(FileName3,pred_data)



#------------- Results -------------

cm1 = confusion_matrix(np.argmax(test_label_rep,axis=1), pred_data)
average_precision = average_precision_score(np.argmax(test_label_rep,axis=1), pred_data)
precision, recall, _ = precision_recall_curve(np.argmax(test_label_rep,axis=1), pred_data)
total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+cm1[1,1])/total1
sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])

print('\nTraining score:', scoreTr[0])
print('Training accuracy:', scoreTr[1])
print('Confusion Matrix : \n', cm1)
print ('Accuracy : ', accuracy1)
print('Sensitivity : ', sensitivity1)
print('Specificity : ', specificity1)
print ('AUC: '+str(roc_auc_score(np.argmax(test_label_rep,axis=1), pred_data)))
print ('AUPR: '+str(average_precision))
print ('Clasification report:\n', classification_report(np.argmax(test_label_rep,axis=1), pred_data))
print ('\nprecision score:',precision_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
print ('\nrecall score:',recall_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
print ('\nf1 score:',f1_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
print ('\ncohen kappa score:',cohen_kappa_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

FileNameR = 'Results/Results.txt'
Result_file=open(FileNameR,'w')
Result_file.write("%s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))
Result_file.write("Sensitivity: %.2f%%\n" %( sensitivity1*100))
Result_file.write("Specificity: %.2f%%\n" %(specificity1*100))
Result_file.write('AUPR: '+str(average_precision))
Result_file.write('\nAUC: '+str(roc_auc_score(np.argmax(test_label_rep,axis=1), pred_data)))
Result_file.close()	

#------------- Plotting -------------

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(average_precision))
PlotFileName = 'Plots/AUC.pdf'
plt.savefig(PlotFileName)

