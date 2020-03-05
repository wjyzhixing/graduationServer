#6/Dec/2017
#In this function we are going to train the genome data of variant species using 2D-CNN



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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"	
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high, dnn.conv.algo_bwd_filter=deterministic, dnn.conv.algo_bwd_data=deterministic"

#If you don't have GPU un-comment the following line
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,floatX=float32,exception_verbosity=high, dnn.conv.algo_bwd_filter=deterministic, dnn.conv.algo_bwd_data=deterministic"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#--------------- Import libraries --------------

import keras
import keras.layers.core as core
import keras.models as models
import keras.layers.convolutional as conv
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, merge, concatenate, LSTM, Bidirectional
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D, Conv2D, Conv1D, MaxPooling1D, Convolution1D, Convolution2D
from keras.regularizers import l1, l2
from keras.utils import np_utils
from keras.optimizers import adam, SGD
from keras.initializers import RandomUniform
from attention import Attention,myFlatten


from keras import backend as k

#th->theano（100,3,16,32）first->tensorflow（100,16,32,3）tf last
#k.set_image_dim_ordering('th')
k.set_image_data_format('channels_first')

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import scipy.io as spio
import os.path

import DataRep

import matplotlib.pyplot as plt

#--------------- Arguments Parser --------------

import argparse

parser = argparse.ArgumentParser(description='Parameters to be used for 2D-CNN training.')
parser.add_argument('--inputFile', help='Input file', required=True)
parser.add_argument('--Shuffle', help='Shuffle samples', required=False, default=False)

args = parser.parse_args()


#--------------- Define some parameters --------------

filtersize1=1
filtersize2=9
filtersize3=10
filter1=200
filter2=150
filter3=200
dropout1=0.4
dropout2=0.4
dropout4=0.4
dropout5=0.4
dropout6=0.4
L1CNN=0
nb_classes=2
batch_size=128
actfun="relu"; 
optimization='adam';
attentionhidden_x=10
attentionhidden_xr=8
attention_reg_x=0.151948
attention_reg_xr=2
dense_size1=149
dense_size2=8
dropout_dense1=0.298224
dropout_dense2=0
epochs=500
val_split = 0.1

#------------- data definition -------------

FileName2='Data/Inputs.npy'
FileName3='Data/Targets.npy'

#In case the file representation was generated previously 
if os.path.isfile(FileName2) & os.path.isfile(FileName3): 
	X=np.load(FileName2)
	Y=np.load(FileName3)
else:
	[X, x, Y, y]=DataRep.Data2Image(args.inputFile)
	

#------------- data preparation -------------

Y=Y.astype(int)
X=X.astype('float32')

channel= 1
rows=X.shape[1]
cols=X.shape[2]
#np.arange()函数返回一个有终点和起点的固定步长的排列
indices = np.arange(X.shape[0])

#In case of all variant data we have to shuffle so no consequence variants
if args.Shuffle==True:
	np.random.shuffle(indices)
	X = X[indices]
	Y = Y[indices]
	
print('rows: ', rows )
print('columns: ', cols)
print('channels: ', channel )

print('data original shapes: '+str(X.shape)+'\n')

#‘channels_first’模式下，输入形如（samples,channels，rows，cols）的4D张量
#X = X.reshape(X.shape[0], channel, rows, cols)

#将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以categorical_crossentropy为目标函数的模型中.
#Y的规则转换为[0,1] or [1,0]
'''
[0,1,0,1...0,1]转为两个类别
转onehot [
	[1,0]
	[0,1]
]
'''
Y = np_utils.to_categorical(Y, nb_classes)
print('data re-shape: '+str(X.shape)+'\n')
	
X_train, X_test, y_train, y_test, i_train, i_test  = train_test_split(X, Y, indices, test_size=0.1, random_state=seed)
dataFile=open(args.inputFile,'r')
data=dataFile.readlines()

TestFileName = 'TestingFiles/test_data.txt' 
TestFile = open(TestFileName, "w")

for i in i_test:
	TestFile.write(data[i])

dataFile.close()
TestFile.close()

test_label_rep = y_test
print('training samples: ', X_train.shape[0])
print('testing samples: ', X_test.shape[0])


#--------------- Creating the model --------------
print("\nCreating the model ...\n")
input = Input(shape=(rows,cols))
x = conv.Convolution1D(filter1, filtersize1,kernel_initializer='he_normal',kernel_regularizer= l1(L1CNN),padding="same")(input) 
x = Dropout(dropout1)(x)
x = Activation(actfun)(x)
x = conv.Convolution1D(filter2,filtersize2,kernel_initializer='he_normal',kernel_regularizer= l1(L1CNN),padding="same")(x)
x = Dropout(dropout2)(x)
x = Activation(actfun)(x)
x = conv.Convolution1D(filter3,filtersize3,kernel_initializer='he_normal',kernel_regularizer= l1(L1CNN),padding="same")(x)
x = Activation(actfun)(x)
x_reshape=core.Reshape((x._keras_shape[2],x._keras_shape[1]))(x)
x = Dropout(dropout4)(x)
x_reshape=Dropout(dropout5)(x_reshape)
decoder_x = Attention(hidden=attentionhidden_x,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_x)) # success  
decoded_x=decoder_x(x)
output_x = myFlatten(x._keras_shape[2])(decoded_x)
decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_xr))
decoded_xr=decoder_xr(x_reshape)
output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)
output=concatenate([output_x,output_xr])
output=Dropout(dropout6)(output)
output=Dense(dense_size1,kernel_initializer='he_normal',activation='relu')(output)
output=Dropout(dropout_dense1)(output)
output=Dense(dense_size2,activation="relu",kernel_initializer='he_normal')(output)
output=Dropout(dropout_dense2)(output)
out=Dense(nb_classes,kernel_initializer='he_normal',activation='sigmoid')(output)
cnn=Model(input,out)



#打印出模型概况
cnn.summary()
#返回包含模型配置信息的Python字典。模型也可以从它的config信息中重构回去
cnn.get_config()

# Compile model
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model		  
filepath="Models/bestModel.h5"
#设置监视值为val_acc，展示，有改进则保存当前模型 
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_weights_only='True')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
hist=cnn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=val_split, callbacks=[early_stopping, checkpoint])
#model.fit在运行结束后返回一个History对象，其中含有的history属性包含了训练过程中损失函数的值以及其他度量指标。
print(hist)

FileName2 = 'Models/Model.h5' 
cnn.save(FileName2) 
FileName3 = 'Models/Weights.h5' 
cnn.save_weights(FileName3)

# evaluate the model
#对模型进行评估看模型指标是否符合要求	  
#返回的是 损失值和你选定的指标值(accuracy)
scores = cnn.evaluate(X_test, y_test, verbose=1)
scoreTr = cnn.evaluate(X_train, y_train, verbose=1)
#对新的数据进行预测
pred_data = cnn.predict(X_test)
pred_data = argmax(pred_data,axis=1)

FileName2 = 'TestingFiles/test_label_rep.txt' 
np.savetxt(FileName2,test_label_rep)
FileName3 = 'TestingFiles/pred_data.txt' 
np.savetxt(FileName3,pred_data)

#------------- Results -------------

#混淆矩阵
cm1 = confusion_matrix(np.argmax(test_label_rep,axis=1), pred_data)
#precision_recall_curve只用于二分类中。而average_precision_score可用于二分类或multilabel指示器格式
average_precision = average_precision_score(np.argmax(test_label_rep,axis=1), pred_data)
precision, recall, _ = precision_recall_curve(np.argmax(test_label_rep,axis=1), pred_data)
total1=sum(sum(cm1))
accuracy1=(cm1[0,0]+cm1[1,1])/total1
sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])

print('\nTraining score(损失值):', scoreTr[0])
print('Training accuracy:', scoreTr[1])
print('Confusion Matrix : \n', cm1)
print ('Accuracy : ', accuracy1)
print('Sensitivity(敏感度) : ', sensitivity1)
print('Specificity(特异性) : ', specificity1)
#Receiver Operating Characteristic(ROC)曲线和AUC常被用来评价一个二值分类器的优劣
print ('AUC: '+str(roc_auc_score(np.argmax(test_label_rep,axis=1), pred_data)))
print ('AUPR: '+str(average_precision))
#显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。
print ('Clasification report:\n', classification_report(np.argmax(test_label_rep,axis=1), pred_data))
print ('\nprecision score:',precision_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
print ('\nrecall score:',recall_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
print ('\nf1 score:',f1_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
#Kappa系数是基于混淆矩阵的计算得到的模型评价参数
print ('\ncohen kappa score:',cohen_kappa_score(np.argmax(test_label_rep,axis=1), pred_data), '\n')
print("%s: %.2f%%" % (cnn.metrics_names[1], scores[1]*100))

FileNameR = 'Results/Results.txt' 
Result_file=open(FileNameR,'w')
Result_file.write("%s: %.2f%%\n" % (cnn.metrics_names[1], scores[1]*100))
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
