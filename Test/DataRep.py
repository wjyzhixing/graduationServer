#12 Mar 2017
#In this file we are going to prepare data for CNN in different formats


from numpy import *
import numpy as np
import sys
import scipy.io as spio
from attention import Attention,myFlatten


np.set_printoptions(threshold=np.inf)

#------------- Function Definition -------------
def FreqMat():
	ntype=['A', 'C', 'G', 'T']
	nuc=[]
	
	for n in ntype:
		nuc.append(n)
		
	for n in ntype:
		for m in ntype:
			nuc.append(n+m)
			
	for n in ntype:
		for m in ntype:
			for x in ntype:
				nuc.append(n+m+x)
				
	return nuc
	


#In this function we are going to add spaces between sequences' nt to make each sequence 
#look like a sentence of words and the representation will be overlapped but all the words
#in the same sentence
#This just like the idea behind BioVec
def Data2Text(posFile,negFile,WordLength,tag):

	WordLength=int(WordLength)
	FileName=tag+'_Data2Text_word%d.txt' %WordLength
	
	#load data from files
	PosSamples = list(open(posFile).readlines())
	PosSamples = [s.strip() for s in PosSamples] #remove \n

	NegSamples = list(open(negFile).readlines())
	NegSamples = [s.strip() for s in NegSamples] #remove \n
	
	print('PosSamples= '+str(len(PosSamples)))
	print('NegSamples= '+str(len(NegSamples)))
		
	#add space between the sequence to convert it into sentence of words
	PosSentences = []
	NegSentences = []
	
	#-1 because of the label at the end of the samples 
	for s in PosSamples:
		x=''
		for i in range(0,len(s)-1):
			#check if it is the last word
			if i==len(s)-1-WordLength:
				x=x+s[i:i+WordLength]
				break
			else:
				x=x+s[i:i+WordLength]+" "
		PosSentences.append(x)

	for s in NegSamples:
		x=''
		for i in range(0,len(s)-1):
			#check if it is the last word
			if i==len(s)-1-WordLength:
				x=x+s[i:i+WordLength]
				break
			else:
				x=x+s[i:i+WordLength]+" "
		NegSentences.append(x)
		
	x='Data/pos_'+FileName
	np.savetxt(x,PosSentences,fmt='%s')
	y='Data/neg_'+FileName
	np.savetxt(y,NegSentences,fmt='%s')	
	
	#Split by words
	data = PosSentences + NegSentences
	
	z='Data/data_'+FileName
	np.save(z,data)
	
	#Generate labels
	PosLabels = [[0, 1] for _ in PosSentences]
	NegLabels = [[1, 0] for _ in NegSentences]
	
	labels = np.concatenate([PosLabels, NegLabels])

	m='Data/labels_'+FileName
	np.save(m,labels)

	
	return [x,len(PosSentences),y,len(NegSentences),data,labels]
	
	
	


#In this function we are going to convert each sequence into image-like representation in 
#which each sequence will be represented as a matrix where the rows is the sequence and the
#columns are tri-nucleotides 
#------------- one-hot vector with 1 channel in 2D rep for Tri-nucleotides -------------
def Data2Image(fileName):
	
	nuc=FreqMat()
	print(nuc)
	seqLength=41  #edit this to 41 for other data
	
	#http://stackoverflow.com/questions/8009882/how-to-read-large-file-line-by-line-in-python
	with open(fileName) as f:
		motifCount=sum(1 for _ in f) #find number of samples
		print(motifCount)
		a=np.zeros((motifCount,seqLength-2,64)) #matrix initializations
		b=np.zeros(motifCount)
#		for i in range(1,motifCount,1):	
		for i in range(1,motifCount,2):	
			b[i]=1
			
		print('a shape = '+str(a.shape))
		print('b shape = '+str(b.shape))
		
		x='Data/'+'Targets.npy'
		
	#	print("b: ")
	#	print(b)
		np.save(x,b)
	#	np.savetxt("Data/Targets.txt",b)
	
	with open(fileName) as f:
		k=-1 #index for the samples
		for line in f:
			k=k+1
			for i in range(seqLength):
				for j in range(20,len(nuc)):
					if line[i:i+3]==nuc[j]:
						a[k,i,j-20]=1
		
	#	print("a: ")
	#	print(a)
		y='Data/'+'Inputs.npy'				
		np.save(y,a)
#		np.savetxt("Data/Inputs.txt",a)

	return [a, y, b, x]



