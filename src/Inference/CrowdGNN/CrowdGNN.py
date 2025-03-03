#import lib
import os
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from numpy.linalg import *
from collections import Counter
import dataGenerate
import myMethod 

#super parameter
#classNum, workerNum, itemNum, Miu, Theta, labelPerWorker

#all data should be converted into one-Hot Code


class CrowdGNN_Model():
	"""docstring for CrowdGNN_Model"""
	def __init__(self, config, graph, depth):
		#config must contain classNum
		#depth is an integar
		super(CrowdGNN_Model, self).__init__()
		
		self.classNum = int(config[0])
		self.workerNum = int(config[1])
		self.graph = graph
		self.depth = depth
		self.GCC = myMethod.graphConvolutionCoefficient(self.graph)
		self.ConvWeight = np.ones((self.depth,self.classNum,self.workerNum,self.workerNum),dtype=float)
		self.ConncWeight = np.ones((self.classNum,self.workerNum),dtype = float)

	def forward(self,data):
		#raw Data
		I = np.zeros((self.depth+1,self.workerNum,self.classNum),dtype = float)
		I[0] = data * 1
		#Convolution Layers
		for i in range(self.depth):
			I[i+1] = self.Relu(self.EwLayer(I[i],i))
		#Connection Layers
		X = self.denseLayer(I[self.depth])
		#Output
		opt = self.softmax(X)
		return opt,X,I

	def Relu(self,X):
		for i in range(self.workerNum):
			for j in range(self.classNum):
				if X[i,j] < 0:
					X[i,j] = 0
		return X

	def EwLayer(self,data,layercnt):
		ret = data * 1
		for i in range(self.classNum):
			ret[:,i] = (self.ConvWeight[layercnt,i] * self.GCC).dot(ret[:,i])
		return ret

	def softmax(self,vec):
		sm = 0
		ret = vec * 1
		for i in range(self.classNum):
			ret[i] = math.exp(vec[i])
			sm += ret[i]
		return ret/sm

	def denseLayer(self,data):
		ret = np.zeros(self.classNum)
		for i in range(self.classNum):
			ret[i] = self.ConncWeight[i].dot(data[:,i])
		return ret

	def backward(self,data,truelabel):
		#forward propagation
		opt,X,I = self.forward(data)
		#zero gradients
		grd_ConncWeight =  np.zeros((self.classNum,self.workerNum),dtype = float)
		grd_ConvWeight = np.zeros((self.depth,self.classNum,self.workerNum,self.workerNum),dtype=float)

		err = np.zeros((self.depth+1,self.workerNum,self.classNum))
		#backward propagation
		grd_ConncWeight,err[self.depth] = self.denseLayerGradient(I[self.depth],truelabel-opt)
		for i in range(self.depth):
			grd_ConvWeight[self.depth-1-i],err[self.depth-1-i] = self.ConvLayerGradient(I[self.depth-1-i],err[self.depth-i],(self.depth-1-i))

		return grd_ConvWeight,grd_ConncWeight

	def denseLayerGradient(self,input_,error):
		grd = (error*(input_)).T
		err = error * self.ConncWeight.T
		return grd,err

	def ConvLayerGradient(self,input_,error,layercnt):
		grd = np.zeros((self.classNum,self.workerNum,self.workerNum))
		err = np.zeros(np.shape(input_))

		temp = self.EwLayer(input_,layercnt)
		temp_error = error * self.ReluGradient(temp)

		for i in range(self.classNum):
			grd[i] = input_[:,i]*(temp_error[:,i]*self.GCC.T).T

		for i in range(self.classNum):
			err[:,i] = temp_error[:,i].dot(self.ConvWeight[layercnt,i]*self.GCC)

		return grd,err

	def ReluGradient(self,input_):
		ret = input_ * 1
		ret = np.where(ret<0,ret,1)
		ret = np.where(ret>0,ret,0)
		return ret

	def train(self,trDataSet,trTrueLabelSet,step,alphal,trType,tsDataSet,tsTrueLabelSet):
		itemNum,_ = np.shape(trTrueLabelSet)
		if trType == 'sgd':
			loss,trAcc,tsAcc = self.sgd(trDataSet,trTrueLabelSet,step,alphal,tsDataSet,tsTrueLabelSet)
		elif trType == 'bgd':
			loss,trAcc,tsAcc = self.bgd(trDataSet,trTrueLabelSet,step,alphal,tsDataSet,tsTrueLabelSet)

		self.show(loss)
		self.show(trAcc)
		self.show(tsAcc)
		return None

	def sgd(self,trDataSet,trTrueLabelSet,step,alphal,tsDataSet,tsTrueLabelSet):
		itemNum,_ = np.shape(trTrueLabelSet)	
		loss = np.zeros(step)	
		trAcc = np.zeros(step)
		tsAcc = np.zeros(step)

		#print('start training in sgd:')

		for i in range(step):
			d = round((i/step)*100,3)
			print('\r%f%%'%d,end="")

			loss[i] = self.calLoss(trDataSet,trTrueLabelSet)
			trAcc[i] = self.calacc(trDataSet,trTrueLabelSet)
			tsAcc[i] = self.calacc(tsDataSet,tsTrueLabelSet)

			index = random.randint(0,itemNum-1)
			ConncWeightGradient = np.zeros((self.classNum,self.workerNum),dtype = float)
			ConvWeightGradient = np.zeros((self.depth,self.classNum,self.workerNum,self.workerNum),dtype=float)
			ConvWeightGradient,ConncWeightGradient = self.backward(trDataSet[index],trTrueLabelSet[index])			
			self.ConvWeight += alphal * ConvWeightGradient
			self.ConncWeight += alphal * ConncWeightGradient
		print('\r')
		return loss,trAcc,tsAcc

	def bgd(self,trDataSet,trTrueLabelSet,step,alphal,tsDataSet,tsTrueLabelSet):

		itemNum,_ = np.shape(trTrueLabelSet)
		loss = np.zeros(step)	
		trAcc = np.zeros(step)
		tsAcc = np.zeros(step)


		for i in range(step):
			d = round((i/step)*100,3)
			print('\r%f%%'%d,end="")

			loss[i] = self.calLoss(trDataSet,trTrueLabelSet)
			trAcc[i] = self.calacc(trDataSet,trTrueLabelSet)
			tsAcc[i] = self.calacc(tsDataSet,tsTrueLabelSet)

			ConncWeightGradient = np.zeros((self.classNum,self.workerNum),dtype = float)
			ConvWeightGradient = np.zeros((self.depth,self.classNum,self.workerNum,self.workerNum),dtype=float)

			for j in range(itemNum):
					temp_ConvWeightGradient,temp_ConncWeightGradient = self.backward(trDataSet[j],trTrueLabelSet[j])
					ConncWeightGradient += temp_ConncWeightGradient
					ConvWeightGradient += temp_ConvWeightGradient	

			self.ConvWeight += alphal * ConvWeightGradient
			self.ConncWeight += alphal * ConncWeightGradient
		print('\r')
		return loss,trAcc,tsAcc

	def calLoss(self,trDataSet,trTrueLabelSet):
		itemNum,_ = np.shape(trTrueLabelSet)

		loss = 0
		for i in range(itemNum):
			opt,_,_ = self.forward(trDataSet[i])
			loss += -(trTrueLabelSet[i].dot(np.log(opt)))
		return loss

	def show(self,vec):
		lenth, = np.shape(vec)
		x = np.zeros(lenth)
		for i in range(lenth):
			x[i] = i+1

		fig = plt.figure()
		plt.plot(x,vec,color='red')
		plt.show()

	def calacc(self,trDataSet,trTrueLabelSet):
		itemNum,_ = np.shape(trTrueLabelSet)
		cnt = 0
		for i in range(itemNum):
			ret,_,_ = self.forward(trDataSet[i])
			index = np.where(ret == max(ret))[0][0]
			if trTrueLabelSet[i,index] == 1:
				cnt += 1
		return float(cnt/itemNum)


def dataOHC(data,classNum):
	workerNum,itemNum = np.shape(data)
	ret = np.zeros((itemNum,workerNum,classNum))
	for i in range(itemNum):
		for j in range(workerNum):
			if data[j,i] != -1:
				ret[i,j,int(data[j,i])] = 1
	return ret

def trueLabelOHC(trueLabel,classNum):
	itemNum, = np.shape(trueLabel)
	ret = np.zeros((itemNum,classNum))
	for i in range(itemNum):
		ret[i,int(trueLabel[i])] = 1
	return ret

def standardVec(Quality):
	data = Quality * 1
	_range = np.max(Quality)-np.min(Quality)
	return (data-np.min(Quality))/_range

def labelingSimilarity(a,b):
	itemNum, = np.shape(a)
	tot = 0
	cnt = 0
	for i in range(itemNum):
		if a[i]!=-1 and b[i]!=-1:
			tot += 1
			if a[i] == b[i]:
				cnt += 1
	if tot == 0:
		return 0.5
	return cnt/tot

def labGraph(trDataSet): #not the one-hot code
	workerNum,itemNum = np.shape(trDataSet)
	graph = np.zeros((workerNum,workerNum))
	for i in range(workerNum):
		for j in range(workerNum):
			graph[i,j] = labelingSimilarity(trDataSet[:,i],trDataSet[:,j])
	return graph