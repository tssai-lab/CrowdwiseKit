#import lib
import os
import numpy as np
import math
import random
#import matplotlib.pyplot as plt
import time
from numpy.linalg import *
#from collections import Counter

#Readm:
#-------------------------------------------------------
'''
this code file only include those data generation methods with no specified number
'''
#----

#methods

def workerStimulation(n,Miu,Theta,order):
	T = np.random.normal(Miu,Theta,n)
	for i in range(n):
		if T[i]<=0 or T[i]>1:
			while 1:
				T[i] = np.random.normal(Miu,Theta,1)
				if T[i]<=1 and T[i] >0:
					break
	if order == 1:
		T[::-1].sort()
	return T


def trueLabelStimulation(n,classNumber,order):
	trueLabel = np.zeros(n)
	for i in range(n):
		trueLabel[i] = i%classNumber
	return trueLabel

def oneHotConvert(Metric,type,classNumber):
	if type == 1: #Vector
		n, = np.shape(Metric)
		oneHotVec = np.zeros(n*classNumber)
		for i in range(n):
			oneHotVec[i*classNumber + int(Metric[i])] = 1
		return oneHotVec
	elif type == 2: #Metrix
		cols,lines = np.shape(Metric)
		oneHotMatric = np.zeros(cols,lines*classNumber)
		for i in range(cols):
			for j in range(lines):
				if Metric[i,j] != -1:
					oneHotMatric[i,j*classNumber+int(Metric[i,j])]= 1
		return oneHotMatric

def stimulateLabel(ability,trueLabel,labelArray):
	classNumber = int(np.size(labelArray))
	prob = np.zeros(classNumber)
	temp = (1-ability)/(classNumber-1)
	for i in range(classNumber):
		prob[i] = temp
	prob[int(trueLabel)] = ability
	return np.random.choice(labelArray,1,p=prob.ravel())

def workerLabelStimulation(abilityMetirc,trueLabel,classNumber,labelPerWorker,type):
	workerNum, = np.shape(abilityMetirc)
	itemNum, = np.shape(trueLabel)
	labelArray = np.zeros(classNumber)
	for i in range(classNumber):
		labelArray[i] = i
	workerLabel = np.zeros((workerNum,itemNum))
	for i in range(workerNum):
		for j in range(itemNum):
			workerLabel[i,j] = -1
	
	if type == 1:#direct random
		
		itemMark = np.zeros(itemNum)
		for i in range(itemNum):
			itemMark[i] = labelPerWorker
		
		for j in range(itemNum):
			while itemMark[j]:
				i = random.randint(0,workerNum-1)
				if workerLabel[i,j] != -1:
					continue
				workerLabel[i,j] = stimulateLabel(abilityMetirc[i],trueLabel[j],labelArray)
				itemMark[j] -= 1

	elif type == 2:#controlled random

		itemMark = np.zeros(itemNum)
		for i in range(itemNum):
			itemMark[i] = labelPerWorker

		tempZ = int(itemNum*labelPerWorker/workerNum)
		tempR = (itemNum*labelPerWorker)%workerNum

		workerMark = np.zeros(workerNum)
		for i in range(workerNum):
			workerMark[i] = tempZ		
		while tempR:
			workerMark[tempR] += 1
			tempR -= 1

		probI = np.zeros(workerNum)
		probJ = np.zeros(itemNum)
		arrayI = np.zeros(workerNum)
		arrayJ = np.zeros(itemNum)
		for i in range(workerNum):
			arrayI[i] = i
		for j in range(itemNum):
			arrayJ[j] = j

		while max(itemMark):
			for i in range(workerNum):
				probI[i] =workerMark[i]/sum(workerMark)
			for j in range(itemNum):
				probJ[j] = itemMark[j]/sum(itemMark)

			WLi = int(np.random.choice(arrayI,1,p=probI.ravel()))
			WLj = int(np.random.choice(arrayJ,1,p=probJ.ravel()))

			if workerLabel[WLi,WLj] != -1:
				continue

			workerLabel[WLi,WLj] = stimulateLabel(abilityMetirc[WLi],trueLabel[WLj],labelArray)
			#print(workerLabel)
			itemMark[WLj] -= 1
			workerMark[WLi] -= 1

	return workerLabel

#111