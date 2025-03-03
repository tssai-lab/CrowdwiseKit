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
this code file only include those methods can be used generally
'''
#-------------------------------------------------------

#Methods

def graphConvolutionCoefficient(Graph):

	cols,_=np.shape(Graph)

	graph = Graph * 1
	'''
	unit = np.zeros((cols,cols))
	for i in range(cols):
		unit[i,i] = 1
	graph += unit
	'''

	degM = np.zeros((cols,cols))
	for i in range(cols):
		degM[i,i] = np.sum(graph[i])

	adjM = graph * 1
	D_a = np.sqrt(inv(degM))
	return D_a.dot(adjM).dot(D_a)

def similarity(a,b,type):
	if type == 1:
		return 1-abs(a-b)
	if type == 2:
		TotalNumber = 0.0
		SameNumber = 0.0
		n, = np.shape(a)
		for i in range(n):
			if a[i] != -1 and b[i] != -1:
				TotalNumber += 1
				if a[i] == b[i]:
					SameNumber += 1
		if TotalNumber == 0:
			return 0.0
		else:
			return SameNumber/TotalNumber

def oneHotConvert(Metric,type,classNumber):
	if type == 1: 
		n, = np.shape(Metric)
		oneHotVec = np.zeros(n*classNumber)
		for i in range(n):
			oneHotVec[i*classNumber + int(Metric[i])] = 1
		return oneHotVec
	elif type == 2: 
		cols,lines = np.shape(Metric)
		oneHotMatric = np.zeros((cols,lines*classNumber))
		for i in range(cols):
			for j in range(lines):
				if Metric[i,j] != -1:
					oneHotMatric[i,j*classNumber+int(Metric[i,j])]= 1
		return oneHotMatric

def generateGraph(vec):
	workerNum, = np.shape(vec)
	graph = np.zeros((workerNum,workerNum))
	for i in range(workerNum):
		for j in range(workerNum):
			graph[i,j] = similarity(vec[i],vec[j],1)
	return graph

def Kapproach(vec,i,graph,k,classNumber):
	tempVec = vec * 1
	relate = graph[i] * 1
	index = np.where(tempVec == -1)
	relate = np.delete(relate,index)
	tempVec = np.delete(tempVec,index)
	kVec = np.zeros(k)
	for m in range(k):
		kVec[m]=-1
	flag = 1
	
	for m in range(k):
		if not np.size(tempVec):
			flag = 0
			break
		maxIndex = np.where(relate == max(relate))[0][0]
		kVec[m] = tempVec[maxIndex]
		tempVec = np.delete(tempVec,maxIndex)
		relate = np.delete(relate,maxIndex)

	if flag == 0:
		kVec = np.delete(kVec,np.where(kVec == -1))
	lengthKVec, = np.shape(kVec)
	result = np.zeros(classNumber)
	for m in range(lengthKVec):
		result[int(kVec[m])] += 1
	
	return np.where(result == max(result))[0][0]

def calGroupAvgJaccardSimilarity(group,workerLabel):
	workerNum, = np.shape(group)
	jcdSimilarity = 0
	cnt = 0
	for i in range(workerNum):
		if group[i] == 1:
			for j in range(i+1,workerNum):
				if group[j] == 1:
					cnt += 1
					jcdSimilarity += similarity(workerLabel[i],workerLabel[j],2)
	if cnt == 0:
		return 0

	return jcdSimilarity/cnt

def groupDivide(graph,workerLabel,type):
	workerNum,_ = np.shape(graph)
	
	if type == 1:
		index = np.where(graph == np.min(graph))
		minPointI = index[0][0]
		minPointJ = index[1][0]
		groupA = np.zeros(workerNum)
		groupB = np.zeros(workerNum)

		for i in range(workerNum):
			if graph[i,minPointI] >=graph[i,minPointJ]:
				groupA[i] = 1
			else:
				groupB[i] = 1
		
	jaccardSimilarityA = calGroupAvgJaccardSimilarity(groupA,workerLabel)
	jaccardSimilarityB = calGroupAvgJaccardSimilarity(groupB,workerLabel)
	if jaccardSimilarityB > jaccardSimilarityA:
		temp = groupB
		groupB = groupA
		groupA = temp

	return groupA,groupB

def calAvgAbility(group,workerAbility):
	workerNum, = np.shape(workerAbility)
	abilitySum = 0
	cnt = 0
	for i in range(workerNum):
		if group[i] == 1:
			abilitySum += workerAbility[i]
			cnt += 1
	return abilitySum/cnt

def calAccuracy(vec,trueLabel):
	lengthVec, =np.shape(vec)
	lengthTrueLabel, = np.shape(trueLabel)
	if lengthVec != lengthTrueLabel:
		return -1
	
	accNum = 0
	for i in range(lengthVec):
		if vec[i] == trueLabel[i]:
			accNum += 1

	return accNum/lengthVec

def MinOrMaxGroup(group,workerAbility,type):
	workerNum, = np.shape(group)
	if type == 1:#return min
		minValue = 1
		for i in range(workerNum):
			if group[i] == 1 and workerAbility[i] < minValue:
				minValue = workerAbility[i]
		return minValue
	if type == 2:
		maxValue = 0
		for i in range(workerNum):
			if group[i] == 1 and workerAbility[i] > maxValue:
				maxValue = workerAbility[i]
		return maxValue

def biasToVoting(biasWorkerLabel,classNum):
	workerNum,itemNum = np.shape(biasWorkerLabel)
	itemNum = int(itemNum/classNum)

	votingWorkerLabel = np.zeros((workerNum,itemNum))
	for i in range(workerNum):
		for j in range(itemNum):

			temp = np.zeros(classNum)

			for k in range(classNum):
				temp[k] = biasWorkerLabel[i,j*classNum+k]

			if sum(temp) == 0:
				votingWorkerLabel[i,j] = -1
			else:
				index = np.where(temp == np.max(temp))[0][0]
				votingWorkerLabel[i,j] = index
	return votingWorkerLabel

def showMatric(M):
	I,J = np.shape(M)
	temp = M * 1
	for i in range(I):
		for j in range(J):
			temp[i][j] = round(M[i,j] , 3)
	print(temp)



