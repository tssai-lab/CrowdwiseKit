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
from tqdm import tqdm
import shutil
from pathlib import Path

#hyper parameter
classNum = 3		
workerNum = 50			
itemNum = 2000			
Miu = 0.7				
Theta = 0.15			
labelPerWorker = 3		

def generateData(workerNum,itemNum,classNum,Miu,Theta,labelPerWorker,order,trainRatio):
	workerAbility = dataGenerate.workerStimulation(workerNum,Miu,Theta,order)
	trueLabel = dataGenerate.trueLabelStimulation(itemNum,classNum,0)
	workerLabel = dataGenerate.workerLabelStimulation(workerAbility,trueLabel,classNum,labelPerWorker,1)

	trainSampleNum = int(itemNum * trainRatio)
	trainTrueLabel  = dataGenerate.trueLabelStimulation(trainSampleNum,classNum,0)
	trainWorkerLabel = dataGenerate.workerLabelStimulation(workerAbility,trainTrueLabel,classNum,labelPerWorker,1)

	return workerAbility,trueLabel,workerLabel,trainTrueLabel,trainWorkerLabel

def fileSave(workerAbility,trueLabel,workerLabel,trainTrueLabel,trainWorkerLabel,Path):
	np.save(Path+'\\'+'workerAbility',workerAbility)
	np.save(Path+'\\'+'trueLabel',trueLabel)
	np.save(Path+'\\'+'workerLabel',workerLabel)
	np.save(Path+'\\'+'trainTrueLabel',trainTrueLabel)
	np.save(Path+'\\'+'trainWorkerLabel',trainWorkerLabel)
	return 1

def stdParameters(classNum,workerNum,itemNum,Miu,Theta,labelPerWorker):
	classNum = int(3)			
	workerNum = int(50)			
	itemNum = int(2000)		
	Miu = float(0.7)		
	Theta = float(0.15)			
	labelPerWorker = int(3)     
	return classNum,workerNum,itemNum,Miu,Theta,labelPerWorker

fileName = ['classNum','workerNum','itemNum','Miu','Theta','labelPerWorker']

names = locals()

classNumData = np.array([2,3,4,5,6,7,8,9,10,15],dtype = int)
workerNumData = np.array([10,20,30,40,50,75,100,150,200,300],dtype = int)
itemNumData = np.array([300,500,700,1000,2000,3000,4000,5000,7000,10000],dtype = int)
MiuData = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],dtype = float)
ThetaData = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.8],dtype = float)
labelPerWorkerData = np.array([1,2,3,4,5,7,9,11,13,15],dtype = int)

filePath = os.getcwd()
baseDataFilePath = filePath+"\\BaseData"
tp = Path(baseDataFilePath)
if tp.exists():
	shutil.rmtree(baseDataFilePath,True)
os.makedirs(baseDataFilePath)

for i in range(6):
	tempFilePath = str(baseDataFilePath)+'\\'+str(fileName[i])
	os.makedirs(tempFilePath)
	classNum,workerNum,itemNum,Miu,Theta,labelPerWorker=stdParameters(classNum,workerNum,itemNum,Miu,Theta,labelPerWorker)
	
	for j in range(10):
		tempDataPath = tempFilePath+'\\'+str(j)
		for k in range(10):
			spcDataPath = tempDataPath+'\\'+str(k)
			os.makedirs(spcDataPath)
			names[fileName[i]] = names[fileName[i]+"Data"][j]
			workerAbility,trueLabel,workerLabel,trainTrueLabel,trainWorkerLabel = generateData(workerNum,itemNum,classNum,Miu,Theta,labelPerWorker,0,0.1)
			fileSave(workerAbility,trueLabel,workerLabel,trainTrueLabel,trainWorkerLabel,spcDataPath)
	

