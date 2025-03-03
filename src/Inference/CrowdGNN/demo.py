import numpy as np
import CrowdGNN
import myMethod

# K - multi classes for sample data
K = 3
#convolution layer depth
MaxDep = 3
#training hyper parameters
epoch = 50
alphal = 1	#learning rate
#sample data Path
dataPath = "./Sample_Data/"

def readData(Path):
	workerAbility = np.load(Path+"workerAbility.npy")
	trueLabel = np.load(Path+"trueLabel.npy")
	workerLabel = np.load(Path+"workerLabel.npy")
	trainWorkerLabel= np.load(Path+"trainWorkerLabel.npy")
	trainTrueLabel= np.load(Path+"trainTrueLabel.npy")
	return workerAbility,trueLabel,workerLabel,trainWorkerLabel,trainTrueLabel


workerAbility,trueLabel,workerLabel,trainWorkerLabel,trainTrueLabel = readData(dataPath)
J,_ = np.shape(workerLabel)
workerQuality = CrowdGNN.standardVec(workerAbility)	

#Original Graph generation
#cosin similarity
graph_cosine = myMethod.generateGraph(workerQuality)
#labeling similarity
graph_labeling = CrowdGNN.labGraph(trainWorkerLabel)

#one-hot encoding
t_data_OH = CrowdGNN.dataOHC(trainWorkerLabel,K)
t_truelabel_OH = CrowdGNN.trueLabelOHC(trainTrueLabel,K)

workerLabel_OH = CrowdGNN.dataOHC(workerLabel,K)
trueLabel_OH = CrowdGNN.trueLabelOHC(trueLabel,K)

#config CrowdGNN
config = np.array([K,J])
model = CrowdGNN.CrowdGNN_Model(config,graph_cosine,MaxDep)

#training with Batch gradient descent
model.train(t_data_OH,t_truelabel_OH,epoch,alphal,'bgd',workerLabel_OH,trueLabel_OH)







