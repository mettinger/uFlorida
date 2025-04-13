import torch
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from eegUtils import *

def makeModel(modelType, nChannel, numSampleInput, numSampleOutput, dataTensor):
    
    if modelType == 'fullyConnected':
        inSize = nChannel * numSampleInput
        outSize = nChannel * numSampleOutput
        hiddenLayerSizeList = [300, 300, 300, 300, 300]
        layerSizeList = [inSize] + hiddenLayerSizeList + [outSize]
        model = fullyConnected(nChannel, numSampleInput, layerSizeList)
        dataset = datasetFullyConnected(dataTensor, numSampleInput)
        lossFunction = torch.nn.MSELoss()
    elif modelType == 'conv1d':
        model = conv1d(nChannel, numSampleInput)
        dataset = datasetConv1d(dataTensor, numSampleInput)
        lossFunction = torch.nn.MSELoss()
    elif modelType == 'fourier':
        model = fourierModel(nChannel, numSampleInput)
        dataset = datasetFourier(dataTensor, numSampleInput)
        lossFunction = torch.nn.MSELoss()
    elif modelType == 'kmeans':
        nCentroids = 2**16
        embeddingInit = None
        
        model = conv1dKmeans(nChannel, nCentroids, embeddingInit)
        dataset = datasetConv1dKmeans(dataTensor)
        lossFunction = model.lossFunction
    
    return model, dataset, lossFunction


def datasetMake(dateTensor, numSampleInput, typeCode):
    if typeCode == 0:
        dataset = datasetFullyConnected(dateTensor, numSampleInput)
    elif typeCode == 1:
        dataset = datasetConv1d(dateTensor, numSampleInput)
    elif typeCode == 3:
        dataset = datasetFourier(dateTensor, numSampleInput)
    return dataset

class datasetFullyConnected(Dataset):
    def __init__(self, dataTensor, numSampleInput):
        self.dataTensor = dataTensor
        self.numSampleInput = numSampleInput
        self.nChannel, self.nSample = dataTensor.shape
        
    def __len__(self):
        return self.nSample - self.numSampleInput
    
    def __getitem__(self, idx):
        inputBlock = self.dataTensor[:,idx : idx + self.numSampleInput]
        label = self.dataTensor[:,idx + self.numSampleInput]
        inputBlockReshape = inputBlock.flatten()
        return inputBlockReshape, label

class fullyConnected(torch.nn.Module):
    def __init__(self, nChannel, numSampleInput, layerSizeList):
        super().__init__()
        self.typeCode = 0
        self.nChannel = nChannel
        self.numSampleInput = numSampleInput
        self.layerList = sizeToLayerList(layerSizeList)
        self.myNet = torch.nn.Sequential(*self.layerList)
    
    def forward(self, x):
        return torch.squeeze(self.myNet(x))

    
##############################################


class datasetConv1d(Dataset):
    def __init__(self, dataTensor, numSampleInput):
        self.dataTensor = dataTensor
        self.numSampleInput = numSampleInput
        self.nChannel, self.nSample = dataTensor.shape
        
    def __len__(self):
        return self.nSample - self.numSampleInput
    
    def __getitem__(self, idx):
        inputBlock = self.dataTensor[:,idx : idx + self.numSampleInput]
        label = self.dataTensor[:,idx + self.numSampleInput]
        return inputBlock, label

class conv1d(torch.nn.Module):
    def __init__(self, nChannel, numSampleInput):
        super().__init__()
        self.typeCode = 1
        self.nChannel = nChannel
        self.numSampleInput = numSampleInput
        self.layerList = [torch.nn.Conv1d(in_channels=nChannel, out_channels=50, kernel_size=3),
                          torch.nn.LeakyReLU(),
                          torch.nn.Conv1d(in_channels=50, out_channels=50, kernel_size=3),
                          torch.nn.LeakyReLU(),
                          torch.nn.Conv1d(in_channels=50, out_channels=50, kernel_size=3),
                          torch.nn.LeakyReLU(),
                          torch.nn.Flatten(),
                          torch.nn.Linear(700,nChannel)]
        for thisLayer in [0,2,4,7]:
            torch.nn.init.xavier_uniform_(self.layerList[thisLayer].weight)
            self.myNet = torch.nn.Sequential(*self.layerList)
            # torch.nn.MaxPool1d(kernel_size=2)
    
    def forward(self, input):
        return self.myNet(input)

##############################################


class fanLayer(nn.Module):
    def __init__(self, inFeatures, outFourier, outLinear):
        super().__init__()
        self.weightLinear = nn.Parameter(torch.randn(outLinear, inFeatures))
        self.biasLinear = nn.Parameter(torch.randn(outLinear))
        
        self.weightFourier = nn.Parameter(torch.randn(outFourier, inFeatures))

    def forward(self, x):
        linear = F.relu(F.linear(x, self.weightLinear, self.biasLinear))
        cos = torch.cos(F.linear(x, self.weightFourier))
        sin = torch.sin(F.linear(x, self.weightFourier))
        
        phi = torch.cat((cos, sin, linear), dim=1)
        return phi
    
class datasetFourier(Dataset):
    def __init__(self, dataTensor, numSampleInput):
        self.dataTensor = dataTensor
        self.numSampleInput = numSampleInput
        self.nChannel, self.nSample = dataTensor.shape
        
    def __len__(self):
        return self.nSample - self.numSampleInput
    
    def __getitem__(self, idx):
        inputBlock = self.dataTensor[:,idx : idx + self.numSampleInput]
        label = self.dataTensor[:,idx + self.numSampleInput]
        return inputBlock, label
    
    
class fourierModel(nn.Module):
    def __init__(self, nChannel, numSampleInput):
        super().__init__()
        self.typeCode = 1
        self.nChannel = nChannel
        self.numSampleInput = numSampleInput
        
        self.fourier1 = fanLayer(10, 5, 8)
       
    def forward(self, x):
        y = self.fourier1(x)
        return y
    
    
#########################################################################

class datasetConv1dKmeans(Dataset):
    def __init__(self, dataTensor):
        self.dataTensor = dataTensor
        self.nChannel, self.nSample = dataTensor.shape
        
    def __len__(self):
        return self.nSample - 1
    
    def __getitem__(self, idx):
        inputBlock = self.dataTensor[:,idx : idx + 1]
        label = self.dataTensor[:,idx + 1]
        return inputBlock, label

class conv1dKmeans(torch.nn.Module):
    def __init__(self, nChannel, nCentroids, embeddingInit):
        super().__init__()
        self.typeCode = 1
        self.nChannel = nChannel
        self.nCentroids = nCentroids
        self.embeddingInit = embeddingInit
        self.layerList = [torch.nn.Conv1d(in_channels=nChannel, out_channels=50, kernel_size=3),
                          torch.nn.LeakyReLU(),
                          torch.nn.Conv1d(in_channels=50, out_channels=50, kernel_size=3),
                          torch.nn.LeakyReLU(),
                          torch.nn.Conv1d(in_channels=50, out_channels=50, kernel_size=3),
                          torch.nn.LeakyReLU(),
                          torch.nn.Flatten(),
                          torch.nn.Linear(700,nChannel)]
        for thisLayer in [0,2,4,7]:
            torch.nn.init.xavier_uniform_(self.layerList[thisLayer].weight)
        
        self.myNet = torch.nn.Sequential(*self.layerList)
            
        if embeddingInit == None:
            self.embedding = nn.Embedding(self.nCentroids, self.nChannel)
        else:
            weight = torch.FloatTensor(self.embeddingInit)
            self.embedding = nn.Embedding.from_pretrained(weight)
            
    def forward(self, input):
        return self.myNet(input)
    
    def lossFunction(self, prediction, label):
        prediction = prediction.cpu()
        for i in range(self.nCentroids):
            thisNorm = torch.linalg.vector_norm(prediction - self.embedding.cpu()(torch.LongTensor([i])))
            if i == 0:
                bestNorm = thisNorm
                bestIndex = 0
            elif thisNorm < bestNorm:
                bestNorm = thisNorm
                bestIndex = i
        thisNorm = torch.linalg.vector_norm(prediction - self.embedding.cpu()(torch.LongTensor([bestIndex])))
        return thisNorm
    
    

