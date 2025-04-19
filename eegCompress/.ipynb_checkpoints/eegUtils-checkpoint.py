import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Dataset

import datetime
import pytz
timeZone = pytz.timezone('America/Los_Angeles')

import torchModels

def predictEEG(model, interval, data, batch_size=128):
    model.eval()
    
    nChannel, nSample = data.shape
    # if None predict the entire dataset
    if interval == None:
        start = model.numSampleInput
        stop = nSample
    else:
        start, stop = interval
        
    predicted = np.zeros((model.nChannel, stop - start))
    dataTensorTruncated = torch.tensor(data[:,start - model.numSampleInput:stop])
    if torch.cuda.is_available():
        dataTensorTruncated = dataTensorTruncated.to('cuda')
        
    dataset = torchModels.datasetMake(dataTensorTruncated, model.numSampleInput, model.typeCode)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None)
    
    with torch.no_grad():
        i = 0
        for modelInput, label in loader:
            modelOutput = model(modelInput).detach().cpu().numpy().copy()
            if modelOutput.ndim == 1:
                predicted[:,i] = modelOutput
                i += 1
            else:
                for j in range(modelOutput.shape[0]):
                    predicted[:,i] = modelOutput[j,:]
                    i += 1
                    
        if interval == None:
            predicted = np.concatenate((data[:,0:model.numSampleInput], predicted), axis = 1)
            
    model.train()
    return predicted

def timeSeriesCompare(original, predicted, start, samplesToPLot, channel = 0, plotOption="both"):
    originalChannel = original[channel, start:start + samplesToPlot]
    predictedChannel = predicted[channel, start:start + samplesToPlot]

    fig = plt.figure()
    if plotOption == "both":

        plt.plot(originalChannel, label='original')
        plt.plot(predictedChannel, label='predicted')
        thisMin = np.min([originalChannel, predictedChannel])
        thisMax = np.max([originalChannel, predictedChannel])
        plt.ylim([thisMin, thisMax])
        plt.legend()
    elif plotOption == "orig":
        plt.plot(originalChannel)
        thisMin = np.min([originalChannel])
        thisMax = np.max([originalChannel])
        plt.ylim([thisMin, thisMax])
        plt.title('original')
    else:
        plt.plot(predictedChannel)
        thisMin = np.min([predictedChannel])
        thisMax = np.max([predictedChannel])
        plt.ylim([thisMin, thisMax])
        plt.title('predicted')

    return fig
    
def saveModel(model, optimizer, epoch, loss, predicted):
    directoryPath = '/blue/gkalamangalam/jmark.ettinger/eegCompress/models/'
    saveName = 'savedModel_' + str(datetime.datetime.now().astimezone(timeZone).strftime('%m-%d %H:%M')) + '_' + f"{np.mean(predicted):.3f}" + '.pt'
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, directoryPath + saveName)
    # save structure information
    structureFileName = directoryPath + 'structure_' + str(datetime.datetime.now().astimezone(timeZone).strftime('%m-%d %H:%M')) + '.txt'
    with open(structureFileName, "w") as text_file:
        print("Network structure: {}".format(str(model)), file=text_file)
    print("Model has been saved: " + saveName)

def loadModel(path, model, optimizer, trainBool = True):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    if trainBool:
        model.train()
    else:
        model.eval()
        
    return model, optimizer, epoch, loss


def sizeToLayerList(layerSizeList):
    layerList = []
    for i in range(0, len(layerSizeList) - 1):
        thisLayer = torch.nn.Linear(layerSizeList[i], layerSizeList[i + 1])
        torch.nn.init.xavier_uniform_(thisLayer.weight)
        layerList.append(thisLayer)
        if i < len(layerSizeList) - 2:
            layerList.append(torch.nn.LeakyReLU())
            
    return layerList

def samplerMake(model, numSampleInput, data):
    nChannel, nSample = data.shape
    predicted = predictEEG(model, None, data)
    residual = np.abs(data - predicted)
    residualMeasure = np.mean(residual, axis=0)[numSampleInput:]
    sampler = torch.utils.data.WeightedRandomSampler(weights=residualMeasure, num_samples=nSample)
    return sampler, predicted, residualMeasure

def modelSize(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb


'''
def timeSeriesCompare(model, start, secondsToPlot, sFreq, data, numSampleInput, channel = 0, plotOption="both"):
    # currently only works for outputSamples = 1

    if secondsToPlot == None:
        original = data
    else:
        samplesToPlot = secondsToPlot * sFreq
        original = data[:,start:start + samplesToPlot]

    predicted = predictEEG(model, (start, start + samplesToPlot), data)
    originalChannel = original[channel,:]
    predictedChannel = predicted[channel,:]

    fig = plt.figure()
    if plotOption == "both":

        plt.plot(originalChannel, label='original')
        plt.plot(predictedChannel, label='predicted')
        thisMin = np.min([originalChannel, predictedChannel])
        thisMax = np.max([originalChannel, predictedChannel])
        plt.ylim([thisMin, thisMax])
        plt.legend()
    elif plotOption == "orig":
        plt.plot(originalChannel)
        thisMin = np.min([originalChannel])
        thisMax = np.max([originalChannel])
        plt.ylim([thisMin, thisMax])
        plt.title('original')
    else:
        plt.plot(predictedChannel)
        thisMin = np.min([predictedChannel])
        thisMax = np.max([predictedChannel])
        plt.ylim([thisMin, thisMax])
        plt.title('predicted')

    return fig, original, predicted
    '''