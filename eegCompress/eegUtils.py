import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Dataset

import datetime
import pytz
timeZone = pytz.timezone('America/Los_Angeles')



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

  dataset = datasetMake(dataTensorTruncated, model.numSampleInput, model.typeCode)
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

def saveModel(model, optimizer, epoch, loss, predicted):

  directoryPath = '/content/drive/MyDrive/NeuroResearch/Data/eegCompress/models/'
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
  #residualMeasure = np.max(residual, axis=0)[numSampleInput:]
  residualMeasure = np.mean(residual, axis=0)[numSampleInput:]
  sampler = torch.utils.data.WeightedRandomSampler(weights=residualMeasure, num_samples=nSample)

  print("Residual measure: " + str(np.max(residualMeasure)))
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

def datasetMake(dateTensor, numSampleInput, typeCode):
  if typeCode == 0:
    dataset = dataset_0(dateTensor, numSampleInput)
  elif typeCode == 1:
    dataset = dataset_1(dateTensor, numSampleInput)
  return dataset

class dataset_0(Dataset):
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

class model_0(torch.nn.Module):
  def __init__(self, nChannel, numSampleInput, layerSizeList):
    super().__init__()
    self.typeCode = 0
    self.nChannel = nChannel
    self.numSampleInput = numSampleInput

    self.layerList = sizeToLayerList(layerSizeList)
    self.myNet = torch.nn.Sequential(*self.layerList)

  def forward(self, x):
    return torch.squeeze(self.myNet(x))


class dataset_1(Dataset):
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

class model_1(torch.nn.Module):
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
