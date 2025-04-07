import torch
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



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