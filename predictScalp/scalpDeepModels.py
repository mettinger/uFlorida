import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

class NeuralNetwork(nn.Module):
    
    def __init__(self, inputSize, hiddenLayerSizes):
        super().__init__()
        self.model = nn.Sequential(listToOrderedDict(inputSize, hiddenLayerSizes))
        
    def forward(self, x):
        return self.model(x)
    
# GIVEN A LIST OF LAYER SIZES MAKE AN ORDERED DICTIONARY FOR INITIALIZING A PYTORCH NET

def listToOrderedDict(inputSize, sizeList):
    n = len(sizeList)
    tupleList = []
    tupleList.append(('in', nn.Linear(inputSize, sizeList[0])))
    for i in range(n - 1):
        tupleList.append(('bn%s' % str(i), nn.BatchNorm1d(sizeList[i])))
        tupleList.append(('l%s' % str(i), nn.Linear(sizeList[i], sizeList[i+1])))
        tupleList.append(('r%s' % str(i), nn.ReLU()))
        tupleList.append(('d%s' % str(i), nn.Dropout(.5)))
    return OrderedDict(tupleList[:-2] + [('out', nn.Linear(sizeList[-1],1))])

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        return loss
            
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)