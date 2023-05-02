import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import neptune

class timeFreqNet(nn.Module):
    def __init__(self):
        
        totalSize = 11397
        self.stftSize = 67
        self.p1 = 512

        super(residualBlockConcat, self).__init__()
        self.linear1 = nn.Linear(totalSize, totalSize)
        self.linear2 = nn.Linear(totalSize, totalSize)
        self.linear3 = nn.Linear(totalSize, self.p1)
        self.linearLast = nn.Linear(self.p1 + self.stftSize,1)
        
    def forward(self, x):
        l1 = F.relu(self.linear1(x))
        l2 = F.relu(self.linear2(l1))
        l3 = F.relu(self.linear3(l2))

        istft = torch.istft(torch.tensor(a), 
               n_fft = nperseg, 
               hop_length = 1, 
               return_complex=True, 
               normalized=False, 
               onesided=True, 
               pad_mode='constant')

        out = F.relu(self.linearLast(torch.concat([l3,istft])))
        return out

class residualBlockConcat(nn.Module):
    def __init__(self, inputSize, linearOutputSize):
        
        super(residualBlockConcat, self).__init__()
        self.linear = nn.Linear(inputSize,linearOutputSize)
        self.bn = nn.BatchNorm1d(inputSize)

    def forward(self, x):
        residual = x
        out = self.linear(self.bn(x))
        out = torch.concat([F.relu(out), residual], dim=1)
        return out
    
def residualConcatDict(inputSize, linearSizeList):
    n = len(linearSizeList)
    tupleList = []
    inSize = inputSize
    outSize = linearSizeList[0]
    tupleList.append(('in', residualBlockConcat(inSize, outSize)))
    for i in range(n - 1):
        inSize = outSize + inSize
        outSize = linearSizeList[i]
        tupleList.append(('residConcat%s' % str(i), residualBlockConcat(inSize, outSize)))
    return OrderedDict(tupleList + [('out', nn.Linear(outSize + inSize,1))])

class residualBlockAdd(nn.Module):
    def __init__(self, inputSize):
        
        super(residualBlockAdd, self).__init__()
        self.linear = nn.Linear(inputSize, inputSize)
        self.bn = nn.BatchNorm1d(inputSize)

    def forward(self, x):
        residual = x
        out = self.linear(self.bn(x))
        out = out + residual
        out = F.relu(out)
        return out

# GIVEN A LIST OF LAYER SIZES MAKE AN ORDERED DICTIONARY FOR INITIALIZING A PYTORCH NET

def residualAddDict(inputSize, residSize, nLayer):
    tupleList = []
    tupleList.append(('in', nn.Linear(inputSize, residSize)))
    for i in range(nLayer):
        tupleList.append(('residAdd%s' % str(i), residualBlockAdd(residSize)))
    return OrderedDict(tupleList + [('out', nn.Linear(residSize,1))])



def listToOrderedDict_1(sizeList):
    n = len(sizeList)
    tupleList = []
    #tupleList.append(('in', nn.Linear(inputSize, sizeList[0])))
    for i in range(n -  1):
        tupleList.append(('bn%s' % str(i), nn.BatchNorm1d(sizeList[i])))
        tupleList.append(('l%s' % str(i), nn.Linear(sizeList[i], sizeList[i+1])))
        tupleList.append(('r%s' % str(i), nn.ReLU()))
        tupleList.append(('d%s' % str(i), nn.Dropout(.5)))
    #return OrderedDict(tupleList[:-2])
    return OrderedDict(tupleList[:-3] + [('out', nn.Linear(sizeList[-1],1))])

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

def neptuneBestRun():
    project = neptune.init_project(project="jettinger35/predictScalp")
    df = project.fetch_runs_table().to_pandas()
    return df[['sys/id','best_test_loss']]

'''
def getModelByID(modelID):
    model = torch.jit.load(modelPath + 'model_%s.pt' % str(modelID))
    bestTestLoss = test(validDataLoader, model, loss_fn, device)

def modelDownloadNeptune(neptuneProject, api_token, neptuneRunID, modelPath):
    run = neptune.init_run(
        project=neptuneProject,
        api_token=api_token,  
        capture_hardware_metrics=True,
        capture_stderr=True,
        capture_stdout=True,
        with_id=neptuneRunID
    )

    try:
        destinationPathModel = modelPath
        run["model_best"].download(destinationPathModel)
        print("model download success...")
        run.stop()
    except Exception as error:
        print("model download failure...")
        print(error)
        run.stop()
'''