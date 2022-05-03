#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor

#%%

# CREATE SIMULATED DATA

numSamples = 10000
numPeriod = 100
dim = 2
lag = int((numSamples/(numPeriod * 2 * np.pi)) * np.pi/2)

t = np.linspace(0, numPeriod * 2 * np.pi, numSamples)
deltaT = t[1] - t[0]
x = np.sin(t)

plt.figure()
plt.plot(x[0:200])
plt.xlabel('Time')
plt.ylabel('Observation')
plt.title('Original Signal')

xlag = np.vstack([x[(dim - 1 - i) * lag : len(x) - (i * lag)] for i in range(dim)]).transpose()
xdot = np.diff(xlag, axis = 0)/ deltaT
xdot = np.array([(xdot[i,:] + xdot[i + 1,:])/2 for i in range(xdot.shape[0] - 1)])

plt.figure()
plt.plot(xlag[:,0], xlag[:,1], '.')
plt.title('Lag Embedding')

batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

tensor_x = torch.Tensor(xlag[1:-1,:])
tensor_y = torch.Tensor(xdot)

training_data = TensorDataset(tensor_x,tensor_y)
train_dataloader = DataLoader(training_data, batch_size=batch_size)

#%%

# DEFINE THE NEURAL NETWORK VECTOR FIELD ESTIMATOR

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        yHat = self.linear_relu_stack(x)
        return yHat

def train(dataloader, model, loss_fn, optimizer):
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
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#%%

# FIT THE MODEL TO THE DATA

epochs = 50

model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")

predictions = model(tensor_x.to(device)).cpu().detach().numpy()
resid = predictions - xdot

# %%

# GRAPH THE ESTIMATED FIELD

gap = 5

xx = xlag[::gap,0]
yy = xlag[::gap,1]
uu = predictions[::gap,0]
vv = predictions[::gap,1]

fig = ff.create_quiver(xx, yy, uu, vv) 
fig.update_layout(title="Estimated Phase Portrait")

# %%

# GRAPH THE ESTIMATED FIELD

grid = np.array([(i,j) for i in range(-10,10) for j in range(-10,10)])
gridPredictions = model(torch.Tensor(grid).to(device)).cpu().detach().numpy()

xx = grid[:,0]
yy = grid[:,1]
uu = gridPredictions[:,0]
vv = gridPredictions[:,1]

fig = ff.create_quiver(xx, yy, uu, vv) 
fig.update_layout(title="Estimated Phase Portrait")

