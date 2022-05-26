#%%
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns

#%%
def orderParameterGet(phiVector):
    return np.abs(sum(np.exp(phiVector * (0+1j))))

#%%
kString = "5"
odeSolutionCSV = "c:\\Users\\the_m\\github\\uFlorida\\data\\sphere\\Cauchy_0_1\\kuramoto_K_%s.0_Cauchy_0_1.csv" % kString

df = pd.read_csv(odeSolutionCSV, header=None)

time = df.values[:,0]
odePhi = df.values[:,1:]
thetaDot = np.diff(odePhi, axis=0) * 10

#%%
orderParameter = [orderParameterGet(odePhi[i,:])for i in range(len(time))]
plt.plot(time, orderParameter)
plt.title("Order Parameter.  K: " + kString);

#%%
index = 0
plt.figure()
plt.plot(odePhi[:,index])
plt.plot(thetaDot[:,index])
plt.title("Phase and Velocity for oscilattor index: " + str(index));

# %%
timeValue = 999
colorLower = np.min(thetaDot)
colorUpper = np.max(thetaDot)

px.scatter(x = np.cos(odePhi[timeValue,:]), 
           y = np.sin(odePhi[timeValue,:]), 
           color=thetaDot[timeValue,:],
           range_color=(colorLower, colorUpper),
           title="Theta for time: " + str(timeValue))
           

#%% 

index = 0
px.histogram(thetaDot[index,:], title='Angular velocity for time: ' + str(index))

#%%

np.argmin(thetaDot[0,:])


# %%
