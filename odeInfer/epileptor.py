#%%
import numpy as np
import pysindy as ps

# %%

# MAIN AND ACCESSORY FUNCTIONS FOR EPILEPTOR MODEL
def f1(x1, x2, z):
    if x1 < 0:
        temp = x1**3 - (3 * x1**2)
        return temp
    else:
        temp = (x2 - (.6 * ((z - 4)**2))) * x1 
        return temp

def f2(x2):
    if x2 < -.25:
        return 0
    else:
        temp = 6 * (x2 + .25)
        return temp
        
def epileptor(x1, y1, z, x2, y2, u):

    x0 = -1.6
    y0 = 1
    tao0 = 2857
    tao1 = 1
    tao2 = 10
    Irest1 = 3.1
    Irest2 = .45
    gamma = .01

    x1Dot = y1 - f1(x1, x2, z) - z + Irest1
    y1Dot = y0 -(5 * x1**2) - y1
    zDot = (1/tao0) * ((4 * (x1 - x0)) - z)
    x2Dot = -y2 + x2 - x2**3 + Irest2 + (2*u) - (.3 * (z - 3.5))
    y2Dot = (1/tao2) * (-y2 + f2(x2))
    uDot = -gamma * (u - (.1 * x1))

    return x1Dot, y1Dot, zDot, x2Dot, y2Dot, uDot
# %%

x1s = np.linspace(0,100, 10)
y1s = np.linspace(0,100, 10)
zs = np.linspace(0,100, 10)
x2s = np.linspace(0,100, 10)
y2s = np.linspace(0,100, 10)
us = np.linspace(0,100, 10)

x = [(x1, y1, z, x2, y2, u) for x1 in x1s for y1 in y1s for z in zs for x2 in x2s for y2 in y2s for u in us]
x_dot = [epileptor(*i) for i in x]

# %%

model = ps.SINDy()
model.fit(x, t = None, x_dot=x_dot, multiple_trajectories=True)
model.print()





# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso

from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control
import pysindy as ps

# bad code but allows us to ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# %%
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12
dt = .002

t_train = np.arange(0, 10, dt)
x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(lorenz, t_train_span, x0_train,
                    t_eval=t_train, **integrator_keywords).y.T

# %%
model = ps.SINDy()
model.fit(x_train, t=dt)
model.print()
# %%
