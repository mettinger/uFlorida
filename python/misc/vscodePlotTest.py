#%%
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np

import plotly.io as pio
#pio.renderers.default = 'plotly_mimetype+notebook' 
# %%

pio.renderers

# %%
plt.plot(range(10), range(10))
# %%


px.scatter(x = range(10), y = range(10))
# %%
