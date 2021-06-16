from jupyterthemes import jtplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import pmdarima as pm
import statsmodels.tsa.api as tsa
from pandas.plotting import autocorrelation_plot, lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
pd.set_option('display.max_columns', 0)

# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.statespace.sarimax import SARIMAX


# import warnings
# from statsmodels.tools.sm_exceptions import ConvergenceWarning
# warnings.simplefilter('ignore', UserWarning)
# warnings.simplefilter('ignore', ConvergenceWarning)


# OG plot
jtplot.reset()
# jtplot.style(theme='monokai', context='notebook', ticks='True', grid='False')
plt.style.use('ggplot')
# https://matplotlib.org/stable/tutorials/introductory/customizing.html
font = {'weight': 'normal', 'size': 8}
text = {'color': 'white'}
axes = {'labelcolor': 'white'}
xtick = {'color': 'white'}
ytick = {'color': 'white'}
legend = {'facecolor': 'black', 'framealpha': 0.6}

# legend={'facecolor':'black'}
# legend1={'framealpha':    0.6}
# mpl.rc('legend', **legend1)
# mpl.rc('legend', **legend)


mpl.rc('legend', **legend)
mpl.rc('text', **text)
mpl.rc('xtick', **xtick)
mpl.rc('ytick', **ytick)
mpl.rc('axes', **axes)
mpl.rc('font', **font)

# NOTE: if you visualizations are too cluttered to read, try calling 'plt.gcf().autofmt_xdate()'!
# - option to reset style
# https://ozzmaker.com/add-colour-to-text-in-python/
# ctrl+k, ctrl+f OR shift+alt+f
# https://github.com/erikgregorywebb/nyc-housing/blob/master/Data/nyc-zip-codes.csv
