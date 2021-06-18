from jupyterthemes import jtplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import pmdarima as pm
import statsmodels.tsa.api as tsa
# from pandas.plotting import autocorrelation_plot, lag_plot
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
# plt.style.use('seaborn-talk')
# https://matplotlib.org/stable/tutorials/introductory/customizing.html
font = {'weight': 'normal', 'size': 8}
text = {'color': 'white'}
axes = {'labelcolor': 'white'}
xtick = {'color': 'white'}
ytick = {'color': 'white'}
legend = {'facecolor': 'black', 'framealpha': 0.6}

# mpl.rcParams['figure.facecolor'] = '#232323' # marplotlib over-rights this
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



                            #    X=None,
                            #    start_p=0,
                            #    d=d,
                            #    start_q=0,
                            #    max_p=2,
                            #    max_d=2,
                            #    max_q=2,
                            #    start_P=0,
                            #    D=D,
                            #    start_Q=0,
                            #    max_P=2,
                            #    max_D=2,
                            #    max_Q=2,
                            #    max_order=None,
                            #    m=12,
                            #    seasonal=True,
                            #    stationary=False,
                            #    information_criterion='oob',
                            #    alpha=0.05,
                            # #    test='dft',
                            # #    seasonal_test='ocsb',  # 'OCSB'
                            #    stepwise=True,
                            #    suppress_warnings=True,
                            #    error_action='warn',
                            #    trace=trace,
                            #    out_of_sample_size=12,
                            #    scoring='mse',
                            #    method='lbfgs',
                            # trend?
                            #    )