# imports
import folium
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib as mpl
import statsmodels.tsa.api as tsa
from IPython.display import display
import numpy as np
import pmdarima as pm
import warnings
import json
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

# functions


def forecast_to_df(forecast, zipcode):
    test_pred = forecast.conf_int()
    test_pred[zipcode] = forecast.predicted_mean
    test_pred.columns = ['lower', 'upper', 'prediction']
    return test_pred


def plot_train_test_pred(train, test, pred_df, figsize=(15, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train, label='Train', marker='o')
    ax.plot(test, label='Test', marker='o')
    ax.plot(pred_df['prediction'], label='prediction', ls='--', marker='o')
    ax.fill_between(x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'])
    ax.legend()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_acf_pacf(ts, figsize=(15, 8), lags=24):
    """
    Plots acf and pacf
    """
    fig, ax = plt.subplots(nrows=3, figsize=figsize)
    # Plot ts
    ts.plot(ax=ax[0], color='#ff6961', lw=5)
    # Plot acf, pavf
    plot_acf(ts, ax=ax[1], lags=lags, color='#ff6961', lw=5)
    plot_pacf(ts, ax=ax[2], lags=lags, method='ld', color='#ff6961', lw=5)
    fig.tight_layout()
    fig.suptitle(f"Zipcode: {ts.name}", y=1.02, fontsize=15)
    for a in ax[1:]:
        a.xaxis.set_major_locator(
            mpl.ticker.MaxNLocator(min_n_ticks=lags, integer=True))
        a.xaxis.grid()
    plt.show()
    return fig, ax

# def melt_data(df):
#     """
#     Reshapes long form dataset to wide format.
#     _________________________________________________________________________
#     Predefined function.
#     defaults:
#         id_vars=['RegionID', 'SizeRank', 'RegionName', 'City', 'State',
#                  'Metro', 'CountyName','ROI_10yr','ROI_5yr','ROI_3yr']
#         var_name='date'

#     Parameters:
#     ===========
#     df = pandas.DataFrame; No default.

#     --version 0.0.2--
#     """
#     melted = pd.melt(df,
#                      id_vars=[
#                          'RegionID', 'SizeRank', 'RegionName', 'City', 'State',
#                          'Metro', 'CountyName','ROI_10yr','ROI_5yr','ROI_3yr'
#                      ],
#                      var_name='date')
#     melted['date'] = pd.to_datetime(melted['date'], infer_datetime_format=True)
#     melted = melted.dropna(subset=['value'])
#     return melted


def adfuller_test_df(ts, index=['AD Fuller Results']):
    """Returns the AD Fuller Test Results and p-values for the null hypothesis
    that there the data is non-stationary (that there is a unit root in the data)"""

    df_res = tsa.stattools.adfuller(ts)

    names = ['Test Statistic', 'p-value',
             '#Lags Used', '# of Observations Used']
    res = dict(zip(names, df_res[:4]))

    res['p<.05'] = res['p-value'] < .05
    res['Stationary?'] = res['p<.05']

    if isinstance(index, str):
        index = [index]
    res_df = pd.DataFrame(res, index=index)
    res_df = res_df[['Test Statistic', '#Lags Used',
                     '# of Observations Used', 'p-value', 'p<.05',
                     'Stationary?']]
    return res_df


def stationarity_check(TS, window=8, plot=True, index=['ADF Result']):
    """Adapted from https://github.com/learn-co-curriculum/dsc-removing-trends-lab/tree/solution"""

    # Calculate rolling statistics
    roll_mean = TS.rolling(window=window, center=False).mean()
    roll_std = TS.rolling(window=window, center=False).std()

    # Perform the Dickey Fuller Test
    dftest = adfuller_test_df(TS, index=index)

    if plot:

        # Building in contingency if not a series with a freq
        try:
            freq = TS.index.freq
        except:
            freq = 'N/A'

        # Plot rolling statistics:
        fig = plt.figure(figsize=(12, 6))
        plt.plot(TS, color='blue', label=f'Original (freq={freq}')
        plt.plot(roll_mean,
                 color='red',
                 label=f'Rolling Mean (window={window})')
        plt.plot(roll_std,
                 color='black',
                 label=f'Rolling Std (window={window})')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        # display(dftest)
        plt.show(block=False)

    return dftest


def return_on_investment(df, start_date, end_date):
    """
    The logarithmic return or continuously compounded return, also known as force of interest.

    Parameters:
    ===========
    df = pandas.DataFrame; No default.
    start_date = str, date in year-month format; No default.
    end_date = str, date in year-month format; No default.

    """
    return np.log(df[end_date] / df[start_date])


def melt_data(df):
    melted = pd.melt(df,
                     id_vars=['RegionName', 'State',
                              'City', 'Metro', 'CountyName'],
                     var_name='date')
    melted['date'] = pd.to_datetime(melted['date'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted


def model_builder(train, test, order, seasonal_order, zipcode, figsize=(15, 8)):
    # model
    model = tsa.SARIMAX(train, order=order,
                        seasonal_order=seasonal_order).fit()
    display(model.summary())
    print('\033[1m \033[5;30;47m' +
          f'{" "*70}Model Diagonostics of {zipcode}{" "*70}'+'\033[0m')
    model.plot_diagnostics(figsize=figsize)
    plt.tight_layout()
    plt.show()
    # forecast
    forecast = model.get_forecast(steps=len(test))
    pred_df = forecast_to_df(forecast, zipcode)
    print('\033[1m \033[5;30;47m' +
          f'{" "*70}Performance on test data of {zipcode}{" "*70}'+'\033[0m')
    plot_train_test_pred(train, test, pred_df)
    plt.show()

    return model


def plot_train_test_pred_forecast(train, test, pred_df_test, pred_df, zipcode):
    fig, ax = plt.subplots(figsize=(15, 5))
    kws = dict(marker='*')
    ax.plot(train, label='Train', **kws)
    ax.plot(test, label='Test', **kws)

    ax.plot(pred_df_test['prediction'],
            label='prediction',
            ls='--',
            **kws,
            color='gold')
    ax.fill_between(x=pred_df_test.index,
                    y1=pred_df_test['lower'],
                    y2=pred_df_test['upper'],
                    color='silver')

    ax.plot(pred_df['prediction'],
            label='forecast',
            ls='--',
            **kws,
            color='silver')
    ax.fill_between(x=pred_df.index,
                    y1=pred_df['lower'],
                    y2=pred_df['upper'],
                    color='r')
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.tight_layout()
    plt.show()
    mean_roi = np.log(pred_df[-1:]['prediction'][0] / test[-1:][0])
    lower_roi = np.log(pred_df[-1:]['lower'][0] / test[-1:][0])
    upper_roi = np.log(pred_df[-1:]['upper'][0] / test[-1:][0])
    std_roi = np.std([lower_roi, upper_roi])
    roi_df = pd.DataFrame([{
        'zipcode': zipcode,
        'mean_forecasted_roi': mean_roi,
        'lower_forecasted_roi': lower_roi,
        'upper_forecasted_roi': upper_roi,
        'std_forecasted_roi': std_roi
    }])
    display(roi_df)
    return fig, ax, roi_df


def grid_search(ts, train, test, figsize=(15, 5), trace=True):
    # grid searching using pyramid arima
    auto_model = pm.auto_arima(y=train,
                               X=None,
                               start_p=0,
                               d=1,
                               start_q=0,
                               max_p=2,
                               max_d=2,
                               max_q=2,
                               start_P=0,
                               D=1,
                               start_Q=0,
                               max_P=2,
                               max_D=2,
                               max_Q=2,
                               max_order=None,
                               m=12,
                               seasonal=True,
                               stationary=False,
                               information_criterion='oob',
                               alpha=0.05,
                               test='kpss',
                               seasonal_test='OCSB',
                               stepwise=True,
                               suppress_warnings=True,
                               error_action='warn',
                               trace=trace,
                               out_of_sample_size=36,
                               scoring='mse',
                               method='lbfgs',
                               )
    # display results of grid search
    display(auto_model.summary())
    auto_model.plot_diagnostics(figsize=figsize)
    plt.tight_layout()
    plt.show()
    # fitting model on train data with the best params found by grid search
    best_model = tsa.SARIMAX(train,
                             order=auto_model.order,
                             seasonal_order=auto_model.seasonal_order, maxiter=500,
                             enforce_invertibility=False).fit()
    forecast = best_model.get_forecast(steps=len(test))
    pred_df_test = pd.DataFrame([forecast.conf_int(
    ).iloc[:, 0], forecast.conf_int().iloc[:, 1], forecast.predicted_mean]).T
    pred_df_test.columns = ["lower", 'upper', 'prediction']
    plot_train_test_pred(train, test, pred_df_test)
    plt.show()

    # fitting on entire data
    best_model_all = tsa.SARIMAX(ts,
                                 order=auto_model.order,
                                 seasonal_order=auto_model.seasonal_order, maxiter=500,
                                 enforce_invertibility=False).fit()
    forecast = best_model_all.get_forecast(steps=36)
    pred_df = pd.DataFrame([forecast.conf_int(
    ).iloc[:, 0], forecast.conf_int().iloc[:, 1], forecast.predicted_mean]).T
    pred_df.columns = ["lower", 'upper', 'prediction']
    plot_train_test_pred(train, test, pred_df)
    plt.show()
    zipcode = ts.name
    plot_train_test_pred_forecast(train, test, pred_df_test, pred_df, zipcode)

    return auto_model, pred_df


def model_loop(ts_df,
               zipcode_list,
               train_size=.8,
               steps=36,
               display_results=True,
               plot_diagnostics=False):
    # store results
    RESULTS = {}
    # store ROI information
    ROI = pd.DataFrame(columns=[
        'zipcode', 'mean_forecasted_roi', 'lower_forecasted_roi',
        'upper_forecasted_roi', 'std_forecasted_roi'
    ])
    # store param information
    PARAMS = {}
    # loop counter
    n = 0
    for zipcode in zipcode_list:
        # loop counter
        n = n + 1
        len_ = len(zipcode_list)
        print(f"""Working on #{n} out of {len_} zipcodes.""")
        print('Working on:', zipcode)
        # make empty dicts for storing data
        temp_dict = {}
        temp_dict_1 = {}
        # make a copy of time series data
        ts = ts_df[zipcode].dropna().copy()
        # train-test split
        train_size = train_size
        split_idx = round(len(ts) * train_size)
        # split
        train = ts.iloc[:split_idx]
        test = ts.iloc[split_idx:]

        # Get best params using auto_arima on train-test data
        model, pred_df = grid_search(
            ts, train, test, figsize=(15, 5), trace=True)

        # storing data in RESULTS
        temp_dict['pred_df'] = pred_df
        temp_dict['model'] = model
        temp_dict['train'] = train
        temp_dict['test'] = test
        RESULTS[zipcode] = temp_dict
        # storing data in PARAMS
        PARAMS[zipcode] = [model.order, model.seasonal_order]
        # storing data in ROI
        mean_roi = np.log(pred_df[-1:]['prediction'][0] / test[-1:][0])
        lower_roi = np.log(pred_df[-1:]['lower'][0] / test[-1:][0])
        upper_roi = np.log(pred_df[-1:]['upper'][0] / test[-1:][0])
        std_roi = np.std([lower_roi, upper_roi])
        roi_df = pd.DataFrame([{
            'zipcode': zipcode,
            'mean_forecasted_roi': mean_roi,
            'lower_forecasted_roi': lower_roi,
            'upper_forecasted_roi': upper_roi,
            'std_forecasted_roi': std_roi
        }])
        ROI = ROI.append(roi_df, ignore_index=True)
        print('-' * 90, end='\n')
    return RESULTS, ROI, PARAMS


def zip_code_map(roi_df):
    """Return an interactive map of zip codes colorized to reflect
    expected return on investment.
    """
    # geojason_url = 'https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ny_new_york_zip_codes_geo.min.json'
    geojason= json.load(
        open('./data/ny_new_york_zip_codes_geo.min.json', 'r'))
    zip_code_map = folium.Map(
        location=[40.7027, -73.7890], width=1330, height=820, zoom_start=11)
    folium.Choropleth(
        geo_data=geojason,
        name='choropleth',
        data=roi_df,
        columns=['zipcode', 'mean_forecasted_roi'],
        key_on='feature.properties.ZCTA5CE10',
        fill_color='BuGn',
        fill_opacity=0.7,
        nan_fill_opacity=0
    ).add_to(zip_code_map)
    return zip_code_map
