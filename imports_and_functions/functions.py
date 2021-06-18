# imports
from pmdarima.arima.utils import ndiffs
from pmdarima.arima.utils import nsdiffs
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


def plot_test_pred(test, pred_df, figsize=(15, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(test, label='Test', marker='o', color='#ff6961', lw=4)
    ax.plot(pred_df['prediction'], label='prediction',
            ls='--', marker='o', color='#639388', lw=4)
    ax.fill_between(
        x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'], color='#938863', alpha=.5)
    ax.legend()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_train_test_pred(train, test, pred_df, figsize=(15, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train, label='Train', marker='o', color='#886393')
    ax.plot(test, label='Test', marker='o', color='#ff6961')
    ax.plot(pred_df['prediction'], label='prediction',
            ls='--', marker='o', color='#639388')
    ax.fill_between(
        x=pred_df.index, y1=pred_df['lower'], y2=pred_df['upper'], color='#938863', alpha=.5)
    ax.legend()
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_train_test_pred_forecast(train, test, pred_df_test, pred_df, zipcode, figsize=(15, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    kws = dict(marker='*')
    # train
    ax.plot(train, label='Train', **kws, color='#886393')
    # test
    ax.plot(test, label='Test', **kws, color='#ff6961')
    # prediction
    ax.plot(pred_df_test['prediction'],
            label='prediction',
            ls='--',
            **kws,
            color='#639388')
    ax.fill_between(x=pred_df_test.index,
                    y1=pred_df_test['lower'],
                    y2=pred_df_test['upper'],
                    color='#938863', alpha=.5)
    # forecast
    ax.plot(pred_df['prediction'],
            label='forecast',
            ls='--',
            **kws,
            color='#ffd700')
    ax.fill_between(x=pred_df.index,
                    y1=pred_df['lower'],
                    y2=pred_df['upper'],
                    color='#0028ff', alpha=.5)
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax.set_title('train-test-pred-forecast plot', size=15)
    fig.tight_layout()
    plt.show()
    # ROI
    mean_roi = (pred_df[-1:]['prediction'][0] - test[-1:][0])/test[-1:][0]
    lower_roi = (pred_df[-1:]['lower'][0] - test[-1:][0])/test[-1:][0]
    upper_roi = (pred_df[-1:]['upper'][0] - test[-1:][0])/test[-1:][0]
    std_roi = np.std([lower_roi, upper_roi])
    roi_df = pd.DataFrame([{
        'zipcode': zipcode,
        'mean_forecasted_roi': round(mean_roi*100, 2),
        'lower_forecasted_roi': round(lower_roi*100, 2),
        'upper_forecasted_roi': round(upper_roi*100, 2),
        'std_forecasted_roi': round(std_roi*100, 2)
    }])
    display(roi_df)
    return fig, ax, roi_df


def plot_acf_pacf(ts, figsize=(15, 8), lags=24):
    """
    Plots acf and pacf
    """
    fig, ax = plt.subplots(nrows=3, figsize=figsize)
    # Plot ts
    ts.plot(ax=ax[0], color='#886393', lw=5)
    # Plot acf, pavf
    plot_acf(ts, ax=ax[1], lags=lags, color='#886393', lw=5)
    plot_pacf(ts, ax=ax[2], lags=lags, method='ld', color='#886393', lw=5)
    fig.tight_layout()
    fig.suptitle(f"Zipcode: {ts.name}", y=1.02, fontsize=15)
    for a in ax[1:]:
        a.xaxis.set_major_locator(
            mpl.ticker.MaxNLocator(min_n_ticks=lags, integer=True))
        a.xaxis.grid()
    plt.show()
    return fig, ax


def adfuller_test_df(ts, index=['AD Fuller Results']):
    """Returns the AD Fuller Test Results and p-values for the null hypothesis
    that there the data is non-stationary (that there is a unit root in the data)

    Adapted from https://github.com/learn-co-curriculum/dsc-removing-trends-lab/tree/solution
    """

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


def stationarity_check(TS, window=8, plot=True, figsize=(15, 5), index=['ADF Result']):
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
        fig = plt.figure(figsize=figsize)
        plt.plot(TS, color='#886393', label=f'Original (freq={freq}')
        plt.plot(roll_mean,
                 color='#ff6961',
                 label=f'Rolling Mean (window={window})')
        plt.plot(roll_std,
                 color='#333333',
                 label=f'Rolling Std (window={window})')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation', size=15)
        # display(dftest)
        plt.show(block=False)

    return dftest


# def return_on_investment(df, start_date, end_date):
#     """
#     The logarithmic return or continuously compounded return, also known as force of interest.

#     Parameters:
#     ===========
#     df = pandas.DataFrame; No default.
#     start_date = str, date in year-month format; No default.
#     end_date = str, date in year-month format; No default.

#     """
#     return np.log(df[end_date] / df[start_date])


def melt_data(df):
    melted = pd.melt(df,
                     id_vars=['RegionName', 'State',
                              'City', 'Metro', 'CountyName'],
                     var_name='date')
    melted['date'] = pd.to_datetime(melted['date'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted


def model_builder(train, test, order, seasonal_order, zipcode, figsize=(15, 8), show_summary=True, show_diagnostics=True, show_prediction=True):
    # model
    model = tsa.SARIMAX(train, order=order,
                        seasonal_order=seasonal_order).fit()
    if show_summary:
        display(model.summary())
    print('\033[1m \033[5;30;47m' +
          f'{" "*70}Model Diagonostics of {zipcode}{" "*70}'+'\033[0m')
    if show_diagnostics:
        model.plot_diagnostics(figsize=figsize)
        plt.tight_layout()
        plt.show()
    # forecast
    forecast = model.get_forecast(steps=len(test))
    pred_df = forecast_to_df(forecast, zipcode)
    if show_prediction:
        print('\033[1m \033[5;30;47m' +
              f'{" "*70}Performance on test data of {zipcode}{" "*70}'+'\033[0m')
        _, ax = plot_test_pred(
            test, pred_df, figsize=(figsize[0], figsize[1]/2))
        ax.set_title('test-pred plot', size=15)
        _, ax = plot_train_test_pred(
            train, test, pred_df, figsize=(figsize[0], figsize[1]-2))
        ax.set_title('train-test-pred plot', size=15)
        plt.show()

    return model, pred_df


def grid_search(ts, train, test, forecast_steps=36, figsize=(15, 5), trace=True, display_results=True, display_roi_results=True):
    """# grid searching using pyramid arima"""

    d = ndiffs(train, alpha=0.05, test='adf', max_d=4)
    D = nsdiffs(train, m=12, test='ocsb', max_D=4)

    auto_model = pm.auto_arima(y=train,
                               X=None,
                               start_p=0,
                               d=d,
                               start_q=0,
                               max_p=3,
                               #    max_d=d+1,
                               max_q=3,
                               start_P=0,
                               D=D,
                               start_Q=0,
                               max_P=3,
                               #    max_D=D+1,
                               max_Q=3,
                               #    max_order=None,
                               m=12,
                               seasonal=True,
                               stationary=False,
                               information_criterion='oob',  # 'oob','aicc', 'bic', 'hqic',
                               #    alpha=0.05,
                               #    test='dft',
                               #    seasonal_test='ocsb',  # 'ch'
                               stepwise=True,
                               suppress_warnings=True,
                               error_action='warn',
                               trace=trace,
                               out_of_sample_size=12,
                               scoring='mse',
                               method='lbfgs',
                               )
    # display results of grid search
    zipcode = ts.name
    if display_results:
        print('\033[1m \033[5;30;47m' +
              f'{" "*70}Model Diagonostics of {zipcode}{" "*70}'+'\033[0m')
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
    if display_results:
        print('\033[1m \033[5;30;47m' +
              f'{" "*70}Performance on test data of {zipcode}{" "*70}'+'\033[0m')
        _, ax = plot_test_pred(
            test, pred_df_test, figsize=(figsize[0], figsize[1]/2))
        ax.set_title('test-pred plot', size=15)

        _, ax = plot_train_test_pred(
            train, test, pred_df_test, figsize=figsize)
        ax.set_title('train-test-pred plot', size=15)
        plt.show()

    # fitting on entire data
    best_model_all = tsa.SARIMAX(ts,
                                 order=auto_model.order,
                                 seasonal_order=auto_model.seasonal_order, maxiter=500,
                                 enforce_invertibility=False).fit()
    forecast = best_model_all.get_forecast(steps=forecast_steps)
    pred_df = pd.DataFrame([forecast.conf_int(
    ).iloc[:, 0], forecast.conf_int().iloc[:, 1], forecast.predicted_mean]).T
    pred_df.columns = ["lower", 'upper', 'prediction']
    if display_results:
        print('\033[1m \033[5;30;47m' +
              f'{" "*70}Forecast of {zipcode}{" "*70}'+'\033[0m')
        _, ax = plot_train_test_pred(train, test, pred_df, figsize=figsize)
        ax.set_title('train-test-forecast plot', size=15)
        plt.show()
    if display_roi_results:
        plot_train_test_pred_forecast(
            train, test, pred_df_test, pred_df, zipcode, figsize=figsize)

    return auto_model, pred_df_test, pred_df


def model_loop(ts_df,
               zipcode_list,
               train_size=.8, show_grid_search_steps=True,
               forecast_steps=36, figsize=(15, 5),
               display_details=False):
    # store results
    RESULTS = {}
    # store ROI information
    ROI = pd.DataFrame(columns=[
        'zipcode', 'mean_forecasted_roi', 'lower_forecasted_roi',
        'upper_forecasted_roi', 'std_forecasted_roi'
    ])
    # # store param information
    # PARAMS = {}

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
        # temp_dict_1 = {}

        # make a copy of time series data
        ts = ts_df[zipcode].dropna().copy()
        # train-test split
        train_size = train_size
        split_idx = round(len(ts) * train_size)
        # split
        train = ts.iloc[:split_idx]
        test = ts.iloc[split_idx:]

        # Get best params using auto_arima on train-test data
        if display_details:
            display_results_gs = True
            display_roi_results_gs = True
        else:
            display_results_gs = False
            display_roi_results_gs = False

        model, pred_df_test, pred_df = grid_search(ts,
                                                   train,
                                                   test,
                                                   forecast_steps=forecast_steps,
                                                   figsize=figsize,
                                                   trace=show_grid_search_steps,
                                                   display_results=display_results_gs,
                                                   display_roi_results=display_roi_results_gs)

        # storing data in RESULTS
        temp_dict['model'] = model # .to_dict() can be used to save space
        temp_dict['train'] = train
        temp_dict['test'] = test
        temp_dict['pred_df_test'] = pred_df_test
        temp_dict['pred_df'] = pred_df

        RESULTS[zipcode] = temp_dict

        # storing data in ROI
        mean_roi = (pred_df[-1:]['prediction'][0] - test[-1:][0])/test[-1:][0]
        lower_roi = (pred_df[-1:]['lower'][0] - test[-1:][0])/test[-1:][0]
        upper_roi = (pred_df[-1:]['upper'][0] - test[-1:][0])/test[-1:][0]
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
    print('Looping completed.')
    return RESULTS, ROI


def zip_code_map(roi_df):
    """Return an interactive map of zip codes colorized to reflect
    expected return on investment.
    """
    geojason = json.load(
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


def map_zipcodes_return(df, plot_style='interactive', geojson_file_path='./data/ny_new_york_zip_codes_geo.min.json'):
    """ """
    import plotly.express as px
    geojason = json.load(open(geojson_file_path, 'r'))
    for feature in geojason['features']:
        feature['id'] = feature['properties']['ZCTA5CE10']
    fig = px.choropleth_mapbox(data_frame=df,
                               geojson=geojason,
                               locations='zipcode',
                               color='mean_forecasted_roi',
                               mapbox_style="stamen-terrain",
                               zoom=10.5,color_continuous_scale=['#FF3D70', '#FFE3E8', '#039E0F'],#range_color=[-20,20],
                               color_continuous_midpoint= 0, hover_name='Neighborhood',
                               hover_data=[
                                   'mean_forecasted_roi', 'lower_forecasted_roi',
                                   'upper_forecasted_roi', 'std_forecasted_roi'
                               ],
                               title='Zipcode by Average Price',
                               opacity=.85,
                               height=800,
                               center={
                                   'lat': 40.7027,
                                   'lon': -73.7890
                               })
    fig.update_geos(fitbounds='locations', visible=True)
    fig.update_layout(margin={"r": 0, "l": 0, "b": 0})
    if plot_style == 'interactive':
        fig.show()
    if plot_style == 'static':
        # import plotly.io as plyIo
        img_bytes = fig.to_image(format="png", width=1400, height=800, scale=1)
        from IPython.display import Image
        display(Image(img_bytes))
    if plot_style == 'dash':
        import dash
        import dash_core_components as dcc
        import dash_html_components as html
        app = dash.Dash()
        app.layout = html.Div([dcc.Graph(figure=fig)])
        app.run_server(debug=True, use_reloader=False)
