import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn-whitegrid')

import statsmodels.api as sm

def plot_trend_seasonal_residual(data):
    decomposition = sm.tsa.seasonal_decompose(data['Price'], model='additive')
    fig = decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()

def plot_ACF_PACF(data):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('ACF & PACF for Train set')
    sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=30, ax=ax[0])
    sm.graphics.tsa.plot_pacf(data.values.squeeze(), lags=30, ax=ax[1])

def plot_model_fit(data, train_data , model, type='raw'):
    if type=='raw':
        fit_arima_value_tbl = pd.DataFrame(model.predict_in_sample(), index=train_data.index)
    elif type=='log':
        fit_arima_value_tbl = pd.DataFrame(np.exp(model.predict_in_sample()), index=train_data.index)
    plt.figure(figsize=(10, 5))
    plt.plot(data, label='Real Data', color='k')
    plt.plot(fit_arima_value_tbl, label='ARIMA Forecast', color='r')
    plt.legend(loc='upper left')
    plt.show()

def plot_model_ACF_PACF(model):
    fig, ax = plt.subplots(1, 2, figsize=(15, 3))
    fig.suptitle('ACF & PACF', size=15)
    sm.graphics.tsa.plot_acf(model.resid()[1:], lags=12, ax=ax[0])
    sm.graphics.tsa.plot_pacf(model.resid()[1:], lags=12, ax=ax[1])

def plot_rolling_predictions(ret, rolling_predictions_arch, rolling_predictions_garch):
    plt.figure(figsize=(20, 10))
    true, = plt.plot(ret[-365 * 2:], 'gray')
    preds, = plt.plot(rolling_predictions_arch, 'r')
    preds, = plt.plot(rolling_predictions_garch, 'b')
    plt.title('ARCH & GARCH : Volatility Prediction - Rolling Forecast', fontsize=20)
    plt.legend(['True Returns', 'Predicted Volatility'], fontsize=20)