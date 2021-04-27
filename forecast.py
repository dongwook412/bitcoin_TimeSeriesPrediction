import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from arch import arch_model

def forecast(model, test_data, type='raw'):
    pred_arima = model.predict(len(test_data), return_conf_int=True)
    if type=='raw':
        pred_arima_value = pred_arima[0]
        pred_arima_ub = pred_arima[1][:, 0]
        pred_arima_lb = pred_arima[1][:, 1]
        rmse_arima = sqrt(mean_squared_error(test_data, pred_arima_value))
    elif type=='log':
        pred_arima_value = np.exp(pred_arima[0])
        pred_arima_ub = np.exp(pred_arima[1][:, 0])
        pred_arima_lb = np.exp(pred_arima[1][:, 1])
        rmse_arima = sqrt(mean_squared_error(np.exp(test_data), pred_arima_value))
    pred_arima_idx = list(test_data.index)

    return pred_arima_value, pred_arima_ub, pred_arima_lb, pred_arima_idx, rmse_arima

def plot_forecast(data, model, test_data, type='raw'):
    pred_arima_value, pred_arima_ub, pred_arima_lb, pred_arima_idx, rmse_arima = forecast(model, test_data, type)

    pred_arima_value_tbl = pd.DataFrame(pred_arima_value, index=test_data.index)

    plt.figure(figsize=(20, 10))
    plt.plot(data, label='Real Data', color='k')
    plt.plot(pred_arima_value_tbl, label='SARIMA Forecast', color='r')
    plt.fill_between(pred_arima_idx, pred_arima_lb, pred_arima_ub, color='r', alpha=0.1,
                     label='95% Prediction Interval')
    plt.legend(loc='upper left')
    plt.suptitle(f'SARIMA {model.order} /. MSE : {round(rmse_arima, 2)}', size=30)
    plt.show()

def rolling_prediction(data, ret, lag_p, lag_q):
    rolling_predictions = []
    test_size = 365 * 2

    for i in range(test_size):
        train = ret[:-(test_size - i)]
        res = arch_model(train, p=lag_p, q=lag_q).fit(disp='off')
        pred = res.forecast(horizon=1)
        rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))

    rolling_predictions = pd.Series(rolling_predictions, index=data.index[-365 * 2:])

    return rolling_predictions
