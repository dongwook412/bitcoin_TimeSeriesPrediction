import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

def ADF_test(data, diff_data, log_diff_data):
    res_adf_raw = adfuller(data['Price'])
    res_adf_diff = adfuller(diff_data['Price'])
    res_adf_log_diff = adfuller(log_diff_data['Price'])
    print('raw: %f' % res_adf_raw[1])
    print('diff: %f' % res_adf_diff[1])
    print('log: %f' % res_adf_log_diff[1])  # p_value

def KPSS_test(data, diff_data, log_diff_data):
    res_kpss_raw = kpss(data['Price'], lags='auto')
    res_kpss_diff = kpss(diff_data['Price'], lags='auto')
    res_kpss_log_diff = kpss(log_diff_data['Price'], lags='auto')
    print('raw: %f' % res_kpss_raw[1])
    print('diff: %f' % res_kpss_diff[1])
    print('log: %f' % res_kpss_log_diff[1])  # p_value

def arch_ADF_KPSS_test(data):
    res_adf_raw = adfuller(data['Return'] ** 2)
    res_kpss_raw = kpss(data['Return'] ** 2, lags='auto')
    print('ADF P-value for raw: %f' % res_adf_raw[1])
    print('KPSS P-value for raw: %f' % res_kpss_raw[1])