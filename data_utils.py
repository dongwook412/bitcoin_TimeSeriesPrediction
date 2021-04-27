import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_load(directory, test_proportion=0.2, type='raw'):
    data = pd.read_csv(directory)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

    if type == 'difference':
        data = data.diff().dropna()

    if type == 'log_difference':
        data = np.log(data).diff().dropna()

    if type == 'log':
        data = np.log(data)

    train_data, test_data = train_test_split(data, test_size=test_proportion, shuffle=False)

    return data, train_data, test_data

def arch_data_load(data, test_proportion=0.2):
    price = data.Price.dropna()
    ret = pd.DataFrame(np.diff(np.log(data.Price.dropna())), data.index[1:]) * 100
    ret.columns = ['Return']

    train_data, test_data = train_test_split(ret, test_size=test_proportion, shuffle=False)

    return price, ret, train_data, test_data