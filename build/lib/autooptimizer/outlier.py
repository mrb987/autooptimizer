import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.simplefilter('ignore')

def interquartile_removal(dataset, axis=0, output='df'):
    dataset = pd.DataFrame(dataset)
    column = np.array([])
    index = np.array([])
    for col in dataset.columns:
        if dataset[col].dtypes != object:
            array = dataset[col]
            Q1 = np.quantile(array, 0.25)
            Q3 = np.quantile(array, 0.75)
            IQR = Q3 - Q1
            lower_extreme = Q1 - (1.5 * IQR)
            upper_extreme = Q3 + (1.5 * IQR)
            for i in array:
                if i < lower_extreme or i > upper_extreme:
                    index_number = (dataset[col][dataset[col] == i].index[0])
                    index = np.append(index, index_number)
                    column = np.append(column, col)
    index = np.unique(index)
    if axis==0:
        dataset.drop(index, axis=0, inplace=True)
    elif axis==1:
        dataset.drop(column, axis=1, inplace=True)
    if output == 'df':
        return dataset
    elif output == 'ar':
        return np.array(dataset)
    
def zscore_removal(dataset, axis=0, output='df'):
    dataset = pd.DataFrame(dataset)
    column = np.array([])
    index = np.array([])
    for col in dataset.columns:
        if dataset[col].dtypes !=object:
            array = dataset[col].values
            threshold = 3
            mean = np.mean(array)
            std = np.std(array, ddof=0)
            for i in array:
                if abs((i - mean) / std) > threshold:
                    index_number = (dataset[col][dataset[col] == i].index[0])
                    index = np.append(index, index_number)
                    column = np.append(column, col)
    index = np.unique(index)
    if axis==0:
        dataset.drop(index, axis=0,inplace=True)
    elif axis==1:
        dataset.drop(column, axis=1, inplace=True)
    if output =='df':
        return dataset
    elif output =='ar':
        return np.array(dataset)
    
def std_removal(dataset, axis=0, output='df', threshold=3):
    dataset = pd.DataFrame(dataset)
    column = np.array([])
    index = np.array([])
    for col in dataset.columns:
        if dataset[col].dtypes !=object:
            array = dataset[col].values
            mean = np.mean(array)
            std = np.std(array, ddof=0)
            upper_limit = mean + threshold * std
            lower_limit = mean - threshold * std
            for i in array:
                if i > upper_limit or i < lower_limit:
                    index_number = (dataset[col][dataset[col]==i].index[0])
                    index = np.append(index, index_number)
                    column = np.append(column, col)
    index = np.unique(index)
    if axis==0:
        dataset.drop(index, axis=0,inplace=True)
    elif axis==1:
        dataset.drop(column, axis=1, inplace=True)
    if output =='df':
        return dataset
    elif output =='ar':
        return np.array(dataset)
    
def lof_removal(dataset, output='df', threshold=10):
    dataset = pd.DataFrame(dataset)
    LOF = LocalOutlierFactor(n_neighbors=threshold)
    Y_hat = LOF.fit_predict(dataset)
    mask = Y_hat != -1
    dataset = dataset[mask]
    if output =='df':
        return dataset
    elif output =='ar':
        return np.array(dataset)
