import pandas as pd
import numpy as np
import os 
import csv
from sklearn.decomposition import PCA

'''

SupportFUnctions.py contains all support functions used 
in Thomas Theodor Kjølbye's Master Thesis. 

'''


def partition_dates(iteration):
    
    """
    Update data split according to iteration 
    
 
    Parameters
    ----------
    iteration : integer
        Iteration
        
        
    Returns
    -------
        Split dates as integers in tuple 
        
    """
    
    tr_length = 18; v_length = 12; t_length = 1
    
    tr_start = 195701 + 100 * iteration # f.o.m.
    tr_end = tr_start + 100 * tr_length # til
    
    v_start = tr_end # f.o.m.
    v_end = v_start + 100 * v_length  # til
    
    t_start = v_end # f.o.m.
    t_end = t_start + 100 * t_length  # til
    
    
    return tr_start, tr_end, v_start, v_end, t_start, t_end


def data_partition(data, start_date, end_date):
    
    """
    Splits data according to specified dates 
        
    """
    
    return data[(data["date"] >= start_date) & 
                (data["date"] < end_date)].set_index(["permno", "date"])
    

def firm_macro_data(data):
    
    """
    Separates firm and macro data
        
    """
    
    macro_columns = ["constant", "dp_macro", 
                     "ep_macro", "bm_macro", 
                     "ntis", "tbl", "tms", 
                     "dfy", "svar"]
    
    return data.drop(macro_columns, axis = 1), data[macro_columns]

def drop_NaN_or_constant(data):
    
    """
    Drops columns for which there are only constant 
    or NaN observations
        
    """
    
    #drop_zero = [column for column in data.columns if (data[column] == 0).all()]
    drop_NaN = [column for column in data.columns if ((data[column] == 0) | (data[column].isna())).all()]
    drop_constant = [column for column in data.columns if (pd.Series.nunique(data[column]) == 1)]

    for i in drop_NaN:
        if i not in drop_constant:
            drop_constant.append(i)
    
    return drop_constant


def downcast(data, output = True): 
    
    """
    Downcast converts 64bit float and ints to 32bits thus reducing the 
    memory usage and computational time of subsequent operations. 
    
 
    Parameters
    ----------
    data : pd.dataframe 
        The data to convert
        
    output : boolean
        Boolan indicating whether to print out memory usage and dtypes
        before and after conversion
        
        
    Returns
    -------
        None
        
    """
    
    if output:
        print("Before downcast: {:1.3f} GB and {}".format(data.memory_usage().sum()/(1024 ** 3), data.dtypes.value_counts()))
        #print("Before downcast: {}".format(data.dtypes.value_counts()))
        
    i = 0
    
    for column in data:
        
        i+= 1
        #print(i)
        
        if data[column].dtype == 'float64':
            data[column]=pd.to_numeric(data[column], downcast='float')
        if data[column].dtype == 'int64':
            data[column]=pd.to_numeric(data[column], downcast='integer')
        
        
    
    if output:
        print(f'After downcast: {data.memory_usage().sum()/(1024 ** 3):1.3f} GB and {data.dtypes.value_counts()}')
        #print(f'After downcast: {data.dtypes.value_counts()}')
        

def industry_partition(data, start_date, end_date):
    
    """
    Separates industry data
        
    """
    
    return pd.get_dummies(data[(data.index.get_level_values("date") >= start_date) & 
                                                  (data.index.get_level_values("date") < end_date)].sic2)


def drop_missing_industry(tr_ind, v_ind, t_ind):
    
    """
    Drops non-repped industries
        
    """
    
    # Industries in each set
    tr_industry = [col for col in tr_ind.columns]
    v_industry = [col for col in v_ind.columns]
    t_industry = [col for col in t_ind.columns]
    
    # Industries represented in all sets
    # Could've used validation or test
    # set as well
    repped_industries = [col for col in tr_industry if col in tr_industry
                         and col in v_industry and col in t_industry]
    
    # Keep repped industries
    tr_ind = tr_ind[repped_industries]
    v_ind = v_ind[repped_industries]
    t_ind = t_ind[repped_industries]
    
    return tr_ind, v_ind, t_ind


def returns_partition(data, start_date, end_date):
    
    """
    Separates returns data
        
    """
    
    return data[(data.index.get_level_values("date") >= start_date) & 
                (data.index.get_level_values("date") < end_date)].reset_index()