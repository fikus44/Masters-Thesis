import pandas as pd
import numpy as np


def normalization(data):
    # Tænkte det måske skulle være en matrix man ganger med? efter man har fået ranks 
    None



def rank_transformation(data):
    
    """
    This function implements the rank transformation and normalization by 
    Gu, Kelly, and Xiu (2020). In particular, first, the function cross-
    sectionally ranks one firm-specific characteristic at a single point 
    in time across all firms. Then, it normalizes (maps) the ranks into
    the [-1:1] interval. 
    
    
    Parameters
    ----------
    data : pd.dataframe 
        The data set to transform
        
    transformation : boolean, default=1
        Determines which transformation is employed. 1 = Gu, Kelly, and
        Xiu (2020). Else linear interpolation. 
        
        
    Returns
    -------
    pd.dataframe of transformed data set 
    
    """
    # Initialize rank-transformed dataframe and partition data 
    data_transform = data[["date", "permno"]].copy()
    data_char = data.loc[:, "mvel1":"zerotrade"]
    
    for i in range(data_char.shape[1]):
        
        data_temp = data[["date", "permno"]].copy()
        data_temp["char"] = data_char.iloc[:, i]
        data_temp = data_temp.groupby("date").apply(lambda x: x.iloc[:, 2].rank(axis = 0, method = 'first')) # x refers to the df
        
        data_transform

        

        
        
    
    
    # grouper by date og så apply() https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
    
    # ellers kan jeg tage en søjle ad gangen med date + permno og så group den by date, og så sorte den ascending så give dem rank i stedet for værdi og så apply normalization. 
                         