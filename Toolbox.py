import pandas as pd
import numpy as np
import os 
import csv

def downcast(data, output = True):
    
    """
    downcast converts 64bit float and ints to 32bits thus reducing the 
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
    
    for column in data:
        if data[column].dtype == 'float64':
            data[column]=pd.to_numeric(data[column], downcast='float')
        if data[column].dtype == 'int64':
            data[column]=pd.to_numeric(data[column], downcast='integer')
    
    if output:
        print(f'After downcast: {data.memory_usage().sum()/(1024 ** 3):1.3f} GB and {data.dtypes.value_counts()}')
        #print(f'After downcast: {data.dtypes.value_counts()}')


def dim_reduction(data, threshold, corr_matrix = False):
    
    """
    dim_reduction reduces the dimensionalty of a data set by filtering out
    highly correlated variables above a threshold value. 
    
    
    Parameters
    ----------
    data : pd.dataframe 
        The data set to filter
        
    threshold : scalar
        Threshold value determining whether a covariate gets filtered out
        
        
    Returns
    -------
        pd.dataframe of filtered data 
        
    """
    # Correlation matrix & upper triangle
    cor_matrix = data.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    
    # Columns to drop 
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Filter correlated variables out 
    #filtered_data = data.drop(data.columns[to_drop], axis = 1)
    
    if corr_matrix:
        return to_drop, cor_matrix#, filtered_data
    else:
        return to_drop#, filtered_data
                           
    
def interaction(firm_data, macro_data):
    
    # Initialize empty dataframe
    interaction = pd.DataFrame(columns = range(0), index = range(firm_data.shape[0]))
    
    for count, value in enumerate(macro_data.columns):
                          
        macro_ite = macro_data[value].values # .values returns np.array not pd.series
        product_ite = macro_ite.reshape(-1,1) * firm_data.values # 2D to make compatible for element-wise multiplication       
        column_ite = [str(col) + f'X{value}' for col in firm_data.columns] 
        df_ite = pd.DataFrame(product_ite, columns = column_ite)
        interaction = pd.concat([interaction, df_ite], axis = 1)
    

    # Save mean and standardization for validation and test set 
    mean = interaction.mean().values.reshape(-1,1).T
    std = interaction.std().values
    
    # standardize and imputate data in preperation of PCA
    interaction = interaction.apply(lambda x: (x - np.mean(x)) / np.std(x), axis = 0)
    interaction = interaction.fillna(0)
    
    return interaction, mean, std


def interaction_noRAM(firm_data, macro_data, mean, std, filename):
    
    for i in range(firm_data.shape[0]):
    
        firm_row = firm_data.iloc[i, :].values.reshape(-1,1).T # 1x85
        macro_row = macro_data.iloc[i, :].values.reshape(-1,1) # 9x1 
    
        interaction = macro_row @ firm_row # 9x1 @ 1x85 = 9x85
        interaction_flat = interaction.reshape((1, firm_data.shape[1] * macro_data.shape[1])) # 1x765
        interaction_std = (interaction_flat - mean) / std # Elementwise
        interaction_std = np.nan_to_num(interaction_std)
    
        to_append = (interaction_std[0]).tolist() 
        # https://stackoverflow.com/questions/39694318/difference-between-single-and-double-bracket-numpy-array
    
        with open(os.path.dirname(os.getcwd()) + "\\" + filename, "a", newline = "") as file:

            writer = csv.writer(file)
            writer.writerow(to_append)
    
    return None
 

def data_processing(data, start, end):
    
    # Add column of ones (useful for computing interaction term later on)
    data["constant"] = 1
    
    # Split data up according to dates
    data_training = data[data["date"] < start].set_index(["permno", "date"])
    data_validation = data[(data["date"] >= start) & (data["date"] < end)].set_index(["permno", "date"])
    data_test = data[(data["date"] >= end) & (data["date"] < 201701)].set_index(["permno", "date"])
    
    # Split data up in firm characteristics and macro data
    firm_to_drop = ["constant", "dp_macro", "ep_macro", "bm_macro", "ntis", "tbl", "tms", "dfy", "svar"]
    firm_training = data_training.drop(firm_to_drop, axis = 1)
    firm_validation = data_validation.drop(firm_to_drop, axis = 1)
    firm_test = data_test.drop(firm_to_drop, axis = 1)
    
    macro_to_keep = ["constant" ,"dp_macro", "ep_macro", "bm_macro", "ntis", "tbl", "tms", "dfy", "svar"]
    macro_training = data_training[macro_to_keep]
    macro_validation = data_validation[macro_to_keep]
    macro_test = data_test[macro_to_keep]
    
    # Drop columns based on korrelation matrix
    to_drop = dim_reduction(data = firm_training, threshold = 0.90, corr_matrix = False)
    firm_training = firm_training.drop(to_drop, axis = 1)
    firm_valiation = firm_validation.drop(to_drop, axis = 1)
    firm_test = firm_test.drop(to_drop, axis = 1)
    
    # Drop columns for which NaN or 0 are the only observations
    to_drop_nan = [column for column in firm_training.columns if (firm_training[column] == 0).all()]
    to_drop_zero = [column for column in firm_training.columns if ((firm_training[column] == 0) | (firm_training[column].isna())).all()]
    to_drop_sum = to_drop_nan + to_drop_zero
    
    firm_training = firm_training.drop(to_drop_sum, axis = 1)
    firm_validation = firm_validation.drop(to_drop_sum, axis = 1)
    firm_test = firm_test.drop(to_drop_sum, axis = 1)
    
    return firm_training, firm_validation, firm_test, macro_training, macro_validation, macro_test

    '''
    # Compute interaction terms for training and validation data and standardize
    training_data, mean, std = interaction(firm_training, macro_training)
    validation_data, _, _ = interaction(firm_validation, macro_validation)
    
    
    # Compute interaction terms for test data and standardize 
    interaction_noRAM(firm_test, macro_test, mean = mean, std = std, filename = filename)
    
    
    # Test data is saved to disc
    return training_data, validation_data 
    '''

def loadtxt(name):
    with open(name, "r") as file:
        for line in file:
            line_int = np.fromstring(line, sep = ",").reshape(-1,1).T # 1xK 2D array
            yield line_int
            
            
def pca_each_line(name, pc):  
    
    for line in loadtxt(name):
        pca_line = np.dot(line, pc)
        to_append = (pca_line[0]).tolist()
        
        yield to_append
        
        
def save_txt(name, newfilename, pc):
    with open(os.path.dirname(os.getcwd()) + "\\" + newfilename, "a", newline = "") as newfile:
        writer = csv.writer(newfile)
        for newline in pca_each_line(name, pc):
            writer.writerow(newline)
        
# For pænt setup: https://github.com/bkelly-lab/ipca/blob/master/ipca/ipca.py

        
        
    
    
 