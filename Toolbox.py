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
    
    """
    interaction_noRAM computes the interaction terms of firm_data and macro_data
    one line at a time, i.e. without loading either of the two files to memory. 
    Instad, it saves the interaction terms directly to disc. 
    
    
    Parameters
    ----------
    firm_data : pd.dataframe 
        First data set from which to compute interaction terms
        
    macro_data : pd.dataframe
        Second data set from which to compute interaction terms
        
    mean : np.array
        Mean of all columns post interaction term computation
        
    std : np.array
        Standard deviation off all columns post interaction term computation
    
    filename : string
        Name of file to save to
        
        
    Returns
    -------
        None
        
    """
    
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
 
    

def data_processing(data, TV_date, V_date, T_date):
    
    """
    data_processing separates the data across two dimensions: firm vs macro 
    and traning vs validation vs test for a total of six data sets leveraged
    as input in the functions interaction() and interaction_noRAM(). Additionally,
    the function also cleans the data for NaN or observations and removes highly
    correlated variables. 
        
    Parameters
    ----------
    data : pd.dataframe 
        The data set to process
        
    TV_date : int
        The training valuation date is the date separating the training and validation
        set
        
    V_date : int
        The validation date marks the end of the validation set
        
    T_date : int
        The test date marks the beginning of the test set. Due to forward-chaining CV
        the V_date and T_date need not be identical. I refer to Gu, Kelly, and Xiu (2020)
        or my thesis for more information. 
        
        
    Returns
    -------
        6 pd.dataframes of processed data ready for the computation of the interaction terms. 
        
    """
    
    # Split data up according to dates
    data_training = data[data["date"] < TV_date].set_index(["permno", "date"])
    data_validation = data[(data["date"] >= TV_date) & (data["date"] < V_date)].set_index(["permno", "date"])
    data_test = data[(data["date"] >= T_date) & (data["date"] < 201701)].set_index(["permno", "date"])
    
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
    firm_validation = firm_validation.drop(to_drop, axis = 1)
    firm_test = firm_test.drop(to_drop, axis = 1)
    
    # Drop columns for which NaN or 0 are the only observations
    to_drop_nan = [column for column in firm_training.columns if (firm_training[column] == 0).all()]
    to_drop_zero = [column for column in firm_training.columns if ((firm_training[column] == 0) | (firm_training[column].isna())).all()]
    to_drop_sum = to_drop_nan + to_drop_zero
    
    firm_training = firm_training.drop(to_drop_sum, axis = 1)
    firm_validation = firm_validation.drop(to_drop_sum, axis = 1)
    firm_test = firm_test.drop(to_drop_sum, axis = 1)
    
    return firm_training, firm_validation, firm_test, macro_training, macro_validation, macro_test



def loadtxt(name):
    
    """
    loadtxt loads the file "name" and yields each line at a time as a 
    np.array
    
    
    Parameters
    ----------
    name : string 
        File to load
        
        
    Yields
    -------
        One line of file "name" 
        
    """
    
    with open(name, "r") as file:
        
        for line in file:
            line_int = np.fromstring(line, sep = ",").reshape(-1,1).T # 1xK 2D array
            
            yield line_int

            
            
def pca_each_line(name, pc):
    
    """
    pca_each_line employs PCA on each line of the file "name". One line 
    at a time. The principcal components are computed from the training 
    data
    
    
    Parameters
    ----------
    name : string
        File to load
        
    pc : np.array
        Principal components of PCA
        
        
    Yields
    -------
        One line of file "name" post-PCA 
        
    """
    
    for line in loadtxt(name):
        pca_line = np.dot(line, pc)
        to_append = (pca_line[0]).tolist()
        
        yield to_append       
        

        
def save_txt(name, newfilename, pc):
    
    """
    save_txt saves the PCA transformed file "name" to a new file
    "newfilename" 
    
    
    Parameters
    ----------
    name : string  
        File to load
        
    pc : np.array
        Principal components of PCA
        
    newfilname : string
        New file to save too
        
        
    Returns / Yields
    -------
        None
        
    """
    
    with open(os.path.dirname(os.getcwd()) + "\\" + newfilename, "a", newline = "") as newfile:
        writer = csv.writer(newfile)
        for newline in pca_each_line(name, pc):
            writer.writerow(newline)
            
            
            
def dummies(data, TV_date, V_date, T_date):
    
    """
    dummies converts the 1-column vector of industry categories to a 
    P-dimensional matrix of industry dummies with P corresponding to the
    number of industries. Moreover, it filters out categories not represented
    in the training data (cannot predict on industries not in the training data)
    
    
    Parameters
    ----------
        Parameters
    ----------
    data : pd.dataframe 
        The data set (dummies) to process
        
    TV_date : int
        The training valuation date is the date separating the training and validation
        set
        
    V_date : int
        The validation date marks the end of the validation set
        
    T_date : int
        The test date marks the beginning of the test set. Due to forward-chaining CV
        the V_date and T_date need not be identical. I refer to Gu, Kelly, and Xiu (2020)
        or my thesis for more information. 
        
        
    Returns
    -------
        3 pd.dataframes of 
        
        
    """
    
    # Split industry dummies according to dates
    industry_dummies_t = pd.get_dummies(data[data.index.get_level_values("date") < TV_date])
    industry_dummies_v = pd.get_dummies(data[(data.index.get_level_values("date") >= TV_date) & 
                                                  (data.index.get_level_values("date") < V_date)])
    industry_dummies_tt = pd.get_dummies(data[data.index.get_level_values("date") >= T_date])
    
    # Drop industries not represented in the training data
    col_training = [col for col in industry_dummies_t.columns]
    col_validation = [col for col in industry_dummies_v.columns] 
    col_test =  [col for col in industry_dummies_tt.columns] 
    
    # To drop
    to_keep_validation = [col for col in col_training and col_validation if col in col_training and col_validation]
    to_keep_test = [col for col in col_training and col_test if col in col_training and col_test]
    
    # Filter
    industry_dummies_v = industry_dummies_v[to_keep_validation]
    industry_dummies_tt = industry_dummies_tt[to_keep_test]
    
    return industry_dummies_t.reset_index(), industry_dummies_v.reset_index(), industry_dummies_tt.reset_index()

  
        

        
        

# For pænt setup: https://github.com/bkelly-lab/ipca/blob/master/ipca/ipca.py

        
        
    
    
 