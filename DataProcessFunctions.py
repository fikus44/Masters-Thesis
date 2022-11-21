import pandas as pd
import numpy as np
import os 
import csv
import SupportFunctions as supp
from sklearn.decomposition import PCA


'''

DataProcessFunctions.py contains all data 
processing functions used in Thomas Theodor 
Kjølbye's Master Thesis. 

The functions are designed such that I need 
only run data_processing_complete() to 
process the entire data set

'''

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
    
    
def data_processing_new(data, iteration):
    
    """
    data_processing_new(data, iteration) calls the
    support functions:
    partition_dates(iteration), 
    data_partition(data, start_date, end_date),
    firm_macro_data(data), 
    drop_NaN_or_constant(data)
    
    as well as the dim_reduction(data, threshold,
    corr_matrix) to split, clean, and do
    preliminary data processing
    
    
    Parameters
    ----------
    data : pd.dataframe 
        The data set for which we compute interaction terms
        
    iteration : integer
        Data split iteration
        
        
    Returns
    -------
        pd.dataframe of the training, validation, and test set
        for both the firm and macro data
        
    """
    
    # Dates partitioning the training, validation, and test set 
    # according to Gu, Kelly, and Xiu (2020) by rolling-windows approach
    tr_start, tr_end, v_start, v_end, t_start, t_end = supp.partition_dates(iteration)
    
    # Split data according to dates 
    tr_data = supp.data_partition(data, tr_start, tr_end)
    v_data = supp.data_partition(data, v_start, v_end)
    t_data = supp.data_partition(data, t_start, t_end)
    
    # Separate firm and macro data
    tr_firm, tr_macro = supp.firm_macro_data(tr_data)
    v_firm, v_macro = supp.firm_macro_data(v_data)
    t_firm, t_macro = supp.firm_macro_data(t_data)
    
    # Drop columns based on korrelation matrix of training set
    drop_corr = dim_reduction(data = tr_firm, 
                            threshold = 0.90, 
                            corr_matrix = False)
    
    tr_firm = tr_firm.drop(drop_corr, axis = 1)
    v_firm = v_firm.drop(drop_corr, axis = 1)
    t_firm = t_firm.drop(drop_corr, axis = 1)
    
    # Drop columns for which NaN or 0 are the only observations
    # in the training set
    nan_or_constant = supp.drop_NaN_or_constant(tr_firm)
    
    tr_firm = tr_firm.drop(nan_or_constant, axis = 1)
    v_firm = v_firm.drop(nan_or_constant, axis = 1)
    t_firm = t_firm.drop(nan_or_constant, axis = 1)
    
    assert all(tr_firm.columns == t_firm.columns)
    assert all(tr_firm.columns == v_firm.columns)
    assert all(v_firm.columns == t_firm.columns)
    
    assert tr_firm.shape[0] == tr_macro.shape[0]
    assert v_firm.shape[0] == v_macro.shape[0]
    assert t_firm.shape[0] == t_macro.shape[0]
    
    return tr_firm, v_firm, t_firm, tr_macro, v_macro, t_macro
    
    
def interaction_new(data, iteration):
    
    """
    interaction_new(data, iteration) calls the
    data_processing_new() and interaction_terms()
    functions to compute the interaction terms of
    the training, validation, and test set
    
    
    Parameters
    ----------
    data : pd.dataframe 
        The data set for which we compute interaction terms
        
    iteration : integer
        Data split iteration
        
        
    Returns
    -------
        pd.dataframe of interaction terms for the training,
        validation, and test set
        
    """
    
    # Load training, validation, and test data 
    # for firm and macro data
    tr_firm, v_firm, t_firm, tr_macro, v_macro, t_macro = data_processing_new(data, iteration)
    
    # Compute interaction terms 
    tr_interaction, mean, std = interaction_terms(tr_firm, 
                                                  tr_macro, 
                                                  mean = None, 
                                                  std = None, 
                                                  training_set = True)
    
    v_interaction = interaction_terms(v_firm,
                                      v_macro,
                                      mean = mean,
                                      std = std,
                                      training_set = False)
    
    t_interaction = interaction_terms(t_firm,
                                      t_macro,
                                      mean = mean,
                                      std = std,
                                      training_set = False)
    
    # Check no NaN values 
    assert tr_interaction.isnull().values.any() == False
    assert v_interaction.isnull().values.any() == False
    assert t_interaction.isnull().values.any() == False
    
    return tr_interaction, v_interaction, t_interaction


def interaction_terms(firm_data, macro_data, mean, std, training_set = False): 
    
    """
    interaction_terms(firm_data, macro_data, mean, 
    std, training_set = False) computes the inter-
    action terms of the macro and firm data for 
    the training, validation, and test set.
    
    
    Parameters
    ----------
    firm_data : pd.dataframe 
        One of two data sets for which we compute interaction terms
        
    macro_data : pd.dataframe 
        Other of two data sets for which we compute interaction terms
        
    mean : pd.Series
        p-dimensional vector of means. p is equal to number
        of columns (covariates) after computing interaction
        terms
        
    std : pd.Series
        p-dimensional vector of standard deviations. p is equal 
        to number of columns (covariates) after computing 
        interaction terms
        
    training_set : Boolean
        Boolean indicating whether the interaction terms
        are computed for the training set
        
        
    Returns
    -------
        pd.dataframe of interaction terms for either
        training, validation or test set
        
    """
    
    # Initialize empty dataframe
    interaction = pd.DataFrame(columns = range(0), index = range(firm_data.shape[0]))
    
    # Computer interaction terms
    for count, value in enumerate(macro_data.columns):
        
        # .values returns np.array not pd.series
        macro_ite = macro_data[value].values
        # 2D to make compatible for element-wise multiplication
        product_ite = macro_ite.reshape(-1,1) * firm_data.values 
        
        column_ite = [str(col) + f'X{value}' for col in firm_data.columns] 
        
        df_ite = pd.DataFrame(product_ite, columns = column_ite)
        interaction = pd.concat([interaction, df_ite], axis = 1)
        
    
    # Save mean and standard deviation for validation and test set
    # Impute data and standardize training data in preperation
    # of PCA
    if training_set:
        mean = interaction.mean()#.values.reshape(-1,1).T
        std = interaction.std()#.values
        
        interaction = interaction.fillna(0)
        interaction = interaction.apply(lambda x: (x - np.mean(x)) / np.std(x), axis = 0)
                
        return interaction, mean, std
                    
    interaction = interaction.fillna(0)
    interaction = interaction.apply(lambda x: (x - mean) / std, axis = 1)                      
                          
    return interaction


def pca(data, iteration): 
    
    """
    pca(data, iteration) calls the PCA module
    from the scikit-learn API to employ PCA
    on the interaction terms.
    
    
    Parameters
    ----------
    data : pd.dataframe 
        The data set for which we compute interaction terms
        
    iteration : integer
        Data split iteration
        
        
    Returns
    -------
        pd.dataframe of post-pca for the training,
        validation, and test set
        
    """
    
    # Compute interaction terms 
    tr_interaction, v_interaction, t_interaction = interaction_new(data, iteration)
    
    # Create instance of PCA class object
    # and fit on training data
    pca = PCA(n_components = 0.95)
    pca.fit(tr_interaction)
    
    # Keep only principal components with 
    # eigenvalues greater than 1
    PC_eigenvalues = pca.components_[:pca.explained_variance_[pca.explained_variance_ >= 1].shape[0], :]
    
    # Transform data
    tr_pca = np.dot(tr_interaction, PC_eigenvalues.T) # NxK @ KxP = NxP
    v_pca = np.dot(v_interaction, PC_eigenvalues.T) 
    t_pca = np.dot(t_interaction, PC_eigenvalues.T)
    
    # Return as pd.DataFrame 
    # with column names
    col_names = ['PC' + str(x) for x in range(1, tr_pca.shape[1]+1)]
    
    tr_pca = pd.DataFrame(tr_pca, columns = col_names)
    v_pca = pd.DataFrame(v_pca, columns = col_names)
    t_pca = pd.DataFrame(t_pca, columns = col_names)
    
    return tr_pca, v_pca, t_pca


def process_dummies(data, iteration):
    
    """
    process_dummies(data, iteration) calls the support
    functions partition_dates(iteration), industry_
    partition(data, start_date, end_date), and
    drop_missing_industry(tr_ind, v_ind, t_ind) to process
    and one-hot encode the industries as well as drop
    non-repped industries.
    
    
    Parameters
    ----------
    data : pd.dataframe 
        The data set for which we compute interaction terms
        
    iteration : integer
        Data split iteration
        
        
    Returns
    -------
        pd.dataframe of one-hot encoded industry
        dummies
        
    """
    
    # Dates partitioning the training, validation, and test set 
    # according to Gu, Kelly, and Xiu (2020) by rolling-windows approach
    tr_start, tr_end, v_start, v_end, t_start, t_end = supp.partition_dates(iteration)
    
    # Split industry dummies according to dates
    tr_ind = supp.industry_partition(data, tr_start, tr_end)
    v_ind = supp.industry_partition(data, v_start, v_end)
    t_ind = supp.industry_partition(data, t_start, t_end)
    
    # Drop industries not represented in the 
    # training, validation, and test data
    tr_ind, v_ind, t_ind = supp.drop_missing_industry(tr_ind, v_ind, t_ind)
    
    return tr_ind.reset_index(), v_ind.reset_index(), t_ind.reset_index()


def process_returns(data, iteration):
    
    """
    process_returns(data, iteration) calls the support
    functions parition_dates(iteration) and 
    returns_partition(data, start_date, end_date)
    to process returns
    
    
    Parameters
    ----------
    data : pd.dataframe 
        The data set for which we compute interaction terms
        
    iteration : integer
        Data split iteration
        
        
    Returns
    -------
        pd.dataframe of one-hot encoded industry
        dummies
        
    """
    
    # Dates partitioning the training, validation, and test set 
    # according to Gu, Kelly, and Xiu (2020) by rolling-windows approach
    tr_start, tr_end, v_start, v_end, t_start, t_end = supp.partition_dates(iteration)
    
    # Split returns into training, validation, and test set
    tr_ret = supp.returns_partition(data, tr_start, tr_end)
    v_ret = supp.returns_partition(data, v_start, v_end)
    t_ret = supp.returns_partition(data, t_start, t_end)
    
    return tr_ret, v_ret, t_ret


def complete_data_process(industry_data, returns_data, FM_data, iteration):
    
    """
    complete_data_process(industry_data, returns_data, FM_data,
    iteration) processes all data making it ready for the
    ML models
    
    
    Parameters
    ----------
    industry_data : pd.dataframe 
        Industry dataframe
        
    returns_data : pd.dataframe 
        returns dataframe
        
    FM_data : pd.dataframe 
        Firm and macro dataframe
        
    iteration : integer
        Data split iteration
        
    
    Returns
    -------
        pd.dataframe of processed and
        cleaned data
        
    """
    
    # Industry dummies
    tr_ind, v_ind, t_ind = process_dummies(data = industry_data, iteration = iteration) # industry code data 
    
    # Returns data
    tr_ret, v_ret, t_ret = process_returns(data = returns_data, iteration = iteration) # returns data
    
    # PCA data
    tr_pca, v_pca, t_pca = pca(data = FM_data, iteration = iteration) # FM_data
    
    # Merge
    tr_data = pd.concat([tr_pca, tr_ind], axis = 1)
    tr_data = tr_data.merge(tr_ret, on = ["permno", "date"], how = "inner").set_index(["permno", "date"])
    
    v_data = pd.concat([v_pca, v_ind], axis = 1)
    v_data = v_data.merge(v_ret, on = ["permno", "date"], how = "inner").set_index(["permno", "date"])
    
    t_data = pd.concat([t_pca, t_ind], axis = 1)
    t_data = t_data.merge(t_ret, on = ["permno", "date"], how = "inner").set_index(["permno", "date"])
    
    '''
    print(
    f'32-/64-bit {insert}, '
    f'Columns: {insert}, '
    f'Rows: {insert}, '
    f'Size: {insert}, '
    )
    '''
    
    return tr_data, v_data, t_data