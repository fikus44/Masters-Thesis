import pandas as pd
import numpy as np


def loss_function(predicted, actual):
    
    """
    loss_function() computes the loss as the squared euclidian 
    distance of the predicted values and the actual values
    
 
    Parameters
    ----------
    predicted : np.array  
        Predicted values
        
    actual : np.array
        Actual values
        
        
    Returns
    -------
        Squared euclidian distance between predicted valued
        and the actual values (scalar)
        
    """
    
    # Convert DataFrame to NumPy array
    if type(predicted) != np.ndarray:
        predicted = predicted.to_numpy()
        
    if type(actual) != np.ndarray:
        actual = actual.to_numpy()
      
    if predicted.ndim != 2:
        predicted = predicted[:, np.newaxis]
        
    if actual.ndim != 2:
        actual = actual[:, np.newaxis]
        
   
    return np.mean((predicted - actual) ** 2)
   

def explained_variation(predicted, actual):
    
    """
    explained_variation() computes the out-of-sample
    R^2 presented in equation (19) in Gu, Kelly, 
    and Xiu (2020)
    
 
    Parameters
    ----------
    predicted : np.array  
        Predicted values
        
    actual : np.array
        Actual values
        
        
    Returns
    -------
        Out-of-sample R^2 (scalar)
        
    """
    
    # Convert DataFrame to NumPy array
    if type(predicted) != np.ndarray:
        predicted = predicted.to_numpy()
        
    if type(actual) != np.ndarray:
        actual = actual.to_numpy()
        
    if predicted.ndim != 2:
        predicted = predicted[:, np.newaxis]
        
    if actual.ndim != 2:
        actual = actual[:, np.newaxis]
        
    N = predicted.shape[0]
    vector_of_ones = np.ones(N)
    
    numerator = ((actual - predicted) ** 2).reshape(-1,1).T @ vector_of_ones # 1xN @ Nx1 = 1x1
    denominator = (actual ** 2).reshape(-1,1).T @ vector_of_ones # 1xN @ Nx1 = 1x1
     
        
    return 1 - numerator / denominator



def lambda_grid(X, 
                y, 
                eps = 1e-3, 
                n_lambdas = 100
):
    
    """
    Compute the gripd of lambda values for lasso parameter search.
    The function leverages the same, albeit a simpler, approach as
    the _alpha_grid() function of the scikit-learn package
    
 
    Parameters
    ----------
    X : np.array  
        Covariates
        
    y : np.array
        Target values
        
    eps : float, default=1e-3
        Length of the path.
        
    n_alphas : int, default=100
        Number of alphas along the regularization path
        
        
    Returns
    -------
        Out-of-sample R^2 (scalar)
        
    """
    
    N = len(y)
    
    Xy = np.dot(X.T, y)
    
    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]
    
    lambda_max = np.sqrt(np.sum(Xy**2, axis = 1)).max() / N
    
    return np.logspace(np.log10(lambda_max * eps), np.log10(lambda_max), num = n_lambdas)
    

    
    
    