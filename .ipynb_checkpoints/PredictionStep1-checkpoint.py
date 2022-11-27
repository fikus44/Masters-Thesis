import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def XY_split(data):
    
    """
    XY_split(data) splits data in X and Y
    
 
    Parameters
    ----------
    data : pd.DataFrame  
        Dataset to split
 
 
    Returns
    -------
        Pd.DataFrame of X's and 
        Pd.Series of Y's
        
    """
    
    
    # Split X and Y
    data_x = data.iloc[:, :-1]
    data_y = data.ret
    
    return data_x, data_y



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


def to_append(pred, test_y):
    
    """
    to_append(pred, test_y) processes results to
    be appended to the dataframes outside of the loop
    
 
    Parameters
    ----------
    pred : np.Ndarray 
        prediction or ML model
        
    test_y : Pd.DataFrame
        actual returns in test set
 
    Returns
    -------
        Results to be appended
        
    """
    
    # Monthly loss, xplained variation
    loss = loss_function(pred, test_y)
    xplained_variation = explained_variation(pred, test_y)
    
    # Pred and actual returns
    pred_actual = test_y.reset_index()
    pred_actual["pred"] = pred
    
    # Annual loss, xplained variation
    annualised = pred_actual.groupby("permno").sum()
    
    # Annual loss
    annual_loss = np.mean((annualised.pred - annualised.ret) ** 2)
    
    # Annual xplained variation
    numerator = np.asarray((annualised.ret - annualised.pred) ** 2).reshape(-1,1).T @ np.ones(annualised.shape[0]) # 1xN @ Nx1 = 1x1
    denominator = np.asarray((annualised.ret) ** 2).reshape(-1,1).T @ np.ones(annualised.shape[0])  # 1xN @ Nx1 = 1x1
    annual_xplained_variation = 1 - numerator / denominator

    return loss, xplained_variation, pred_actual, annual_loss, annual_xplained_variation
    
    
def portfolio_sorts_1(pred_actual):
    
    """
    portfolio_sorts_1(pred_actual) computes the
    average returns of all deciles at each
    point in time
    
 
    Parameters
    ----------
    pred_actual : pd.DataFrame 
        The predicted returns of the model
        and the actual returns
        
 
    Returns
    -------
        pd.DataFrame of average returns of 
        all deciles at all points in time
        
    """
    
    # Rank predicted returns at each point in time
    pred_actual["rank"] = pred_actual.groupby("date")["pred"].rank(method = 'first')
    
    # Sanity check: Max rank is equal to the no of observations of returns in each group
    # i.e. each point in time 
    assert all(pred_actual.groupby("date").size() == pred_actual.groupby("date")["rank"].max())
    
    # Sort predicted returns at each point in time into deciles
    pred_actual["decile"] = pred_actual.groupby("date")["rank"].transform(lambda x: pd.qcut(x, 10, labels = range(1, 11)))
    
    # Compute average return for each decile at all points in time
    deciles = pred_actual.groupby(["date", "decile"])[["ret", "pred"]].mean()
    
    return deciles 


def portfolio_sorts_acc_return(deciles):
    
    
    """
    portfolio_sorts_acc_return() takes
    the output of portfolio_sorts_1() as
    input and returns the accumulated return
    of each decile over the period 
    
 
    Parameters
    ----------
    deciles : pd.DataFrame 
        The decile portfolios 
        
 
    Returns
    -------
        pd.Series of accmumuated returns
        
    """
    
    return deciles.groupby("decile").sum()
    

    

def portfolio_sorts_SR(deciles):
    
    """
    portfolio_sorts_SR() takes
    the output of portfolio_sorts_1() as
    input and returns the annualized Sharpe
    Ratio, the monthly average return, and 
    monthly standard deviation of the predicted 
    and actual returns
    
    Parameters
    ----------
    deciles : pd.DataFrame 
        The decile portfolios 
        
 
    Returns
    -------
        pd.Series of SR, mean, and std
        
    """
        
    # Compute monthly average return of each decile
    # during the entire period
    avg_return = deciles.groupby("decile").mean()
    
    # Compute monthly stanrd deviation of each 
    # decile during the entire period
    std = deciles.groupby("decile").std()
    
    # Annualized Sharpe Ratio
    SR = (avg_return / std) * np.sqrt(12)
    
    return avg_return, std, SR


def cumulative_ret_fig(data, name, save_fig = False, hide = False):
    
    """
    cumulative_ret_fig(data, name, save_fig, hide) computes
    the cumulative return figure for a given machine
    learning model. 
    
    Parameters
    ----------
    data : pd.DataFrame 
        Time series data of returns (predicted and actual) 
        for each decile 
    
    name : String 
        Name of figure when saving to disk
        
    save_fig : Boolean 
        Save figure to disk
        
    hide : Boolean 
        Prevent from printing figure

        
    Returns
    -------
        None
        
    """
    
    # Returns lowest (1) and highest (10) deciles
    low_decile = data.index.get_level_values(1).min()
    high_decile = data.index.get_level_values(1).max()
    
    # Running returns (cumulative)
    run_ret = [list(it.accumulate(data[data.index.get_level_values(1) == x].ret)) for x in range(low_decile, high_decile + 1)]
    dates =  list(data.index.get_level_values(0).unique())
    dates = pd.to_datetime(dates, format = '%Y%m')
    
    n = len(run_ret)
    
    # Setup
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Create figure
    fig, ax = plt.subplots(figsize = (15,8))

    # Make iterable 
    color = iter(cm.viridis(np.linspace(0.1, 0.6, n)))
    label = iter([f'{ite}.' + ' Decile' for ite in range(1, n+1)])

    for i in range(n):
    
        c = next(color)
        l = next(label)
    
        ax.plot(dates, 
                run_ret[i], 
                c = c, 
                label = l);
        
    # Add labels and change font 
    plt.xlabel("Time", size = 13.5)
    plt.ylabel("Percentage Return", size = 13.5)
    
    # Legend
    ax.legend(loc = 'upper center', ncol = 10, bbox_to_anchor = (0.5, 1.05), fontsize = 11.5, columnspacing=0.8, frameon=False)
    
    # Horizontal lines
    ax.axhline(y=0, color='black', linewidth=0.75)  
    #ax.axhline(y=1, color='black', linewidth=0.5)  
    #ax.axhline(y=2, color='black', linewidth=0.5)  
    #ax.axhline(y=3, color='black', linewidth=0.5) 
    
    # Tick text, tick length and width
    ax.tick_params(axis='both', which='major', labelsize=12.5, width = 1.2, length = 4)
    
    # Recession 1:
    R1_start = pd.datetime(1990, 7, 1)
    R1_end = pd.datetime(1991, 2, 1)
    plt.axvspan(R1_start, R1_end, color="grey", alpha=0.25)
    
    # Recession 2:
    R2_start = pd.datetime(2001, 6, 1)
    R2_end = pd.datetime(2001, 11, 1)
    plt.axvspan(R2_start, R2_end, color="grey", alpha=0.25)
    
    # Save as PDF
    if save_fig:
        plt.savefig(f'{name}' + '.pdf', bbox_inches='tight')
        
    if hide:
        plt.close()
    else:
        plt.show()
    
    return None 
    
    
    