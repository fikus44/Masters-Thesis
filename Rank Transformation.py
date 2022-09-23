import pandas as pd
import numpy as np


class rank_transformation():
    
    """
    This class implements the rank transformation by Gu, Kelly, and Xiu (2020). 
    
    Parameters
    ----------
    
    n_factors : int, default=1
        The total number of factors to estimate. Note, the number of
        estimated factors is automatically reduced by the number of
        pre-specified factors. For example, if n_factors = 2 and one
        pre-specified factor is passed, then InstrumentedPCA will estimate
        one factor estimated in addition to the pre-specified factor.
        
    intercept : boolean, default=False
        Determines whether the model is estimated with or without an intercept
        
    max_iter : int, default=10000
        Maximum number of alternating least squares updates before the
        estimation is stopped
        
    iter_tol : float, default=10e-6
        Tolerance threshold for stopping the alternating least squares
        procedure
        
    alpha : scalar
        Regularizing constant for Gamma estimation.  If this is set to
        zero then the estimation defaults to non-regularized.
        
    l1_ratio : scalar
        Ratio of l1 and l2 penalties for elastic net Gamma fit.
        
    n_jobs : scalar
        number of jobs for F step estimation in ALS, if set to one no
        parallelization is done
        
    backend : str
        label for Joblib backend used for F step in ALS    
    """

    def __init__(self, n_factors=1, intercept=False, max_iter=10000,
                 iter_tol=10e-6, alpha=0., l1_ratio=1., n_jobs=1,
                 backend="loky"):