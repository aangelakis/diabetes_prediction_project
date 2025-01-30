from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from plot_functions import plot_feature_selection

def lasso_feature_selection(X, y, alpha_range=np.logspace(-4, 1, 50), cv_splits=5, importance_threshold=0.005):
    """
    Perform feature selection using LASSO regression.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.array
        The feature matrix.
    y : pandas.Series or numpy.array
        The target vector.
    alpha_range : numpy.array, optional
        Range of alpha values for LASSO regularization (default is np.logspace(-4, 1, 50)).
    cv_splits : int, optional
        Number of cross-validation splits (default is 5).
    importance_threshold : float, optional
        Minimum coefficient value to consider a feature as important (default is 0.005).

    Returns
    -------
    selected_features : list
        List of selected feature names.
    """
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    # Define LASSO with cross-validation
    lasso = Lasso()
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    param_grid = {'alpha': alpha_range}
    
    lasso_cv = GridSearchCV(lasso, param_grid=param_grid, cv=kf, scoring='r2')
    lasso_cv.fit(X, y)
    
    best_alpha = lasso_cv.best_params_['alpha']
    print(f'Best alpha: {best_alpha:.5f}')
    
    # Fit LASSO with best alpha
    best_lasso = Lasso(alpha=best_alpha)
    best_lasso.fit(X, y)
    
    best_lasso_coef = np.abs(best_lasso.coef_)
    feature_names = X.columns
    
    # Plot feature selection results
    plot_feature_selection(feature_names, best_lasso_coef)
    
    # Select features with coefficients above the threshold
    selected_features = feature_names[best_lasso_coef > importance_threshold].tolist()
    print(f"Selected Features: {selected_features}")
    
    return selected_features

def backward_elimination(X, y, threshold=0.05):
    """
    Perform feature selection using Backward Elimination with OLS regression.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.array
        The feature matrix.
    y : pandas.Series or numpy.array
        The target vector.
    threshold : float, optional
        The p-value threshold for feature selection (default is 0.05).

    Returns
    -------
    selected_features : list
        List of selected feature names.
    """
    X = pd.DataFrame(X)
    X = sm.add_constant(X)  # Add intercept term
    y = pd.Series(y)
    
    while True:
        model = sm.OLS(y, X).fit()
        p_values = model.pvalues
        
        # Find the feature with the highest p-value (excluding intercept)
        max_p_value = p_values.drop('const').max()
        worst_feature = p_values.drop('const').idxmax()
        
        if max_p_value > threshold:
            print(f"Removing {worst_feature} with p-value {max_p_value:.4f}")
            X = X.drop(columns=[worst_feature])
        else:
            break  # Stop when all features have p-values below the threshold
    
    selected_features = X.columns.tolist()
    selected_features.remove('const')  # Remove intercept from final feature list
    print(f"Selected Features using Backward Elimination: {selected_features}")
    
    return selected_features
