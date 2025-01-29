from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def lasso_feature_selection(X, y):
    """
    Perform feature selection using LASSO.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.array
        The feature matrix.
    y : pandas.DataFrame or numpy.array
        The target vector.

    Returns
    -------
    selected_features : list
        The names of the features that were selected.
    """
    
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    params = {'alpha': np.logspace(-4, 1, 50)}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lasso = Lasso()
    
    # Create the LASSO model for feature selection
    lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
    lasso_cv.fit(X, y)
    
    best_alpha = lasso_cv.best_params_['alpha']
    print('Best alpha: {}'.format(best_alpha))
    
    best_lasso = Lasso(alpha=best_alpha)
    best_lasso.fit(X, y)
    
    # Using np.abs() to make coefficients positive.  
    best_lasso_coef = np.abs(best_lasso.coef_)
    
    names=X.columns
    print("Column Names: {}".format(names.values))

    # plotting the Column Names and Importance of Columns. 
    plt.bar(names, best_lasso_coef)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on Lasso")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, 0.15)
    # plt.show() 
    plt.savefig('lasso_feature_selection.png')
    plt.close()

    # Subsetting the features which has more than 0.001 importance.
    feature_subset=np.array(names)[best_lasso_coef>0.005]
    
    selected_features = list(feature_subset)
    print("Selected Features: {}".format(selected_features))
    return selected_features


def backward_elimination(X, y, threshold=0.05):
    """
    Perform feature selection using Backward Elimination with OLS regression.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.array
        The feature matrix.
    y : pandas.DataFrame or numpy.array
        The target vector.
    threshold : float, optional
        The p-value threshold for feature selection (default is 0.05).

    Returns
    -------
    selected_features : list
        The names of the features that were selected.
    """
    X = pd.DataFrame(X)
    X = sm.add_constant(X)  # Add intercept term
    y = pd.Series(y)
    print(X)

    while True:
        model = sm.OLS(y, X).fit()
        print(model.summary())
        p_values = model.pvalues

        
        # Find the feature with the highest p-value (excluding intercept)
        max_p_value = p_values.drop('const').max()
        worst_feature = p_values.drop('const').idxmax()

        # If the highest p-value is above the threshold, drop that feature
        if max_p_value > threshold:
            print(f"Removing {worst_feature} with p-value {max_p_value:.4f}")
            X = X.drop(columns=[worst_feature])
        else:
            break  # Stop when all features have p-values below the threshold

    selected_features = X.columns.tolist()
    selected_features.remove('const')  # Remove intercept from final feature list
    print("Selected Features using Backward Elimination:", selected_features)
    return selected_features

