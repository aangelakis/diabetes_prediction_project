from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import matplotlib.pyplot as plt

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
