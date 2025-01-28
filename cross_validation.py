from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score


def nested_CV(X, y, configurations, outer_k_folds=10, inner_k_folds=5):
    """
    Perform nested cross-validation using scikit-learn's GridSearchCV for the inner loop.

    Parameters
    ----------
    X : Pandas DataFrame or array-like
        The feature set.
    y : Pandas Series or array-like
        The target labels.
    configurations : list of tuples
        A list where each tuple contains a model class and a dictionary of hyperparameters.
    outer_k_folds : int, optional
        The number of folds for the outer CV (default is 10).
    inner_k_folds : int, optional
        The number of folds for the inner CV (default is 5).

    Returns
    -------
    dict
        A dictionary containing the best model, best parameters, and average AUC across outer folds.
    """
    outer_cv = StratifiedKFold(n_splits=outer_k_folds, shuffle=True, random_state=42)
    outer_scores = []

    # Store the best configuration across all outer folds
    best_configuration = {
        'model': None,
        'params': None,
        'average_auc': -np.inf
    }

    # Outer loop
    for train_idx, test_idx in tqdm(outer_cv.split(X, y), desc="Outer folds", total=outer_k_folds):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_inner_model = None
        best_inner_params = None
        best_inner_auc = -np.inf

        # Inner loop: hyperparameter tuning
        for model, param_grid in configurations:
            inner_cv = StratifiedKFold(n_splits=inner_k_folds, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=model(),
                param_grid=param_grid,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1  # Use all available CPU cores
            )
            grid_search.fit(X_train, y_train)

            # Check if this model's best configuration is better than the current best
            if grid_search.best_score_ > best_inner_auc:
                best_inner_auc = grid_search.best_score_
                best_inner_model = grid_search.best_estimator_
                best_inner_params = grid_search.best_params_

        # Evaluate the best model from the inner loop on the outer test set
        best_inner_model.fit(X_train, y_train)
        y_pred_proba = best_inner_model.predict_proba(X_test)[:, 1]
        outer_auc = roc_auc_score(y_test, y_pred_proba)
        outer_scores.append(outer_auc)

        # Update the best configuration if this outer fold's AUC is the best so far
        if outer_auc > best_configuration['average_auc']:
            best_configuration['model'] = best_inner_model.__class__.__name__
            best_configuration['params'] = best_inner_params
            best_configuration['average_auc'] = outer_auc

    # Print the best configuration and its AUC
    print("Best Configuration:")
    print(f"  Model: {best_configuration['model']}")
    print(f"  Parameters: {best_configuration['params']}")
    print(f"  Average AUC: {np.mean(outer_scores):.3f}")

    return best_configuration
