from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, average_precision_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             make_scorer, accuracy_score)

def nested_CV(X, y, configurations, outer_k_folds=10, inner_k_folds=5):
    """
    Perform nested cross-validation with hyperparameter tuning.

    Parameters
    ----------
    X : pd.DataFrame or np.array
        Feature set.
    y : pd.Series or np.array
        Target labels.
    configurations : list of tuples
        Each tuple contains a model class and a dictionary of hyperparameters.
    outer_k_folds : int, optional
        Number of outer CV folds (default is 10).
    inner_k_folds : int, optional
        Number of inner CV folds (default is 5).

    Returns
    -------
    tuple
        - results_df: DataFrame with results per fold.
        - best_configuration: Dictionary with best model and hyperparameters.
        - best_final_model: Best trained model on full dataset.
    """
    outer_cv = StratifiedKFold(n_splits=outer_k_folds, shuffle=True, random_state=42)
    scoring_metrics = {
        'roc_auc': 'roc_auc',
        'F1': make_scorer(f1_score),
        'F1_macro': 'f1_macro',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'average_precision': 'average_precision',
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'accuracy': 'accuracy',
        'matthews_corrcoef': make_scorer(matthews_corrcoef),
    }
    
    configurations = [
        (model, [
            {key: value for key, value in param_grid.items() if not (key == 'class_weight' or key == 'scale_pos_weight' and value is None)}
            for param_grid in param_list
        ])
        for model, param_list in configurations
    ]
    
    results_per_fold = []
    best_configuration = {'model': None, 'params': None, 'average_auc': -np.inf}
    
    # Outer loop
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(outer_cv.split(X, y), desc="Outer folds", total=outer_k_folds)):
        print(f'Processing Fold {fold_idx + 1}/{outer_k_folds}...')
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        best_inner_model, best_inner_params, best_inner_metrics = None, None, None
        best_inner_auc = -np.inf
        
        # Inner loop: Hyperparameter tuning
        for model, param_grid in configurations:
            inner_cv = StratifiedKFold(n_splits=inner_k_folds, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=model(),
                param_grid=param_grid,
                cv=inner_cv,
                scoring=scoring_metrics,
                refit='roc_auc',
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            
            # Update best inner model based on AUC
            if grid_search.best_score_ > best_inner_auc:
                best_inner_auc = grid_search.best_score_
                best_inner_model = grid_search.best_estimator_
                best_inner_params = grid_search.best_params_
                best_inner_metrics = {
                    metric: np.max(grid_search.cv_results_[f'mean_test_{metric}']) 
                    for metric in scoring_metrics.keys()
                }
        
        # Evaluate best model from inner loop on outer test set
        y_pred_proba = best_inner_model.predict_proba(X_test)[:, 1]
        y_pred = best_inner_model.predict(X_test)
        
        outer_metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'F1': f1_score(y_test, y_pred),
            'F1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'average_precision': average_precision_score(y_test, y_pred_proba),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
        }
        
        # Store results for this fold
        results_per_fold.append({
            'fold': fold_idx + 1,
            'model': best_inner_model.__class__.__name__,
            'params': best_inner_params,
            'inner_cv_metrics': best_inner_metrics,
            'outer_test_metrics': outer_metrics
        })
        
        # Update best overall configuration
        if outer_metrics['roc_auc'] > best_configuration['average_auc']:
            best_configuration.update({
                'model': best_inner_model.__class__,
                'params': best_inner_params,
                'average_auc': outer_metrics['roc_auc'],
            })
        
        print(f'Completed Fold {fold_idx + 1}')
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_per_fold)
    
    # Print best model summary
    print("\nBest Model Configuration Across Outer Folds:")
    print(f"  Model: {best_configuration['model'].__name__}")
    print(f"  Parameters: {best_configuration['params']}")
    print(f"  Average AUC: {best_configuration['average_auc']:.3f}")
    
    # Train final model on full dataset
    best_final_model = best_configuration['model'](**best_configuration['params'])
    best_final_model.fit(X, y)
    
    return results_df, best_configuration, best_final_model
