import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from trivial_classifier import trivial_train, trivial_predict
from sklearn.tree import DecisionTreeClassifier

# Data structure that stores information about the data preprocessing (apply standardization or not), which classifier is going to be trained and the hyper-paramters of each classifier
CONFIGURATIONS = [
    (True, RandomForestClassifier, [
        {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20, 30], 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]}
    ]),
    (False, RandomForestClassifier, [
        {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20, 30], 'min_samples_leaf': [1, 2, 4], 'min_samples_split': [2, 5, 10]}
    ]),
    (True, DecisionTreeClassifier, [
        {'max_depth': [None, 5, 10, 15], 'min_samples_leaf': [1, 5, 10], 'min_samples_split': [2, 5, 10], 'max_leaf_nodes': [None, 10, 20, 50]}
    ]),
    (False, DecisionTreeClassifier, [
        {'max_depth': [None, 5, 10, 15], 'min_samples_leaf': [1, 5, 10], 'min_samples_split': [2, 5, 10], 'max_leaf_nodes': [None, 10, 20, 50]}
    ])
]


def create_folds(X, y, k_folds=5):
    """
    Splits the dataset into stratified folds for cross-validation.
    
    Parameters
    ----------
    X : Pandas DataFrame
        The feature set.
    y : Pandas Series or array-like
        The target labels.
    n_folds : int, optional
        The number of folds to create (default is 5).
    
    Returns
    -------
    list of tuples
        A list where each element is a tuple containing the training
        and testing splits: (X_train, y_train, X_test, y_test).
    """
    skf = StratifiedKFold(n_splits=k_folds)
    folds = []
    for train_index, test_index in skf.split(X, y):
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Append the split to the folds list
        folds.append((X_train, y_train, X_test, y_test))
    return folds


def nested_CV(X, y, outer_k_folds=10, inner_k_folds=5):
    """
    Cross-validation function to find the best combination of preprocessing,
    model and hyperparameters for a given dataset.

    This function performs a nested cross-validation to find the best
    combination of preprocessing, model and hyperparameters. It returns the
    best preprocessing, model, hyperparameters, average AUC score and the
    final model trained on all data.

    Parameters
    ----------
    X : Pandas DataFrame
        The feature set.
    y : Pandas Series or array-like
        The target labels.
    outer_k_folds : int, optional
        The number of folds to create in the outer loop (default is 5).
    inner_k_folds : int, optional
        The number of folds to create in the inner loop (default is 5).

    Returns
    -------
    tuple
        A tuple containing the best preprocessing, model, hyperparameters,
        average AUC score and the final model trained on all data.
    """
    # Create a dictionary to store the performance of each configuration
    config_performance = {}

    # Iterate through the CONFIGURATIONS list
    for standardization, model, hyperparameters in CONFIGURATIONS:
        # Create a top-level key for standardization
        if standardization not in config_performance:
            config_performance[standardization] = {}

        # Create a second-level key for the model name
        model_name = model.__name__
        if model_name not in config_performance[standardization]:
            config_performance[standardization][model_name] = {}

        # Create third-level keys for each parameter configuration
        for param_grid in hyperparameters:
            for params in ParameterGrid(param_grid):
                # Convert params to a tuple of sorted items
                params_key = tuple(sorted(params.items()))
                if params_key not in config_performance[standardization][model_name]:
                    config_performance[standardization][model_name][params_key] = []

    # Split the data into stratified folds for the outer loop
    outer_folds = create_folds(X, y, outer_k_folds)

    # Iterate through the outer folds
    for outer_fold in tqdm(outer_folds, desc="Outer folds"):
        outer_X_train, outer_y_train, outer_X_test, outer_y_test = outer_fold

        # Initialize variables to track the best configuration and its AUC
        best_model = None
        best_params = None
        best_standardization = None
        best_auc = -np.inf

        # Split the data into stratified folds for the inner loop
        inner_folds = create_folds(outer_X_train, outer_y_train, inner_k_folds)

        # Iterate through the inner folds
        for inner_fold in inner_folds:
            inner_X_train, inner_y_train, inner_X_test, inner_y_test = inner_fold

            # Iterate through the hyperparameters for the current model
            for standardization, model, hyperparameters in CONFIGURATIONS:

                # Apply standardization if needed
                if standardization:
                    scaler = StandardScaler()
                    inner_X_train = scaler.fit_transform(inner_X_train)
                    inner_X_test = scaler.transform(inner_X_test)

                # Iterate through the hyperparameters and train a model
                for param_grid in hyperparameters:
                    for params in ParameterGrid(param_grid):
                        # print(f"Training model {model.__name__} with params: {params}")
                        params_key = tuple(sorted(params.items()))
                        clf = model(**params)
                        clf.fit(inner_X_train, inner_y_train)
                        # config_performance[standardization][model.__name__][params_key].append(roc_auc_score(inner_y_test, clf.predict_proba(inner_X_test)[:, 1]))
                        auc_score = roc_auc_score(inner_y_test, clf.predict_proba(inner_X_test)[:, 1])
                        if auc_score > best_auc:
                            best_auc = auc_score
                            best_model = model
                            best_params = params
                            best_standardization = standardization

        # Apply standardization if needed
        if best_standardization:
            scaler = StandardScaler()
            outer_X_train = scaler.fit_transform(outer_X_train)
            outer_X_test = scaler.transform(outer_X_test)

        # Train a model with the best hyperparameters
        best_clf = best_model(**best_params)
        best_clf.fit(outer_X_train, outer_y_train)
        y_pred_proba = best_clf.predict_proba(outer_X_test)[:, 1]
        outer_auc = roc_auc_score(outer_y_test, y_pred_proba)
        best_params_key = tuple(sorted(best_params.items()))
        config_performance[best_standardization][best_model.__name__][best_params_key].append(outer_auc)


    # Find the best configuration and report the performance
    # Initialize variables to track the best configuration and its AUC
    best_auc = -float('inf')  # Start with a very low value
    best_configuration = None  # To store the details of the best configuration

    # Iterate through the config_performance dictionary
    for standardization, model_dict in config_performance.items():
        for model_name, params_dict in model_dict.items():
            for params_key, auc_list in params_dict.items():
                avg_auc = np.mean(auc_list)
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    best_configuration = {
                        'standardization': standardization,
                        'model': [model for _, model, _ in CONFIGURATIONS if model.__name__ == model_name][0],
                        'parameters': dict(params_key),
                        'average_auc': avg_auc
                    }

    # Print the best configuration and its AUC
    print("Best Configuration:")
    print(f"  Standardization: {best_configuration['standardization']}")
    print(f"  Model: {best_configuration['model']}")
    print(f"  Parameters: {best_configuration['parameters']}")
    print(f"  Average AUC: {best_configuration['average_auc']:.3f}")

    # Construct a final model trained on all data
    best_clf = best_configuration['model'](**best_configuration['parameters'])
    if best_configuration['standardization']:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    best_clf.fit(X, y)

    return scaler, best_configuration['model'], best_configuration['parameters'], best_configuration['average_auc'], best_clf


def CV(X, y, k_folds=10):
    """
    Cross-validation function to find the best combination of preprocessing,
    model and hyperparameters for a given dataset.

    This function performs a nested cross-validation to find the best
    combination of preprocessing, model and hyperparameters. It returns the
    best preprocessing, model, hyperparameters, average AUC score and the
    final model trained on all data.

    Parameters
    ----------
    X : Pandas DataFrame
        The feature set.
    y : Pandas Series or array-like
        The target labels.
    k_folds : int, optional
        The number of folds to create (default is 5).

    Returns
    -------
    tuple
        A tuple containing the best preprocessing, model, hyperparameters,
        average AUC score and the final model trained on all data.
    """
    # Create a dictionary to store the performance of each configuration
    config_performance = {}
    
    # Iterate through the configurations and create the necessary keys
    for standardization, model, hyperparameters in CONFIGURATIONS:
        # Create a top-level key for standardization
        if standardization not in config_performance:
            config_performance[standardization] = {}
        
        # Create a second-level key for the model name
        model_name = model.__name__
        if model_name not in config_performance[standardization]:
            config_performance[standardization][model_name] = {}
        
        # Create third-level keys for each parameter configuration
        for param_grid in hyperparameters:
            for params in ParameterGrid(param_grid):
                # Convert params to a tuple of sorted items
                params_key = tuple(sorted(params.items()))
                if params_key not in config_performance[standardization][model_name]:
                    config_performance[standardization][model_name][params_key] = []

    # 1. Split the data into stratified k folds
    folds = create_folds(X, y, k_folds)
    
    # 2. Use CV to find the best configuration
    for fold in tqdm(folds, desc="Folds"):
        X_train, y_train, X_test, y_test = fold
            
        for standardization, model, hyperparameters in CONFIGURATIONS:

            if standardization:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            for param_grid in hyperparameters:
                for params in ParameterGrid(param_grid):
                    # Train the model and store the performance
                    params_key = tuple(sorted(params.items()))
                    clf = model(**params)
                    clf.fit(X_train, y_train)
                    config_performance[standardization][model.__name__][params_key].append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))    

    # 3. Find the best configuration and report the performance
    models_performance = []
    for standardization, model, hyperparameters in CONFIGURATIONS:
            for param_grid in hyperparameters:
                for params in ParameterGrid(param_grid):
                    params_key = tuple(sorted(params.items()))
                    avg_auc = np.mean(config_performance[standardization][model.__name__][params_key])
                    models_performance.append((standardization, model, params, avg_auc))
                    
    best_configuration = max(models_performance, key=lambda x: x[3])
    print("Best Configuration:")
    print(f"  Standardization: {best_configuration[0]}")
    print(f"  Model: {best_configuration[1].__name__}")
    print(f"  Parameters: {best_configuration[2]}")
    print(f"  Average AUC: {best_configuration[3]:.3f}")
    
    # 4. Construct a final model trained on all data
    best_clf = best_configuration[1](**best_configuration[2])
    if best_configuration[0]:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    best_clf.fit(X, y)
    
    return scaler, best_configuration[1], best_configuration[2], best_configuration[3], best_clf

    
def main():
    """
    Main function to load datasets, perform cross-validation, evaluate models,
    and plot ROC curves for both the trained and trivial classifiers.
    """
    # Load the training dataset
    dataset = pd.read_csv('Dataset5.A_XY.csv', header=None)
    X_train = dataset.iloc[:, :-1]
    y_train = dataset.iloc[:, -1]

    # Perform cross-validation to find the best model configuration
    # scaler, model, params, cv_auc, trained_clf = CV(X_train, y_train, k_folds=5)
    scaler, model, params, cv_auc, trained_clf = nested_CV(X_train, y_train, outer_k_folds=10, inner_k_folds=5)
    print("Standardization:", scaler is not None)
    print("Model:", model)
    print("Params:", params)
    print("CV AUC: %.3f" % cv_auc)
    print("Classifier:", trained_clf)

    # Load the test dataset
    test_dataset = pd.read_csv('Dataset5.B_XY.csv', header=None)
    X_test = test_dataset.iloc[:, :-1]
    y_test = test_dataset.iloc[:, -1]

    # Apply standardization to the test data if applicable
    if scaler is not None:
        X_test = scaler.transform(X_test)

    # Predict probabilities on the test set using the trained classifier
    y_pred_proba = trained_clf.predict_proba(X_test)[:, 1]
    holdout_auc = roc_auc_score(y_test, y_pred_proba)
    print("Hold-out test set AUC:", holdout_auc)

    # Train a trivial classifier and evaluate its performance
    trivial_model = trivial_train(X_train, y_train)
    y_pred_proba_trivial = trivial_predict(trivial_model, X_test) * np.ones(y_test.shape)
    trivial_auc = roc_auc_score(y_test, y_pred_proba_trivial)
    print("Trivial classifier Hold-out test set AUC:", trivial_auc)

    # Convert y_test to binary (assuming class "2.0" as positive class)
    y_test = (y_test == 2.0).astype(int)

    # Calculate ROC curves for both classifiers
    fpr_trivial, tpr_trivial, _ = roc_curve(y_test, y_pred_proba_trivial)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # Plot the ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve, AUC = %.3f' % holdout_auc)
    plt.plot(fpr_trivial, tpr_trivial, color='red', lw=2, label='Trivial ROC curve, AUC = %.3f' % trivial_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.show()
    

if __name__ == '__main__':
    main()
