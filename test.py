import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import os
from xgboost import XGBClassifier
from tqdm import tqdm
from lightgbm import LGBMClassifier 
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from feature_selection import *
from sklearn.model_selection import ParameterGrid
from cross_validation import *


def plot_roc_curve(y_test, y_pred_proba, y_pred_trivial, title='ROC Curve'):
    """
    Plot the ROC curve for the given true labels and predicted probabilities.

    Parameters
    ----------
    y_test : numpy array
        True labels of the test set.
    y_pred_proba : numpy array
        Predicted probabilities of the positive class for the test set.
    y_pred_trivial : numpy array
        Predicted probabilities of the trivial classifier.
    title : str, optional
        Title of the plot (default is 'ROC Curve').

    Returns
    -------
    None
    """
    # Compute false positive rate and true positive rate for the classifier
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # Compute false positive rate and true positive rate for the trivial classifier
    fpr_trivial, tpr_trivial, _ = roc_curve(y_test, y_pred_trivial)
    
    # Plot the ROC curve for the classifier
    plt.plot(fpr, tpr, label='Classifier')
    # Plot the ROC curve for the trivial classifier
    plt.plot(fpr_trivial, tpr_trivial, linestyle='--', label='Trivial Classifier')
    # Set the x-axis label
    plt.xlabel('False Positive Rate')
    # Set the y-axis label
    plt.ylabel('True Positive Rate')
    # Set the title of the plot
    plt.title(title)
    # Add a legend to the plot
    plt.legend()
    # Save the plot to a file
    if not os.path.exists('roc_curve'):
        os.makedirs('roc_curve')
    plt.savefig(f'roc_curve/{title}.png')
    # Close the plot to free up memory
    plt.close()
    
    
def plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix'):
    """
    Plot the confusion matrix of the given true labels and predicted labels.

    Parameters
    ----------
    y_test : numpy array
        True labels of the test set
    y_pred : numpy array
        Predicted labels of the test set
    """
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d')

    # Add axis labels
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Add title
    plt.title(title)

    if not os.path.exists('confusion_matrix'):
        os.makedirs('confusion_matrix')
    # Save the plot
    plt.savefig(f'confusion_matrix/{title}.png')

    # Close the plot
    plt.close()

    # Return the confusion matrix
    return cm 


def trivial_train(X, Y):
    """
    Method to train a trivial classifier that predicts the most frequent class.

    Inputs:
        X (numpy array): A IxM matrix of categorical variables. Rows correspond to samples and columns to variables.
        Y (numpy array): A Ix1 vector. Y is the class variable to predict.

    Outputs:
        model (dict): This model should contain all the parameters required by the trivial classifierto classify new samples.
    """
    # Number of classes
    possible_classes = np.unique(Y)
    samples_in_classes = [np.sum(Y == c) for c in possible_classes]
    frequent_class = possible_classes[np.argmax(samples_in_classes)]

    # Create a dictionary to store the model parameters
    model = {'frequent_class': frequent_class}

    # Return the model parameters
    return model


def trivial_predict(model, X):
    """
    Method to predict the class of new samples using a trivial classifier.

    Inputs:
        X (numpy array): A IxM matrix of categorical variables. Rows correspond to samples and columns to variables.
        model (dict): This model should contain all the parameters required by the trivial classifierto classify new samples.

    Outputs:
        Y_pred (numpy array): A Ix1 vector of the predicted class labels.

    """
    # Number of samples
    I = X.shape[0]
        
    # Get the most frequent class
    frequent_class = model['frequent_class']
    
    # Initialize the predictions
    Y_pred = np.zeros(I)
    
    # Predict the most frequent class for each sample
    for i in range(I):
        Y_pred[i] = frequent_class
        
    return Y_pred


def load_preprocess_data(path):
    """
    Loads and preprocesses the dataset

    Parameters
    ----------
    path : str
        Path to the dataset

    Returns
    -------
    X : numpy array
        Preprocessed features
    y : numpy array
        Target

    """
    df = pd.read_csv(path)
    print(df.head())
    
    # Check for missing values
    print(df.isnull().sum())

    # Remove the 'OTHER' gender
    df.loc[~df['gender'].isin(['Male', 'Female']), 'gender'] = None
    df.dropna(inplace=True)
    
    # One-hot encode the gender variable
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)

    # Frequency encode the smoking_history variable
    df['smoking_history'] = df['smoking_history'].map(df['smoking_history'].value_counts(normalize=True))
    
    # Reorder columns: Move dummy variables next to original position
    col_order = ['gender_Male', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
    df = df[col_order]  # Manually reorder
        
    target = df.iloc[:, -1]     # Last column
    class_distribution = target.value_counts(normalize=True).to_dict()
    print(f'Class distribution: {class_distribution}')  # Print the class distribution (percentage of each class in the dataset)
    if class_distribution[0]/class_distribution[1] > 2:
        print('Heavily Imbalanced Dataset')

    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = StandardScaler() 

    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    print(df.head())
    
    print(f'Preprocessed dataset shape: {df.shape}')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def stratified_kfold_cv(X, y, model, cv=10):
    accuracy_scores = []
    auc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    balanced_accuracy_scores = []
    
    i = 0
    for train_index, test_index in tqdm(cv.split(X, y), desc='Cross-Validation', total=cv.get_n_splits()):
        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        trivial_clf = trivial_train(X_train_fold, y_train_fold)
        model.fit(X_train_fold, y_train_fold)
        
        y_pred_fold_trivial = trivial_predict(trivial_clf, X_test_fold)
        y_pred_fold = model.predict(X_test_fold)
        
        accuracy_scores.append(accuracy_score(y_test_fold, y_pred_fold))
        auc_scores.append(roc_auc_score(y_test_fold, y_pred_fold))
        f1_scores.append(f1_score(y_test_fold, y_pred_fold))
        precision_scores.append(precision_score(y_test_fold, y_pred_fold))
        recall_scores.append(recall_score(y_test_fold, y_pred_fold))
        balanced_accuracy_scores.append(balanced_accuracy_score(y_test_fold, y_pred_fold))

        i += 1
        plot_confusion_matrix(y_test_fold, y_pred_fold, title=f'Confusion Matrix {i}')
        plot_roc_curve(y_test_fold, y_pred_fold, y_pred_fold_trivial, title=f'ROC Curve {i}')
        

    print(f'Accuracy: {np.mean(accuracy_scores)}' + ' ± ' + f'{np.std(accuracy_scores)}')
    print(f'AUC: {np.mean(auc_scores)}' + ' ± ' + f'{np.std(auc_scores)}')
    print(f'F1: {np.mean(f1_scores)}' + ' ± ' + f'{np.std(f1_scores)}')
    print(f'Precision: {np.mean(precision_scores)}' + ' ± ' + f'{np.std(precision_scores)}')
    print(f'Recall: {np.mean(recall_scores)}' + ' ± ' + f'{np.std(recall_scores)}')
    print(f'Balanced Accuracy: {np.mean(balanced_accuracy_scores)}' + ' ± ' + f'{np.std(balanced_accuracy_scores)}')    


def main():
    X, y = load_preprocess_data('diabetes_prediction_dataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature selection using LASSO
    selected_features = lasso_feature_selection(X_train, y_train)
    
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    configurations = [
        (LogisticRegression, [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'], 'class_weight': ['balanced']}]),
        (SVC, [{'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto'], 'class_weight': ['balanced']}]),
        (RandomForestClassifier, [{'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10], 'class_weight': ['balanced']}]),
        (XGBClassifier, [{'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'scale_pos_weight': [scale_pos_weight]}]),
        (LGBMClassifier, [{'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'num_leaves': [31, 50, 100], 'is_unbalance': [True]}]),
        (CatBoostClassifier, [{'learning_rate': [0.01, 0.1, 0.2], 'iterations': [100, 200, 500], 'depth': [4, 6, 10], 'auto_class_weights': ['Balanced']}]),
        (MLPClassifier, [{'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001], 'class_weight': ['balanced']}])
    ]


    # DO NESTED CV TO SELECT THE BEST MODEL AND HYPERPARAMETERS 
    # IF CLASS WEIGHT IS NOT WORKING AND I STILL HAVE VERY LOW RECALL USE SMOTE
    best_configuration, best_clf = nested_CV(X_train[selected_features], y_train, configurations, outer_k_folds=10, inner_k_folds=5)   

    # Evaluate the best model on the test set, chehck GPT's answer for more
    # also compare class_weight with SMOTE
    y_pred_proba = best_clf.predict_proba(X_test[selected_features])[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f'Test AUC: {test_auc}')

    # Plot the ROC curve for the test set
    y_pred_trivial = trivial_predict(best_configuration, X_test[selected_features])
    plot_roc_curve(y_test, y_pred_proba, y_pred_trivial, title='Test ROC Curve')

    # Plot the confusion matrix for the test set
    plot_confusion_matrix(y_test, y_pred_proba, title='Test Confusion Matrix')
        

if __name__ == '__main__':
    main()