import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, matthews_corrcoef, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from feature_selection import *
from cross_validation import *
from plot_functions import *
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

WEIGHT_BALANCE = True
SMOTE_FLAG = False
UNDERSAMPLING = False


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
    df['gender_Male'] = df['gender_Male'].astype(int)

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


def main():
    X, y = load_preprocess_data('diabetes_prediction_dataset.csv')

    # 90-10 train-test split because we have a large dataset and we use StratifiedKFold CV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print(f'Train set class distribution: {y_train.value_counts(normalize=True)} and shape: {X_train.shape}')

    # Feature selection using LASSO
    selected_features = lasso_feature_selection(X_train, y_train)
    
    # Feature selection using backward elimination algorithm
    # selected_features = backward_elimination(X_train, y_train, threshold=0.05)

    if SMOTE_FLAG:
        print('Using SMOTE to balance the classes in the training set')
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f'Train set class distribution after SMOTE: {y_train.value_counts(normalize=True)} and shape: {X_train.shape}',)
    elif UNDERSAMPLING:
        print('Using Random Undersampling to balance the classes in the training set')
        undersample = RandomUnderSampler(random_state=42)
        X_train, y_train = undersample.fit_resample(X_train, y_train)
        print(f'Train set class distribution after Random Undersampling: {y_train.value_counts(normalize=True)} and shape: {X_train.shape}',)

    if WEIGHT_BALANCE == True:
        print('Using class weight to balance the classes in the training set')
        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()
        scale_pos_weight = negative_count / positive_count

        configurations = [
            (LogisticRegression, [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'], 'class_weight': ['balanced']}]),
            (SVC, [{'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto'], 'class_weight': ['balanced']}]),
            (RandomForestClassifier, [{'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10], 'class_weight': ['balanced']}]),
            (XGBClassifier, [{'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'scale_pos_weight': [scale_pos_weight]}]),
            (DecisionTreeClassifier, [{'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10], 'class_weight': ['balanced']}])
        ]
        confusion_matrix_title = 'Test Confusion Matrix with Class Weight'
        roc_curve_title = 'Test ROC Curve with Class Weight'
        results_per_fold_title = 'results_per_fold_with_class_weight.csv'
        metric_values_title = 'metric_values_with_class_weight.csv'
        best_config_title = 'best_configuration_with_class_weight.csv'
        feautre_importance_title = 'feature_importance_with_class_weight'
    else:
        configurations = [
            (LogisticRegression, [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}]),
            (SVC, [{'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto']}]),
            (RandomForestClassifier, [{'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20]}]),
            (XGBClassifier, [{'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}]),
            (DecisionTreeClassifier, [{'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]}]),
            (KNeighborsClassifier, [{'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski']}]),
            (GaussianNB, [{'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}])
        ]
        if SMOTE_FLAG:
            confusion_matrix_title = 'Test Confusion Matrix with SMOTE'
            roc_curve_title = 'Test ROC Curve with SMOTE'
            results_per_fold_title = 'results_per_fold_with_smote.csv'
            metric_values_title = 'metric_values_with_smote.csv'
            best_config_title = 'best_configuration_with_smote.csv'
            feautre_importance_title = 'feature_importance_with_smote'
        elif UNDERSAMPLING:
            confusion_matrix_title = 'Test Confusion Matrix with Random Undersampling'
            roc_curve_title = 'Test ROC Curve with Random Undersampling'
            results_per_fold_title = 'results_per_fold_with_undersampling.csv'
            metric_values_title = 'metric_values_with_undersampling.csv'
            best_config_title = 'best_configuration_with_undersampling.csv'
            feautre_importance_title = 'feature_importance_with_undersampling'
        else:
            print('No class weight or undersampling applied')
            confusion_matrix_title = 'Test Confusion Matrix without Class Weight or undersampling'
            roc_curve_title = 'Test ROC Curve without Class Weight or undersampling'
            results_per_fold_title = 'results_per_fold_without_class_weight_or_undersampling.csv'
            metric_values_title = 'metric_values_without_class_weight_or_undersampling.csv'
            best_config_title = 'best_configuration_without_class_weight_or_undersampling.csv'
            feautre_importance_title = 'feature_importance_without_class_weight_or_undersampling'


    results_per_fold, best_configuration, best_clf = nested_CV(X_train[selected_features], y_train, configurations, outer_k_folds=10, inner_k_folds=5)   

    plot_feature_importance(best_clf, selected_features, title=feautre_importance_title)

    print(f'RESULTS PER OUTER FOLD: {results_per_fold}')
    results_per_fold.to_csv(results_per_fold_title, index=False)

    print(f'BEST CONFIGURATION: {best_configuration}')
    best_config_df = pd.DataFrame([best_configuration])
    best_config_df.to_csv(best_config_title, index=False)

    # Define metrics correctly
    score_metrics = {
        'roc_auc': roc_auc_score,  # Uses probabilities
        'average_precision': average_precision_score,  # Uses probabilities
        'F1': f1_score,  # Uses predicted labels
        'F1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score,  # Uses predicted labels
        'recall': recall_score,  # Uses predicted labels
        'balanced_accuracy': balanced_accuracy_score,  # Uses predicted labels
        'accuracy': accuracy_score,  # Uses predicted labels
        'matthews_corrcoef': matthews_corrcoef,  # Uses predicted labels
    }

    # Get predicted probabilities for positive class (only for AUC-based metrics)
    y_pred_proba = best_clf.predict_proba(X_test[selected_features])[:, 1]

    # Get discrete class predictions (binary 0/1 or multiclass labels)
    y_pred = best_clf.predict(X_test[selected_features])

    metric_values = {}  # Store metrics in a dictionary

    # Compute and store each metric correctly
    for metric, func in score_metrics.items():
        if metric in ['roc_auc', 'average_precision']:  # Metrics requiring probabilities
            metric_value = func(y_test, y_pred_proba)
        else:  # Metrics requiring discrete class predictions
            metric_value = func(y_test, y_pred)

        print(f'{metric}: {metric_value:.4f}')
        metric_values[metric] = metric_value  # Store values in dictionary

    # Convert dictionary to DataFrame
    metric_values_df = pd.DataFrame([metric_values])

    # Save to CSV
    metric_values_df.to_csv(metric_values_title, index=False)

    # Plot the ROC curve for the test set
    trivia_model = trivial_train(X_train[selected_features], y_train)
    y_pred_trivial = trivial_predict(trivia_model, X_test[selected_features])
    plot_roc_curve(y_test, y_pred_proba, y_pred_trivial, title=roc_curve_title)

    # Plot the confusion matrix for the test set
    plot_confusion_matrix(y_test, y_pred, title=confusion_matrix_title)
        

if __name__ == '__main__':
    main()