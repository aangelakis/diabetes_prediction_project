import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from trivial_model import *
from feature_selection import lasso_feature_selection
from cross_validation import nested_CV
from plot_functions import plot_feature_importance, plot_roc_curve, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_preprocess_data(path):
    """
    Load and preprocess the dataset.

    Parameters
    ----------
    path : str
        Path to the dataset file.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target variable.
    """
    df = pd.read_csv(path)
    print(df.head(), df.isnull().sum())
    
    df.loc[~df['gender'].isin(['Male', 'Female']), 'gender'] = None
    df.dropna(inplace=True)
    
    df = pd.get_dummies(df, columns=['gender'], drop_first=True)
    df['gender_Male'] = df['gender_Male'].astype(int)
    
    df['smoking_history'] = df['smoking_history'].map(df['smoking_history'].value_counts(normalize=True))
    
    col_order = [
        'gender_Male', 'age', 'hypertension', 'heart_disease', 'smoking_history',
        'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes'
    ]
    df = df[col_order]
    
    target = df.iloc[:, -1]
    class_distribution = target.value_counts(normalize=True).to_dict()
    print(f'Class distribution: {class_distribution}')
    if class_distribution[0] / class_distribution[1] > 2:
        print('Heavily Imbalanced Dataset')
    
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    df[numeric_features] = StandardScaler().fit_transform(df[numeric_features])
    
    print(f'Preprocessed dataset shape: {df.shape}')
    return df.iloc[:, :-1], df.iloc[:, -1]

def main():
    WEIGHT_BALANCE, SMOTE_FLAG, UNDERSAMPLING = True, False, False
    
    X, y = load_preprocess_data('diabetes_prediction_dataset.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    print(f'Train set class distribution: {y_train.value_counts(normalize=True)}, Shape: {X_train.shape}')
    
    selected_features = lasso_feature_selection(X_train, y_train)
    
    if SMOTE_FLAG:
        print('Applying SMOTE')
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    elif UNDERSAMPLING:
        print('Applying Random Undersampling')
        X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if WEIGHT_BALANCE else None
    configurations = [
        (LogisticRegression, [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga'], 'class_weight': ['balanced'] if WEIGHT_BALANCE else None}]),
        (SVC, [{'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto'], 'class_weight': ['balanced'] if WEIGHT_BALANCE else None}]),
        (RandomForestClassifier, [{'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20], 'class_weight': ['balanced'] if WEIGHT_BALANCE else None}]),
        (XGBClassifier, [{'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'scale_pos_weight': scale_pos_weight if WEIGHT_BALANCE else None}]),
        (DecisionTreeClassifier, [{'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10], 'class_weight': ['balanced'] if WEIGHT_BALANCE else None}]),
        (KNeighborsClassifier, [{'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski']}]),
        (GaussianNB, [{'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}])
    ]
    
    results_per_fold, best_configuration, best_clf = nested_CV(X_train[selected_features], y_train, configurations, outer_k_folds=10, inner_k_folds=5)
    
    plot_feature_importance(best_clf, selected_features, title='Feature Importance')
    print(f'Best Configuration: {best_configuration}')
    
    y_pred_proba, y_pred = best_clf.predict_proba(X_test[selected_features])[:, 1], best_clf.predict(X_test[selected_features])
    score_metrics = {
        'roc_auc': roc_auc_score, 'average_precision': average_precision_score, 'F1': f1_score,
        'precision': precision_score, 'recall': recall_score, 'balanced_accuracy': balanced_accuracy_score,
        'accuracy': accuracy_score, 'matthews_corrcoef': matthews_corrcoef
    }
    
    metric_values = {metric: func(y_test, y_pred_proba if metric in ['roc_auc', 'average_precision'] else y_pred) for metric, func in score_metrics.items()}
    for metric, value in metric_values.items():
        print(f'{metric}: {value:.4f}')
    
    pd.DataFrame([metric_values]).to_csv('model_metrics.csv', index=False)
    
    baseline_model = trivial_train(X_train[selected_features], y_train)
    y_pred_baseline = trivial_predict(baseline_model, X_test[selected_features])
    
    plot_roc_curve(y_test, y_pred_proba, y_pred_baseline, title='ROC Curve')
    plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix')
    
if __name__ == '__main__':
    main()