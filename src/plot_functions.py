from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

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
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fpr_trivial, tpr_trivial, _ = roc_curve(y_test, y_pred_trivial)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='Classifier', linewidth=2)
    plt.plot(fpr_trivial, tpr_trivial, linestyle='--', label='Trivial Classifier', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs('roc_curve', exist_ok=True)
    plt.savefig(f'roc_curve/{title}.png')
    plt.close()

def plot_confusion_matrix(y_test, y_pred, title='Confusion Matrix'):
    """
    Plot the confusion matrix of the given true labels and predicted labels.

    Parameters
    ----------
    y_test : numpy array
        True labels of the test set.
    y_pred : numpy array
        Predicted labels of the test set.
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    os.makedirs('confusion_matrix', exist_ok=True)
    plt.savefig(f'confusion_matrix/{title}.png')
    plt.close()
    
    return cm 

def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """
    Plot the feature importance of a given model.

    Parameters
    ----------
    model : object
        Trained model with a `feature_importances_` attribute.
    feature_names : list
        List of feature names.
    title : str, optional
        Title of the plot (default is 'Feature Importance').
    """
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    os.makedirs('feature_importance', exist_ok=True)
    plt.savefig(f'feature_importance/{title}.png')
    plt.close()
    
    return importance_df

def plot_feature_selection(names, best_lasso_coef, title='LASSO Feature Selection'):
    """
    Plot the importance of selected features using LASSO.

    Parameters
    ----------
    names : list
        List of feature names.
    best_lasso_coef : numpy array
        Coefficients of the LASSO model.
    title : str, optional
        Title of the plot (default is 'LASSO Feature Selection').
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(x=best_lasso_coef, y=names, palette='coolwarm')
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title(title)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    os.makedirs('lasso_feature_selection', exist_ok=True)
    plt.savefig(f'lasso_feature_selection/{title}.png')
    plt.close()
