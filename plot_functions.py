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

    if not os.path.exists('confusion_matrix'):
        os.makedirs('confusion_matrix')
    # Save the plot
    plt.savefig(f'confusion_matrix/{title}.png')

    # Close the plot
    plt.close()

    # Return the confusion matrix
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
    # Get the feature importances from the model
    importances = model.feature_importances_

    # Create a DataFrame with the feature names and importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Create a bar plot of the feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{title}.png')
    plt.close()
    
    # Return the DataFrame
    return importance_df


def plot_feature_selection(names, best_lasso_coef):

    # plotting the Column Names and Importance of Columns. 
    plt.figure(figsize=(12, 10))
    plt.bar(names, best_lasso_coef)
    plt.xticks(rotation=25, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, 0.15)
    plt.savefig('lasso_feature_selection.png')
    plt.close()

