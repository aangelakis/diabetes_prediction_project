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


def plot_feature_importance(names, importances, title='Feature Importance'):
    """
    Plot the feature importances for a given set of feature names.

    Parameters
    ----------
    names : list
        The names of the features.
    importances : list
        The importances of the features.
    title : str, optional
        The title of the plot (default is 'Feature Importance').
    """
    # Create a DataFrame of feature names and importances
    df = pd.DataFrame({'Feature': names, 'Importance': importances})
    # Sort the features by importance
    df = df.sort_values('Importance', ascending=False)
    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    # Save the plot
    if not os.path.exists('feature_importance'):
        os.makedirs('feature_importance')
    plt.savefig(f'feature_importance/{title}.png')
    # Close the plot
    plt.close()