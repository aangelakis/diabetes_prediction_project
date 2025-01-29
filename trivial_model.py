import numpy as np

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