import numpy as np


def random_train_test_split(X, y, proportion_train):
    split_point = int(
        proportion_train * X.shape[0]
    )  # numer of samples in the training set

    shuffled_indices = np.random.permutation(X.shape[0])

    train_indices = shuffled_indices[:split_point]
    test_indices = shuffled_indices[split_point:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


def stratified_train_test_split(X, y, train_prop):
    """
    Split randomly the dataset into training and test sets while preserving class proportions.

    Args:
    X: array of samples. Shape (n_samples, n_features)
    y: array of labels. Shape (n_samples, ) or (n_samples, 1)
    train_prop: proportion of samples to include in the training set. Must be between 0 and 1.

    Returns:
    X_train: array of training samples. Shape (n_train_samples, n_features)
    X_test: array of test samples. Shape (n_test_samples, n_features)
    y_train: array of training labels. Shape (n_train_samples, ) or (n_train_samples, 1)
    y_test: array of test labels. Shape (n_test_samples, ) or (n_test_samples, 1)
    """
    unique_classes = np.unique(
        y
    )  # get unique class labels (i.e., the different classes)

    train_indices = []
    test_indices = []

    for cls in unique_classes:
        indices = np.where(y == cls)[
            0
        ]  # find indices of samples belonging to class cls
        np.random.shuffle(indices)  # shuffle the indices for randomness
        split_idx = int(indices.shape[0] * train_prop)  # determine split index

        cls_train_indices = indices[
            :split_idx
        ].tolist()  # get the first split_idx indices for training for class cls
        cls_test_indices = indices[
            split_idx:
        ].tolist()  # get the remaining indices for testing for class cls

        train_indices = (
            train_indices + cls_train_indices
        )  # concatenate the training indices for all classes
        test_indices = (
            test_indices + cls_test_indices
        )  # concatenate the test indices for all classes

    # shuffle again to avoid unintended ordering (right now the indices are ordered by class)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


def accuracy(y_true, y_pred):
    """
    Args:
    y_true: array of true labels. Shape (n_samples, ) or (n_samples, 1)
    y_pred: array of predicted labels. Shape (n_samples, ) or (n_samples, 1)
    """
    total_predictions = len(y_true)
    is_correct = y_true == y_pred
    n_correct_pred = np.sum(is_correct)
    accuracy = n_correct_pred / total_predictions
    return accuracy
