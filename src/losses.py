import numpy as np


def binary_log_loss(y_true, y_pred, eps=1e-15):
    """
    Binary Cross-Entropy (Log Loss).

    Parameters:
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)
        eps    : numerical stability constant

    Returns:
        scalar log loss
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )


def binary_log_loss_gradient(y_true, y_pred, eps=1e-15):
    """
    Gradient of Log Loss with respect to predictions.

    Note:
    This returns element-wise gradients.
    In Day 06, this will be combined with X^T
    to compute parameter gradients:
        grad_w = X^T @ (y_hat - y) / n

    Parameters:
        y_true : array-like, shape (n,)
        y_pred : array-like, shape (n,)

    Returns:
        gradient array, shape (n,)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))
