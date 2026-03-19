"""
Loss functions for the legacy Compact-RIEnet package.

This module belongs to the deprecated Compact-RIEnet package. For maintained
development, install ``rienet`` and use the code hosted at
https://github.com/bongiornoc/RIEnet.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
def variance_loss_function(covariance_true: tf.Tensor,
                          weights_predicted: tf.Tensor) -> tf.Tensor:
    """
    Portfolio variance loss function for training Compact-RIEnet models.

    Deprecated
    ----------
    ``compact_rienet`` is no longer maintained. For new projects, install
    ``rienet`` and refer to https://github.com/bongiornoc/RIEnet.

    This loss function computes the global minimum-variance objective using the true
    covariance matrix and predicted portfolio weights.

    The portfolio variance is calculated as:
    variance = weights^T @ Σ @ weights

    where Σ is the true covariance matrix and weights are the predicted portfolio weights.

    Parameters
    ----------
    covariance_true : tf.Tensor
        Covariance matrices of shape (batch_size, n_assets, n_assets)
    weights_predicted : tf.Tensor
        Predicted portfolio weights of shape (batch_size, n_assets, 1)

    Returns
    -------
    tf.Tensor
        Portfolio variance loss, shape (batch_size, 1, 1)

    Notes
    -----
    The loss function assumes:
    - Daily returns data (annualized by factor of 252 in preprocessing)
    - Portfolio weights are expected to sum to one (enforced by the layer)
    - Covariance matrices are positive (semi) definite
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from compact_rienet.losses import variance_loss_function
    >>> 
    >>> # Sample data: 32 batches, 10 assets
    >>> covariance = tf.random.normal((32, 10, 10))
    >>> covariance = tf.matmul(covariance, covariance, transpose_b=True)  # PSD
    >>> weights = tf.random.normal((32, 10, 1))
    >>> weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)  # Normalize
    >>> 
    >>> # Compute loss
    >>> loss = variance_loss_function(covariance, weights)
    >>> print(f"Portfolio variance: {loss.shape}")  # (32, 1, 1)

    """
    dtype = weights_predicted.dtype
    n = tf.cast(tf.shape(covariance_true)[-1], dtype=dtype)

    # Portfolio variance: n * w^T Σ w (Eq. 6 in the paper)
    portfolio_variance = n * tf.matmul(
        weights_predicted,
        tf.matmul(covariance_true, weights_predicted),
        transpose_a=True
    )

    return portfolio_variance

__all__ = [
    'variance_loss_function',
]
