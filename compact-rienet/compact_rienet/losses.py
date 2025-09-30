"""
Loss functions module for Compact-RIEnet.

This module contains specialized loss functions for portfolio optimization,
including variance-based losses and other financial metrics.

References:
-----------
Please cite the following papers when using this code:
[Paper references to be provided by the author]

Copyright (c) 2025
"""

import tensorflow as tf
from keras import backend as K
from typing import Optional


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
def variance_loss_function(covariance_true: tf.Tensor, 
                          weights_predicted: tf.Tensor, 
                          penalty: float = 0.) -> tf.Tensor:
    """
    Portfolio variance loss function for training Compact-RIEnet models.
    
    This loss function computes the portfolio variance using the true covariance matrix
    and predicted portfolio weights. It normalizes by the system size and optionally
    includes a penalty term for excessive leverage.
    
    The portfolio variance is calculated as:
    variance = weights^T @ Σ @ weights
    
    where Σ is the true covariance matrix and weights are the predicted portfolio weights.
    
    Parameters
    ----------
    covariance_true : tf.Tensor
        True covariance matrices of shape (batch_size, n_assets, n_assets)
        These should be the actual realized covariance matrices
    weights_predicted : tf.Tensor  
        Predicted portfolio weights of shape (batch_size, n_assets, 1)
        These are the output weights from the Compact-RIEnet model
    penalty : float, default 0.0
        Penalty coefficient for excessive leverage. If > 0, adds a penalty
        term for gross leverage exceeding 1.0
        
    Returns
    -------
    tf.Tensor
        Portfolio variance loss, shape (batch_size, 1, 1)
        The loss is normalized by the number of assets when penalty > 0
        
    Notes
    -----
    The loss function assumes:
    - Daily returns data (annualized by factor of 252 in preprocessing)
    - Portfolio weights sum to 1 (enforced by the model architecture)
    - Covariance matrices are positive definite
    
    When penalty > 0, the function adds a leverage penalty:
    penalty_term = penalty * mean((gross_leverage - 1)^2)
    
    where gross_leverage = sum(|weights|) is the L1 norm of weights.
    
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
    
    References
    ----------
    Please cite the following papers when using this loss function:
    [Paper references to be provided by the author]
    """
    # Normalization factor
    if penalty > 0:
        n = tf.cast(tf.shape(covariance_true)[-1], dtype=tf.float32)
    else:
        n = 1.0

    # Portfolio variance: w^T @ Σ @ w
    portfolio_variance = n * tf.matmul(
        weights_predicted, 
        tf.matmul(covariance_true, weights_predicted), 
        transpose_a=True
    )

    # Add leverage penalty if specified
    if penalty > 0:
        gross_leverage = tf.reduce_sum(tf.abs(weights_predicted), axis=-2, keepdims=True)
        excess_leverage = gross_leverage - 1.0
        leverage_penalty = penalty * tf.reduce_mean(tf.square(excess_leverage))
        portfolio_variance += leverage_penalty

    return portfolio_variance


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
def buy_and_hold_volatility_loss(returns_out: tf.Tensor, 
                                 weights_predicted: tf.Tensor) -> tf.Tensor:
    """
    Buy-and-hold portfolio volatility loss function.
    
    This loss function computes the annualized volatility of a buy-and-hold
    portfolio strategy, where the initial weights are held constant and the
    portfolio value evolves according to the asset returns.
    
    The function assumes daily returns and annualizes the volatility by
    multiplying by sqrt(252 * n_assets).
    
    Parameters
    ----------
    returns_out : tf.Tensor
        Asset returns of shape (batch_size, n_assets, n_periods)
        These should be the daily returns data
    weights_predicted : tf.Tensor
        Portfolio weights of shape (batch_size, n_assets, 1) 
        Initial allocation weights for the buy-and-hold strategy
        
    Returns
    -------
    tf.Tensor
        Annualized portfolio volatility, scalar value averaged across batch
        
    Notes
    -----
    The computation follows these steps:
    1. Compute cumulative price series from returns: P_t = P_0 * ∏(1 + r_s) 
    2. Calculate portfolio value: V_t = Σ(w_i * P_i,t)
    3. Compute portfolio returns: R_t = (V_t - V_{t-1}) / V_{t-1}
    4. Calculate sample standard deviation and annualize
    
    The annualization uses the factor sqrt(252 * n_assets) which is common
    in financial applications for daily data.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from compact_rienet.losses import buy_and_hold_volatility_loss
    >>> 
    >>> # Sample data: 32 batches, 10 assets, 60 days
    >>> returns = tf.random.normal((32, 10, 60)) * 0.02  # 2% daily vol
    >>> weights = tf.random.normal((32, 10, 1))
    >>> weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
    >>> 
    >>> # Compute volatility loss
    >>> vol_loss = buy_and_hold_volatility_loss(returns, weights)
    >>> print(f"Annualized volatility: {vol_loss}")
    """
    n_stocks = tf.shape(returns_out)[1]
        
    # Compute cumulative relative prices: cumprod(1 + r)
    cum_rel = tf.math.cumprod(1.0 + returns_out, axis=-1)
    
    # Add initial price of 1.0 at the beginning
    price_series = tf.concat([tf.ones_like(cum_rel[..., :1]), cum_rel], axis=-1)

    # Portfolio value over time: V_t = Σ(w_i * P_i,t)
    port_val = tf.reduce_sum(price_series * weights_predicted, axis=1)
    
    # Ensure portfolio value is never zero
    port_val = tf.clip_by_value(port_val, K.epsilon(), tf.float32.max)

    # Portfolio returns: R_t = (V_t - V_{t-1}) / V_{t-1}
    port_returns = tf.divide(port_val[:, 1:], port_val[:, :-1]) - 1.0

    # Sample standard deviation and annualization
    # Uses sqrt(252 * n_stocks) for annualization factor
    volatility = (
        tf.math.reduce_std(port_returns, axis=-1) * 
        tf.sqrt(252.0 * tf.cast(n_stocks, dtype=tf.float32))
    )

    # Return batch average (scalar loss for training)
    return tf.reduce_mean(volatility)


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
def frobenius_loss_function(covariance_true: tf.Tensor, 
                           covariance_predicted: tf.Tensor) -> tf.Tensor:
    """
    Frobenius norm loss between true and predicted covariance matrices.
    
    This loss function computes the Frobenius norm of the difference between
    predicted and true covariance matrices, normalized by the system size.
    
    The Frobenius norm is defined as:
    ||A||_F = sqrt(Σ_ij |A_ij|^2) = sqrt(trace(A^T @ A))
    
    Parameters
    ----------
    covariance_true : tf.Tensor
        True covariance matrices of shape (batch_size, n_assets, n_assets)
    covariance_predicted : tf.Tensor
        Predicted covariance matrices of shape (batch_size, n_assets, n_assets)
        
    Returns
    -------
    tf.Tensor
        Frobenius norm of the difference, normalized by sqrt(n_assets)
        Shape: (batch_size,)
        
    Notes
    -----
    The normalization by n_assets ensures the loss is scale-invariant with
    respect to the number of assets in the portfolio.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from compact_rienet.losses import frobenius_loss_function
    >>> 
    >>> # Sample covariance matrices: 32 batches, 10 assets
    >>> cov_true = tf.random.normal((32, 10, 10))
    >>> cov_true = tf.matmul(cov_true, cov_true, transpose_b=True)  # PSD
    >>> cov_pred = tf.random.normal((32, 10, 10))
    >>> cov_pred = tf.matmul(cov_pred, cov_pred, transpose_b=True)  # PSD
    >>> 
    >>> # Compute Frobenius loss
    >>> frob_loss = frobenius_loss_function(cov_true, cov_pred)
    >>> print(f"Frobenius loss shape: {frob_loss.shape}")  # (32,)
    """
    n = tf.cast(tf.shape(covariance_true)[-1], dtype=tf.float32)
    
    # Compute Frobenius norm: ||A - B||_F = sqrt(Σ_ij (A_ij - B_ij)^2)
    diff_squared = tf.square(covariance_predicted - covariance_true)
    frobenius_norm = tf.sqrt(tf.reduce_sum(diff_squared, axis=[-2, -1]))
    
    # Normalize by system size
    return frobenius_norm / n