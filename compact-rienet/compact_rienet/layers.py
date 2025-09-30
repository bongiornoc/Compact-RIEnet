"""
Compact-RIEnet: A Compact Recurrent-Invariant Eigenvalue Network for Portfolio Optimization

This module implements the Compact-RIEnet layer, a neural network architecture for 
portfolio optimization that processes financial time series data and outputs portfolio weights.

The architecture is based on eigenvalue decomposition of covariance matrices and recurrent
neural networks to capture temporal dependencies in financial data.

References:
-----------
Please cite the following papers when using this code:
[Paper references to be provided by the author]

Copyright (c) 2025
"""

import tensorflow as tf
from keras import layers
from typing import Optional, List, Tuple, Union
import numpy as np

from .custom_layers import (
    LagTransformLayer,
    StandardDeviationLayer, 
    CovarianceLayer,
    SpectralDecompositionLayer,
    DimensionAwareLayer,
    DeepRecurrentLayer,
    DeepLayer,
    CustomNormalizationLayer,
    EigenProductLayer,
    NormalizedSum
)


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class CompactRIEnetLayer(layers.Layer):
    """
    Compact Recurrent-Invariant Eigenvalue Network (Compact-RIEnet) Layer.
    
    This layer implements a neural network architecture for portfolio optimization
    that processes daily returns and outputs portfolio weights. The architecture
    combines eigenvalue decomposition of covariance matrices with recurrent neural
    networks to capture both cross-sectional relationships and temporal dependencies
    in financial data.
    
    The layer automatically scales input daily returns by 252 (annualization factor)
    and applies a series of transformations including:
    - Lag transformation for temporal preprocessing
    - Covariance matrix estimation and eigenvalue decomposition
    - Recurrent processing of eigenvalues with GRU networks
    - Standard deviation transformation with dense networks
    - Portfolio weight computation through matrix operations
    
    Parameters
    ----------
    output_type : str, default 'weights'
        Type of output to return. Options:
        - 'weights': Portfolio weights (normalized to sum to 1)
        - 'precision': Transformed precision matrix (inverse covariance)
        
    name : str, optional
        Name of the layer.
        
    **kwargs : dict
        Additional keyword arguments passed to the base Layer class.
        
    Input Shape
    -----------
    (batch_size, n_stocks, n_days) : tf.Tensor
        Daily returns data where:
        - batch_size: Number of samples in the batch
        - n_stocks: Number of assets/stocks 
        - n_days: Number of time periods (days)
        
    Output Shape
    ------------
    If output_type='weights':
        (batch_size, n_stocks, 1) : tf.Tensor
            Portfolio weights normalized to sum to 1
            
    If output_type='precision':
        (batch_size, n_stocks, n_stocks) : tf.Tensor
            Transformed precision matrix (inverse covariance)
            
    Notes
    -----
    The architecture uses fixed hyperparameters optimized for portfolio optimization:
    - Hidden layer size: 8 units
    - Recurrent layer size: 32 units  
    - Recurrent model: GRU (Gated Recurrent Unit)
    - Bidirectional processing for temporal modeling
    
    The input returns are automatically scaled by 252 to account for annualization
    effects commonly used in financial modeling.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from compact_rienet import CompactRIEnetLayer
    >>> 
    >>> # Create layer for portfolio weights
    >>> layer = CompactRIEnetLayer(output_type='weights')
    >>> 
    >>> # Generate sample daily returns data  
    >>> returns = tf.random.normal((32, 10, 60))  # 32 samples, 10 stocks, 60 days
    >>> 
    >>> # Get portfolio weights
    >>> weights = layer(returns)
    >>> print(f"Portfolio weights shape: {weights.shape}")  # (32, 10, 1)
    
    References
    ----------
    Please cite the following papers when using this code:
    [Paper references to be provided by the author]
    """
    
    def __init__(self, 
                 output_type: str = 'weights',
                 name: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Compact-RIEnet layer.
        
        Parameters
        ----------
        output_type : str, default 'weights'
            Type of output ('weights' or 'precision')
        name : str, optional
            Layer name
        **kwargs : dict
            Additional arguments for base Layer
        """
        super().__init__(name=name, **kwargs)
        
        if output_type not in ['weights', 'precision']:
            raise ValueError("output_type must be 'weights' or 'precision'")
            
        self.output_type = output_type
        
        # Fixed architecture parameters optimized for portfolio optimization
        self._hidden_layer_sizes = [8]
        self._recurrent_layer_sizes = [32] 
        self._recurrent_model = 'GRU'
        self._direction = 'bidirectional'
        self._dimensional_features = ['n_stocks', 'n_days', 'q']
        self._annualization_factor = 252.0
        
        # Initialize component layers
        self._build_layers()
        
    def _build_layers(self):
        """Build the internal layers of the architecture."""
        # Input transformation and preprocessing
        self.lag_transform = LagTransformLayer(
            warm_start=True, 
            name=f"{self.name}_lag_transform"
        )
        
        self.std_layer = StandardDeviationLayer(
            axis=-1, 
            name=f"{self.name}_std"
        )
        
        self.covariance_layer = CovarianceLayer(
            expand_dims=False,
            normalize=True,
            name=f"{self.name}_covariance"
        )
        
        # Eigenvalue decomposition
        self.spectral_decomp = SpectralDecompositionLayer(
            name=f"{self.name}_spectral"
        )
        
        self.dimension_aware = DimensionAwareLayer(
            features=self._dimensional_features,
            name=f"{self.name}_dimension_aware"
        )
        
        # Recurrent processing of eigenvalues
        self.eigenvalue_transform = DeepRecurrentLayer(
            recurrent_layer_sizes=self._recurrent_layer_sizes,
            recurrent_model=self._recurrent_model,
            direction=self._direction,
            dropout=0.0,
            recurrent_dropout=0.0,
            final_hidden_layer_sizes=[],
            normalize='inverse',
            name=f"{self.name}_eigenvalue_rnn"
        )
        
        # Standard deviation transformation
        self.std_transform = DeepLayer(
            hidden_layer_sizes=self._hidden_layer_sizes + [1],
            last_activation='softplus',
            name=f"{self.name}_std_transform"
        )
        
        self.std_normalization = CustomNormalizationLayer(
            axis=-2,
            mode='inverse',
            name=f"{self.name}_std_norm"
        )
        
        # Matrix reconstruction
        self.eigen_product = EigenProductLayer(
            scaling_factor='inverse',
            name=f"{self.name}_eigen_product"
        )
        
        self.covariance_reconstruct = CovarianceLayer(
            normalize=False,
            name=f"{self.name}_cov_reconstruct"
        )
        
        # Portfolio weight computation
        self.portfolio_weights = NormalizedSum(
            axis_1=-1, 
            axis_2=-2, 
            name=f"{self.name}_portfolio_weights"
        )
        
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass of the Compact-RIEnet layer.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, n_stocks, n_days)
            containing daily returns data
        training : bool, optional
            Whether the layer is in training mode
            
        Returns
        -------
        tf.Tensor
            Output tensor - either portfolio weights or precision matrix
            depending on output_type parameter
        """
        # Scale inputs by annualization factor
        scaled_inputs = inputs * self._annualization_factor
        
        # Apply lag transformation
        input_transformed = self.lag_transform(scaled_inputs)
        
        # Compute standard deviation and mean
        std, mean = self.std_layer(input_transformed)
        
        # Standardize returns
        returns = (input_transformed - mean) / std
        
        # Compute covariance matrix
        covariance_matrix = self.covariance_layer(returns)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = self.spectral_decomp(covariance_matrix)
        
        # Add dimensional features
        eigenvalues_enhanced = self.dimension_aware([eigenvalues, scaled_inputs])
        
        # Transform eigenvalues with recurrent network
        transformed_eigenvalues = self.eigenvalue_transform(eigenvalues_enhanced)
        
        # Transform standard deviations
        transformed_std = self.std_transform(std)
        transformed_std = self.std_normalization(transformed_std)
        
        # Reconstruct correlation matrix from transformed eigenvalues
        transformed_correlation = self.eigen_product(
            transformed_eigenvalues, eigenvectors
        )
        
        # Reconstruct covariance matrix
        transformed_covariance = (
            transformed_correlation * 
            self.covariance_reconstruct(transformed_std)
        )
        
        if self.output_type == 'precision':
            return transformed_correlation  # This is actually the precision matrix
        else:  # output_type == 'weights'
            # Compute portfolio weights
            weights = self.portfolio_weights(transformed_covariance)
            return weights
    
    def get_config(self) -> dict:
        """
        Get layer configuration for serialization.
        
        Returns
        -------
        dict
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update({
            'output_type': self.output_type,
        })
        return config
    
    @classmethod
    def from_config(cls, config: dict):
        """
        Create layer from configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        CompactRIEnetLayer
            Layer instance
        """
        return cls(**config)
        
    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute output shape given input shape.
        
        Parameters
        ----------
        input_shape : tuple
            Input shape (batch_size, n_stocks, n_days)
            
        Returns
        -------
        tuple
            Output shape
        """
        batch_size, n_stocks, n_days = input_shape
        
        if self.output_type == 'weights':
            return (batch_size, n_stocks, 1)
        else:  # precision
            return (batch_size, n_stocks, n_stocks)