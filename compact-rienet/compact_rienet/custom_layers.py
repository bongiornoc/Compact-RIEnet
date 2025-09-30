"""
Custom layers module for Compact-RIEnet.

This module contains all the specialized neural network layers required for the
Compact-RIEnet architecture, including layers for covariance estimation,
Rotational Invariant Estimator (RIE) based eigenvalue cleaning, and specialized
transformations for financial data.

References:
-----------
Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025).
"Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage."
Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF '25).

Copyright (c) 2025
"""

import tensorflow as tf
from keras import backend as K
from keras import layers, initializers
from typing import List, Optional, Tuple, Union


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class StandardDeviationLayer(layers.Layer):
    """
    Layer for computing sample standard deviation and mean.
    
    This layer computes the sample standard deviation and mean along a specified axis,
    with optional demeaning for statistical preprocessing.
    
    Parameters
    ----------
    axis : int, default 1
        Axis along which to compute statistics
    demean : bool, default False
        Whether to use an unbiased denominator (n-1)
    epsilon : float, default 1e-6
        Small value added for numerical stability
    name : str, optional
        Layer name
    """
    
    def __init__(self,
                 axis: int = 1,
                 demean: bool = False,
                 epsilon: float = 1e-6,
                 name: Optional[str] = None,
                 **kwargs):
        if name is None:
            raise ValueError("StandardDeviationLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.axis = axis
        self.demean = demean
        self.epsilon = epsilon

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute standard deviation and mean.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor
            
        Returns
        -------
        tuple of tf.Tensor
            (standard_deviation, mean)
        """
        dtype = x.dtype
        epsilon = tf.cast(self.epsilon, dtype)

        sample_size = tf.cast(tf.shape(x)[self.axis], dtype)
        sample_size = tf.maximum(sample_size, 1.0)

        mean = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        centered = x - mean

        if self.demean:
            denom = tf.maximum(sample_size - 1.0, 1.0)
        else:
            denom = sample_size

        variance = tf.reduce_sum(tf.square(centered), axis=self.axis, keepdims=True) / denom
        std = tf.sqrt(tf.maximum(variance, epsilon))

        return std, mean

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'demean': self.demean,
            'epsilon': float(self.epsilon)
        })
        return config


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class CovarianceLayer(layers.Layer):
    """
    Layer for computing covariance matrices.
    
    This layer computes sample covariance matrices from input data with optional
    normalization and dimension expansion.
    
    Parameters
    ----------
    expand_dims : bool, default False
        Whether to expand dimensions of output
    normalize : bool, default True  
        Whether to normalize by sample size
    name : str, optional
        Layer name
    """
    
    def __init__(self, expand_dims: bool = False, normalize: bool = True, 
                 name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("CovarianceLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.expand_dims = expand_dims
        self.normalize = normalize

    def call(self, returns: tf.Tensor) -> tf.Tensor:
        """
        Compute covariance matrix.
        
        Parameters
        ----------
        returns : tf.Tensor
            Input returns data
            
        Returns
        -------
        tf.Tensor
            Covariance matrix
        """
        if self.normalize:
            sample_size = tf.cast(tf.shape(returns)[-1], tf.float32) 
            covariance = tf.matmul(returns, returns, transpose_b=True) / sample_size
        else:
            covariance = tf.matmul(returns, returns, transpose_b=True)
            
        if self.expand_dims:
            covariance = tf.expand_dims(covariance, axis=-3)
            
        return covariance

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'expand_dims': self.expand_dims,
            'normalize': self.normalize
        })
        return config


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class SpectralDecompositionLayer(layers.Layer):
    """
    Layer for eigenvalue decomposition of symmetric matrices.
    
    This layer performs eigenvalue decomposition using tf.linalg.eigh,
    which is optimized for symmetric/Hermitian matrices like covariance matrices.
    
    Parameters
    ----------
    name : str, optional
        Layer name
    """
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("SpectralDecompositionLayer must have a name.")
        super().__init__(name=name, **kwargs)

    def call(self, covariance_matrix: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform eigenvalue decomposition.
        
        Parameters
        ----------
        covariance_matrix : tf.Tensor
            Input covariance matrix
            
        Returns
        -------
        tuple of tf.Tensor
            (eigenvalues, eigenvectors) where eigenvalues have shape [..., n, 1]
        """
        eigenvalues, eigenvectors = tf.linalg.eigh(covariance_matrix)
        # Expand dims to make eigenvalues [..., n, 1] for compatibility
        eigenvalues = tf.expand_dims(eigenvalues, axis=-1)
        return eigenvalues, eigenvectors

    def get_config(self) -> dict:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class DimensionAwareLayer(layers.Layer):
    """
    Layer that adds dimensional features to eigenvalue data.
    
    This layer augments eigenvalue tensors with additional features derived
    from the dimensions of the input data, such as number of stocks, days,
    and their ratios.
    
    Parameters
    ----------
    features : list of str
        List of features to add: 'n_stocks', 'n_days', 'q', 'rsqrt_n_days'
    name : str, optional
        Layer name
    """
    
    def __init__(self, features: List[str], name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("DimensionAwareLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.features = features

    def _set_attribute(self, value: tf.Tensor, shape: tf.Tensor) -> tf.Tensor:
        """Broadcast scalar value to target shape."""
        value = tf.expand_dims(value, axis=-1)
        value = tf.broadcast_to(value, shape)
        return value

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """
        Add dimensional features to eigenvalues.
        
        Parameters
        ----------
        inputs : list of tf.Tensor
            [eigenvalues, original_inputs] where original_inputs has shape [..., n_stocks, n_days]
            
        Returns
        -------
        tf.Tensor
            Enhanced eigenvalues with additional features
        """
        eigen_values, original_inputs = inputs
        n_stocks = tf.cast(tf.shape(original_inputs)[1], tf.float32)
        n_days = tf.cast(tf.shape(original_inputs)[2], tf.float32)
        final_shape = tf.shape(eigen_values)
        
        tensors_to_concat = [eigen_values]
        
        if 'q' in self.features:
            q = n_days / n_stocks
            tensors_to_concat.append(self._set_attribute(q, final_shape))
            
        if 'n_stocks' in self.features:
            tensors_to_concat.append(self._set_attribute(tf.sqrt(n_stocks), final_shape))
            
        if 'n_days' in self.features:
            tensors_to_concat.append(self._set_attribute(tf.sqrt(n_days), final_shape))
            
        if 'rsqrt_n_days' in self.features:
            rsqrt_n_days = tf.math.rsqrt(n_days)
            tensors_to_concat.append(self._set_attribute(rsqrt_n_days, final_shape))
            
        return tf.concat(tensors_to_concat, axis=-1)

    def compute_output_shape(self, input_shape: Tuple[Tuple, Tuple]) -> Tuple:
        """Compute output shape."""
        eigen_values_shape, _ = input_shape
        additional_features = len(self.features)
        return eigen_values_shape[:-1] + (eigen_values_shape[-1] + additional_features,)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'features': self.features})
        return config


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class DeepLayer(layers.Layer):
    """
    Multi-layer dense network with configurable activation and dropout.
    
    This layer implements a sequence of dense layers with specified activations,
    dropout, and flexible configuration for the final layer.
    
    Parameters
    ----------
    hidden_layer_sizes : list of int
        Sizes of hidden layers including output layer
    last_activation : str, default "linear"
        Activation for the final layer
    activation : str, default "leaky_relu"
        Activation for hidden layers
    other_biases : bool, default True
        Whether to use bias in hidden layers
    last_bias : bool, default True
        Whether to use bias in final layer
    dropout_rate : float, default 0.0
        Dropout rate for hidden layers
    kernel_initializer : str, default "glorot_uniform"
        Weight initialization method
    name : str, optional
        Layer name
    """
    
    def __init__(self, hidden_layer_sizes: List[int], last_activation: str = "linear",
                 activation: str = "leaky_relu", other_biases: bool = True, 
                 last_bias: bool = True, dropout_rate: float = 0., 
                 kernel_initializer: str = "glorot_uniform", name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("DeepLayer must have a name.")
        super().__init__(name=name, **kwargs)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.last_activation = last_activation
        self.other_biases = other_biases
        self.last_bias = last_bias
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        # Build hidden layers
        self.hidden_layers = []
        self.dropouts = []
        
        for i, size in enumerate(self.hidden_layer_sizes[:-1]):
            layer_name = f"{self.name}_hidden_{i}"
            dropout_name = f"{self.name}_dropout_{i}"
            
            dense = layers.Dense(
                size,
                activation=self.activation,
                use_bias=self.other_biases,
                kernel_initializer=self.kernel_initializer,
                name=layer_name
            )
            dropout = layers.Dropout(self.dropout_rate, name=dropout_name)
            
            self.hidden_layers.append(dense)
            self.dropouts.append(dropout)

        # Final layer
        self.final_dense = layers.Dense(
            self.hidden_layer_sizes[-1],
            use_bias=self.last_bias,
            activation=self.last_activation,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_output"
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor
        training : bool, optional
            Whether in training mode
            
        Returns
        -------
        tf.Tensor
            Output tensor
        """
        x = inputs
        for dense, dropout in zip(self.hidden_layers, self.dropouts):
            x = dense(x)
            x = dropout(x, training=training)
        return self.final_dense(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'last_activation': self.last_activation,
            'other_biases': self.other_biases,
            'last_bias': self.last_bias,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer
        })
        return config

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """Compute output shape."""
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_layer_sizes[-1]
        return tuple(output_shape)


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class DeepRecurrentLayer(layers.Layer):
    """
    Deep recurrent layer with configurable RNN cells and post-processing.
    
    This layer implements a stack of recurrent layers (LSTM/GRU) with optional
    bidirectional processing, followed by dense layers for final transformation.
    
    Parameters
    ----------
    recurrent_layer_sizes : list of int
        Sizes of recurrent layers
    final_activation : str, default "softplus"
        Activation for final dense layer
    final_hidden_layer_sizes : list of int, default []
        Sizes of dense layers after RNN
    final_hidden_activation : str, default "leaky_relu"
        Activation for final hidden layers
    direction : str, default 'bidirectional'
        RNN direction: 'bidirectional', 'forward', or 'backward'
    dropout : float, default 0.0
        Dropout rate for RNN layers
    recurrent_dropout : float, default 0.0
        Recurrent dropout rate
    recurrent_model : str, default 'LSTM'
        Type of RNN cell: 'LSTM' or 'GRU'
    normalize : str, optional
        Normalization mode: None, 'inverse', or 'sum'
    name : str, optional
        Layer name
    """
    
    def __init__(self, recurrent_layer_sizes: List[int], final_activation: str = "softplus", 
                 final_hidden_layer_sizes: List[int] = [], final_hidden_activation: str = "leaky_relu",
                 direction: str = 'bidirectional', dropout: float = 0., recurrent_dropout: float = 0.,
                 recurrent_model: str = 'LSTM', normalize: Optional[str] = None, 
                 name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("DeepRecurrentLayer must have a name.")
        super().__init__(name=name, **kwargs)

        self.recurrent_layer_sizes = recurrent_layer_sizes
        self.final_activation = final_activation
        self.final_hidden_layer_sizes = final_hidden_layer_sizes
        self.final_hidden_activation = final_hidden_activation
        self.direction = direction
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_model = recurrent_model
        
        if normalize not in [None, 'inverse', "sum"]:
            raise ValueError("normalize must be None, 'inverse', or 'sum'.")
        self.normalize = normalize

        # Build recurrent layers
        RNN = getattr(layers, recurrent_model)
        self.recurrent_layers = []
        
        for i, units in enumerate(self.recurrent_layer_sizes):
            layer_name = f"{self.name}_rnn_{i}"
            cell_name = f"{layer_name}_cell"
            
            if self.direction == 'bidirectional':
                cell = RNN(
                    units=units, 
                    dropout=self.dropout, 
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True, 
                    name=cell_name
                )
                rnn_layer = layers.Bidirectional(cell, name=layer_name)
            elif self.direction == 'forward':
                rnn_layer = RNN(
                    units=units, 
                    dropout=self.dropout, 
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True, 
                    name=layer_name
                )
            elif self.direction == 'backward':
                rnn_layer = RNN(
                    units=units, 
                    dropout=self.dropout, 
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True, 
                    go_backwards=True, 
                    name=layer_name
                )
            else:
                raise ValueError("direction must be 'bidirectional', 'forward', or 'backward'.")
                
            self.recurrent_layers.append(rnn_layer)

        # Final dense layers
        self.final_deep_dense = DeepLayer(
            final_hidden_layer_sizes + [1], 
            activation=final_hidden_activation,
            last_activation=final_activation,
            dropout_rate=dropout,
            name=f"{self.name}_finaldeep"
        )       

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through recurrent layers.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor
        training : bool, optional
            Whether in training mode
            
        Returns
        -------
        tf.Tensor
            Output tensor
        """
        x = inputs
        for layer in self.recurrent_layers:
            x = layer(x, training=training)
            
        outputs = self.final_deep_dense(x, training=training)
        
        if self.normalize is not None:
            outputs = CustomNormalizationLayer(
                mode=self.normalize, 
                axis=-2, 
                name=f"{self.name}_norm"
            )(outputs)
            
        return tf.squeeze(outputs, axis=-1)
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'recurrent_layer_sizes': self.recurrent_layer_sizes,
            'final_activation': self.final_activation,
            'final_hidden_layer_sizes': self.final_hidden_layer_sizes,
            'final_hidden_activation': self.final_hidden_activation,
            'direction': self.direction,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'recurrent_model': self.recurrent_model,
            'normalize': self.normalize
        })
        return config


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class CustomNormalizationLayer(layers.Layer):
    """
    Custom normalization layer with different modes.
    
    This layer applies different types of normalization along specified axes,
    including sum normalization and inverse normalization.
    
    Parameters
    ----------
    mode : str, default 'sum'
        Normalization mode: 'sum' or 'inverse'
    axis : int, default -2
        Axis along which to normalize
    epsilon : float, default 1e-6
        Numerical stability constant
    name : str, optional
        Layer name
    """
    
    def __init__(self,
                 mode: str = 'sum',
                 axis: int = -2,
                 epsilon: float = 1e-6,
                 name: Optional[str] = None,
                 **kwargs):
        if name is None:
            raise ValueError("CustomNormalizationLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.mode = mode
        self.axis = axis
        self.epsilon = epsilon

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply normalization.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor
            
        Returns
        -------
        tf.Tensor
            Normalized tensor
        """
        dtype = x.dtype
        epsilon = tf.cast(self.epsilon, dtype)
        n = tf.cast(tf.shape(x)[self.axis], dtype)

        denom_axis = tf.reduce_sum(x, axis=self.axis, keepdims=True)

        if self.mode == 'sum':
            x = n * x / tf.maximum(denom_axis, epsilon)
        elif self.mode == 'inverse':
            x = tf.maximum(x, epsilon)
            inv = tf.math.reciprocal(x)
            inv_total = tf.reduce_sum(inv, axis=self.axis, keepdims=True)
            inv_normalized = n * inv / tf.maximum(inv_total, epsilon)
            x = tf.math.reciprocal(tf.maximum(inv_normalized, epsilon))
        
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'mode': self.mode,
            'axis': self.axis,
            'epsilon': float(self.epsilon)
        })
        return config


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class EigenProductLayer(layers.Layer):
    """
    Layer for reconstructing matrices from eigenvalue decomposition.
    
    This layer reconstructs matrices from eigenvalues and eigenvectors using
    the formula: Matrix = V @ diag(eigenvalues) @ V^T, with optional scaling
    for precision or covariance matrix reconstruction.
    
    Parameters
    ----------
    scaling_factor : str, default 'none'
        Scaling mode: 'inverse', 'direct', or 'none'
    name : str, optional
        Layer name
    """
    
    def __init__(self, scaling_factor: str = 'none', name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        
        if name is None:
            raise ValueError("EigenProductLayer must have a name.")
        if scaling_factor not in ['inverse', 'direct', 'none']:
            raise ValueError("scaling_factor must be 'inverse', 'direct', or 'none'")
        
        self.scaling_factor = scaling_factor

    def call(self, eigenvalues: tf.Tensor, eigenvectors: tf.Tensor) -> tf.Tensor:
        """
        Reconstruct matrix from eigenvalue decomposition.
        
        Parameters
        ----------
        eigenvalues : tf.Tensor
            Eigenvalues tensor of shape [..., n]
        eigenvectors : tf.Tensor
            Eigenvectors tensor of shape [..., n, n]
            
        Returns
        -------
        tf.Tensor
            Reconstructed matrix
        """
        # Construct base matrix P = V @ diag(s) @ V^T
        if self.scaling_factor == 'inverse':
            # For precision matrix use 1/λ
            s = tf.math.reciprocal(eigenvalues)
        else:
            # For covariance matrix use λ
            s = eigenvalues

        # Scale eigenvectors: each column k of V is scaled by s[..., k]
        V_scaled = eigenvectors * tf.expand_dims(s, axis=-2)  # [..., n, n]
        P = tf.matmul(V_scaled, eigenvectors, transpose_b=True)  # [..., n, n]

        if self.scaling_factor == 'none':
            return P

        # Apply scaling based on mode
        if self.scaling_factor == 'direct':
            # Normalize P so diagonal elements are handled correctly
            diag_P = tf.linalg.diag_part(P)  # [..., n]
            inv_sqrt = tf.math.rsqrt(diag_P)  # [..., n] = 1/√diag
            row = tf.expand_dims(inv_sqrt, axis=-1)  # [..., n, 1]
            col = tf.expand_dims(inv_sqrt, axis=-2)  # [..., 1, n]
            return P * row * col

        else:  # 'inverse'
            # For precision matrix scaling
            diag_Sigma = tf.reduce_sum(
                tf.square(eigenvectors) * tf.expand_dims(eigenvalues, -2),
                axis=-1
            )  # [..., n]
            sqrt_d = tf.sqrt(diag_Sigma)  # [..., n]
            row = tf.expand_dims(sqrt_d, axis=-1)  # [..., n, 1]
            col = tf.expand_dims(sqrt_d, axis=-2)  # [..., 1, n]
            return P * row * col

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'scaling_factor': self.scaling_factor})
        return config


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class NormalizedSum(layers.Layer):
    """
    Layer for computing normalized sums along specified axes.
    
    This layer computes sums along one axis and then normalizes by the sum
    along another axis, commonly used for portfolio weight computation.
    
    Parameters
    ----------
    axis_1 : int, default -1
        First axis for summation
    axis_2 : int, default -2
        Second axis for normalization
    epsilon : float, default 1e-6
        Numerical stability constant
    name : str, optional
        Layer name
    """
    
    def __init__(self,
                 axis_1: int = -1,
                 axis_2: int = -2,
                 epsilon: float = 1e-6,
                 name: Optional[str] = None,
                 **kwargs):
        if name is None:
            raise ValueError("NormalizedSum must have a name.")
        super().__init__(name=name, **kwargs)
        self.axis_1 = axis_1
        self.axis_2 = axis_2
        self.epsilon = epsilon

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute normalized sum.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor
            
        Returns
        -------
        tf.Tensor
            Normalized sum
        """
        dtype = x.dtype
        epsilon = tf.cast(self.epsilon, dtype)
        w = tf.reduce_sum(x, axis=self.axis_1, keepdims=True)
        denominator = tf.reduce_sum(w, axis=self.axis_2, keepdims=True)
        sign = tf.where(denominator >= 0, tf.ones_like(denominator), -tf.ones_like(denominator))
        safe_denominator = tf.where(
            tf.abs(denominator) < epsilon,
            sign * epsilon,
            denominator
        )
        return w / safe_denominator

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'axis_1': self.axis_1,
            'axis_2': self.axis_2,
            'epsilon': float(self.epsilon)
        })
        return config


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class LagTransformLayer(layers.Layer):
    """
    Layer that applies a lag transformation to input financial time series.
    
    This layer applies a non-linear transformation to financial returns that
    accounts for temporal dependencies and lag effects. The transformation
    uses learnable parameters to adaptively weight different time lags.
    
    Parameters
    ----------
    warm_start : bool, default True
        Whether to initialize parameters near target values
    name : str, optional
        Layer name
    eps : float, optional
        Small epsilon value for numerical stability
    """
    
    def __init__(self, warm_start: bool = True, name: Optional[str] = None, 
                 eps: Optional[float] = None, **kwargs):
        if name is None:
            raise ValueError("LagTransformLayer must have a name.")
        super().__init__(name=name, **kwargs)
        
        self.eps = eps if eps is not None else K.epsilon()
        self.warm_start = warm_start
        
        # Target parameter values optimized for financial data
        self._target = dict(c0=2.8, c1=0.20, c2=0.85, c3=0.50, c4=0.05)

    def _inv_softplus(self, y: float) -> float:
        """Inverse softplus function for parameter initialization."""
        return tf.math.log(tf.math.expm1(y))

    def _add_param(self, name: str, target: float) -> tf.Variable:
        """Add a learnable parameter with appropriate initialization."""
        mean_raw = self._inv_softplus(target - self.eps)

        if self.warm_start:
            init = initializers.Constant(mean_raw)
        else:
            # Add ±5% noise in raw space
            init = initializers.RandomNormal(mean_raw, 0.05 * tf.math.abs(mean_raw))

        return self.add_weight(
            shape=(), 
            dtype="float32",
            name=f"raw_{name}",
            initializer=init,
            trainable=True,
        )

    def build(self, input_shape: Tuple) -> None:
        """Build layer parameters."""
        self._raw_c0 = self._add_param("c0", self._target["c0"])
        self._raw_c1 = self._add_param("c1", self._target["c1"])
        self._raw_c2 = self._add_param("c2", self._target["c2"])
        self._raw_c3 = self._add_param("c3", self._target["c3"])
        self._raw_c4 = self._add_param("c4", self._target["c4"])
        super().build(input_shape)

    def _pos(self, x: tf.Tensor) -> tf.Tensor:
        """Apply softplus + epsilon to ensure positive values."""
        return tf.nn.softplus(x) + self.eps

    def call(self, R: tf.Tensor) -> tf.Tensor:
        """
        Apply lag transformation to returns.
        
        Parameters
        ----------
        R : tf.Tensor
            Input returns tensor of shape [..., time_steps]
            
        Returns
        -------
        tf.Tensor
            Transformed returns with same shape as input
        """
        T = tf.shape(R)[-1]  # Time dimension length

        # Create time indices: t = [T, T-1, ..., 1]
        t = tf.cast(tf.range(1, T + 1), R.dtype)  # [1, 2, ..., T]
        t = tf.reverse(t, axis=[0])  # [T, T-1, ..., 1]

        # Get positive parameters via softplus
        c0 = self._pos(self._raw_c0)
        c1 = self._pos(self._raw_c1)
        c2 = self._pos(self._raw_c2)
        c3 = self._pos(self._raw_c3)
        c4 = self._pos(self._raw_c4)

        # Compute lag transformation parameters
        alpha = c0 * tf.pow(t, -c1)  # (T,)
        beta = c2 - c3 * tf.exp(-c4 * t)  # (T,)

        # Reshape for broadcasting
        ndims = tf.rank(R)
        pad_ones = tf.ones(ndims - 1, dtype=tf.int32)
        shape_T = tf.concat([pad_ones, [T]], 0)

        alpha_div_beta = tf.reshape(alpha / (beta + self.eps), shape_T)
        beta = tf.reshape(beta, shape_T)

        # Apply transformation: alpha/beta * tanh(beta * R)
        return alpha_div_beta * tf.tanh(beta * R)
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'eps': self.eps,
            'warm_start': self.warm_start
        })
        return config


__all__ = [
    'StandardDeviationLayer',
    'CovarianceLayer',
    'SpectralDecompositionLayer',
    'DimensionAwareLayer',
    'DeepLayer',
    'DeepRecurrentLayer',
    'CustomNormalizationLayer',
    'EigenProductLayer',
    'NormalizedSum',
    'LagTransformLayer',
]
