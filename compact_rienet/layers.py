"""
Compact-RIEnet: A Compact Rotational Invariant Eigenvalue Network for Portfolio Optimization

This module implements the Compact-RIEnet layer, a neural network architecture for 
portfolio optimization that processes financial time series data and outputs portfolio weights.

The architecture is based on Rotational Invariant Estimators (RIE) of the covariance matrix
combined with recurrent neural networks to capture temporal dependencies in financial data.

References:
-----------
Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025).
"Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage."
Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF '25).

Copyright (c) 2025
"""

import tensorflow as tf
from keras import layers
from typing import Optional, List, Tuple, Union, Sequence

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
from .dtype_utils import epsilon_for_dtype


@tf.keras.utils.register_keras_serializable(package='compact_rienet')
class CompactRIEnetLayer(layers.Layer):
    """
    Compact Rotational Invariant Estimator (RIE) Network layer for GMV portfolios.

    This layer implements the compact network described in Bongiorno et al. (2025) for
    global minimum-variance (GMV) portfolio construction. The architecture couples
    Rotational Invariant Estimators of the covariance matrix with recurrent neural
    networks in order to clean the eigen-spectrum and learn marginal volatilities in a
    parameter-efficient way.

    The layer automatically scales daily returns by 252 (annualisation factor) and
    applies the following stages:

    - Lag transformation with a five-parameter RIE-friendly non-linearity
    - Sample covariance estimation and eigenvalue decomposition
    - Bidirectional recurrent cleaning of eigenvalues (GRU or LSTM)
    - Dense transformation of marginal volatilities
    - Recombination into Σ⁻¹ followed by GMV weight normalisation

    Parameters
    ----------
    output_type : Union[str, Sequence[str]], default 'weights'
        Component(s) to return. Each entry must belong to
        {'weights', 'precision', 'covariance', 'input_transformed'} or the special string
        'all'. When multiple components are requested a dictionary mapping component name
        to tensor is returned.
    recurrent_layer_sizes : Sequence[int], optional
        Hidden sizes of the recurrent cleaning block. Defaults to [16] matching the
        compact GMV network in the paper. If a sequence with multiple integers is
        provided (e.g. [32, 16]) the recurrent cleaning head will apply multiple hidden
        layers in the given order: first a layer with 32 units, then one with 16 units.
    std_hidden_layer_sizes : Sequence[int], optional
        Hidden sizes of the dense network acting on marginal volatilities. Defaults to
        [8] matching the paper. Sequences are interpreted similarly (e.g. [64, 8] ->
        two dense hidden layers with 64 then 8 units).
    recurrent_cell : str, default 'GRU'
        Recurrent cell family used inside the eigenvalue cleaning block. Accepted
        values are 'GRU' and 'LSTM'.
    normalize_transformed_variance : bool, default True
        Whether to normalize the transformed inverse volatilities so that the implied
        covariance diagonal (variance) is centred on 1. Disable only when the network is
        not trained end-to-end on the GMV objective.
    name : str, optional
        Name of the Keras layer instance.
    **kwargs : dict
        Additional keyword arguments propagated to ``tf.keras.layers.Layer``.

    Input Shape
    -----------
    (batch_size, n_stocks, n_days)
        Daily return tensors for each batch element, stock and time step.

    Output Shape
    ------------
    Depends on ``output_type``:
        - 'weights' -> (batch_size, n_stocks, 1)
        - 'precision' or 'covariance' -> (batch_size, n_stocks, n_stocks)
        - 'input_transformed' -> (batch_size, n_stocks, n_days)
        - Multiple components -> ``dict`` mapping component name to the shapes above

    Notes
    -----
    Defaults replicate the compact RIE network optimised for GMV portfolios in the
    reference paper: a single bidirectional GRU layer with 16 units per direction and a
    dense marginal-volatility head with 8 hidden units. Inputs are annualised by 252 and
    the resulting Σ⁻¹ is symmetrised for numerical stability. Training on batches that
    span different asset universes is recommended when deploying on variable-dimension
    portfolios.
    
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
    Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025).
    Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage.
    Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025). End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning (arXiv:2507.01918).
    """
    
    def __init__(self,
                 output_type: Union[str, Sequence[str]] = 'weights',
                 recurrent_layer_sizes: Sequence[int] = (16,),
                 std_hidden_layer_sizes: Sequence[int] = (8,),
                 recurrent_cell: str = 'GRU',
                 normalize_transformed_variance: bool = True,
                 name: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Compact-RIEnet layer.
        
        Parameters
        ----------
        output_type : Union[str, Sequence[str]], default 'weights'
            Requested output component(s).
        recurrent_layer_sizes : Sequence[int], optional (default (16,))
            Hidden sizes of the recurrent cleaning block (defaults to [16]).
            If multiple integers are supplied (for example [32, 16]) the recurrent
            block will create multiple hidden layers applied in sequence: first 32 units,
            then 16 units.
        std_hidden_layer_sizes : Sequence[int], optional (default (8,))
            Hidden sizes of the dense marginal-volatility block (defaults to [8]).
            A sequence such as [64, 8] will be interpreted as two dense hidden layers
            with 64 then 8 units respectively.
        recurrent_cell : str, default 'GRU'
            Type of recurrent cell to use ('GRU' or 'LSTM').
        name : str, optional
            Layer name
        **kwargs : dict
            Additional arguments for base Layer
        """
        super().__init__(name=name, **kwargs)

        allowed_outputs = ('weights', 'precision', 'covariance', 'input_transformed')
        self._output_config = output_type if isinstance(output_type, str) else list(output_type)

        if isinstance(output_type, str):
            if output_type == 'all':
                components = list(allowed_outputs)
            else:
                if output_type not in allowed_outputs:
                    raise ValueError(
                        "output_type must be one of 'weights', 'precision', 'covariance', 'input_transformed', or 'all'"
                    )
                components = [output_type]
        else:
            output_list = list(output_type)
            if not output_list:
                raise ValueError("output_type cannot be an empty sequence")
            expanded: List[str] = []
            for entry in output_list:
                if entry == 'all':
                    expanded.extend(allowed_outputs)
                    continue
                if entry not in allowed_outputs:
                    raise ValueError(
                        "All requested outputs must be in {'weights', 'precision', 'covariance', 'input_transformed', 'all'}"
                    )
                expanded.append(entry)
            seen = set()
            components = []
            for entry in expanded:
                if entry not in seen:
                    components.append(entry)
                    seen.add(entry)

        self.output_components = tuple(components)
        self.output_type = components[0] if len(components) == 1 else tuple(components)

        if recurrent_layer_sizes is None:
            # backward-compatible fallback if caller passes None
            recurrent_layer_sizes = [16]
        else:
            recurrent_layer_sizes = list(recurrent_layer_sizes)
            if not recurrent_layer_sizes:
                raise ValueError("recurrent_layer_sizes must contain at least one positive integer")
        if std_hidden_layer_sizes is None:
            std_hidden_layer_sizes = [8]
        else:
            std_hidden_layer_sizes = list(std_hidden_layer_sizes)
            if not std_hidden_layer_sizes:
                raise ValueError("std_hidden_layer_sizes must contain at least one positive integer")

        for size in recurrent_layer_sizes:
            if size <= 0:
                raise ValueError("recurrent_layer_sizes must contain positive integers")
        for size in std_hidden_layer_sizes:
            if size <= 0:
                raise ValueError("std_hidden_layer_sizes must contain positive integers")

        normalized_cell = recurrent_cell.strip().upper()
        if normalized_cell not in {'GRU', 'LSTM'}:
            raise ValueError("recurrent_cell must be either 'GRU' or 'LSTM'")

        # Architecture parameters (paper defaults preserved if args omitted)
        self._std_hidden_layer_sizes = list(std_hidden_layer_sizes)
        self._recurrent_layer_sizes = list(recurrent_layer_sizes)
        self._recurrent_model = normalized_cell
        self._direction = 'bidirectional'
        self._dimensional_features = ['n_stocks', 'n_days', 'q']
        self._annualization_factor = 252.0
        self._normalize_variance = bool(normalize_transformed_variance)
        self.input_spec = layers.InputSpec(ndim=3)
        
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
            hidden_layer_sizes=self._std_hidden_layer_sizes + [1],
            last_activation='softplus',
            name=f"{self.name}_std_transform"
        )
        
        if self._normalize_variance:
            self.std_normalization = CustomNormalizationLayer(
                axis=-2,
                mode='inverse',
                inverse_power=2.0,
                name=f"{self.name}_std_norm"
            )
        else:
            self.std_normalization = None
        
        # Matrix reconstruction (see Eq. 13-15)
        self.eigen_product = EigenProductLayer(
            scaling_factor='inverse',
            name=f"{self.name}_eigen_product"
        )

        self.correlation_product = EigenProductLayer(
            scaling_factor='direct',
            name=f"{self.name}_correlation"
        )

        self.outer_product = CovarianceLayer(
            normalize=False,
            name=f"{self.name}_inverse_scale_outer"
        )
        
        # Portfolio weight computation
        self.portfolio_weights = NormalizedSum(
            axis_1=-1, 
            axis_2=-2, 
            name=f"{self.name}_portfolio_weights"
        )

    def build(self, input_shape: Tuple[int, int, int]) -> None:
        """Build sub-layers once input dimensionality is known."""
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 3:
            raise ValueError(
                "CompactRIEnetLayer expects inputs with shape (batch, n_stocks, n_days)."
            )

        batch = input_shape[0]
        n_stocks = input_shape[1]

        covariance_shape = tf.TensorShape([batch, n_stocks, n_stocks])
        eigenvalues_shape = tf.TensorShape([batch, n_stocks, 1])
        enhanced_features = 1 + len(self._dimensional_features)
        enhanced_eigen_shape = tf.TensorShape([batch, n_stocks, enhanced_features])
        std_shape = tf.TensorShape([batch, n_stocks, 1])
        eigenvalues_vector_shape = tf.TensorShape([batch, n_stocks])

        self.lag_transform.build(input_shape)
        self.std_layer.build(input_shape)
        self.covariance_layer.build(input_shape)
        self.spectral_decomp.build(covariance_shape)
        self.dimension_aware.build([eigenvalues_shape, input_shape])
        self.eigenvalue_transform.build(enhanced_eigen_shape)
        self.std_transform.build(std_shape)
        if self.std_normalization is not None:
            self.std_normalization.build(std_shape)
        self.eigen_product.build([eigenvalues_vector_shape, covariance_shape])
        self.correlation_product.build([eigenvalues_vector_shape, covariance_shape])
        self.outer_product.build(std_shape)
        self.portfolio_weights.build(covariance_shape)

        super().build(input_shape)
        
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
            Output tensor determined by `output_type`:
            - weights: portfolio weights (batch, n_stocks, 1)
            - precision: cleaned precision matrix Σ^{-1}
            - covariance: pseudo-inverse covariance Σ
        """
        # Scale inputs by annualization factor
        scaled_inputs = inputs * self._annualization_factor
        
        # Apply lag transformation
        input_transformed = self.lag_transform(scaled_inputs)
        
        # Compute standard deviation and mean
        std, mean = self.std_layer(input_transformed)
        
        # Standardize returns
        zscores = (input_transformed - mean) / std
        
        # Compute correlation matrix
        correlation_matrix = self.covariance_layer(zscores)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = self.spectral_decomp(correlation_matrix)
        
        # Add dimensional features
        eigenvalues_enhanced = self.dimension_aware([eigenvalues, scaled_inputs])
        
        # Transform eigenvalues with recurrent network
        transformed_inverse_eigenvalues = self.eigenvalue_transform(eigenvalues_enhanced)
        spectrum_epsilon = epsilon_for_dtype(
            transformed_inverse_eigenvalues.dtype,
            tf.keras.backend.epsilon(),
        )
        transformed_inverse_eigenvalues = tf.maximum(transformed_inverse_eigenvalues, spectrum_epsilon)

        # Transform standard deviations
        transformed_inverse_std = self.std_transform(std)
        if self.std_normalization is not None:
            transformed_inverse_std = self.std_normalization(transformed_inverse_std)
        transformed_inverse_std = tf.maximum(
            transformed_inverse_std,
            epsilon_for_dtype(
                transformed_inverse_std.dtype,
                tf.keras.backend.epsilon(),
            )
        )

        # Build inverse correlation matrix (Eq. 13-14 of the paper)
        inverse_correlation = self.eigen_product(
            transformed_inverse_eigenvalues, eigenvectors
        )

        transformed_eigenvalues = tf.math.reciprocal(transformed_inverse_eigenvalues)
        transformed_eigenvalues = tf.maximum(
            transformed_eigenvalues,
            epsilon_for_dtype(
                transformed_eigenvalues.dtype,
                tf.keras.backend.epsilon(),
            )
        )
        correlation_matrix = self.correlation_product(
            transformed_eigenvalues, eigenvectors
        )
        correlation_matrix = 0.5 * (
            correlation_matrix + tf.linalg.matrix_transpose(correlation_matrix)
        )

        # Combine with marginal inverse volatilities to obtain Σ^{-1}
        inverse_volatility_matrix = self.outer_product(transformed_inverse_std)
        precision_matrix = inverse_correlation * inverse_volatility_matrix
        precision_matrix = 0.5 * (precision_matrix + tf.linalg.matrix_transpose(precision_matrix))

        transformed_std = tf.math.reciprocal(transformed_inverse_std)
        transformed_std = tf.maximum(
            transformed_std,
            epsilon_for_dtype(transformed_std.dtype, tf.keras.backend.epsilon()),
        )
        volatility_matrix = self.outer_product(transformed_std)
        covariance = correlation_matrix * volatility_matrix
        covariance = 0.5 * (covariance + tf.linalg.matrix_transpose(covariance))

        results = {}
        if 'precision' in self.output_components:
            results['precision'] = precision_matrix

        if 'covariance' in self.output_components:
            results['covariance'] = covariance

        if 'weights' in self.output_components:
            weights = self.portfolio_weights(precision_matrix)
            results['weights'] = weights

        if 'input_transformed' in self.output_components:
            results['input_transformed'] = input_transformed

        if len(self.output_components) == 1:
            return results[self.output_components[0]]

        return results
    
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
            'output_type': self._output_config,
            'recurrent_layer_sizes': list(self._recurrent_layer_sizes),
            'std_hidden_layer_sizes': list(self._std_hidden_layer_sizes),
            'recurrent_cell': self._recurrent_model,
            'normalize_transformed_variance': self._normalize_variance,
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
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_size, n_stocks, n_days = input_shape

        def shape_for(component: str) -> Tuple[int, ...]:
            if component == 'weights':
                return (batch_size, n_stocks, 1)
            if component == 'input_transformed':
                return (batch_size, n_stocks, n_days)
            return (batch_size, n_stocks, n_stocks)

        if len(self.output_components) == 1:
            return shape_for(self.output_components[0])

        return {component: shape_for(component) for component in self.output_components}
