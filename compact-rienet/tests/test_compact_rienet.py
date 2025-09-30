"""
Tests for the Compact-RIEnet package.

This module contains comprehensive tests for all components of the Compact-RIEnet
package including the main layer, loss functions, and custom layers.
"""

import pytest
import tensorflow as tf
import numpy as np
from typing import Tuple

# Import all components to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compact_rienet import (
    CompactRIEnetLayer,
    variance_loss_function
)

from compact_rienet.custom_layers import (
    StandardDeviationLayer,
    CovarianceLayer, 
    SpectralDecompositionLayer,
    DimensionAwareLayer,
    DeepLayer,
    DeepRecurrentLayer,
    CustomNormalizationLayer,
    EigenProductLayer,
    NormalizedSum,
    LagTransformLayer
)


class TestCompactRIEnetLayer:
    """Test cases for the main CompactRIEnetLayer."""
    
    def test_layer_initialization(self):
        """Test layer can be initialized with different parameters."""
        # Test default initialization
        layer1 = CompactRIEnetLayer()
        assert layer1.output_type == 'weights'
        
        # Test with precision output
        layer2 = CompactRIEnetLayer(output_type='precision')
        assert layer2.output_type == 'precision'
        
        # Test invalid output type
        with pytest.raises(ValueError):
            CompactRIEnetLayer(output_type='invalid')
    
    def test_weights_output_shape(self):
        """Test that weights output has correct shape."""
        layer = CompactRIEnetLayer(output_type='weights')
        
        # Test various input shapes
        test_cases = [
            (32, 10, 60),  # 32 batches, 10 stocks, 60 days
            (1, 5, 30),    # Single batch, 5 stocks, 30 days
            (16, 20, 120), # 16 batches, 20 stocks, 120 days
        ]
        
        for batch_size, n_stocks, n_days in test_cases:
            inputs = tf.random.normal((batch_size, n_stocks, n_days))
            outputs = layer(inputs)
            
            expected_shape = (batch_size, n_stocks, 1)
            assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"
    
    def test_precision_output_shape(self):
        """Test that precision output has correct shape."""
        layer = CompactRIEnetLayer(output_type='precision')
        
        batch_size, n_stocks, n_days = 16, 8, 50
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        outputs = layer(inputs)
        
        expected_shape = (batch_size, n_stocks, n_stocks)
        assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"

    def test_covariance_output_shape(self):
        """Test that covariance output has correct shape."""
        layer = CompactRIEnetLayer(output_type='covariance')

        batch_size, n_stocks, n_days = 12, 5, 40
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        outputs = layer(inputs)

        expected_shape = (batch_size, n_stocks, n_stocks)
        assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"
    
    def test_weights_normalization(self):
        """Test that portfolio weights sum to 1."""
        layer = CompactRIEnetLayer(output_type='weights')
        
        batch_size, n_stocks, n_days = 8, 6, 40
        inputs = tf.random.normal((batch_size, n_stocks, n_days), stddev=0.02)
        weights = layer(inputs)
        
        # Check weights sum to 1 along the stocks dimension
        weights_sum = tf.reduce_sum(weights, axis=1)  # Sum over stocks
        
        # Should be close to 1 for each sample
        np.testing.assert_allclose(weights_sum.numpy(), 1.0, rtol=1e-6)
    
    def test_input_scaling(self):
        """Test that input scaling by 252 is applied."""
        layer = CompactRIEnetLayer(output_type='weights')
        
        # Small inputs to see scaling effect
        small_inputs = tf.ones((4, 5, 30)) * 0.001
        large_inputs = small_inputs * 252
        
        weights_small = layer(small_inputs)
        weights_large = layer(large_inputs)
        
        # After standardisation the layer is scale-invariant; outputs should match
        diff = tf.reduce_max(tf.abs(weights_small - weights_large))
        assert diff < 1e-6, f"Input scaling should preserve outputs, got {diff}"
    
    def test_layer_serialization(self):
        """Test that layer can be serialized and deserialized."""
        layer = CompactRIEnetLayer(output_type='weights', name='test_layer')
        config = layer.get_config()
        
        # Check config contains expected keys
        assert 'output_type' in config
        assert config['output_type'] == 'weights'
        
        # Test from_config
        new_layer = CompactRIEnetLayer.from_config(config)
        assert new_layer.output_type == layer.output_type

    def test_multiple_outputs(self):
        """Layer should optionally return multiple components."""
        layer = CompactRIEnetLayer(output_type=['weights', 'precision'])

        batch_size, n_stocks, n_days = 3, 4, 20
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        outputs = layer(inputs)

        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {'weights', 'precision'}
        assert outputs['weights'].shape == (batch_size, n_stocks, 1)
        assert outputs['precision'].shape == (batch_size, n_stocks, n_stocks)

    def test_custom_recurrent_configuration(self):
        """Custom recurrent sizes and cell types should be honoured."""
        layer = CompactRIEnetLayer(
            output_type='weights',
            recurrent_layer_sizes=[12, 6],
            std_hidden_layer_sizes=[4, 2],
            recurrent_cell='lstm'
        )

        batch_size, n_stocks, n_days = 2, 3, 15
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        weights = layer(inputs)

        assert weights.shape == (batch_size, n_stocks, 1)
        first_block = layer.eigenvalue_transform.recurrent_layers[0]
        assert isinstance(first_block, tf.keras.layers.Bidirectional)
        assert isinstance(first_block.forward_layer, tf.keras.layers.LSTM)


class TestVarianceLoss:
    """Tests for the variance loss function."""

    def test_variance_loss_function_shape(self):
        """Variance loss should keep batch dimension and output scalar per sample."""
        batch_size, n_assets = 16, 8

        weights = tf.random.normal((batch_size, n_assets, 1))
        weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

        covariance = tf.random.normal((batch_size, n_assets, n_assets))
        covariance = tf.matmul(covariance, covariance, transpose_b=True)

        loss = variance_loss_function(covariance, weights)

        expected_shape = (batch_size, 1, 1)
        assert loss.shape == expected_shape

    def test_variance_loss_is_non_negative(self):
        """Variance loss values should be non-negative."""
        batch_size, n_assets = 8, 5

        weights = tf.random.normal((batch_size, n_assets, 1))
        weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

        covariance = tf.eye(n_assets, batch_shape=[batch_size]) * 0.01

        loss = variance_loss_function(covariance, weights)

        assert tf.reduce_all(loss >= 0), "Variance loss should be non-negative"


class TestCustomLayers:
    """Test cases for custom layers."""
    
    def test_standard_deviation_layer(self):
        """Test StandardDeviationLayer."""
        layer = StandardDeviationLayer(axis=-1, name='test_std')
        
        # Test with known data
        x = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
        std, mean = layer(x)
        
        # Check shapes
        assert std.shape == (1, 2, 1)
        assert mean.shape == (1, 2, 1)
        
        # Check values are reasonable
        assert tf.reduce_all(std > 0), "Standard deviation should be positive"
    
    def test_covariance_layer(self):
        """Test CovarianceLayer."""
        layer = CovarianceLayer(normalize=True, name='test_cov')
        
        batch_size, n_assets, n_days = 4, 3, 20
        returns = tf.random.normal((batch_size, n_assets, n_days))
        
        cov_matrix = layer(returns)
        
        # Check output shape
        expected_shape = (batch_size, n_assets, n_assets)
        assert cov_matrix.shape == expected_shape
        
        # Covariance matrix should be symmetric
        cov_transpose = tf.transpose(cov_matrix, perm=[0, 2, 1])
        np.testing.assert_allclose(cov_matrix.numpy(), cov_transpose.numpy(), rtol=1e-6)
    
    def test_spectral_decomposition_layer(self):
        """Test SpectralDecompositionLayer."""
        layer = SpectralDecompositionLayer(name='test_spectral')
        
        batch_size, n_assets = 4, 5
        
        # Create symmetric positive definite matrix
        A = tf.random.normal((batch_size, n_assets, n_assets))
        cov_matrix = tf.matmul(A, A, transpose_b=True)
        
        eigenvalues, eigenvectors = layer(cov_matrix)
        
        # Check shapes
        assert eigenvalues.shape == (batch_size, n_assets, 1)
        assert eigenvectors.shape == (batch_size, n_assets, n_assets)
        
        # Eigenvalues should be positive for PSD matrix
        assert tf.reduce_all(eigenvalues >= 0), "Eigenvalues should be non-negative for PSD matrix"
    
    def test_dimension_aware_layer(self):
        """Test DimensionAwareLayer."""
        features = ['n_stocks', 'n_days', 'q']
        layer = DimensionAwareLayer(features=features, name='test_dim_aware')
        
        batch_size, n_stocks, n_days, n_eigenvals = 4, 6, 50, 6
        eigenvalues = tf.random.normal((batch_size, n_eigenvals, 1))
        original_inputs = tf.random.normal((batch_size, n_stocks, n_days))
        
        enhanced = layer([eigenvalues, original_inputs])
        
        # Should add 3 features
        expected_shape = (batch_size, n_eigenvals, 1 + len(features))
        assert enhanced.shape == expected_shape
    
    def test_deep_layer(self):
        """Test DeepLayer."""
        layer = DeepLayer(
            hidden_layer_sizes=[16, 8, 4],
            activation='relu',
            last_activation='linear',
            name='test_deep'
        )
        
        batch_size, input_dim = 8, 10
        inputs = tf.random.normal((batch_size, input_dim))
        
        outputs = layer(inputs)
        
        # Should output final layer size
        expected_shape = (batch_size, 4)
        assert outputs.shape == expected_shape
    
    def test_deep_recurrent_layer(self):
        """Test DeepRecurrentLayer."""
        layer = DeepRecurrentLayer(
            recurrent_layer_sizes=[32],
            recurrent_model='GRU',
            direction='bidirectional',
            name='test_deep_rnn'
        )
        
        batch_size, seq_len, input_dim = 4, 20, 8
        inputs = tf.random.normal((batch_size, seq_len, input_dim))
        
        outputs = layer(inputs)
        
        # Output should have sequence dimension
        assert len(outputs.shape) >= 2
        assert outputs.shape[0] == batch_size
    
    def test_eigen_product_layer(self):
        """Test EigenProductLayer."""
        layer = EigenProductLayer(scaling_factor='none', name='test_eigen_product')
        
        batch_size, n_assets = 4, 5
        eigenvalues = tf.random.normal((batch_size, n_assets))
        eigenvectors = tf.random.normal((batch_size, n_assets, n_assets))
        
        # Make eigenvectors orthogonal (approximately)
        eigenvectors, _ = tf.linalg.qr(eigenvectors)
        
        reconstructed = layer(eigenvalues, eigenvectors)
        
        # Should reconstruct matrix of same size
        expected_shape = (batch_size, n_assets, n_assets)
        assert reconstructed.shape == expected_shape
    
    def test_normalized_sum(self):
        """Test NormalizedSum layer."""
        layer = NormalizedSum(axis_1=-1, axis_2=-2, name='test_norm_sum')
        
        batch_size, n_assets = 8, 6
        matrix = tf.random.normal((batch_size, n_assets, n_assets))
        
        weights = layer(matrix)
        
        # Should sum to 1 along the specified axis
        weights_sum = tf.reduce_sum(weights, axis=-2, keepdims=True)
        np.testing.assert_allclose(weights_sum.numpy(), 1.0, rtol=1e-6)
    
    def test_lag_transform_layer(self):
        """Test LagTransformLayer."""
        layer = LagTransformLayer(warm_start=True, name='test_lag_transform')
        
        batch_size, n_stocks, n_days = 4, 5, 30
        returns = tf.random.normal((batch_size, n_stocks, n_days), stddev=0.02)
        
        transformed = layer(returns)
        
        # Should preserve input shape
        assert transformed.shape == returns.shape
        
        # Should be different from input (transformation applied)
        assert not tf.reduce_all(tf.abs(transformed - returns) < 1e-6)


class TestIntegration:
    """Integration tests for the complete package."""
    
    def test_full_model_creation(self):
        """Test creating a complete Keras model with CompactRIEnetLayer."""
        n_stocks = 8
        
        inputs = tf.keras.Input(shape=(n_stocks, None))
        weights = CompactRIEnetLayer(output_type='weights')(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=weights)
        
        # Test model compilation
        model.compile(optimizer='adam', loss='mse')
        
        # Test forward pass
        batch_size, n_days = 16, 60
        test_input = tf.random.normal((batch_size, n_stocks, n_days))
        output = model(test_input)
        
        assert output.shape == (batch_size, n_stocks, 1)
    
    def test_training_compatibility(self):
        """Test that the layer works with standard Keras training."""
        n_stocks = 6
        
        # Build model
        model = tf.keras.Sequential([
            CompactRIEnetLayer(output_type='weights')
        ])
        
        # Compile with custom loss
        def custom_loss(y_true, y_pred):
            # Simple squared difference loss for testing
            return tf.reduce_mean(tf.square(y_true - y_pred))
        
        model.compile(optimizer='adam', loss=custom_loss)
        
        # Generate dummy data
        batch_size, n_days = 32, 50
        X = tf.random.normal((batch_size, n_stocks, n_days))
        y = tf.random.normal((batch_size, n_stocks, 1))  # Dummy targets
        
        # Test that training runs without errors
        history = model.fit(X, y, epochs=1, verbose=0)
        
        assert len(history.history['loss']) == 1
        assert history.history['loss'][0] is not None


# Test fixtures and utilities
@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    batch_size, n_stocks, n_days = 16, 10, 60
    returns = tf.random.normal((batch_size, n_stocks, n_days), stddev=0.02)
    return returns


@pytest.fixture
def sample_covariance():
    """Generate sample covariance matrices for testing."""
    batch_size, n_stocks = 16, 10
    A = tf.random.normal((batch_size, n_stocks, n_stocks))
    covariance = tf.matmul(A, A, transpose_b=True)
    return covariance


if __name__ == "__main__":
    pytest.main([__file__])
