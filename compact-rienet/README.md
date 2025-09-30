# Compact-RIEnet: A Compact Recurrent-Invariant Eigenvalue Network for Portfolio Optimization

[![PyPI version](https://badge.fury.io/py/compact-rienet.svg)](https://badge.fury.io/py/compact-rienet)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compact-RIEnet is a neural network architecture specifically designed for portfolio optimization tasks. It combines eigenvalue decomposition of covariance matrices with recurrent neural networks to capture both cross-sectional relationships and temporal dependencies in financial data.

## Key Features

- **Eigenvalue-Based Processing**: Uses spectral decomposition for robust covariance matrix handling
- **Recurrent Architecture**: GRU-based networks capture temporal dependencies in financial time series
- **Portfolio-Optimized**: Specifically designed for portfolio weight optimization
- **Professional Implementation**: Comprehensive documentation, type hints, and testing
- **Easy to Use**: Simple API with sensible defaults for financial applications

## Installation

Install from PyPI:

```bash
pip install compact-rienet
```

Or install from source:

```bash
git clone https://github.com/author/compact-rienet.git
cd compact-rienet
pip install -e .
```

## Quick Start

### Basic Usage

```python
import tensorflow as tf
import numpy as np
from compact_rienet import CompactRIEnetLayer, variance_loss_function

# Create the model layer
rienet_layer = CompactRIEnetLayer(output_type='weights')

# Sample daily returns data: (batch_size, n_stocks, n_days)
# Shape: (32 samples, 10 stocks, 60 trading days)
returns = tf.random.normal((32, 10, 60), stddev=0.02)

# Get portfolio weights
portfolio_weights = rienet_layer(returns)
print(f"Portfolio weights shape: {portfolio_weights.shape}")  # (32, 10, 1)

# Verify weights sum to 1
weights_sum = tf.reduce_sum(portfolio_weights, axis=1)
print(f"Weights sum: {weights_sum[0].numpy()}")  # Should be close to 1.0
```

### Training with Variance Loss

The package includes specialized loss functions for portfolio optimization:

```python
import tensorflow as tf
from compact_rienet import CompactRIEnetLayer, variance_loss_function

# Build a complete model
def create_portfolio_model(n_stocks):
    inputs = tf.keras.Input(shape=(n_stocks, None))  # (n_stocks, n_days)
    weights = CompactRIEnetLayer(output_type='weights')(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=weights)
    return model

# Create model for 10 stocks
model = create_portfolio_model(n_stocks=10)

# Prepare training data
# Daily returns: (batch_size, n_stocks, n_days)
X_train = tf.random.normal((1000, 10, 60), stddev=0.02)

# True covariance matrices for loss computation: (batch_size, n_stocks, n_stocks)
# In practice, these would be computed from realized returns
Sigma_train = tf.random.normal((1000, 10, 10))
Sigma_train = tf.matmul(Sigma_train, Sigma_train, transpose_b=True)  # Ensure PSD

# Custom training loop with variance loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(returns_batch, covariance_batch):
    with tf.GradientTape() as tape:
        # Forward pass
        predicted_weights = model(returns_batch, training=True)
        
        # Compute variance loss
        loss = variance_loss_function(covariance_batch, predicted_weights)
        loss = tf.reduce_mean(loss)  # Average over batch
    
    # Backward pass
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Training loop
batch_size = 32
n_batches = len(X_train) // batch_size

for epoch in range(10):
    epoch_loss = 0.0
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        batch_returns = X_train[start_idx:end_idx]
        batch_covariance = Sigma_train[start_idx:end_idx]
        
        loss = train_step(batch_returns, batch_covariance)
        epoch_loss += loss
        
    print(f"Epoch {epoch+1}, Average Loss: {epoch_loss/n_batches:.6f}")
```

### Using Different Output Types

The layer can output either portfolio weights or the transformed precision matrix:

```python
# Get portfolio weights (default)
weights_layer = CompactRIEnetLayer(output_type='weights')
weights = weights_layer(returns)  # Shape: (batch_size, n_stocks, 1)

# Get precision matrix (inverse covariance)
precision_layer = CompactRIEnetLayer(output_type='precision') 
precision_matrix = precision_layer(returns)  # Shape: (batch_size, n_stocks, n_stocks)
```

## Loss Functions

### Variance Loss Function

The `variance_loss_function` is the primary loss function for training portfolio optimization models:

```python
from compact_rienet import variance_loss_function

# Compute portfolio variance
loss = variance_loss_function(
    covariance_true=true_covariance,      # (batch_size, n_assets, n_assets)
    weights_predicted=predicted_weights,   # (batch_size, n_assets, 1)
    penalty=0.1                           # Optional leverage penalty
)
```

**Parameters:**
- `covariance_true`: True/realized covariance matrices
- `weights_predicted`: Portfolio weights from the model
- `penalty`: Penalty coefficient for excessive leverage (L1 norm > 1)

**Mathematical Formula:**
```
Loss = w^T Σ w + penalty × max(0, ||w||₁ - 1)²
```

Where:
- `w` are the portfolio weights
- `Σ` is the true covariance matrix
- `||w||₁` is the L1 norm (gross leverage)

### Additional Loss Functions

```python
from compact_rienet import buy_and_hold_volatility_loss, frobenius_loss_function

# Buy-and-hold volatility loss
vol_loss = buy_and_hold_volatility_loss(returns, weights)

# Frobenius norm loss for covariance matrix estimation
frob_loss = frobenius_loss_function(true_covariance, predicted_covariance)
```

## Architecture Details

The Compact-RIEnet architecture processes financial data through several stages:

1. **Input Scaling**: Daily returns are scaled by 252 (annualization factor)
2. **Lag Transformation**: Temporal preprocessing with learnable lag parameters
3. **Covariance Estimation**: Sample covariance matrix computation
4. **Eigenvalue Decomposition**: Spectral decomposition for robust processing
5. **Recurrent Processing**: GRU networks process eigenvalue sequences
6. **Matrix Reconstruction**: Eigen-decomposition reconstruction with learned eigenvalues
7. **Portfolio Weights**: Final weights via normalized summation

### Fixed Hyperparameters

The architecture uses optimized hyperparameters:
- Hidden layer size: 8 units
- Recurrent layer size: 32 units  
- Recurrent model: GRU (Gated Recurrent Unit)
- Direction: Bidirectional processing
- Dimensional features: `['n_stocks', 'n_days', 'q']`

## Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.10.0
- Keras ≥ 2.10.0
- NumPy ≥ 1.21.0

## Development

For development installation:

```bash
git clone https://github.com/author/compact-rienet.git
cd compact-rienet
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

## Citation

**Please cite the following papers when using this code:**

```bibtex
@article{author2025compact,
    title={Compact-RIEnet: A Compact Recurrent-Invariant Eigenvalue Network for Portfolio Optimization},
    author={[Author Names]},
    journal={[Journal Name]},
    year={2025},
    note={[Additional publication details]}
}
```

For software citation:

```bibtex
@software{compact_rienet2025,
    title={Compact-RIEnet: A Compact Recurrent-Invariant Eigenvalue Network for Portfolio Optimization},
    author={[Author Name]},
    year={2025},
    version={1.0.0},
    url={https://github.com/author/compact-rienet}
}
```

You can also print citation information programmatically:

```python
import compact_rienet
compact_rienet.print_citation()
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions, please:
- Open an issue on [GitHub](https://github.com/author/compact-rienet/issues)
- Check the documentation
- Contact the authors

## Acknowledgments

This implementation is based on research in portfolio optimization and neural network architectures for financial applications. Please cite the appropriate academic papers when using this software in research.

---

**Note**: This is a research implementation. Please validate the results for your specific use case and consider the inherent risks in portfolio optimization and financial modeling.