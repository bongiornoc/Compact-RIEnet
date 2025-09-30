# Compact-RIEnet: A Compact Rotational Invariant Estimator Network for GMV Optimisation

[![PyPI version](https://img.shields.io/pypi/v/compact-rienet.svg)](https://pypi.org/project/compact-rienet/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compact-RIEnet is a neural architecture purpose-built for **global minimum-variance (GMV)** portfolio optimisation. The model combines Rotational Invariant Estimator (RIE) techniques for covariance cleaning with recurrent neural networks, allowing it to capture both cross-sectional structure and temporal dynamics in financial data.

## Key Features

- **GMV-First Design** – End-to-end training minimises realised portfolio variance
- **RIE-Based Cleaning** – Rotational Invariant Estimators denoise the covariance spectrum for stable optimisation
- **Dimension Agnostic** – Train on mixed asset universes and deploy on unseen sizes without architectural changes
- **Configurable Recurrent Core** – Switch between GRU/LSTM cleaners and customise hidden stacks while retaining paper defaults
- **Professional Implementation** – Comprehensive documentation, type hints, and test coverage

## Installation

Install from PyPI:

```bash
pip install compact-rienet
```

Or install from source:

```bash
git clone https://github.com/bongiornoc/Compact-RIEnet.git
cd Compact-RIEnet
pip install -e .
```

## Quick Start

### Basic Usage

```python
import tensorflow as tf
from compact_rienet import CompactRIEnetLayer, variance_loss_function

# Defaults reproduce the compact GMV architecture (bidirectional GRU with 16 units)
rienet_layer = CompactRIEnetLayer(output_type=['weights', 'precision'])

# Sample data: (batch_size, n_stocks, n_days)
returns = tf.random.normal((32, 10, 60), stddev=0.02)

# Retrieve GMV weights and cleaned precision in one pass
outputs = rienet_layer(returns)
weights = outputs['weights']          # (32, 10, 1)
precision = outputs['precision']      # (32, 10, 10)

# GMV training objective
covariance = tf.random.normal((32, 10, 10))
covariance = tf.matmul(covariance, covariance, transpose_b=True)
loss = variance_loss_function(covariance, weights)
print(loss.shape)  # (32, 1, 1)
```

### Training with the GMV Variance Loss

```python
import tensorflow as tf
from compact_rienet import CompactRIEnetLayer, variance_loss_function

def create_portfolio_model():
    inputs = tf.keras.Input(shape=(None, None))
    weights = CompactRIEnetLayer(output_type='weights')(inputs)
    return tf.keras.Model(inputs=inputs, outputs=weights)

model = create_portfolio_model()

# Synthetic training data
X_train = tf.random.normal((1000, 10, 60), stddev=0.02)
Sigma_train = tf.random.normal((1000, 10, 10))
Sigma_train = tf.matmul(Sigma_train, Sigma_train, transpose_b=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=variance_loss_function)

model.fit(X_train, Sigma_train, epochs=10, batch_size=32, verbose=True)
```

> **Tip:** When you intend to deploy Compact-RIEnet on portfolios of varying size, train on batches that span different asset universes. The RIE-based architecture is dimension agnostic and benefits from heterogeneous training shapes.

### Using Different Output Types

```python
# GMV weights only
weights = CompactRIEnetLayer(output_type='weights')(returns)

# Precision matrix only
precision_matrix = CompactRIEnetLayer(output_type='precision')(returns)

# Both precision and covariance in one pass
outputs = CompactRIEnetLayer(output_type=['precision', 'covariance'])(returns)
precision_matrix = outputs['precision']
covariance_matrix = outputs['covariance']
```

## Loss Function

### Variance Loss Function

```python
from compact_rienet import variance_loss_function

loss = variance_loss_function(
    covariance_true=true_covariance,    # (batch_size, n_assets, n_assets)
    weights_predicted=predicted_weights # (batch_size, n_assets, 1)
)
```

**Mathematical Formula:**
```
Loss = n_assets × wᵀ Σ w
```

Where `w` are the portfolio weights and `Σ` is the realised covariance matrix.

## Architecture Details

The Compact-RIEnet pipeline consists of:

1. **Input Scaling** – Annualise returns by 252
2. **Lag Transformation** – Five-parameter memory kernel for temporal weighting
3. **Covariance Estimation** – Sample covariance across assets
4. **Eigenvalue Decomposition** – Spectral analysis of the covariance matrix
5. **Recurrent Cleaning** – Bidirectional GRU/LSTM processing of eigen spectra
6. **Marginal Volatility Head** – Dense network forecasting inverse standard deviations
7. **Matrix Reconstruction** – RIE-based synthesis of Σ⁻¹ and GMV weight normalisation

Paper defaults use a single bidirectional GRU layer with 16 units per direction and a marginal-volatility head with 8 hidden units, matching the compact network described in Bongiorno et al. (2025).

## Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.10.0
- Keras ≥ 2.10.0
- NumPy ≥ 1.21.0

## Development

```bash
git clone https://github.com/bongiornoc/Compact-RIEnet.git
cd Compact-RIEnet
pip install -e ".[dev]"
pytest tests/
```

## Citation

Please cite the following references when using Compact-RIEnet:

```bibtex
@inproceedings{bongiorno2025compact,
  title={Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage},
  author={Bongiorno, Christian and Manolakis, Efstratios and Mantegna, Rosario N.},
  booktitle={Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF '25)},
  year={2025}
}

@article{bongiorno2025covariance,
  title={End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning},
  author={Bongiorno, Christian and Manolakis, Efstratios and Mantegna, Rosario N.},
  journal={arXiv preprint arXiv:2507.01918},
  year={2025}
}
```

For software citation:

```bibtex
@software{compact_rienet2025,
  title={Compact-RIEnet: A Compact Rotational Invariant Estimator Network for Global Minimum-Variance Optimisation},
  author={Bongiorno, Christian},
  year={2025},
  version={1.0.3},
  url={https://github.com/bongiornoc/Compact-RIEnet}
}
```

You can print citation information programmatically:

```python
import compact_rienet
compact_rienet.print_citation()
```

## Support

For questions, issues, or contributions, please:

- Open an issue on [GitHub](https://github.com/bongiornoc/Compact-RIEnet/issues)
- Check the documentation
- Contact Prof. Christian Bongiorno (<christian.bongiorno@centralesupelec.fr>) for calibrated model weights or collaboration requests
