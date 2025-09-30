"""
Compact-RIEnet: A Compact Recurrent-Invariant Eigenvalue Network for Portfolio Optimization

This package provides a neural network architecture specifically designed for portfolio
optimization tasks. The Compact-RIEnet layer processes financial time series data and
outputs optimized portfolio weights using eigenvalue decomposition and recurrent networks.

Key Features:
- Eigenvalue-based covariance matrix processing
- Recurrent neural networks for temporal modeling  
- Specialized loss functions for portfolio optimization
- Professional implementation with comprehensive documentation

Main Components:
- CompactRIEnetLayer: Main neural network layer for portfolio optimization
- Loss functions: Specialized loss functions including variance_loss_function
- Custom layers: Internal building blocks for the architecture

References:
-----------
Please cite the following papers when using this code:
[Paper references to be provided by the author]

Examples:
---------
>>> import tensorflow as tf
>>> from compact_rienet import CompactRIEnetLayer, variance_loss_function
>>> 
>>> # Create the layer
>>> layer = CompactRIEnetLayer(output_type='weights')
>>> 
>>> # Sample data: 32 batches, 10 stocks, 60 days of returns
>>> returns = tf.random.normal((32, 10, 60)) 
>>> 
>>> # Get portfolio weights
>>> weights = layer(returns)
>>> 
>>> # Use with variance loss for training
>>> true_cov = tf.eye(10, batch_shape=[32])  # Example covariance
>>> loss = variance_loss_function(true_cov, weights)

Copyright (c) 2025
"""

from .layers import CompactRIEnetLayer
from .losses import (
    variance_loss_function,
    buy_and_hold_volatility_loss,
    frobenius_loss_function
)

# Version information
__version__ = "1.0.0"
__author__ = "Author Name"
__email__ = "author@email.com"

# Public API
__all__ = [
    'CompactRIEnetLayer',
    'variance_loss_function', 
    'buy_and_hold_volatility_loss',
    'frobenius_loss_function',
    '__version__'
]

# Citation reminder
def print_citation():
    """Print citation information for academic use."""
    citation = """
    Please cite the following papers when using Compact-RIEnet:
    
    [Paper references to be provided by the author]
    
    For software citation:
    
    @software{compact_rienet2025,
        title={Compact-RIEnet: A Compact Recurrent-Invariant Eigenvalue Network for Portfolio Optimization},
        author={Author Name},
        year={2025},
        version={1.0.0},
        url={https://github.com/author/compact-rienet}
    }
    """
    print(citation)

# Display citation on import
print("Compact-RIEnet v{} loaded.".format(__version__))
print("Please cite the appropriate papers when using this software.")
print("Use compact_rienet.print_citation() for citation information.")