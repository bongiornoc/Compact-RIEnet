"""
Compact-RIEnet: A Compact Rotational Invariant Estimator Network for Global Minimum-Variance Optimisation

Compact-RIEnet implements the compact RIE-based architecture introduced in Bongiorno et al. (2025) for global minimum-variance (GMV) portfolio construction. It processes financial return tensors and outputs optimised GMV portfolio weights using Rotational Invariant Estimator (RIE) techniques for covariance cleaning combined with recurrent neural networks.

Key Features:
- GMV-focused objective with a differentiable variance loss
- RIE-denoised covariance spectrum for dimension-agnostic deployment
- Configurable recurrent cleaning block (GRU/LSTM) with paper-aligned defaults
- Professional implementation with documentation, tests, and type hints

Main Components:
- `CompactRIEnetLayer`: GMV layer returning weights, precision, and/or covariance
- `variance_loss_function`: Training objective for variance minimisation
- `custom_layers`: Internal building blocks of the architecture

References:
-----------
- Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025). Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage. ICAIF '25.
- Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025). End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning. arXiv:2507.01918.

Contact Prof. Christian Bongiorno (<christian.bongiorno@centralesupelec.fr>) for calibrated weights or collaboration requests.

Copyright (c) 2025
Project URL: https://github.com/author/compact-rienet
"""

from .layers import CompactRIEnetLayer
from .losses import variance_loss_function
from . import custom_layers, losses
from .version import __version__

# Author information
__author__ = "Christian Bongiorno"
__email__ = "christian.bongiorno@centralesupelec.fr"

# Public API
__all__ = [
    'CompactRIEnetLayer',
    'variance_loss_function',
    'print_citation',
    'custom_layers',
    'losses',
    '__version__'
]

# Citation reminder
def print_citation():
    """Print citation information for academic use."""
    citation = """
    Please cite the following references when using Compact-RIEnet:

    Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025).
    Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage.
    Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF '25).

    Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025).
    End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning.
    arXiv preprint arXiv:2507.01918.

    For software citation:

    @software{compact_rienet2025,
        title={Compact-RIEnet: A Compact Rotational Invariant Estimator Network for Global Minimum-Variance Optimisation},
        author={Christian Bongiorno},
        year={2025},
        version={VERSION},
        url={https://github.com/bongiornoc/Compact-RIEnet}
    }
    """
    print(citation.replace("VERSION", __version__))
