"""
Legacy Compact-RIEnet package for global minimum-variance portfolio optimisation.

This repository is deprecated and no longer maintained. Active development has
moved to RIEnet:

- Install the maintained package with ``pip install rienet``
- Repository: https://github.com/bongiornoc/RIEnet

Compact-RIEnet implements the compact RIE-based architecture introduced in
Bongiorno et al. (2025) for global minimum-variance (GMV) portfolio
construction. It processes financial return tensors and outputs optimised GMV
portfolio weights using Rotational Invariant Estimator (RIE) techniques for
covariance cleaning combined with recurrent neural networks.
"""

import warnings

_ACTIVE_REPOSITORY_URL = "https://github.com/bongiornoc/RIEnet"
_REPLACEMENT_INSTALL_COMMAND = "pip install rienet"
_DEPRECATION_WARNING = (
    "compact_rienet is deprecated and no longer maintained. "
    f"Install the actively maintained replacement with `{_REPLACEMENT_INSTALL_COMMAND}` "
    f"and see {_ACTIVE_REPOSITORY_URL}."
)

warnings.warn(_DEPRECATION_WARNING, FutureWarning)

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
    """Print citation information for the legacy Compact-RIEnet package."""
    citation = """
    Compact-RIEnet is deprecated and no longer maintained.
    Install the actively maintained replacement with `pip install rienet`.
    Active repository: https://github.com/bongiornoc/RIEnet

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
