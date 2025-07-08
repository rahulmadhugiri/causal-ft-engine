"""
Causal Fine-Tuning Engine

A neural network framework for incorporating do-operator-based causal reasoning
directly into the forward and backward passes of neural networks.

Key Components:
- CausalUnit: Neural layers with do-operator support
- CausalMLP: Multi-layer causal networks  
- DAGRewiring: Causal graph manipulation
- CausalLosses: Intervention-aware loss functions
"""

from .causal_unit import CausalUnit, CausalMLP
from .rewiring import DAGRewiring, CausalLayerRewiring
from .loss_functions import CausalLosses, CausalMetrics

__version__ = "0.1.0"
__author__ = "Causal Fine-Tuning Team"

__all__ = [
    "CausalUnit",
    "CausalMLP", 
    "DAGRewiring",
    "CausalLayerRewiring",
    "CausalLosses",
    "CausalMetrics"
]
