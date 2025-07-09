"""
Causal Fine-Tuning Engine - Phase 3

A neural network framework for incorporating do-operator-based causal reasoning
directly into the forward and backward passes of neural networks with novel
mathematical innovations.

Phase 3 Key Components:
- CausalUnit: Neural layers with custom autograd and gradient blocking
- CausalUnitNetwork: Dynamic network assembly with intervention APIs  
- Custom autograd functions for precise causal interventions
- Runtime graph rewiring and symbolic-continuous hybrid learning
"""

from .causal_unit import CausalUnit
from .loss_functions import CausalLosses, CausalMetrics

__version__ = "0.3.0"
__author__ = "Causal Fine-Tuning Team"

__all__ = [
    "CausalUnit",
    "CausalLosses",
    "CausalMetrics"
]
