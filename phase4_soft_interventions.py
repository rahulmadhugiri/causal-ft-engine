#!/usr/bin/env python3
"""
Phase 4: Soft Interventions Implementation

Goal: Replace hard do(X=v) interventions with learnable blended interventions:
(1 - alpha) * x_orig + alpha * do_value

This should provide smoother optimization and better performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from causal_unit_network import CausalUnitNetwork
from experiments.utils import generate_synthetic_data

class SoftInterventionCausalUnit(nn.Module):
    """
    Enhanced CausalUnit with soft interventions using learnable alpha parameters.
    
    Key Innovation: Instead of hard do(X=v), we use:
    x_intervened = (1 - alpha) * x_original + alpha * do_value
    
    Where alpha is learnable and can be different for each node.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, 
                 activation: str = 'relu', node_id: Optional[str] = None,
                 enable_structure_learning: bool = True, enable_gradient_surgery: bool = True):
        super(SoftInterventionCausalUnit, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.node_id = node_id or f"node_{id(self)}"
        self.enable_structure_learning = enable_structure_learning
        self.enable_gradient_surgery = enable_gradient_surgery
        
        # Core neural computation layers
        if hidden_dim is None:
            self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(output_dim))
            self.hidden_weights = None
            self.hidden_bias = None
        else:
            self.weights = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.hidden_weights = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
            self.hidden_bias = nn.Parameter(torch.zeros(output_dim))
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # SOFT INTERVENTION PARAMETERS
        # Alpha parameters for learnable intervention strength (one per input dimension)
        self.alpha = nn.Parameter(torch.full((input_dim,), 0.5))  # Initialize to 0.5 (balanced)
        
        # Learnable adjacency matrix for structure learning
        if enable_structure_learning:
            self.adj_logits = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)
            self.adj_temperature = nn.Parameter(torch.ones(1) * 1.0)
        else:
            self.adj_logits = None
            self.adj_temperature = None
        
        # Intervention tracking
        self.last_interventions = {}
        self.intervention_history = []
        
    def get_intervention_strength(self) -> torch.Tensor:
        """Get current intervention strength (alpha values) with sigmoid activation."""
        return torch.sigmoid(self.alpha)  # Bound to [0, 1]
    
    def get_adjacency_matrix(self, hard: bool = False, temperature: Optional[float] = None) -> torch.Tensor:
        """Get the current adjacency matrix."""
        if self.adj_logits is None:
            return torch.eye(self.input_dim, device=self.weights.device)
        
        if hard:
            return torch.sigmoid(self.adj_logits) > 0.5
        else:
            temp = temperature if temperature is not None else self.adj_temperature
            return torch.sigmoid(self.adj_logits / temp)
    
    def apply_soft_intervention(self, 
                              input_tensor: torch.Tensor,
                              do_mask: torch.Tensor,
                              do_values: torch.Tensor) -> torch.Tensor:
        """
        Apply soft intervention with learnable alpha parameters.
        
        Args:
            input_tensor: Original input values (batch_size, input_dim)
            do_mask: Intervention mask (batch_size, input_dim)
            do_values: Intervention values (batch_size, input_dim)
            
        Returns:
            Soft intervened tensor
        """
        batch_size = input_tensor.shape[0]
        
        # Get intervention strength (alpha) for each dimension
        alpha = self.get_intervention_strength()  # (input_dim,)
        alpha = alpha.unsqueeze(0).expand(batch_size, -1)  # (batch_size, input_dim)
        
        # Apply soft intervention only where do_mask is True
        # x_soft = (1 - alpha) * x_original + alpha * do_value
        soft_intervention = (1 - alpha) * input_tensor + alpha * do_values
        
        # Use soft intervention where mask is True, original values otherwise
        result = torch.where(do_mask.bool(), soft_intervention, input_tensor)
        
        return result
    
    def forward(self, input_tensor: torch.Tensor, parent_values: Optional[torch.Tensor] = None,
                adj_mask: Optional[torch.Tensor] = None, do_mask: Optional[torch.Tensor] = None,
                do_values: Optional[torch.Tensor] = None, 
                interventions: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Forward pass with soft causal interventions.
        """
        batch_size = input_tensor.shape[0]
        
        # Handle multiple interventions
        if interventions is not None:
            combined_mask = torch.zeros(batch_size, self.input_dim, device=input_tensor.device)
            combined_values = torch.zeros(batch_size, self.input_dim, device=input_tensor.device)
            
            for name, (mask, values) in interventions.items():
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0).expand(batch_size, -1)
                if values.dim() == 1:
                    values = values.unsqueeze(0).expand(batch_size, -1)
                    
                combined_mask = torch.logical_or(combined_mask.bool(), mask.bool()).float()
                combined_values = torch.where(mask.bool(), values, combined_values)
            
            do_mask = combined_mask
            do_values = combined_values
        
        # Apply soft interventions
        if do_mask is not None and do_values is not None:
            if do_mask.dim() == 1:
                do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
            if do_values.dim() == 1:
                do_values = do_values.unsqueeze(0).expand(batch_size, -1)
                
            # Apply soft intervention
            intervened_input = self.apply_soft_intervention(input_tensor, do_mask, do_values)
        else:
            intervened_input = input_tensor
        
        # Forward computation
        if self.hidden_weights is None:
            # Single layer
            output = torch.matmul(intervened_input, self.weights) + self.bias
        else:
            # Multi-layer
            hidden = torch.matmul(intervened_input, self.weights) + self.bias
            hidden = self.activation(hidden)
            output = torch.matmul(hidden, self.hidden_weights) + self.hidden_bias
        
        # Track interventions
        if do_mask is not None and do_values is not None:
            self.last_interventions = {
                'mask': do_mask.detach().cpu(),
                'values': do_values.detach().cpu(),
                'alpha': self.get_intervention_strength().detach().cpu(),
                'timestamp': len(self.intervention_history)
            }
            self.intervention_history.append(self.last_interventions)
        
        return output
    
    def get_intervention_info(self) -> Dict:
        """Get detailed information about interventions and alpha parameters."""
        return {
            'last_interventions': self.last_interventions,
            'intervention_count': len(self.intervention_history),
            'alpha_values': self.get_intervention_strength().detach().cpu(),
            'alpha_raw': self.alpha.detach().cpu(),
            'node_id': self.node_id
        }
    
    def reset_intervention_history(self):
        """Reset intervention tracking."""
        self.intervention_history = []
        self.last_interventions = {}


class SoftInterventionNetwork(nn.Module):
    """
    Network using soft intervention causal units.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 activation: str = 'relu', lambda_reg: float = 0.01):
        super(SoftInterventionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.lambda_reg = lambda_reg
        
        # Create soft intervention units
        self.units = nn.ModuleList()
        
        # First unit: input -> hidden[0]
        self.units.append(SoftInterventionCausalUnit(
            input_dim=input_dim,
            output_dim=hidden_dims[0],
            activation=activation,
            node_id=f"soft_unit_0"
        ))
        
        # Hidden units
        for i in range(1, len(hidden_dims)):
            self.units.append(SoftInterventionCausalUnit(
                input_dim=hidden_dims[i-1],
                output_dim=hidden_dims[i],
                activation=activation,
                node_id=f"soft_unit_{i}"
            ))
        
        # Output unit
        self.units.append(SoftInterventionCausalUnit(
            input_dim=hidden_dims[-1],
            output_dim=output_dim,
            activation=activation,
            node_id=f"soft_unit_output"
        ))
        
        # Intervention schedule
        self.intervention_schedule = []
    
    def forward(self, x: torch.Tensor, interventions: Optional[List[Dict]] = None) -> torch.Tensor:
        """Forward pass through soft intervention network."""
        
        current_activation = x
        
        # Apply interventions only to the first unit (input layer)
        for i, unit in enumerate(self.units):
            if i == 0 and interventions is not None:
                # Apply interventions to first unit
                combined_mask, combined_values = self._combine_interventions(interventions, x.shape[0])
                current_activation = unit(
                    input_tensor=current_activation,
                    do_mask=combined_mask,
                    do_values=combined_values
                )
            else:
                # No interventions for hidden/output units
                current_activation = unit(input_tensor=current_activation)
        
        return current_activation
    
    def _combine_interventions(self, interventions: List[Dict], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine multiple interventions into single mask and values."""
        combined_mask = torch.zeros(batch_size, self.input_dim)
        combined_values = torch.zeros(batch_size, self.input_dim)
        
        for i, intervention_dict in enumerate(interventions):
            for name, (mask, values) in intervention_dict.items():
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                if values.dim() == 1:
                    values = values.unsqueeze(0)
                
                # Expand to batch size
                if mask.shape[0] == 1:
                    mask = mask.expand(batch_size, -1)
                if values.shape[0] == 1:
                    values = values.expand(batch_size, -1)
                
                # Update only the i-th sample
                combined_mask[i] = torch.logical_or(combined_mask[i].bool(), mask[i].bool()).float()
                combined_values[i] = torch.where(mask[i].bool(), values[i], combined_values[i])
        
        return combined_mask, combined_values
    
    def get_alpha_summary(self) -> Dict:
        """Get summary of alpha parameters across all units."""
        alpha_info = {}
        for i, unit in enumerate(self.units):
            alpha_info[f'unit_{i}'] = {
                'alpha_values': unit.get_intervention_strength().detach().cpu(),
                'alpha_raw': unit.alpha.detach().cpu(),
                'mean_alpha': unit.get_intervention_strength().mean().item(),
                'std_alpha': unit.get_intervention_strength().std().item()
            }
        return alpha_info


def test_soft_interventions():
    """Test the soft interventions implementation."""
    
    print("=== TESTING SOFT INTERVENTIONS ===")
    
    # Generate test data
    x_np, y_np, true_adjacency_np = generate_synthetic_data(100, n_nodes=4, graph_type='chain', noise_level=0.3)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    
    print(f"Data shapes: x={x.shape}, y={y.shape}")
    
    # Create soft intervention network
    network = SoftInterventionNetwork(
        input_dim=x.shape[1],
        hidden_dims=[8, 4],
        output_dim=y.shape[1],
        activation='relu',
        lambda_reg=0.1
    )
    
    print(f"Network created with {len(network.units)} units")
    
    # Test forward pass without interventions
    output_no_int = network(x)
    print(f"Output shape (no interventions): {output_no_int.shape}")
    
    # Test forward pass with interventions
    interventions = []
    for i in range(10):  # First 10 samples
        if i % 2 == 0:
            mask = torch.zeros(x.shape[1])
            values = torch.zeros(x.shape[1])
            mask[0] = 1.0
            values[0] = 2.0
            interventions.append({'test': (mask, values)})
        else:
            interventions.append({})
    
    # Pad interventions to match batch size
    while len(interventions) < x.shape[0]:
        interventions.append({})
    
    output_with_int = network(x, interventions=interventions)
    print(f"Output shape (with interventions): {output_with_int.shape}")
    
    # Check alpha parameter values
    alpha_summary = network.get_alpha_summary()
    print(f"\nAlpha parameters summary:")
    for unit_name, info in alpha_summary.items():
        print(f"  {unit_name}: mean={info['mean_alpha']:.3f}, std={info['std_alpha']:.3f}")
    
    # Test training step
    print(f"\nTesting training step...")
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    # Forward pass
    output = network(x[:32], interventions=interventions[:32])  # Small batch
    loss = loss_fn(output, y[:32])
    
    print(f"Loss before training: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients on alpha parameters
    for i, unit in enumerate(network.units):
        if unit.alpha.grad is not None:
            alpha_grad_norm = torch.norm(unit.alpha.grad).item()
            print(f"  Unit {i} alpha gradient norm: {alpha_grad_norm:.4f}")
    
    # Training step
    optimizer.step()
    
    # Forward pass after training
    output_after = network(x[:32], interventions=interventions[:32])
    loss_after = loss_fn(output_after, y[:32])
    
    print(f"Loss after training: {loss_after.item():.4f}")
    
    # Check if alpha values changed
    alpha_summary_after = network.get_alpha_summary()
    print(f"\nAlpha parameters after training:")
    for unit_name, info in alpha_summary_after.items():
        print(f"  {unit_name}: mean={info['mean_alpha']:.3f}, std={info['std_alpha']:.3f}")
    
    print(f"\n=== SOFT INTERVENTIONS TEST COMPLETE ===")


if __name__ == "__main__":
    test_soft_interventions() 