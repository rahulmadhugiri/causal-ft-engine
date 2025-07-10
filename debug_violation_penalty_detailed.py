#!/usr/bin/env python3
"""
Detailed debug script with print statements in backward pass to understand violation penalty computation.
"""

import torch
import numpy as np
from causal_unit_network import CausalUnitNetwork
from experiments.utils import generate_synthetic_data
from engine.causal_unit import CausalInterventionFunction

# Monkey patch the backward method to add debug prints
original_backward = CausalInterventionFunction.backward

@staticmethod
def debug_backward(ctx, grad_output):
    """Debug version of backward pass with print statements."""
    
    print(f"\n=== BACKWARD PASS DEBUG ===")
    
    input_tensor, parent_values, adj_mask, do_mask, do_values, weights, bias = ctx.saved_tensors
    
    print(f"Tensors shapes:")
    print(f"  input_tensor: {input_tensor.shape}")
    print(f"  parent_values: {parent_values.shape if parent_values is not None else None}")
    print(f"  adj_mask: {adj_mask.shape if adj_mask is not None else None}")
    print(f"  do_mask: {do_mask.shape if do_mask is not None else None}")
    print(f"  do_values: {do_values.shape if do_values is not None else None}")
    print(f"  weights: {weights.shape if weights is not None else None}")
    print(f"  grad_output: {grad_output.shape}")
    
    if adj_mask is not None:
        print(f"adj_mask:\\n{adj_mask}")
    
    if do_mask is not None:
        print(f"do_mask:\\n{do_mask}")
        print(f"do_values:\\n{do_values}")
    
    # Call original backward
    result = original_backward(ctx, grad_output)
    
    print(f"Violation penalty after backward: {ctx.causal_violation_penalty}")
    print(f"Class-level violation penalty: {CausalInterventionFunction.last_violation_penalty}")
    print(f"=== END BACKWARD PASS DEBUG ===\n")
    
    return result

# Apply the monkey patch
CausalInterventionFunction.backward = debug_backward

def debug_violation_penalty_detailed():
    """Debug the violation penalty computation with detailed output."""
    
    print("=== DETAILED DEBUGGING CAUSAL VIOLATION PENALTY ===")
    
    # Generate small synthetic data
    x_np, y_np, true_adjacency_np = generate_synthetic_data(10, n_nodes=4, graph_type='chain', noise_level=0.3)
    
    # Convert to PyTorch tensors
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    true_adjacency = torch.tensor(true_adjacency_np, dtype=torch.float32)
    
    print(f"Data shapes: x={x.shape}, y={y.shape}, adjacency={true_adjacency.shape}")
    print(f"True adjacency:\\n{true_adjacency}")
    
    # Create network with high lambda_reg
    network = CausalUnitNetwork(
        input_dim=x.shape[1],
        hidden_dims=[4],
        output_dim=y.shape[1],
        activation='relu',
        lambda_reg=1.0,
        enable_structure_learning=True,
        enable_gradient_surgery=True
    )
    
    # Create simple training loop to debug
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    
    print("\\n=== SIMPLE FORWARD/BACKWARD TEST ===")
    
    # Create a very small batch
    batch_x = x[:3]
    batch_y = y[:3]
    
    print(f"Batch shapes: x={batch_x.shape}, y={batch_y.shape}")
    
    # Create simple interventions
    interventions = []
    for i in range(batch_x.shape[0]):
        if i == 0:  # Apply intervention to first sample only
            mask = torch.zeros(batch_x.shape[1])
            values = torch.zeros(batch_x.shape[1])
            mask[2] = 1.0  # Intervene on THIRD feature (node 2, which has parents)
            values[2] = 5.0  # Set to value 5.0
            interventions.append({'test': (mask, values)})
        else:
            interventions.append({})
    
    print(f"Interventions: {interventions}")
    
    # Reset violation penalty
    CausalInterventionFunction.last_violation_penalty = 0.0
    
    # Forward pass
    network.train()
    output = network(batch_x, interventions=interventions)
    
    print(f"Output: {output}")
    print(f"Violation penalty after forward: {CausalInterventionFunction.last_violation_penalty}")
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(output, batch_y)
    print(f"Loss: {loss.item()}")
    
    # Backward pass
    print("\\n=== CALLING BACKWARD ===")
    optimizer.zero_grad()
    loss.backward()
    
    print(f"\\nFinal violation penalty: {CausalInterventionFunction.last_violation_penalty}")
    
    print("\\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_violation_penalty_detailed() 