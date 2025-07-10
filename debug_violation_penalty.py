#!/usr/bin/env python3
"""
Debug script to understand why the causal violation penalty isn't being computed.
"""

import torch
import numpy as np
from causal_unit_network import CausalUnitNetwork
from train_causalunit import CausalUnitTrainer
from experiments.utils import generate_synthetic_data
from engine.causal_unit import CausalInterventionFunction

def debug_violation_penalty():
    """Debug the violation penalty computation."""
    
    print("=== DEBUGGING CAUSAL VIOLATION PENALTY ===")
    
    # Generate small synthetic data
    x_np, y_np, true_adjacency_np = generate_synthetic_data(100, n_nodes=4, graph_type='chain', noise_level=0.3)
    
    # Convert to PyTorch tensors
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    true_adjacency = torch.tensor(true_adjacency_np, dtype=torch.float32)
    
    print(f"Data shapes: x={x.shape}, y={y.shape}, adjacency={true_adjacency.shape}")
    print(f"True adjacency:\n{true_adjacency}")
    
    # Create network with high lambda_reg
    network = CausalUnitNetwork(
        input_dim=x.shape[1],
        hidden_dims=[8],
        output_dim=y.shape[1],
        activation='relu',
        lambda_reg=1.0,  # High regularization
        enable_structure_learning=True,
        enable_gradient_surgery=True
    )
    
    print(f"Network created with lambda_reg=1.0")
    print(f"Network structure: {network.get_network_info()['network_structure']}")
    
    # Create simple training loop to debug
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    
    print("\n=== MANUAL TRAINING STEP ===")
    
    # Forward pass
    network.train()
    
    # Create a small batch
    batch_x = x[:10]  # Small batch
    batch_y = y[:10]
    
    print(f"Batch shapes: x={batch_x.shape}, y={batch_y.shape}")
    
    # Forward pass with manual interventions
    print("\nForward pass with manual interventions...")
    
    # Create manual interventions
    interventions = []
    for i in range(batch_x.shape[0]):
        if i % 2 == 0:  # Apply intervention to every other sample
            mask = torch.zeros(batch_x.shape[1])
            values = torch.zeros(batch_x.shape[1])
            mask[0] = 1.0  # Intervene on first node
            values[0] = 2.0  # Set to value 2.0
            interventions.append({'manual': (mask, values)})
        else:
            interventions.append({})
    
    print(f"Created {len(interventions)} interventions")
    print(f"Intervention example: {interventions[0]}")
    
    # Reset violation penalty
    CausalInterventionFunction.last_violation_penalty = 0.0
    
    # Forward pass
    output = network(batch_x, interventions=interventions)
    
    print(f"Output shape: {output.shape}")
    print(f"Violation penalty after forward: {CausalInterventionFunction.last_violation_penalty}")
    
    # Compute loss
    loss = torch.nn.functional.mse_loss(output, batch_y)
    print(f"Loss: {loss.item()}")
    
    # Backward pass
    print("\nBackward pass...")
    optimizer.zero_grad()
    loss.backward()
    
    print(f"Violation penalty after backward: {CausalInterventionFunction.last_violation_penalty}")
    
    # Check network's get_causal_violation_penalty method
    network_penalty = network.get_causal_violation_penalty()
    print(f"Network penalty: {network_penalty}")
    
    # Check individual unit penalties
    print("\nUnit penalties:")
    for i, unit in enumerate(network.units):
        unit_penalty = unit.get_causal_violation_penalty()
        print(f"Unit {i}: {unit_penalty}")
    
    # Check gradients
    print("\nGradient information:")
    for i, unit in enumerate(network.units):
        if unit.weights.grad is not None:
            print(f"Unit {i} weights grad norm: {torch.norm(unit.weights.grad).item()}")
        if unit.bias.grad is not None:
            print(f"Unit {i} bias grad norm: {torch.norm(unit.bias.grad).item()}")
    
    # Check adjacency gradients
    learned_adj = network.get_adjacency_matrix()
    print(f"\nLearned adjacency:\n{learned_adj}")
    
    if hasattr(network, 'learned_adjacency') and network.learned_adjacency is not None:
        if network.learned_adjacency.grad is not None:
            print(f"Adjacency gradient norm: {torch.norm(network.learned_adjacency.grad).item()}")
    
    # Test with different intervention scenarios
    print("\n=== TESTING DIFFERENT SCENARIOS ===")
    
    # Test 1: No interventions
    print("\nTest 1: No interventions")
    CausalInterventionFunction.last_violation_penalty = 0.0
    output_no_int = network(batch_x, interventions=None)
    loss_no_int = torch.nn.functional.mse_loss(output_no_int, batch_y)
    optimizer.zero_grad()
    loss_no_int.backward()
    print(f"Violation penalty (no interventions): {CausalInterventionFunction.last_violation_penalty}")
    
    # Test 2: All samples intervened
    print("\nTest 2: All samples intervened")
    CausalInterventionFunction.last_violation_penalty = 0.0
    all_interventions = []
    for i in range(batch_x.shape[0]):
        mask = torch.zeros(batch_x.shape[1])
        values = torch.zeros(batch_x.shape[1])
        mask[1] = 1.0  # Intervene on second node
        values[1] = 1.5  # Set to value 1.5
        all_interventions.append({'all': (mask, values)})
    
    output_all_int = network(batch_x, interventions=all_interventions)
    loss_all_int = torch.nn.functional.mse_loss(output_all_int, batch_y)
    optimizer.zero_grad()
    loss_all_int.backward()
    print(f"Violation penalty (all interventions): {CausalInterventionFunction.last_violation_penalty}")
    
    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_violation_penalty() 