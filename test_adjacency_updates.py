#!/usr/bin/env python3
"""
Test script to verify adjacency matrix updates during training.

This tests the critical gradient flow:
1. Counterfactual structure loss computes gradients w.r.t. adjacency matrix
2. Optimizer updates adjacency matrix parameters
3. Adjacency matrix values actually change during training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_unit_network import CausalUnitNetwork
from engine.loss_functions import CausalLosses
from experiments.utils import generate_synthetic_data

def test_adjacency_matrix_updates():
    """Test that adjacency matrix parameters are updated during training."""
    print("=" * 60)
    print("TESTING ADJACENCY MATRIX PARAMETER UPDATES")
    print("=" * 60)
    
    # Generate synthetic data
    X_train, y_train, true_adjacency = generate_synthetic_data(
        n_samples=200,
        n_nodes=3,
        graph_type='chain',
        noise_level=0.1
    )
    
    print(f"True adjacency matrix:")
    print(true_adjacency)
    print(f"Data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Create network
    network = CausalUnitNetwork(
        input_dim=3,
        hidden_dims=[8],
        output_dim=1,
        enable_structure_learning=True,
        enable_gradient_surgery=True,
        lambda_reg=0.01
    )
    
    # Print initial adjacency matrix
    initial_adjacency = network.get_adjacency_matrix(hard=False).detach().clone()
    print(f"\nInitial adjacency matrix:")
    print(initial_adjacency)
    
    # Create optimizer that includes adjacency matrix parameters
    optimizer = optim.Adam(network.parameters(), lr=0.1)  # Higher LR to see changes
    
    # Create loss function
    causal_losses = CausalLosses(
        counterfactual_structure_weight=1.0,  # High weight to force structure updates
        structure_weight=0.1
    )
    
    # Test data
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    print(f"\nTesting adjacency matrix parameter updates...")
    
    # Track adjacency matrix changes
    adjacency_history = []
    loss_history = []
    
    for step in range(50):  # More steps to see changes
        optimizer.zero_grad()
        
        # Get current adjacency matrix
        current_adjacency = network.get_adjacency_matrix(hard=False)
        adjacency_history.append(current_adjacency.detach().clone())
        
        # Factual forward pass
        factual_output = network(X_tensor)
        
        # Create counterfactual with intervention
        batch_size = X_tensor.shape[0]
        intervention_mask = torch.zeros(batch_size, 3)
        intervention_vals = torch.zeros(batch_size, 3)
        
        # Intervene on random samples and nodes
        for i in range(min(5, batch_size)):
            intervention_mask[i, 1] = 1.0  # Intervene on node 1
            intervention_vals[i, 1] = torch.randn(1).item() * 2.0
        
        # Counterfactual forward pass
        interventions = []
        for i in range(batch_size):
            if intervention_mask[i].sum() > 0:
                interventions.append({'counterfactual': (intervention_mask[i], intervention_vals[i])})
            else:
                interventions.append({})
        
        counterfactual_output = network(X_tensor, interventions=interventions)
        
        # Compute counterfactual structure loss
        structure_loss = causal_losses.counterfactual_structure_loss(
            factual_output, counterfactual_output, intervention_mask, intervention_vals, current_adjacency
        )
        
        # Add prediction loss for stability
        prediction_loss = nn.MSELoss()(factual_output, y_tensor)
        
        # Total loss
        total_loss = prediction_loss + structure_loss
        loss_history.append(total_loss.item())
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients on adjacency matrix
        if hasattr(network, 'learned_adjacency') and network.learned_adjacency is not None:
            if network.learned_adjacency.grad is not None:
                adj_grad_norm = network.learned_adjacency.grad.norm().item()
                print(f"Step {step}: Loss={total_loss.item():.6f}, Adj Grad Norm={adj_grad_norm:.6f}")
            else:
                print(f"Step {step}: Loss={total_loss.item():.6f}, Adj Grad=None")
        
        # Update parameters
        optimizer.step()
        
        # Print every 10 steps
        if step % 10 == 0 or step < 5:
            print(f"Step {step}: Current adjacency matrix:")
            print(current_adjacency.detach())
            print()
    
    # Final adjacency matrix
    final_adjacency = network.get_adjacency_matrix(hard=False).detach().clone()
    print(f"Final adjacency matrix:")
    print(final_adjacency)
    
    # Calculate change
    adjacency_change = torch.abs(final_adjacency - initial_adjacency).max().item()
    print(f"\nMaximum adjacency matrix change: {adjacency_change:.6f}")
    
    # Test results
    print(f"\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    
    if adjacency_change > 1e-4:
        print("✅ PASS: Adjacency matrix parameters are being updated!")
        print(f"   Maximum change: {adjacency_change:.6f}")
    else:
        print("❌ FAIL: Adjacency matrix parameters are NOT being updated!")
        print(f"   Maximum change: {adjacency_change:.6f}")
    
    # Check if gradients are flowing
    if hasattr(network, 'learned_adjacency') and network.learned_adjacency is not None:
        if network.learned_adjacency.grad is not None:
            print("✅ PASS: Gradients are flowing to adjacency matrix!")
        else:
            print("❌ FAIL: No gradients flowing to adjacency matrix!")
    else:
        print("❌ FAIL: No learnable adjacency matrix found!")
    
    return adjacency_change > 1e-4

if __name__ == "__main__":
    success = test_adjacency_matrix_updates()
    sys.exit(0 if success else 1) 