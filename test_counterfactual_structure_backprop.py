#!/usr/bin/env python3
"""
Test script for counterfactual structure backpropagation.

This tests the critical feedback loop:
1. Predict counterfactual effects based on current structure
2. Compare to actual observed effects
3. Update structure based on prediction errors
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_unit_network import CausalUnitNetwork
from engine.loss_functions import CausalLosses
from train_causalunit import CausalUnitTrainer
from experiments.utils import generate_synthetic_data

def test_counterfactual_structure_backprop():
    """Test counterfactual structure backpropagation functionality."""
    
    print("🧪 Testing Counterfactual Structure Backpropagation")
    print("=" * 60)
    
    # Setup
    device = 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create simple network (input_dim should match X_train.shape[1])
    network = CausalUnitNetwork(
        input_dim=3,  # X_train has 3 features
        hidden_dims=[8],
        output_dim=1,
        enable_structure_learning=True,
        enable_gradient_surgery=True,
        lambda_reg=0.1
    )
    
    # Generate synthetic data
    X_train, y_train, true_adjacency = generate_synthetic_data(
        n_samples=100,
        n_nodes=4,
        graph_type='chain',
        noise_level=0.1
    )
    
    # Create trainer with counterfactual structure loss
    trainer = CausalUnitTrainer(
        network=network,
        intervention_prob=0.3,
        counterfactual_weight=1.0,
        structure_weight=0.1,
        learning_rate=0.01,
        device=device
    )
    
    # Test 1: Verify counterfactual structure loss computation
    print("\n1. Testing counterfactual structure loss computation...")
    
    # Create test batch
    x_batch = torch.tensor(X_train[:8], dtype=torch.float32)
    y_batch = torch.tensor(y_train[:8], dtype=torch.float32)
    
    # Test counterfactual loss computation
    cf_loss, cf_info = trainer.compute_counterfactual_loss(x_batch, y_batch, true_adjacency)
    
    print(f"   ✓ Counterfactual loss computed: {cf_loss.item():.6f}")
    print(f"   ✓ Traditional CF loss: {cf_info['traditional_cf_loss']:.6f}")
    print(f"   ✓ Structure loss: {cf_info['structure_loss']:.6f}")
    print(f"   ✓ Consistency loss: {cf_info['consistency_loss']:.6f}")
    print(f"   ✓ Number of counterfactuals: {cf_info['num_counterfactuals']}")
    
    # Test 2: Verify gradient flow to adjacency matrix
    print("\n2. Testing gradient flow to adjacency matrix...")
    
    # Get adjacency matrix and compute loss
    adjacency = network.get_adjacency_matrix(hard=False)
    print(f"   Initial adjacency matrix:\n{adjacency}")
    
    # Enable gradient tracking
    adjacency.requires_grad_(True)
    
    # Forward pass and compute counterfactual structure loss
    factual_output = network(x_batch, interventions=None)
    
    # Create intervention
    intervention_mask = torch.zeros(8, 3)
    intervention_vals = torch.zeros(8, 3)
    intervention_mask[0, 1] = 1.0  # Intervene on node 1
    intervention_vals[0, 1] = 2.0  # Set to value 2.0
    
    interventions = [{'test': (intervention_mask[i], intervention_vals[i])} for i in range(8)]
    counterfactual_output = network(x_batch, interventions=interventions)
    
    # Compute structure loss
    causal_losses = CausalLosses(counterfactual_structure_weight=0.5)
    structure_loss = causal_losses.counterfactual_structure_loss(
        factual_output, counterfactual_output, 
        intervention_mask, intervention_vals, adjacency
    )
    
    print(f"   ✓ Structure loss: {structure_loss.item():.6f}")
    
    # Backward pass
    structure_loss.backward()
    
    if adjacency.grad is not None:
        print(f"   ✓ Gradient flow detected to adjacency matrix")
        print(f"   ✓ Gradient norm: {torch.norm(adjacency.grad).item():.6f}")
    else:
        print(f"   ❌ No gradient flow to adjacency matrix")
    
    # Test 3: Verify structure updates through training
    print("\n3. Testing structure updates through training...")
    
    # Create data loader
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Get initial adjacency
    initial_adjacency = network.get_adjacency_matrix(hard=False).detach().clone()
    print(f"   Initial adjacency matrix:\n{initial_adjacency}")
    
    # Train for a few epochs
    for epoch in range(3):
        epoch_metrics = trainer.train_epoch(train_loader, true_adjacency, epoch)
        print(f"   Epoch {epoch+1}: CF Structure Loss = {epoch_metrics['cf_structure_loss']:.6f}")
    
    # Get final adjacency
    final_adjacency = network.get_adjacency_matrix(hard=False).detach().clone()
    print(f"   Final adjacency matrix:\n{final_adjacency}")
    
    # Check if adjacency changed
    adjacency_change = torch.norm(final_adjacency - initial_adjacency).item()
    print(f"   ✓ Adjacency matrix change: {adjacency_change:.6f}")
    
    if adjacency_change > 1e-6:
        print("   ✅ Structure learning through counterfactual feedback: WORKING")
    else:
        print("   ❌ Structure learning through counterfactual feedback: NOT WORKING")
    
    # Test 4: Verify counterfactual consistency
    print("\n4. Testing counterfactual consistency...")
    
    # Create two similar interventions
    batch_size = 4
    x_test = torch.tensor(X_train[:batch_size], dtype=torch.float32)
    
    # Intervention 1: Set node 1 to value 1.0
    mask1 = torch.zeros(batch_size, 3)
    vals1 = torch.zeros(batch_size, 3)
    mask1[0, 1] = 1.0
    vals1[0, 1] = 1.0
    
    # Intervention 2: Set node 1 to value 2.0 (double the first)
    mask2 = torch.zeros(batch_size, 3)
    vals2 = torch.zeros(batch_size, 3)
    mask2[0, 1] = 1.0
    vals2[0, 1] = 2.0
    
    # Get outputs
    factual = network(x_test, interventions=None)
    cf1 = network(x_test, interventions=[{'test': (mask1[i], vals1[i])} for i in range(batch_size)])
    cf2 = network(x_test, interventions=[{'test': (mask2[i], vals2[i])} for i in range(batch_size)])
    
    # Compute consistency loss
    consistency_loss = causal_losses.counterfactual_consistency_loss(
        factual, [cf1, cf2], [mask1, mask2], [vals1, vals2]
    )
    
    print(f"   ✓ Consistency loss: {consistency_loss.item():.6f}")
    
    # Test 5: Verify directional effects
    print("\n5. Testing directional effects...")
    
    # Create intervention with known direction
    effect1 = cf1[0] - factual[0]
    effect2 = cf2[0] - factual[0]
    
    print(f"   Effect of intervention 1.0: {effect1.item():.6f}")
    print(f"   Effect of intervention 2.0: {effect2.item():.6f}")
    print(f"   Effect ratio (should be ~2): {(effect2.item() / effect1.item()) if effect1.item() != 0 else 'undefined'}")
    
    # Directional loss
    directional_loss = causal_losses.directional_effect_loss(
        factual, cf1, mask1, vals1
    )
    
    print(f"   ✓ Directional loss: {directional_loss.item():.6f}")
    
    print("\n" + "=" * 60)
    print("✅ Counterfactual Structure Backpropagation Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_counterfactual_structure_backprop() 