#!/usr/bin/env python3
"""
Test script for enhanced counterfactual consistency loss.

This tests all the consistency checks:
1. Proportionality: Similar interventions should have proportional effects
2. Directional consistency: Effects should align with causal graph structure
3. No spurious effects: Interventions on non-parent nodes shouldn't affect outputs
4. Monotonicity: Increasing intervention values should produce monotonic effects
5. Effect magnitude consistency: Large interventions should have larger effects
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.loss_functions import CausalLosses

def test_counterfactual_consistency():
    """Test enhanced counterfactual consistency loss function."""
    print("=" * 60)
    print("TESTING ENHANCED COUNTERFACTUAL CONSISTENCY LOSS")
    print("=" * 60)
    
    # Create causal losses instance
    causal_losses = CausalLosses()
    
    # Test data dimensions
    batch_size = 4
    input_dim = 3
    output_dim = 3
    
    # Create simple chain adjacency matrix: 0->1->2
    adjacency_matrix = torch.tensor([
        [0.0, 1.0, 0.0],  # Node 0 -> Node 1
        [0.0, 0.0, 1.0],  # Node 1 -> Node 2
        [0.0, 0.0, 0.0]   # Node 2 -> nothing
    ], dtype=torch.float32)
    
    print(f"Adjacency matrix (chain 0->1->2):")
    print(adjacency_matrix)
    print()
    
    # Factual output (no intervention)
    factual_output = torch.randn(batch_size, output_dim)
    
    # Test 1: PROPORTIONALITY CHECK
    print("Test 1: PROPORTIONALITY CHECK")
    print("-" * 40)
    
    # Create two interventions on the same node with different values
    mask1 = torch.zeros(batch_size, input_dim)
    mask1[0, 1] = 1.0  # Intervene on node 1
    values1 = torch.zeros(batch_size, input_dim)
    values1[0, 1] = 1.0  # Intervention value = 1.0
    
    mask2 = torch.zeros(batch_size, input_dim)
    mask2[0, 1] = 1.0  # Intervene on node 1
    values2 = torch.zeros(batch_size, input_dim)
    values2[0, 1] = 2.0  # Intervention value = 2.0
    
    # Create counterfactual outputs
    # Good case: Effect is proportional to intervention value
    cf_output1_good = factual_output.clone()
    cf_output1_good[0, 1] += 0.5  # Effect = 0.5 for intervention = 1.0
    
    cf_output2_good = factual_output.clone()
    cf_output2_good[0, 1] += 1.0  # Effect = 1.0 for intervention = 2.0 (proportional)
    
    # Bad case: Effect is not proportional
    cf_output2_bad = factual_output.clone()
    cf_output2_bad[0, 1] += 0.3  # Effect = 0.3 for intervention = 2.0 (not proportional)
    
    # Test good case
    consistency_loss_good = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output1_good, cf_output2_good], 
        [mask1, mask2], 
        [values1, values2],
        adjacency_matrix
    )
    
    # Test bad case
    consistency_loss_bad = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output1_good, cf_output2_bad], 
        [mask1, mask2], 
        [values1, values2],
        adjacency_matrix
    )
    
    print(f"Good proportionality loss: {consistency_loss_good.item():.6f}")
    print(f"Bad proportionality loss: {consistency_loss_bad.item():.6f}")
    print(f"Bad case penalty: {consistency_loss_bad.item() - consistency_loss_good.item():.6f}")
    print()
    
    # Test 2: DIRECTIONAL CONSISTENCY CHECK
    print("Test 2: DIRECTIONAL CONSISTENCY CHECK")
    print("-" * 40)
    
    # Intervention on node 1 (has causal edge to node 2)
    mask_causal = torch.zeros(batch_size, input_dim)
    mask_causal[0, 1] = 1.0
    values_causal = torch.zeros(batch_size, input_dim)
    values_causal[0, 1] = 1.0  # Positive intervention
    
    # Good case: Effect in same direction as intervention
    cf_output_dir_good = factual_output.clone()
    cf_output_dir_good[0, 2] += 0.5  # Positive effect on downstream node 2
    
    # Bad case: Effect in opposite direction
    cf_output_dir_bad = factual_output.clone()
    cf_output_dir_bad[0, 2] -= 0.5  # Negative effect (wrong direction)
    
    consistency_loss_dir_good = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output_dir_good], 
        [mask_causal], 
        [values_causal],
        adjacency_matrix
    )
    
    consistency_loss_dir_bad = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output_dir_bad], 
        [mask_causal], 
        [values_causal],
        adjacency_matrix
    )
    
    print(f"Good directional consistency loss: {consistency_loss_dir_good.item():.6f}")
    print(f"Bad directional consistency loss: {consistency_loss_dir_bad.item():.6f}")
    print(f"Direction penalty: {consistency_loss_dir_bad.item() - consistency_loss_dir_good.item():.6f}")
    print()
    
    # Test 3: NO SPURIOUS EFFECTS CHECK
    print("Test 3: NO SPURIOUS EFFECTS CHECK")
    print("-" * 40)
    
    # Intervention on node 2 (no outgoing causal edges)
    mask_spurious = torch.zeros(batch_size, input_dim)
    mask_spurious[0, 2] = 1.0
    values_spurious = torch.zeros(batch_size, input_dim)
    values_spurious[0, 2] = 1.0
    
    # Good case: No effect (as expected for node with no outgoing edges)
    cf_output_no_spurious = factual_output.clone()  # No change
    
    # Bad case: Spurious effect
    cf_output_spurious = factual_output.clone()
    cf_output_spurious[0, 0] += 0.5  # Spurious effect on node 0
    
    consistency_loss_no_spurious = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output_no_spurious], 
        [mask_spurious], 
        [values_spurious],
        adjacency_matrix
    )
    
    consistency_loss_spurious = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output_spurious], 
        [mask_spurious], 
        [values_spurious],
        adjacency_matrix
    )
    
    print(f"No spurious effects loss: {consistency_loss_no_spurious.item():.6f}")
    print(f"With spurious effects loss: {consistency_loss_spurious.item():.6f}")
    print(f"Spurious effect penalty: {consistency_loss_spurious.item() - consistency_loss_no_spurious.item():.6f}")
    print()
    
    # Test 4: MONOTONICITY CHECK
    print("Test 4: MONOTONICITY CHECK")
    print("-" * 40)
    
    # Three interventions on the same node with increasing values
    mask_mono = torch.zeros(batch_size, input_dim)
    mask_mono[0, 1] = 1.0
    
    values_mono1 = torch.zeros(batch_size, input_dim)
    values_mono1[0, 1] = 1.0
    
    values_mono2 = torch.zeros(batch_size, input_dim)
    values_mono2[0, 1] = 2.0
    
    values_mono3 = torch.zeros(batch_size, input_dim)
    values_mono3[0, 1] = 3.0
    
    # Good case: Monotonic effects
    cf_output_mono1 = factual_output.clone()
    cf_output_mono1[0, 1] += 0.5  # Effect = 0.5 for intervention = 1.0
    
    cf_output_mono2 = factual_output.clone()
    cf_output_mono2[0, 1] += 1.0  # Effect = 1.0 for intervention = 2.0
    
    cf_output_mono3 = factual_output.clone()
    cf_output_mono3[0, 1] += 1.5  # Effect = 1.5 for intervention = 3.0
    
    # Bad case: Non-monotonic effects
    cf_output_mono3_bad = factual_output.clone()
    cf_output_mono3_bad[0, 1] += 0.3  # Effect = 0.3 for intervention = 3.0 (decreasing)
    
    consistency_loss_mono_good = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output_mono1, cf_output_mono2, cf_output_mono3], 
        [mask_mono, mask_mono, mask_mono], 
        [values_mono1, values_mono2, values_mono3],
        adjacency_matrix
    )
    
    consistency_loss_mono_bad = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output_mono1, cf_output_mono2, cf_output_mono3_bad], 
        [mask_mono, mask_mono, mask_mono], 
        [values_mono1, values_mono2, values_mono3],
        adjacency_matrix
    )
    
    print(f"Good monotonicity loss: {consistency_loss_mono_good.item():.6f}")
    print(f"Bad monotonicity loss: {consistency_loss_mono_bad.item():.6f}")
    print(f"Monotonicity penalty: {consistency_loss_mono_bad.item() - consistency_loss_mono_good.item():.6f}")
    print()
    
    # Test 5: EFFECT MAGNITUDE CONSISTENCY
    print("Test 5: EFFECT MAGNITUDE CONSISTENCY")
    print("-" * 40)
    
    # Large intervention should have significant effect
    mask_large = torch.zeros(batch_size, input_dim)
    mask_large[0, 1] = 1.0
    values_large = torch.zeros(batch_size, input_dim)
    values_large[0, 1] = 5.0  # Large intervention
    
    # Good case: Large effect for large intervention
    cf_output_large_good = factual_output.clone()
    cf_output_large_good[0, 1] += 2.0  # Significant effect
    
    # Bad case: Tiny effect for large intervention
    cf_output_large_bad = factual_output.clone()
    cf_output_large_bad[0, 1] += 0.001  # Tiny effect
    
    consistency_loss_large_good = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output_large_good], 
        [mask_large], 
        [values_large],
        adjacency_matrix
    )
    
    consistency_loss_large_bad = causal_losses.counterfactual_consistency_loss(
        factual_output, 
        [cf_output_large_bad], 
        [mask_large], 
        [values_large],
        adjacency_matrix
    )
    
    print(f"Good magnitude consistency loss: {consistency_loss_large_good.item():.6f}")
    print(f"Bad magnitude consistency loss: {consistency_loss_large_bad.item():.6f}")
    print(f"Magnitude penalty: {consistency_loss_large_bad.item() - consistency_loss_large_good.item():.6f}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY OF CONSISTENCY CHECKS")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    if consistency_loss_bad.item() > consistency_loss_good.item():
        print("✅ Proportionality check: WORKING (bad case penalized)")
        tests_passed += 1
    else:
        print("❌ Proportionality check: FAILED")
    
    if consistency_loss_dir_bad.item() > consistency_loss_dir_good.item():
        print("✅ Directional consistency check: WORKING (wrong direction penalized)")
        tests_passed += 1
    else:
        print("❌ Directional consistency check: FAILED")
    
    if consistency_loss_spurious.item() > consistency_loss_no_spurious.item():
        print("✅ Spurious effects check: WORKING (spurious effects penalized)")
        tests_passed += 1
    else:
        print("❌ Spurious effects check: FAILED")
    
    if consistency_loss_mono_bad.item() > consistency_loss_mono_good.item():
        print("✅ Monotonicity check: WORKING (non-monotonic effects penalized)")
        tests_passed += 1
    else:
        print("❌ Monotonicity check: FAILED")
    
    if consistency_loss_large_bad.item() > consistency_loss_large_good.item():
        print("✅ Magnitude consistency check: WORKING (weak effects penalized)")
        tests_passed += 1
    else:
        print("❌ Magnitude consistency check: FAILED")
    
    print(f"\nOverall: {tests_passed}/{total_tests} consistency checks working")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = test_counterfactual_consistency()
    sys.exit(0 if success else 1) 