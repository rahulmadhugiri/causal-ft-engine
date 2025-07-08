#!/usr/bin/env python3
"""
Debug Phase 2: Isolated Structure Learning and Counterfactual Testing

This script debugs the Phase 2 implementation by:
1. Isolating structure learning (no prediction/counterfactual losses)
2. Using simplified datasets with known structure
3. Testing on tiny datasets to verify memorization
4. Adding detailed logging and visualization

Usage:
    python experiments/debug_phase2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.structure_learning import StructureLearner, DifferentiableDAG, create_true_dag, generate_dag_data
from engine.counterfactuals import CounterfactualSimulator
from engine.causal_unit import CausalMLP, StructureAwareCausalMLP
from engine.loss_functions import CausalLosses


def create_simple_chain_dag(num_variables=3):
    """
    Create a simple chain DAG: X1 → X2 → X3
    
    Returns:
        adjacency: True adjacency matrix
    """
    adjacency = torch.zeros(num_variables, num_variables)
    
    # Create chain: 0 → 1 → 2
    for i in range(num_variables - 1):
        adjacency[i, i + 1] = 1.0
    
    return adjacency


def generate_simple_chain_data(n_samples=100, noise_std=0.1):
    """
    Generate data from simple chain: X1 → X2 → X3
    
    X1 ~ N(0, 1)
    X2 = 2*X1 + noise
    X3 = -1.5*X2 + noise
    
    Returns:
        X: Input features
        true_adjacency: True adjacency matrix
        true_coefficients: True causal coefficients
    """
    # Generate X1 (root node)
    X1 = torch.randn(n_samples, 1)
    
    # Generate X2 = 2*X1 + noise
    X2 = 2.0 * X1 + noise_std * torch.randn(n_samples, 1)
    
    # Generate X3 = -1.5*X2 + noise
    X3 = -1.5 * X2 + noise_std * torch.randn(n_samples, 1)
    
    # Combine features
    X = torch.cat([X1, X2, X3], dim=1)
    
    # True adjacency matrix
    true_adjacency = create_simple_chain_dag(3)
    
    # True coefficients
    true_coefficients = torch.zeros(3, 3)
    true_coefficients[0, 1] = 2.0    # X1 → X2
    true_coefficients[1, 2] = -1.5   # X2 → X3
    
    return X, true_adjacency, true_coefficients


def debug_structure_learning_only(n_samples=100, num_epochs=500, verbose=True):
    """
    Debug structure learning in isolation (no prediction loss).
    
    Args:
        n_samples: Number of training samples
        num_epochs: Number of training epochs
        verbose: Whether to print progress
        
    Returns:
        results: Dictionary with learned structure and training history
    """
    print(f"\n=== DEBUG: Structure Learning Only (n_samples={n_samples}) ===")
    
    # Generate simple chain data
    X, true_adjacency, true_coefficients = generate_simple_chain_data(n_samples)
    
    print(f"True adjacency matrix:")
    print(true_adjacency.numpy())
    print(f"True coefficients:")
    print(true_coefficients.numpy())
    
    # Initialize structure learner
    structure_learner = StructureLearner(num_variables=3, hidden_dim=8)
    
    # Train with structure learning only
    print(f"\nTraining structure learner for {num_epochs} epochs...")
    
    learned_adjacency, training_history = structure_learner.learn_structure(
        X, 
        num_epochs=num_epochs,
        lr=0.01,
        lambda_acyclic=0.1,
        lambda_sparse=0.01,
        verbose=verbose
    )
    
    print(f"\nLearned adjacency matrix:")
    print(learned_adjacency.numpy())
    
    # Evaluate structure recovery
    structure_metrics = structure_learner.evaluate_structure_recovery(
        learned_adjacency, true_adjacency
    )
    
    print(f"\nStructure Recovery Metrics:")
    for metric, value in structure_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return {
        'learned_adjacency': learned_adjacency,
        'true_adjacency': true_adjacency,
        'structure_metrics': structure_metrics,
        'training_history': training_history,
        'X': X,
        'true_coefficients': true_coefficients
    }


def debug_integrated_structure_learning(n_samples=100, num_epochs=300, verbose=True):
    """
    Debug structure learning integrated with the CausalMLP.
    
    Args:
        n_samples: Number of training samples
        num_epochs: Number of training epochs
        verbose: Whether to print progress
        
    Returns:
        results: Dictionary with results
    """
    print(f"\n=== DEBUG: Integrated Structure Learning (n_samples={n_samples}) ===")
    
    # Generate data
    X, true_adjacency, true_coefficients = generate_simple_chain_data(n_samples)
    
    # Create target (simple linear combination for now)
    y = torch.sum(X * torch.tensor([1.0, 0.5, -0.3]), dim=1, keepdim=True)
    
    print(f"Data shape: {X.shape}, Target shape: {y.shape}")
    print(f"True adjacency matrix:")
    print(true_adjacency.numpy())
    
    # Initialize model
    model = StructureAwareCausalMLP(
        input_dim=3, hidden_dims=[8], output_dim=1,
        learn_structure=True, structure_hidden_dim=8
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    causal_losses = CausalLosses(
        structure_weight=1.0,  # High weight for structure learning
        sparsity_weight=0.05
    )
    
    # Training history
    history = {
        'structure_loss': [],
        'acyclicity_loss': [],
        'sparsity_loss': [],
        'total_loss': []
    }
    
    # Phase 1: Structure learning only
    print("\nPhase 1: Structure Learning Only...")
    model.set_training_mode(structure_learning=True, prediction=False)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass for structure learning
        _, structure_info = model(X, return_structure=True)
        structure_info['original_input'] = X
        
        # Compute structure losses only
        structure_recon_loss = causal_losses.structure_reconstruction_loss(
            structure_info['reconstruction'], X, structure_info['adjacency']
        )
        acyclicity_loss = causal_losses.acyclicity_loss(structure_info['adjacency'])
        sparsity_loss = causal_losses.sparsity_loss(structure_info['adjacency'])
        
        # Total loss (structure only)
        total_loss = (structure_recon_loss + 
                     0.1 * acyclicity_loss + 
                     0.05 * sparsity_loss)
        
        total_loss.backward()
        optimizer.step()
        
        # Record metrics
        history['structure_loss'].append(structure_recon_loss.item())
        history['acyclicity_loss'].append(acyclicity_loss.item())
        history['sparsity_loss'].append(sparsity_loss.item())
        history['total_loss'].append(total_loss.item())
        
        # Print progress
        if verbose and epoch % 50 == 0:
            learned_adj = model.get_learned_adjacency(hard=True)
            print(f"Epoch {epoch:3d}: "
                  f"Recon={structure_recon_loss.item():.4f}, "
                  f"Acyc={acyclicity_loss.item():.4f}, "
                  f"Sparse={sparsity_loss.item():.4f}")
            if epoch % 100 == 0:
                print(f"  Current adjacency matrix:")
                print(f"  {learned_adj.numpy()}")
    
    # Final evaluation
    final_adjacency = model.get_learned_adjacency(hard=True)
    print(f"\nFinal learned adjacency matrix:")
    print(final_adjacency.numpy())
    
    # Evaluate structure recovery
    from engine.loss_functions import CausalMetrics
    structure_metrics = CausalMetrics.structure_recovery_metrics(
        final_adjacency, true_adjacency
    )
    
    print(f"\nStructure Recovery Metrics:")
    for metric, value in structure_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return {
        'model': model,
        'learned_adjacency': final_adjacency,
        'true_adjacency': true_adjacency,
        'structure_metrics': structure_metrics,
        'training_history': history,
        'X': X,
        'y': y
    }


def debug_counterfactual_forward_pass():
    """
    Debug counterfactual forward pass manually.
    """
    print(f"\n=== DEBUG: Counterfactual Forward Pass ===")
    
    # Create simple model
    model = CausalMLP(input_dim=3, hidden_dims=[8], output_dim=1)
    
    # Create sample input
    x = torch.tensor([[1.0, 2.0, 3.0]])
    print(f"Original input: {x}")
    
    # Forward pass without intervention
    y_factual = model(x)
    print(f"Factual output: {y_factual}")
    
    # Forward pass with intervention on x[1] = 5.0
    do_mask = torch.tensor([0.0, 1.0, 0.0])
    do_values = torch.tensor([0.0, 5.0, 0.0])
    
    interventions = {0: (do_mask.unsqueeze(0), do_values.unsqueeze(0))}
    y_counterfactual = model(x, interventions=interventions)
    print(f"Counterfactual output (x[1] = 5.0): {y_counterfactual}")
    
    # Check if output changed
    effect = y_counterfactual - y_factual
    print(f"Causal effect: {effect}")
    
    # Test with CounterfactualSimulator
    cf_simulator = CounterfactualSimulator(model)
    effect_sim, factual_sim, counterfactual_sim = cf_simulator.compute_counterfactual_effect(
        x, do_mask, do_values
    )
    
    print(f"Simulator - Factual: {factual_sim}, Counterfactual: {counterfactual_sim}")
    print(f"Simulator - Effect: {effect_sim}")
    
    return {
        'factual': y_factual,
        'counterfactual': y_counterfactual,
        'effect': effect
    }


def debug_counterfactual_on_chain_data():
    """
    Debug counterfactual reasoning on chain data.
    """
    print(f"\n=== DEBUG: Counterfactual on Chain Data ===")
    
    # Generate chain data
    X, true_adjacency, true_coefficients = generate_simple_chain_data(n_samples=50)
    
    print(f"Chain structure: X1 → X2 → X3")
    print(f"True coefficients: X1→X2={true_coefficients[0,1]:.1f}, X2→X3={true_coefficients[1,2]:.1f}")
    
    # Train a simple model on this data
    model = CausalMLP(input_dim=3, hidden_dims=[16], output_dim=1)
    
    # Create target as X3 (the final node in chain)
    y = X[:, 2:3]  # X3 as target
    
    # Quick training
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        optimizer.step()
    
    print(f"Model trained, final loss: {loss.item():.4f}")
    
    # Test counterfactual: do(X1 = 0) vs do(X1 = 2)
    test_x = X[:5]  # First 5 samples
    
    cf_simulator = CounterfactualSimulator(model)
    
    # Intervention on X1 = 0
    do_mask_x1 = torch.tensor([1.0, 0.0, 0.0])
    do_values_x1_0 = torch.tensor([0.0, 0.0, 0.0])
    do_values_x1_2 = torch.tensor([2.0, 0.0, 0.0])
    
    effect_0, factual_0, cf_0 = cf_simulator.compute_counterfactual_effect(
        test_x, do_mask_x1, do_values_x1_0
    )
    
    effect_2, factual_2, cf_2 = cf_simulator.compute_counterfactual_effect(
        test_x, do_mask_x1, do_values_x1_2
    )
    
    print(f"\nCounterfactual Analysis:")
    print(f"Original X1 values: {test_x[:, 0]}")
    print(f"Factual predictions: {factual_0.squeeze()}")
    print(f"CF predictions (X1=0): {cf_0.squeeze()}")
    print(f"CF predictions (X1=2): {cf_2.squeeze()}")
    print(f"Effect (X1=0): {effect_0.squeeze()}")
    print(f"Effect (X1=2): {effect_2.squeeze()}")
    
    # Expected effect based on true structure:
    # X1 → X2 (coeff=2.0) → X3 (coeff=-1.5)
    # So do(X1=v) should change X3 by approximately 2.0 * (-1.5) * v = -3.0 * v
    expected_effect_0 = -3.0 * (0.0 - test_x[:, 0])  # Change from original X1 to 0
    expected_effect_2 = -3.0 * (2.0 - test_x[:, 0])  # Change from original X1 to 2
    
    print(f"\nExpected effects:")
    print(f"Expected effect (X1=0): {expected_effect_0}")
    print(f"Expected effect (X1=2): {expected_effect_2}")
    
    # Compute correlation
    correlation_0 = torch.corrcoef(torch.stack([effect_0.squeeze(), expected_effect_0]))[0, 1]
    correlation_2 = torch.corrcoef(torch.stack([effect_2.squeeze(), expected_effect_2]))[0, 1]
    
    print(f"\nEffect correlations:")
    print(f"Correlation (X1=0): {correlation_0.item():.4f}")
    print(f"Correlation (X1=2): {correlation_2.item():.4f}")
    
    return {
        'test_x': test_x,
        'factual': factual_0,
        'counterfactual_0': cf_0,
        'counterfactual_2': cf_2,
        'effect_0': effect_0,
        'effect_2': effect_2,
        'expected_effect_0': expected_effect_0,
        'expected_effect_2': expected_effect_2,
        'correlation_0': correlation_0,
        'correlation_2': correlation_2
    }


def plot_debug_results(results_structure, results_integrated, results_cf):
    """
    Plot debug results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Structure learning losses
    if 'training_history' in results_structure:
        history = results_structure['training_history']
        axes[0, 0].plot(history['reconstruction_loss'], label='Reconstruction', alpha=0.8)
        axes[0, 0].plot(history['acyclicity_constraint'], label='Acyclicity', alpha=0.8)
        axes[0, 0].plot(history['sparsity_loss'], label='Sparsity', alpha=0.8)
        axes[0, 0].set_title('Structure Learning Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Integrated structure learning
    if 'training_history' in results_integrated:
        history = results_integrated['training_history']
        axes[0, 1].plot(history['structure_loss'], label='Structure', alpha=0.8)
        axes[0, 1].plot(history['acyclicity_loss'], label='Acyclicity', alpha=0.8)
        axes[0, 1].plot(history['sparsity_loss'], label='Sparsity', alpha=0.8)
        axes[0, 1].set_title('Integrated Structure Learning')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # True vs Learned adjacency
    if 'true_adjacency' in results_structure and 'learned_adjacency' in results_structure:
        axes[0, 2].imshow(results_structure['true_adjacency'].numpy(), cmap='Blues', alpha=0.8)
        axes[0, 2].set_title('True Adjacency Matrix')
        axes[0, 2].set_xlabel('To Variable')
        axes[0, 2].set_ylabel('From Variable')
    
    # Learned adjacency (standalone)
    if 'learned_adjacency' in results_structure:
        im = axes[1, 0].imshow(results_structure['learned_adjacency'].numpy(), cmap='Reds', alpha=0.8)
        axes[1, 0].set_title('Learned Adjacency (Standalone)')
        axes[1, 0].set_xlabel('To Variable')
        axes[1, 0].set_ylabel('From Variable')
        plt.colorbar(im, ax=axes[1, 0])
    
    # Learned adjacency (integrated)
    if 'learned_adjacency' in results_integrated:
        im = axes[1, 1].imshow(results_integrated['learned_adjacency'].numpy(), cmap='Greens', alpha=0.8)
        axes[1, 1].set_title('Learned Adjacency (Integrated)')
        axes[1, 1].set_xlabel('To Variable')
        axes[1, 1].set_ylabel('From Variable')
        plt.colorbar(im, ax=axes[1, 1])
    
    # Counterfactual effects
    if results_cf and 'effect_0' in results_cf:
        axes[1, 2].scatter(results_cf['expected_effect_0'].detach().numpy(), 
                          results_cf['effect_0'].squeeze().detach().numpy(), 
                          alpha=0.7, label='X1=0')
        axes[1, 2].scatter(results_cf['expected_effect_2'].detach().numpy(), 
                          results_cf['effect_2'].squeeze().detach().numpy(), 
                          alpha=0.7, label='X1=2')
        axes[1, 2].plot([-3, 3], [-3, 3], 'k--', alpha=0.5, label='Perfect correlation')
        axes[1, 2].set_xlabel('Expected Effect')
        axes[1, 2].set_ylabel('Predicted Effect')
        axes[1, 2].set_title('Counterfactual Effect Correlation')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_phase2_results.png', dpi=300, bbox_inches='tight')
    print(f"Debug results saved to: debug_phase2_results.png")


def run_debug_phase2():
    """
    Run complete debug analysis of Phase 2.
    """
    print("DEBUG PHASE 2: Systematic Analysis")
    print("=" * 50)
    
    # 1. Test structure learning in isolation
    print("\n1. Testing structure learning in isolation...")
    results_structure_large = debug_structure_learning_only(n_samples=100, num_epochs=500)
    
    # 2. Test on tiny dataset (overfitting)
    print("\n2. Testing structure learning on tiny dataset...")
    results_structure_tiny = debug_structure_learning_only(n_samples=10, num_epochs=1000)
    
    # 3. Test integrated structure learning
    print("\n3. Testing integrated structure learning...")
    results_integrated = debug_integrated_structure_learning(n_samples=100, num_epochs=300)
    
    # 4. Test counterfactual forward pass
    print("\n4. Testing counterfactual forward pass...")
    results_cf_forward = debug_counterfactual_forward_pass()
    
    # 5. Test counterfactual on chain data
    print("\n5. Testing counterfactual on chain data...")
    results_cf_chain = debug_counterfactual_on_chain_data()
    
    # 6. Generate plots
    print("\n6. Generating debug plots...")
    plot_debug_results(results_structure_large, results_integrated, results_cf_chain)
    
    # 7. Summary
    print("\n" + "=" * 50)
    print("DEBUG SUMMARY")
    print("=" * 50)
    
    print(f"\nStructure Learning (Large Dataset):")
    if 'structure_metrics' in results_structure_large:
        metrics = results_structure_large['structure_metrics']
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
    
    print(f"\nStructure Learning (Tiny Dataset):")
    if 'structure_metrics' in results_structure_tiny:
        metrics = results_structure_tiny['structure_metrics']
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
    
    print(f"\nIntegrated Structure Learning:")
    if 'structure_metrics' in results_integrated:
        metrics = results_integrated['structure_metrics']
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
    
    print(f"\nCounterfactual Analysis:")
    if 'correlation_0' in results_cf_chain:
        print(f"  Effect Correlation (X1=0): {results_cf_chain['correlation_0'].item():.3f}")
        print(f"  Effect Correlation (X1=2): {results_cf_chain['correlation_2'].item():.3f}")
    
    return {
        'structure_large': results_structure_large,
        'structure_tiny': results_structure_tiny,
        'integrated': results_integrated,
        'counterfactual_forward': results_cf_forward,
        'counterfactual_chain': results_cf_chain
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = run_debug_phase2() 