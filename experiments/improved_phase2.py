#!/usr/bin/env python3
"""
Improved Phase 2: Fixed Structure Learning and Counterfactual Learning

This script implements the improved Phase 2 with fixes based on debug analysis:
1. Stronger sparsity constraints to prevent overfitting
2. Better hyperparameter tuning for structure learning
3. Improved training strategy
4. Enhanced evaluation metrics

Usage:
    python experiments/improved_phase2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import BaselineMLP, generate_causal_data
from engine.causal_unit import CausalMLP, StructureAwareCausalMLP
from engine.counterfactuals import CounterfactualSimulator, generate_counterfactual_data, CounterfactualLoss
from engine.structure_learning import StructureLearner, create_true_dag, generate_dag_data
from engine.loss_functions import CausalLosses, CausalMetrics


def create_simple_chain_dag(num_variables=3):
    """Create a simple chain DAG: X1 → X2 → X3"""
    adjacency = torch.zeros(num_variables, num_variables)
    for i in range(num_variables - 1):
        adjacency[i, i + 1] = 1.0
    return adjacency


def generate_simple_chain_data(n_samples=100, noise_std=0.1):
    """Generate data from simple chain: X1 → X2 → X3"""
    X1 = torch.randn(n_samples, 1)
    X2 = 2.0 * X1 + noise_std * torch.randn(n_samples, 1)
    X3 = -1.5 * X2 + noise_std * torch.randn(n_samples, 1)
    
    X = torch.cat([X1, X2, X3], dim=1)
    true_adjacency = create_simple_chain_dag(3)
    
    true_coefficients = torch.zeros(3, 3)
    true_coefficients[0, 1] = 2.0
    true_coefficients[1, 2] = -1.5
    
    return X, true_adjacency, true_coefficients


def improved_structure_learning(n_samples=100, num_epochs=800, verbose=True):
    """
    Improved structure learning with better hyperparameters.
    """
    print(f"\n=== IMPROVED: Structure Learning (n_samples={n_samples}) ===")
    
    # Generate data
    X, true_adjacency, true_coefficients = generate_simple_chain_data(n_samples)
    
    print(f"True adjacency matrix:")
    print(true_adjacency.numpy())
    
    # Initialize with better hyperparameters
    structure_learner = StructureLearner(num_variables=3, hidden_dim=16)
    
    # Improved training with stronger sparsity
    print(f"\nTraining with improved hyperparameters...")
    
    learned_adjacency, training_history = structure_learner.learn_structure(
        X, 
        num_epochs=num_epochs,
        lr=0.005,  # Lower learning rate for stability
        lambda_acyclic=0.5,  # Stronger acyclicity constraint
        lambda_sparse=0.1,   # Much stronger sparsity constraint
        verbose=verbose
    )
    
    print(f"\nLearned adjacency matrix:")
    print(learned_adjacency.numpy())
    
    # Evaluate
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
        'training_history': training_history
    }


def improved_integrated_training(n_samples=200, num_epochs=400, verbose=True):
    """
    Improved integrated training with better strategy.
    """
    print(f"\n=== IMPROVED: Integrated Training (n_samples={n_samples}) ===")
    
    # Generate data
    X, true_adjacency, true_coefficients = generate_simple_chain_data(n_samples)
    
    # Create target as X3 (makes more sense for chain structure)
    y = X[:, 2:3]  # Predict X3 from X1, X2, X3
    
    print(f"Data shape: {X.shape}, Target shape: {y.shape}")
    print(f"True adjacency matrix:")
    print(true_adjacency.numpy())
    
    # Initialize model
    model = StructureAwareCausalMLP(
        input_dim=3, hidden_dims=[16, 8], output_dim=1,
        learn_structure=True, structure_hidden_dim=16
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    causal_losses = CausalLosses(
        structure_weight=0.5,    # Balanced structure weight
        sparsity_weight=0.2,     # Strong sparsity
        counterfactual_weight=0.3
    )
    
    # Training history
    history = {
        'structure_loss': [],
        'acyclicity_loss': [],
        'sparsity_loss': [],
        'prediction_loss': [],
        'total_loss': []
    }
    
    # Phase 1: Structure learning only (longer)
    print("\nPhase 1: Structure Learning (300 epochs)...")
    model.set_training_mode(structure_learning=True, prediction=False)
    
    for epoch in range(300):
        optimizer.zero_grad()
        
        _, structure_info = model(X, return_structure=True)
        structure_info['original_input'] = X
        
        # Structure losses with stronger sparsity
        structure_recon_loss = causal_losses.structure_reconstruction_loss(
            structure_info['reconstruction'], X, structure_info['adjacency']
        )
        acyclicity_loss = causal_losses.acyclicity_loss(structure_info['adjacency'])
        sparsity_loss = causal_losses.sparsity_loss(structure_info['adjacency'])
        
        total_loss = (structure_recon_loss + 
                     0.5 * acyclicity_loss + 
                     0.2 * sparsity_loss)
        
        total_loss.backward()
        optimizer.step()
        
        # Record metrics
        history['structure_loss'].append(structure_recon_loss.item())
        history['acyclicity_loss'].append(acyclicity_loss.item())
        history['sparsity_loss'].append(sparsity_loss.item())
        history['total_loss'].append(total_loss.item())
        
        if verbose and epoch % 50 == 0:
            learned_adj = model.get_learned_adjacency(hard=True)
            print(f"Epoch {epoch:3d}: "
                  f"Recon={structure_recon_loss.item():.4f}, "
                  f"Acyc={acyclicity_loss.item():.4f}, "
                  f"Sparse={sparsity_loss.item():.4f}")
            if epoch % 100 == 0:
                print(f"  Current adjacency matrix:")
                print(f"  {learned_adj.numpy()}")
    
    # Phase 2: Joint training
    print("\nPhase 2: Joint Training (100 epochs)...")
    model.set_training_mode(structure_learning=False, prediction=True)
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        predictions, structure_info = model(X, return_structure=True)
        structure_info['original_input'] = X
        
        # Combined loss
        total_loss, loss_components = causal_losses.phase2_total_loss(
            predictions=predictions, targets=y,
            structure_info=structure_info
        )
        
        total_loss.backward()
        optimizer.step()
        
        # Record metrics
        history['prediction_loss'].append(loss_components['prediction_loss'])
        
        if verbose and epoch % 25 == 0:
            print(f"Joint Epoch {epoch:3d}: "
                  f"Pred={loss_components['prediction_loss']:.4f}, "
                  f"Total={loss_components['total_loss']:.4f}")
    
    # Final evaluation
    final_adjacency = model.get_learned_adjacency(hard=True)
    print(f"\nFinal learned adjacency matrix:")
    print(final_adjacency.numpy())
    
    # Evaluate structure recovery
    structure_metrics = CausalMetrics.structure_recovery_metrics(
        final_adjacency, true_adjacency
    )
    
    print(f"\nStructure Recovery Metrics:")
    for metric, value in structure_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test counterfactual reasoning
    print(f"\nTesting counterfactual reasoning...")
    cf_simulator = CounterfactualSimulator(model)
    
    # Test on first 5 samples
    test_x = X[:5]
    
    # Intervention on X1
    do_mask_x1 = torch.tensor([1.0, 0.0, 0.0])
    do_values_x1_0 = torch.tensor([0.0, 0.0, 0.0])
    do_values_x1_2 = torch.tensor([2.0, 0.0, 0.0])
    
    effect_0, _, _ = cf_simulator.compute_counterfactual_effect(
        test_x, do_mask_x1, do_values_x1_0
    )
    effect_2, _, _ = cf_simulator.compute_counterfactual_effect(
        test_x, do_mask_x1, do_values_x1_2
    )
    
    # Expected effects for chain X1 → X2 → X3
    expected_effect_0 = -3.0 * (0.0 - test_x[:, 0])
    expected_effect_2 = -3.0 * (2.0 - test_x[:, 0])
    
    # Compute correlations
    correlation_0 = torch.corrcoef(torch.stack([effect_0.squeeze(), expected_effect_0]))[0, 1]
    correlation_2 = torch.corrcoef(torch.stack([effect_2.squeeze(), expected_effect_2]))[0, 1]
    
    print(f"Counterfactual correlations:")
    print(f"  X1=0: {correlation_0.item():.4f}")
    print(f"  X1=2: {correlation_2.item():.4f}")
    
    return {
        'model': model,
        'learned_adjacency': final_adjacency,
        'true_adjacency': true_adjacency,
        'structure_metrics': structure_metrics,
        'training_history': history,
        'counterfactual_correlations': {
            'x1_0': correlation_0.item(),
            'x1_2': correlation_2.item()
        }
    }


def comprehensive_counterfactual_test():
    """
    Comprehensive test of counterfactual reasoning on chain data.
    """
    print(f"\n=== COMPREHENSIVE: Counterfactual Testing ===")
    
    # Generate larger dataset
    X, true_adjacency, true_coefficients = generate_simple_chain_data(n_samples=500)
    
    # Train model to predict X3 from all inputs
    model = CausalMLP(input_dim=3, hidden_dims=[32, 16], output_dim=1)
    y = X[:, 2:3]  # X3 as target
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        optimizer.step()
    
    print(f"Model trained, final loss: {loss.item():.6f}")
    
    # Comprehensive counterfactual testing
    cf_simulator = CounterfactualSimulator(model)
    
    # Test multiple interventions
    test_x = X[:20]  # Larger test set
    
    interventions = [
        (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])),  # do(X1=0)
        (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])),  # do(X1=1)
        (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([2.0, 0.0, 0.0])),  # do(X1=2)
        (torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])),  # do(X2=0)
        (torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])),  # do(X2=1)
    ]
    
    results = {}
    
    for i, (do_mask, do_values) in enumerate(interventions):
        effect, _, _ = cf_simulator.compute_counterfactual_effect(
            test_x, do_mask, do_values
        )
        
        # Compute expected effect based on true structure
        if do_mask[0] == 1.0:  # Intervention on X1
            expected_effect = -3.0 * (do_values[0] - test_x[:, 0])
        elif do_mask[1] == 1.0:  # Intervention on X2
            expected_effect = -1.5 * (do_values[1] - test_x[:, 1])
        else:
            expected_effect = torch.zeros_like(effect.squeeze())
        
        # Compute correlation
        correlation = torch.corrcoef(torch.stack([effect.squeeze(), expected_effect]))[0, 1]
        
        intervention_name = f"do(X{torch.argmax(do_mask)+1}={do_values[torch.argmax(do_mask)].item():.1f})"
        results[intervention_name] = {
            'correlation': correlation.item(),
            'effect_magnitude': torch.abs(effect).mean().item(),
            'expected_magnitude': torch.abs(expected_effect).mean().item()
        }
        
        print(f"{intervention_name}: correlation={correlation.item():.4f}, "
              f"effect_mag={torch.abs(effect).mean().item():.4f}")
    
    return results


def plot_improved_results(structure_results, integrated_results, cf_results):
    """
    Plot improved results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Structure learning progress
    if 'training_history' in structure_results:
        history = structure_results['training_history']
        axes[0, 0].plot(history['reconstruction_loss'], label='Reconstruction', alpha=0.8)
        axes[0, 0].plot(history['acyclicity_constraint'], label='Acyclicity', alpha=0.8)
        axes[0, 0].plot(history['sparsity_loss'], label='Sparsity', alpha=0.8)
        axes[0, 0].set_title('Improved Structure Learning')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Integrated training progress
    if 'training_history' in integrated_results:
        history = integrated_results['training_history']
        axes[0, 1].plot(history['structure_loss'], label='Structure', alpha=0.8)
        if history['prediction_loss']:
            axes[0, 1].plot(range(300, 300 + len(history['prediction_loss'])), 
                           history['prediction_loss'], label='Prediction', alpha=0.8)
        axes[0, 1].set_title('Integrated Training Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Structure recovery comparison
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    structure_values = [
        structure_results['structure_metrics']['precision'],
        structure_results['structure_metrics']['recall'],
        structure_results['structure_metrics']['f1_score'],
        structure_results['structure_metrics']['accuracy']
    ]
    integrated_values = [
        integrated_results['structure_metrics']['precision'],
        integrated_results['structure_metrics']['recall'],
        integrated_results['structure_metrics']['f1_score'],
        integrated_results['structure_metrics']['accuracy']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0, 2].bar(x - width/2, structure_values, width, label='Standalone', alpha=0.8)
    axes[0, 2].bar(x + width/2, integrated_values, width, label='Integrated', alpha=0.8)
    axes[0, 2].set_title('Structure Recovery Comparison')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(metrics_names)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # True adjacency matrix
    axes[1, 0].imshow(structure_results['true_adjacency'].numpy(), cmap='Blues', alpha=0.8)
    axes[1, 0].set_title('True Adjacency Matrix')
    axes[1, 0].set_xlabel('To Variable')
    axes[1, 0].set_ylabel('From Variable')
    
    # Learned adjacency matrices
    im1 = axes[1, 1].imshow(structure_results['learned_adjacency'].numpy(), cmap='Reds', alpha=0.8)
    axes[1, 1].set_title('Learned (Standalone)')
    axes[1, 1].set_xlabel('To Variable')
    axes[1, 1].set_ylabel('From Variable')
    plt.colorbar(im1, ax=axes[1, 1])
    
    # Counterfactual correlations
    if cf_results:
        interventions = list(cf_results.keys())
        correlations = [cf_results[key]['correlation'] for key in interventions]
        
        axes[1, 2].bar(interventions, correlations, alpha=0.7, color='lightgreen')
        axes[1, 2].set_title('Counterfactual Correlations')
        axes[1, 2].set_ylabel('Correlation')
        axes[1, 2].set_ylim(0, 1.1)
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_phase2_results.png', dpi=300, bbox_inches='tight')
    print(f"Improved results saved to: improved_phase2_results.png")


def run_improved_phase2():
    """
    Run improved Phase 2 experiment.
    """
    print("IMPROVED PHASE 2: Enhanced Structure Learning & Counterfactual Reasoning")
    print("=" * 80)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Improved structure learning
    print("\n1. Running improved structure learning...")
    structure_results = improved_structure_learning(n_samples=200, num_epochs=800)
    
    # 2. Improved integrated training
    print("\n2. Running improved integrated training...")
    integrated_results = improved_integrated_training(n_samples=200, num_epochs=400)
    
    # 3. Comprehensive counterfactual testing
    print("\n3. Running comprehensive counterfactual testing...")
    cf_results = comprehensive_counterfactual_test()
    
    # 4. Generate plots
    print("\n4. Generating improved visualizations...")
    plot_improved_results(structure_results, integrated_results, cf_results)
    
    # 5. Final summary
    print("\n" + "=" * 80)
    print("IMPROVED PHASE 2 RESULTS")
    print("=" * 80)
    
    print(f"\nStructure Learning (Standalone):")
    sm = structure_results['structure_metrics']
    print(f"  Precision: {sm['precision']:.3f}")
    print(f"  Recall: {sm['recall']:.3f}")
    print(f"  F1-Score: {sm['f1_score']:.3f}")
    print(f"  Learned edges: {sm['learned_edges']:.0f} / True edges: {sm['true_edges']:.0f}")
    
    print(f"\nStructure Learning (Integrated):")
    im = integrated_results['structure_metrics']
    print(f"  Precision: {im['precision']:.3f}")
    print(f"  Recall: {im['recall']:.3f}")
    print(f"  F1-Score: {im['f1_score']:.3f}")
    print(f"  Learned edges: {im['learned_edges']:.0f} / True edges: {im['true_edges']:.0f}")
    
    print(f"\nCounterfactual Correlations (Integrated):")
    cc = integrated_results['counterfactual_correlations']
    print(f"  X1=0: {cc['x1_0']:.3f}")
    print(f"  X1=2: {cc['x1_2']:.3f}")
    
    print(f"\nComprehensive Counterfactual Results:")
    for intervention, metrics in cf_results.items():
        print(f"  {intervention}: correlation={metrics['correlation']:.3f}")
    
    # Check success criteria
    structure_success = im['precision'] >= 0.7 and im['recall'] >= 0.5
    cf_success = cc['x1_0'] >= 0.5 and cc['x1_2'] >= 0.5
    
    print(f"\n" + "=" * 80)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 80)
    print(f"Structure Learning: {'✅ PASS' if structure_success else '❌ FAIL'}")
    print(f"Counterfactual Learning: {'✅ PASS' if cf_success else '❌ FAIL'}")
    
    if structure_success and cf_success:
        print(f"\n🎉 PHASE 2 IMPROVEMENTS SUCCESSFUL!")
        print(f"Ready to proceed to Phase 3!")
    else:
        print(f"\n⚠️  Phase 2 needs further improvements.")
    
    return {
        'structure_results': structure_results,
        'integrated_results': integrated_results,
        'cf_results': cf_results,
        'success': structure_success and cf_success
    }


if __name__ == "__main__":
    results = run_improved_phase2() 