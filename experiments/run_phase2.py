#!/usr/bin/env python3
"""
Phase 2: Counterfactual Learning + Dynamic DAG Discovery Experiment Runner

This script runs the complete Phase 2 experiment demonstrating:
- Counterfactual reasoning capabilities
- Dynamic causal structure learning
- Integration of structure learning with prediction
- Evaluation of structure recovery and counterfactual accuracy

Usage:
    python experiments/run_phase2.py
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


def train_phase2_model(model, X_train, y_train, X_test, y_test, 
                      counterfactual_data=None, true_adjacency=None,
                      num_epochs=200, lr=0.01, verbose=True):
    """
    Train a Phase 2 model with structure learning and counterfactual reasoning.
    
    Args:
        model: StructureAwareCausalMLP model
        X_train, y_train: Training data
        X_test, y_test: Test data
        counterfactual_data: Dictionary with counterfactual examples
        true_adjacency: True adjacency matrix for evaluation
        num_epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print training progress
        
    Returns:
        Dictionary with training history and final metrics
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    causal_losses = CausalLosses(
        intervention_weight=1.0, 
        counterfactual_weight=0.3,
        structure_weight=0.2,
        sparsity_weight=0.01
    )
    
    training_history = {
        'total_loss': [],
        'prediction_loss': [],
        'structure_loss': [],
        'counterfactual_loss': [],
        'test_loss': []
    }
    
    # Two-phase training: structure learning then joint training
    print("Phase 2A: Structure Learning...")
    
    # Phase 1: Focus on structure learning
    model.set_training_mode(structure_learning=True, prediction=False)
    
    for epoch in range(num_epochs // 2):
        optimizer.zero_grad()
        
        # Forward pass for structure learning
        _, structure_info = model(X_train, return_structure=True)
        
        # Structure learning loss
        structure_info['original_input'] = X_train
        structure_loss, loss_components = causal_losses.phase2_total_loss(
            predictions=None, targets=y_train,
            structure_info=structure_info
        )
        
        structure_loss.backward()
        optimizer.step()
        
        # Record metrics
        training_history['structure_loss'].append(loss_components.get('structure_reconstruction_loss', 0.0))
        
        if verbose and epoch % 50 == 0:
            print(f"Structure Epoch {epoch:3d}: "
                  f"Recon={loss_components.get('structure_reconstruction_loss', 0.0):.4f}, "
                  f"Acyc={loss_components.get('acyclicity_loss', 0.0):.4f}, "
                  f"Sparse={loss_components.get('sparsity_loss', 0.0):.4f}")
    
    print("\nPhase 2B: Joint Training (Structure + Prediction + Counterfactuals)...")
    
    # Phase 2: Joint training with prediction and counterfactuals
    model.set_training_mode(structure_learning=False, prediction=True)
    
    for epoch in range(num_epochs // 2):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions, structure_info = model(X_train, return_structure=True)
        
        # Prepare loss inputs
        structure_info['original_input'] = X_train
        
        # Add counterfactual predictions if available
        enhanced_cf_data = None
        if counterfactual_data is not None:
            enhanced_cf_data = {}
            for cf_name, cf_data in counterfactual_data.items():
                # Generate counterfactual prediction using current model
                cf_simulator = CounterfactualSimulator(model)
                pred_cf = cf_simulator.simulate_counterfactual(
                    X_train, cf_data['do_mask'], cf_data['do_values']
                )
                
                enhanced_cf_data[cf_name] = cf_data.copy()
                enhanced_cf_data[cf_name]['predicted_counterfactual'] = pred_cf
        
        # Combined loss
        total_loss, loss_components = causal_losses.phase2_total_loss(
            predictions=predictions, targets=y_train,
            counterfactual_data=enhanced_cf_data,
            structure_info=structure_info
        )
        
        total_loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = nn.MSELoss()(test_pred, y_test)
        
        # Record metrics
        training_history['total_loss'].append(loss_components['total_loss'])
        training_history['prediction_loss'].append(loss_components['prediction_loss'])
        training_history['test_loss'].append(test_loss.item())
        
        if 'counterfactual_loss' in loss_components:
            training_history['counterfactual_loss'].append(loss_components['counterfactual_loss'])
        
        if verbose and epoch % 50 == 0:
            print(f"Joint Epoch {epoch:3d}: "
                  f"Pred={loss_components['prediction_loss']:.4f}, "
                  f"CF={loss_components.get('counterfactual_loss', 0.0):.4f}, "
                  f"Test={test_loss.item():.4f}")
    
    return training_history


def evaluate_phase2_model(model, X_test, y_test, counterfactual_data, 
                         true_adjacency=None, true_coeffs=None):
    """
    Comprehensive evaluation of Phase 2 model capabilities.
    
    Args:
        model: Trained StructureAwareCausalMLP
        X_test, y_test: Test data
        counterfactual_data: Counterfactual examples for evaluation
        true_adjacency: True adjacency matrix
        true_coeffs: True causal coefficients
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    model.eval()
    evaluation_results = {}
    
    # Standard prediction accuracy
    with torch.no_grad():
        predictions = model(X_test)
        prediction_mse = nn.MSELoss()(predictions, y_test)
        evaluation_results['prediction_mse'] = prediction_mse.item()
    
    # Structure recovery evaluation
    learned_adjacency = model.get_learned_adjacency(hard=True)
    if learned_adjacency is not None:
        evaluation_results['learned_adjacency'] = learned_adjacency
        
        if true_adjacency is not None:
            structure_metrics = CausalMetrics.structure_recovery_metrics(
                learned_adjacency, true_adjacency
            )
            evaluation_results['structure_recovery'] = structure_metrics
    
    # Counterfactual evaluation
    cf_simulator = CounterfactualSimulator(model)
    cf_results = {}
    
    for cf_name, cf_data in counterfactual_data.items():
        # Predict counterfactual effect
        effect, factual, counterfactual = cf_simulator.compute_counterfactual_effect(
            X_test, cf_data['do_mask'], cf_data['do_values']
        )
        
        # Compare with true effect
        true_effect = cf_data['true_effect'][:X_test.shape[0]]  # Match test set size
        
        cf_accuracy = CausalMetrics.counterfactual_accuracy(effect, true_effect)
        cf_results[cf_name] = cf_accuracy
    
    evaluation_results['counterfactual_accuracy'] = cf_results
    
    return evaluation_results


def plot_phase2_results(training_history, evaluation_results, 
                       filename="phase2_results_summary.png"):
    """
    Plot Phase 2 training and evaluation results.
    
    Args:
        training_history: Training metrics over time
        evaluation_results: Final evaluation results
        filename: Output filename
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training losses
    axes[0, 0].plot(training_history['total_loss'], label='Total Loss', alpha=0.8)
    axes[0, 0].plot(training_history['prediction_loss'], label='Prediction Loss', alpha=0.8)
    if training_history['counterfactual_loss']:
        axes[0, 0].plot(training_history['counterfactual_loss'], label='Counterfactual Loss', alpha=0.8)
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test loss
    axes[0, 1].plot(training_history['test_loss'], label='Test Loss', color='red', alpha=0.8)
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Structure learning loss
    if training_history['structure_loss']:
        axes[0, 2].plot(training_history['structure_loss'], label='Structure Loss', color='green', alpha=0.8)
        axes[0, 2].set_title('Structure Learning Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Reconstruction Loss')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Learned adjacency matrix heatmap
    if 'learned_adjacency' in evaluation_results:
        im = axes[1, 0].imshow(evaluation_results['learned_adjacency'].cpu().numpy(), 
                              cmap='Blues', aspect='auto')
        axes[1, 0].set_title('Learned Adjacency Matrix')
        axes[1, 0].set_xlabel('To Variable')
        axes[1, 0].set_ylabel('From Variable')
        plt.colorbar(im, ax=axes[1, 0])
    
    # Structure recovery metrics
    if 'structure_recovery' in evaluation_results:
        metrics = evaluation_results['structure_recovery']
        metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        metric_values = [metrics['precision'], metrics['recall'], 
                        metrics['f1_score'], metrics['accuracy']]
        
        axes[1, 1].bar(metric_names, metric_values, alpha=0.7, color='skyblue')
        axes[1, 1].set_title('Structure Recovery Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
    
    # Counterfactual accuracy
    if 'counterfactual_accuracy' in evaluation_results:
        cf_data = evaluation_results['counterfactual_accuracy']
        cf_names = list(cf_data.keys())
        cf_correlations = [cf_data[name]['effect_correlation'] for name in cf_names]
        
        axes[1, 2].bar(cf_names, cf_correlations, alpha=0.7, color='lightcoral')
        axes[1, 2].set_title('Counterfactual Effect Correlation')
        axes[1, 2].set_ylabel('Correlation')
        axes[1, 2].set_ylim(-1, 1)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Phase 2 results plots saved to: {filename}")


def run_phase2_experiment():
    """
    Run the complete Phase 2 experiment.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Running Phase 2: Counterfactual Learning + Dynamic DAG Discovery")
    print("=" * 70)
    
    # 1. Generate data with known structure
    print("\n1. Generating data with known causal structure...")
    
    # Create true DAG structure (3 variables)
    true_adjacency = create_true_dag(num_variables=3, sparsity=0.4)
    print(f"True adjacency matrix:\n{true_adjacency}")
    
    # Generate data from this structure
    X_train, true_coeffs = generate_dag_data(true_adjacency, n_samples=800, noise_std=0.1)
    X_test, _ = generate_dag_data(true_adjacency, n_samples=200, noise_std=0.1)
    
    # Generate target using linear combination (for compatibility)
    y_train = torch.sum(X_train * torch.tensor([1.0, 2.0, -3.0]), dim=1, keepdim=True)
    y_test = torch.sum(X_test * torch.tensor([1.0, 2.0, -3.0]), dim=1, keepdim=True)
    
    # Generate counterfactual data
    _, _, counterfactual_data = generate_counterfactual_data(
        n_samples=X_train.shape[0], input_dim=3
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Counterfactual scenarios: {list(counterfactual_data.keys())}")
    
    # 2. Initialize models
    print("\n2. Initializing models...")
    
    # Baseline model (no structure learning)
    baseline_model = CausalMLP(input_dim=3, hidden_dims=[16, 16], output_dim=1)
    
    # Phase 2 model with structure learning
    phase2_model = StructureAwareCausalMLP(
        input_dim=3, hidden_dims=[16, 16], output_dim=1,
        learn_structure=True, structure_hidden_dim=12
    )
    
    print(f"Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters())}")
    print(f"Phase 2 model parameters: {sum(p.numel() for p in phase2_model.parameters())}")
    
    # 3. Train Phase 2 model
    print("\n3. Training Phase 2 model...")
    
    training_history = train_phase2_model(
        phase2_model, X_train, y_train, X_test, y_test,
        counterfactual_data=counterfactual_data,
        true_adjacency=true_adjacency,
        num_epochs=200, verbose=True
    )
    
    # 4. Comprehensive evaluation
    print("\n4. Evaluating Phase 2 capabilities...")
    
    evaluation_results = evaluate_phase2_model(
        phase2_model, X_test, y_test, counterfactual_data,
        true_adjacency=true_adjacency, true_coeffs=true_coeffs
    )
    
    # 5. Print results
    print("\n" + "="*70)
    print("PHASE 2 EVALUATION RESULTS")
    print("="*70)
    
    print(f"Final Prediction MSE: {evaluation_results['prediction_mse']:.4f}")
    
    if 'structure_recovery' in evaluation_results:
        sr = evaluation_results['structure_recovery']
        print(f"\nStructure Recovery Metrics:")
        print(f"  Precision: {sr['precision']:.3f}")
        print(f"  Recall: {sr['recall']:.3f}")
        print(f"  F1-Score: {sr['f1_score']:.3f}")
        print(f"  Accuracy: {sr['accuracy']:.3f}")
        print(f"  Structural Hamming Distance: {sr['shd']:.0f}")
        print(f"  Learned edges: {sr['learned_edges']:.0f} / True edges: {sr['true_edges']:.0f}")
    
    if 'counterfactual_accuracy' in evaluation_results:
        print(f"\nCounterfactual Accuracy:")
        for cf_name, cf_metrics in evaluation_results['counterfactual_accuracy'].items():
            print(f"  {cf_name}:")
            print(f"    Effect Correlation: {cf_metrics['effect_correlation']:.3f}")
            print(f"    MSE: {cf_metrics['counterfactual_mse']:.4f}")
    
    # 6. Generate plots
    print(f"\n5. Generating result visualizations...")
    plot_phase2_results(training_history, evaluation_results)
    
    # 7. Summary
    print("\n" + "="*70)
    print("PHASE 2 EXPERIMENT COMPLETE")
    print("="*70)
    
    print("Key Achievements:")
    print("- Dynamic causal structure learning implemented")
    print("- Counterfactual reasoning capabilities demonstrated")
    print("- Structure recovery accuracy evaluated")
    print("- Counterfactual effect prediction accuracy measured")
    print("\nReady for Phase 3: Real-world applications and advanced interpretability!")
    
    return {
        'training_history': training_history,
        'evaluation_results': evaluation_results,
        'true_adjacency': true_adjacency,
        'learned_adjacency': evaluation_results.get('learned_adjacency'),
        'model': phase2_model
    }


if __name__ == "__main__":
    results = run_phase2_experiment() 