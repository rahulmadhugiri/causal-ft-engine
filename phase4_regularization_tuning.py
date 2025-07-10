#!/usr/bin/env python3
"""
Phase 4: Causal Regularization Tuning Experiment

Goal: Find optimal lambda_reg values that activate causal violation penalty
while maintaining or improving predictive performance.

Tests lambda_reg values: 0.01, 0.1, 0.5, 1.0
Tracks: test loss, causal violation penalty, structure F1, CF accuracy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from causal_unit_network import CausalUnitNetwork
from train_causalunit import CausalUnitTrainer
from experiments.utils import generate_synthetic_data, create_dag_from_edges


def run_regularization_experiment(lambda_reg, experiment_name, num_epochs=30, n_samples=1000):
    """Run single regularization experiment with given lambda_reg value."""
    
    print(f"\n=== REGULARIZATION EXPERIMENT: {experiment_name} (lambda_reg={lambda_reg}) ===")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    x_np, y_np, true_adjacency_np = generate_synthetic_data(n_samples, n_nodes=4, graph_type='chain', noise_level=0.3)
    
    # Convert to PyTorch tensors
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    true_adjacency = torch.tensor(true_adjacency_np, dtype=torch.float32)
    
    # Create network with specific lambda_reg
    network = CausalUnitNetwork(
        input_dim=x.shape[1],
        hidden_dims=[32, 16],
        output_dim=y.shape[1],
        activation='relu',
        lambda_reg=lambda_reg,  # Key parameter being tested
        enable_structure_learning=True,
        enable_gradient_surgery=True
    )
    
    # Create trainer
    trainer = CausalUnitTrainer(
        network=network,
        learning_rate=0.01,
        counterfactual_weight=0.1,
        structure_weight=0.05,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create test loader (use same data for validation)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    # Train model
    print(f"Training for {num_epochs} epochs...")
    trainer.train(train_loader, test_loader, num_epochs=num_epochs, true_adjacency=true_adjacency)
    
    # Extract final metrics
    final_metrics = trainer.training_history
    
    # Get final values
    final_test_loss = final_metrics['prediction_loss'][-1]
    final_violation_penalty = final_metrics['causal_violation_penalty'][-1]
    final_structure_f1 = final_metrics['structure_accuracy'][-1]
    final_cf_accuracy = final_metrics['counterfactual_accuracy'][-1]
    
    # Calculate average violation penalty over last 10 epochs
    avg_violation_penalty = np.mean(final_metrics['causal_violation_penalty'][-10:])
    
    result = {
        'lambda_reg': lambda_reg,
        'experiment_name': experiment_name,
        'final_test_loss': final_test_loss,
        'final_violation_penalty': final_violation_penalty,
        'avg_violation_penalty': avg_violation_penalty,
        'final_structure_f1': final_structure_f1,
        'final_cf_accuracy': final_cf_accuracy,
        'training_history': final_metrics,
        'total_epochs': num_epochs,
        'n_samples': n_samples
    }
    
    print(f"Results - Test Loss: {final_test_loss:.4f}, "
          f"Violation Penalty: {final_violation_penalty:.6f}, "
          f"Structure F1: {final_structure_f1:.4f}, "
          f"CF Accuracy: {final_cf_accuracy:.4f}")
    
    return result


def plot_regularization_results(results, save_path):
    """Plot comprehensive regularization tuning results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Phase 4: Causal Regularization Tuning Results', fontsize=16)
    
    lambda_values = [r['lambda_reg'] for r in results]
    test_losses = [r['final_test_loss'] for r in results]
    violation_penalties = [r['avg_violation_penalty'] for r in results]
    structure_f1s = [r['final_structure_f1'] for r in results]
    cf_accuracies = [r['final_cf_accuracy'] for r in results]
    
    # Plot 1: Test Loss vs Lambda
    axes[0, 0].plot(lambda_values, test_losses, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Lambda Regularization')
    axes[0, 0].set_ylabel('Final Test Loss')
    axes[0, 0].set_title('Predictive Performance vs Regularization')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Violation Penalty vs Lambda
    axes[0, 1].plot(lambda_values, violation_penalties, 'o-', color='red', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Lambda Regularization')
    axes[0, 1].set_ylabel('Avg Violation Penalty')
    axes[0, 1].set_title('Causal Violation Penalty Activation')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # Plot 3: Structure F1 vs Lambda
    axes[1, 0].plot(lambda_values, structure_f1s, 'o-', color='green', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Lambda Regularization')
    axes[1, 0].set_ylabel('Structure F1 Score')
    axes[1, 0].set_title('Structure Learning Performance')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    
    # Plot 4: CF Accuracy vs Lambda
    axes[1, 1].plot(lambda_values, cf_accuracies, 'o-', color='purple', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Lambda Regularization')
    axes[1, 1].set_ylabel('Counterfactual Accuracy')
    axes[1, 1].set_title('Counterfactual Reasoning Performance')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Regularization tuning plot saved to: {save_path}")


def main():
    """Run comprehensive regularization tuning experiment."""
    
    print("=== PHASE 4: CAUSAL REGULARIZATION TUNING ===")
    print("Testing lambda_reg values: 0.01, 0.1, 0.5, 1.0")
    
    # Test different lambda_reg values
    lambda_values = [0.01, 0.1, 0.5, 1.0]
    results = []
    
    for lambda_reg in lambda_values:
        experiment_name = f"lambda_{lambda_reg}"
        result = run_regularization_experiment(
            lambda_reg=lambda_reg,
            experiment_name=experiment_name,
            num_epochs=30,
            n_samples=1000
        )
        results.append(result)
    
    # Save results
    os.makedirs("results/phase4_regularization", exist_ok=True)
    
    # Save detailed results
    with open("results/phase4_regularization/tuning_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    summary = {
        'experiment_type': 'causal_regularization_tuning',
        'timestamp': datetime.now().isoformat(),
        'lambda_values_tested': lambda_values,
        'num_experiments': len(results),
        'key_findings': {
            'best_test_loss': {
                'lambda_reg': min(results, key=lambda x: x['final_test_loss'])['lambda_reg'],
                'value': min(r['final_test_loss'] for r in results)
            },
            'highest_violation_penalty': {
                'lambda_reg': max(results, key=lambda x: x['avg_violation_penalty'])['lambda_reg'],
                'value': max(r['avg_violation_penalty'] for r in results)
            },
            'best_structure_f1': {
                'lambda_reg': max(results, key=lambda x: x['final_structure_f1'])['lambda_reg'],
                'value': max(r['final_structure_f1'] for r in results)
            },
            'best_cf_accuracy': {
                'lambda_reg': max(results, key=lambda x: x['final_cf_accuracy'])['lambda_reg'],
                'value': max(r['final_cf_accuracy'] for r in results)
            }
        },
        'results_summary': [
            {
                'lambda_reg': r['lambda_reg'],
                'test_loss': r['final_test_loss'],
                'violation_penalty': r['avg_violation_penalty'],
                'structure_f1': r['final_structure_f1'],
                'cf_accuracy': r['final_cf_accuracy']
            }
            for r in results
        ]
    }
    
    with open("results/phase4_regularization/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate plots
    plot_regularization_results(results, "results/phase4_regularization/tuning_analysis.png")
    
    # Print summary
    print("\n=== REGULARIZATION TUNING SUMMARY ===")
    print(f"Best Test Loss: {summary['key_findings']['best_test_loss']['value']:.4f} "
          f"(lambda_reg={summary['key_findings']['best_test_loss']['lambda_reg']})")
    print(f"Highest Violation Penalty: {summary['key_findings']['highest_violation_penalty']['value']:.6f} "
          f"(lambda_reg={summary['key_findings']['highest_violation_penalty']['lambda_reg']})")
    print(f"Best Structure F1: {summary['key_findings']['best_structure_f1']['value']:.4f} "
          f"(lambda_reg={summary['key_findings']['best_structure_f1']['lambda_reg']})")
    print(f"Best CF Accuracy: {summary['key_findings']['best_cf_accuracy']['value']:.4f} "
          f"(lambda_reg={summary['key_findings']['best_cf_accuracy']['lambda_reg']})")
    
    print(f"\nResults saved to: results/phase4_regularization/")
    print("Next step: Implement soft interventions with optimal lambda_reg value")


if __name__ == "__main__":
    main() 