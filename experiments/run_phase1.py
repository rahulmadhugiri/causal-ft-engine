#!/usr/bin/env python3
"""
Phase 1: Causal Fine-Tuning Engine Experiment Runner

This script runs the complete Phase 1 experiment comparing baseline MLP
with causal MLP using do-operator interventions.

Usage:
    python experiments/run_phase1.py
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    generate_causal_data, 
    BaselineMLP, 
    train_model, 
    evaluate_models, 
    plot_training_curves
)
from engine.causal_unit import CausalMLP

def run_phase1_experiment():
    """
    Run the complete Phase 1 experiment.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Running Phase 1: Causal Fine-Tuning Engine")
    print("=" * 50)
    
    # 1. Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    X_train, y_train, true_coeffs = generate_causal_data(n_samples=800)
    X_test, y_test, _ = generate_causal_data(n_samples=200)
    
    print(f"   Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"   Test data shape: X={X_test.shape}, y={y_test.shape}")
    print(f"   True causal coefficients: {true_coeffs}")
    print(f"   Sample input: {X_train[0]}")
    print(f"   Sample output: {y_train[0].item():.3f}")
    
    # 2. Initialize models
    print("\n2. Initializing models...")
    baseline_model = BaselineMLP(input_dim=3, hidden_dim=16, output_dim=1)
    causal_model = CausalMLP(input_dim=3, hidden_dims=[16, 16], output_dim=1)
    
    print(f"   Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters())}")
    print(f"   Causal model parameters: {sum(p.numel() for p in causal_model.parameters())}")
    
    # 3. Train baseline model
    print("\n3. Training baseline model...")
    baseline_train_losses, baseline_test_losses = train_model(
        baseline_model, X_train, y_train, X_test, y_test, 
        is_causal=False, num_epochs=100, verbose=True
    )
    
    # 4. Train causal model
    print("\n4. Training causal model with do(x2=0.5) interventions...")
    causal_train_losses, causal_test_losses = train_model(
        causal_model, X_train, y_train, X_test, y_test, 
        is_causal=True, num_epochs=100, verbose=True
    )
    
    # 5. Evaluate and compare models
    evaluate_models(causal_model, baseline_model, X_test, y_test, true_coeffs)
    
    # 6. Generate and save plots
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    plot_training_curves(
        baseline_train_losses, baseline_test_losses,
        causal_train_losses, causal_test_losses,
        filename="phase1_training_summary.png"
    )
    
    # 7. Summary
    print("\n" + "="*50)
    print("PHASE 1 EXPERIMENT COMPLETE")
    print("="*50)
    
    final_baseline_loss = baseline_test_losses[-1]
    final_causal_loss = causal_test_losses[-1]
    
    print(f"Baseline model final test loss: {final_baseline_loss:.4f}")
    print(f"Causal model final test loss: {final_causal_loss:.4f}")
    
    if final_causal_loss < final_baseline_loss:
        improvement = ((final_baseline_loss - final_causal_loss) / final_baseline_loss * 100)
        print(f"Causal model achieved {improvement:.2f}% improvement!")
    else:
        print("Results show baseline performance (may vary with random seed)")
    
    print("\nReady for Phase 2: Counterfactual learning and dynamic DAG discovery!")
    
    return {
        'baseline_train_losses': baseline_train_losses,
        'baseline_test_losses': baseline_test_losses,
        'causal_train_losses': causal_train_losses,
        'causal_test_losses': causal_test_losses,
        'final_baseline_loss': final_baseline_loss,
        'final_causal_loss': final_causal_loss,
        'true_coeffs': true_coeffs
    }

if __name__ == "__main__":
    results = run_phase1_experiment() 