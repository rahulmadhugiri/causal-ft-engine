#!/usr/bin/env python3
"""
Phase 2 Focused Test: Diagnose and Fix Structure Learning Issues

This script focuses on diagnosing why structure learning is failing
and provides targeted fixes to achieve perfect structure recovery.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.structure_learning import StructureLearner, DifferentiableDAG
from engine.counterfactuals import CounterfactualSimulator
from engine.causal_unit import CausalMLP
from engine.loss_functions import CausalMetrics


def create_simple_chain_dag(num_variables=3):
    """Create a simple chain DAG: X1 → X2 → X3"""
    adjacency = torch.zeros(num_variables, num_variables)
    for i in range(num_variables - 1):
        adjacency[i, i + 1] = 1.0
    return adjacency


def generate_simple_chain_data(n_samples=100, noise_std=0.05):
    """Generate clean chain data with low noise"""
    X1 = torch.randn(n_samples, 1)
    X2 = 2.0 * X1 + noise_std * torch.randn(n_samples, 1)
    X3 = -1.5 * X2 + noise_std * torch.randn(n_samples, 1)
    
    X = torch.cat([X1, X2, X3], dim=1)
    true_adjacency = create_simple_chain_dag(3)
    
    return X, true_adjacency


def diagnose_structure_learning():
    """Diagnose why structure learning is failing"""
    print("=== DIAGNOSING STRUCTURE LEARNING ISSUES ===")
    
    # Generate data
    X, true_adjacency = generate_simple_chain_data(n_samples=200)
    
    print(f"Data shape: {X.shape}")
    print(f"Data statistics:")
    print(f"  Mean: {X.mean(dim=0)}")
    print(f"  Std: {X.std(dim=0)}")
    print(f"  Min: {X.min(dim=0)[0]}")
    print(f"  Max: {X.max(dim=0)[0]}")
    
    print(f"\nTrue adjacency matrix:")
    print(true_adjacency.numpy())
    
    # Test DifferentiableDAG directly
    print(f"\n--- Testing DifferentiableDAG directly ---")
    dag_model = DifferentiableDAG(num_variables=3, hidden_dim=16)
    
    # Forward pass
    reconstruction, adjacency = dag_model(X, hard_adjacency=False)
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Adjacency shape: {adjacency.shape}")
    print(f"Soft adjacency matrix:\n{adjacency.detach().numpy()}")
    
    # Get hard adjacency
    hard_adjacency = dag_model.get_adjacency_matrix(hard=True)
    print(f"Hard adjacency matrix:\n{hard_adjacency.detach().numpy()}")
    
    # Test reconstruction error
    print(f"Reconstruction error: {torch.mean((X - reconstruction)**2).item():.6f}")
    
    # Test training step by step
    print(f"\n--- Testing training step by step ---")
    optimizer = optim.Adam(dag_model.parameters(), lr=0.01)
    
    print("Before training:")
    print(f"  Adjacency: {dag_model.get_adjacency_matrix(hard=True).detach().numpy()}")
    
    for step in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        reconstruction, soft_adjacency = dag_model(X, hard_adjacency=False)
        
        # Reconstruction loss
        recon_loss = torch.mean((X - reconstruction)**2)
        
        # Sparsity loss
        sparsity_loss = torch.mean(soft_adjacency)
        
        # Total loss
        total_loss = recon_loss + 0.1 * sparsity_loss
        
        total_loss.backward()
        optimizer.step()
        
        if step % 2 == 0:
            hard_adj = dag_model.get_adjacency_matrix(hard=True)
            print(f"Step {step}: Loss={total_loss.item():.6f}, "
                  f"Edges={hard_adj.sum().item():.0f}")
    
    final_adjacency = dag_model.get_adjacency_matrix(hard=True)
    print(f"\nAfter 10 steps:")
    print(f"Final adjacency:\n{final_adjacency.detach().numpy()}")
    
    return dag_model, X, true_adjacency


def test_structure_learner():
    """Test StructureLearner with different configurations"""
    print(f"\n=== TESTING STRUCTURE LEARNER ===")
    
    X, true_adjacency = generate_simple_chain_data(n_samples=300)
    
    # Test different configurations
    configs = [
        {"lr": 0.01, "lambda_acyclic": 0.1, "lambda_sparse": 0.01, "name": "Default"},
        {"lr": 0.005, "lambda_acyclic": 0.1, "lambda_sparse": 0.05, "name": "Higher Sparsity"},
        {"lr": 0.02, "lambda_acyclic": 0.2, "lambda_sparse": 0.02, "name": "Higher LR"},
        {"lr": 0.001, "lambda_acyclic": 0.05, "lambda_sparse": 0.001, "name": "Low Constraints"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n--- Testing {config['name']} ---")
        
        structure_learner = StructureLearner(num_variables=3, hidden_dim=16)
        
        learned_adjacency, history = structure_learner.learn_structure(
            X,
            num_epochs=200,
            lr=config["lr"],
            lambda_acyclic=config["lambda_acyclic"],
            lambda_sparse=config["lambda_sparse"],
            verbose=False
        )
        
        metrics = structure_learner.evaluate_structure_recovery(learned_adjacency, true_adjacency)
        
        print(f"Results: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
              f"F1={metrics['f1_score']:.3f}, Edges={metrics['learned_edges']:.0f}")
        print(f"Learned adjacency:\n{learned_adjacency.numpy()}")
        
        results.append({
            'config': config,
            'metrics': metrics,
            'learned_adjacency': learned_adjacency,
            'history': history
        })
    
    return results


def manual_structure_learning():
    """Manual structure learning with careful debugging"""
    print(f"\n=== MANUAL STRUCTURE LEARNING ===")
    
    X, true_adjacency = generate_simple_chain_data(n_samples=500)
    
    print(f"True adjacency:\n{true_adjacency.numpy()}")
    
    # Manual implementation
    class SimpleStructureLearner(nn.Module):
        def __init__(self, num_variables):
            super().__init__()
            self.num_variables = num_variables
            # Initialize adjacency parameters
            self.adjacency_logits = nn.Parameter(torch.randn(num_variables, num_variables) * 0.1)
            # Zero out diagonal
            with torch.no_grad():
                self.adjacency_logits.fill_diagonal_(0)
        
        def get_adjacency(self, hard=False):
            # Zero diagonal
            adj_logits = self.adjacency_logits.clone()
            adj_logits.fill_diagonal_(0)
            
            if hard:
                return (torch.sigmoid(adj_logits) > 0.5).float()
            else:
                return torch.sigmoid(adj_logits)
        
        def forward(self, X):
            adjacency = self.get_adjacency(hard=False)
            
            # Simple linear reconstruction: X_reconstructed = X @ adjacency
            reconstruction = torch.matmul(X, adjacency)
            
            return reconstruction, adjacency
    
    model = SimpleStructureLearner(3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training manual structure learner...")
    
    for epoch in range(500):
        optimizer.zero_grad()
        
        reconstruction, adjacency = model(X)
        
        # Reconstruction loss
        recon_loss = torch.mean((X - reconstruction)**2)
        
        # Sparsity loss
        sparsity_loss = torch.mean(adjacency)
        
        # Acyclicity loss (simple version)
        acyc_loss = torch.trace(torch.matrix_power(adjacency, 3))  # Penalize 3-cycles
        
        # Total loss
        total_loss = recon_loss + 0.1 * sparsity_loss + 0.1 * acyc_loss
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            hard_adj = model.get_adjacency(hard=True)
            print(f"Epoch {epoch}: Loss={total_loss.item():.6f}, "
                  f"Recon={recon_loss.item():.6f}, "
                  f"Edges={hard_adj.sum().item():.0f}")
            if epoch % 200 == 0:
                print(f"  Adjacency:\n{hard_adj.detach().numpy()}")
    
    final_adjacency = model.get_adjacency(hard=True)
    print(f"\nFinal adjacency:\n{final_adjacency.detach().numpy()}")
    
    # Evaluate
    from engine.loss_functions import CausalMetrics
    metrics = CausalMetrics.structure_recovery_metrics(final_adjacency, true_adjacency)
    print(f"Final metrics: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
          f"F1={metrics['f1_score']:.3f}")
    
    return model, final_adjacency


def test_counterfactual_on_perfect_structure():
    """Test counterfactual reasoning with perfect structure knowledge"""
    print(f"\n=== TESTING COUNTERFACTUAL WITH PERFECT STRUCTURE ===")
    
    X, true_adjacency = generate_simple_chain_data(n_samples=300)
    y = X[:, 2:3]  # Predict X3
    
    # Train model with perfect structure knowledge
    model = CausalMLP(input_dim=3, hidden_dims=[32, 16], output_dim=1)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training model to predict X3...")
    for epoch in range(200):
        optimizer.zero_grad()
        pred = model(X)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.6f}")
    
    # Test counterfactual reasoning
    print(f"\nTesting counterfactual reasoning...")
    cf_simulator = CounterfactualSimulator(model)
    
    test_x = X[:20]
    
    # Test interventions
    interventions = [
        (0, -1.0), (0, 0.0), (0, 1.0), (0, 2.0),  # X1 interventions
        (1, -1.0), (1, 0.0), (1, 1.0), (1, 2.0),  # X2 interventions
        (2, -1.0), (2, 0.0), (2, 1.0), (2, 2.0),  # X3 interventions
    ]
    
    perfect_count = 0
    total_count = 0
    
    for var_idx, val in interventions:
        do_mask = torch.zeros(3)
        do_mask[var_idx] = 1.0
        do_values = torch.zeros(3)
        do_values[var_idx] = val
        
        effect, _, _ = cf_simulator.compute_counterfactual_effect(test_x, do_mask, do_values)
        
        # Expected effect based on true chain structure
        if var_idx == 0:  # X1 intervention
            expected_effect = -3.0 * (val - test_x[:, 0])
        elif var_idx == 1:  # X2 intervention
            expected_effect = -1.5 * (val - test_x[:, 1])
        else:  # X3 intervention (no downstream effects)
            expected_effect = torch.zeros_like(effect.squeeze())
        
        # Compute correlation
        try:
            correlation = torch.corrcoef(torch.stack([effect.squeeze(), expected_effect]))[0, 1]
            correlation = abs(correlation.item()) if not torch.isnan(correlation) else 0.0
        except:
            correlation = 0.0
        
        status = "✅" if correlation >= 0.95 else "❌"
        print(f"  do(X{var_idx+1}={val:.1f}): {correlation:.4f} {status}")
        
        if correlation >= 0.95:
            perfect_count += 1
        total_count += 1
    
    success_rate = perfect_count / total_count
    print(f"\nCounterfactual Success Rate: {perfect_count}/{total_count} ({success_rate*100:.1f}%)")
    
    return success_rate


def run_focused_test():
    """Run focused diagnostic test"""
    print("PHASE 2 FOCUSED TEST: Diagnosing Structure Learning Issues")
    print("=" * 70)
    
    # 1. Diagnose structure learning
    dag_model, X, true_adjacency = diagnose_structure_learning()
    
    # 2. Test structure learner configurations
    config_results = test_structure_learner()
    
    # 3. Manual structure learning
    manual_model, manual_adjacency = manual_structure_learning()
    
    # 4. Test counterfactual with perfect structure
    cf_success_rate = test_counterfactual_on_perfect_structure()
    
    # 5. Summary
    print(f"\n" + "=" * 70)
    print("FOCUSED TEST SUMMARY")
    print("=" * 70)
    
    print(f"\nStructure Learning Results:")
    best_config = max(config_results, key=lambda x: x['metrics']['f1_score'])
    print(f"  Best config: {best_config['config']['name']}")
    print(f"  Best F1: {best_config['metrics']['f1_score']:.3f}")
    print(f"  Best Precision: {best_config['metrics']['precision']:.3f}")
    print(f"  Best Recall: {best_config['metrics']['recall']:.3f}")
    
    print(f"\nManual Structure Learning:")
    manual_metrics = CausalMetrics.structure_recovery_metrics(manual_adjacency, true_adjacency)
    print(f"  Precision: {manual_metrics['precision']:.3f}")
    print(f"  Recall: {manual_metrics['recall']:.3f}")
    print(f"  F1-Score: {manual_metrics['f1_score']:.3f}")
    
    print(f"\nCounterfactual Performance:")
    print(f"  Success Rate: {cf_success_rate*100:.1f}%")
    
    # Recommendations
    print(f"\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if best_config['metrics']['f1_score'] < 0.8:
        print("❌ Structure learning needs improvement:")
        print("  - Try different reconstruction methods")
        print("  - Adjust sparsity/acyclicity constraints")
        print("  - Use curriculum learning")
    else:
        print("✅ Structure learning is working well")
    
    if cf_success_rate < 0.9:
        print("❌ Counterfactual reasoning needs improvement:")
        print("  - Ensure model captures true causal relationships")
        print("  - Use better training with counterfactual loss")
    else:
        print("✅ Counterfactual reasoning is excellent")
    
    return {
        'config_results': config_results,
        'manual_results': manual_metrics,
        'cf_success_rate': cf_success_rate
    }


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = run_focused_test() 