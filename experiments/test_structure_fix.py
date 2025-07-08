import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.structure_learning import StructureLearner

def generate_simple_chain_data(n_samples=1000, noise_std=0.01):
    """Generate X1 → X2 → X3 chain data"""
    X1 = torch.randn(n_samples, 1)
    X2 = 0.8 * X1 + noise_std * torch.randn(n_samples, 1)
    X3 = 0.7 * X2 + noise_std * torch.randn(n_samples, 1)
    
    X = torch.cat([X1, X2, X3], dim=1)
    
    # True adjacency matrix: X1 → X2 → X3
    true_adjacency = torch.zeros(3, 3)
    true_adjacency[0, 1] = 1  # X1 → X2
    true_adjacency[1, 2] = 1  # X2 → X3
    
    return X, true_adjacency

def test_sparsity_progression():
    """Test structure learning with progressively increasing sparsity"""
    print("=== TESTING SPARSITY PROGRESSION ===")
    
    X, true_adjacency = generate_simple_chain_data(1000, noise_std=0.01)
    
    print(f"True adjacency matrix:")
    print(true_adjacency.numpy())
    
    # Test different sparsity levels
    sparsity_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    for sparsity in sparsity_levels:
        print(f"\n--- Sparsity λ = {sparsity} ---")
        
        learner = StructureLearner(num_variables=3, hidden_dim=16)
        
        # Learn with current sparsity level
        learned_adj, _ = learner.learn_structure(
            X, num_epochs=500, 
            lr=0.01, lambda_acyclic=0.1, lambda_sparse=sparsity,
            verbose=False
        )
        
        print(f"Learned adjacency matrix:")
        print(learned_adj.numpy())
        
        # Count edges
        num_edges = learned_adj.sum().item()
        print(f"Number of edges: {num_edges}")
        
        # Calculate metrics
        metrics = learner.evaluate_structure_recovery(learned_adj, true_adjacency)
        print(f"Precision: {metrics['precision']:.3f}, "
              f"Recall: {metrics['recall']:.3f}, "
              f"F1: {metrics['f1_score']:.3f}")

def test_no_regularization():
    """Test structure learning with minimal regularization"""
    print("\n=== TESTING NO REGULARIZATION ===")
    
    X, true_adjacency = generate_simple_chain_data(1000, noise_std=0.01)
    
    learner = StructureLearner(num_variables=3, hidden_dim=16)
    
    # Learn with minimal regularization
    learned_adj, history = learner.learn_structure(
        X, num_epochs=1000, 
        lr=0.01, lambda_acyclic=0.001, lambda_sparse=0.001,
        verbose=True
    )
    
    print(f"\nFinal learned adjacency matrix:")
    print(learned_adj.numpy())
    
    # Show soft adjacency too
    learner.dag_model.eval()
    with torch.no_grad():
        soft_adj = learner.dag_model.get_adjacency_matrix(hard=False)
    
    print(f"\nSoft adjacency matrix (before thresholding):")
    print(soft_adj.numpy())
    
    # Test different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        hard_adj = (soft_adj > threshold).float()
        metrics = learner.evaluate_structure_recovery(hard_adj, true_adjacency)
        print(f"Threshold {threshold}: P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

def test_reconstruction_only():
    """Test with only reconstruction loss (no regularization)"""
    print("\n=== TESTING RECONSTRUCTION ONLY ===")
    
    X, true_adjacency = generate_simple_chain_data(1000, noise_std=0.01)
    
    learner = StructureLearner(num_variables=3, hidden_dim=16)
    
    # Learn with only reconstruction loss
    learned_adj, history = learner.learn_structure(
        X, num_epochs=1000, 
        lr=0.01, lambda_acyclic=0.0, lambda_sparse=0.0,
        verbose=True
    )
    
    print(f"\nFinal learned adjacency matrix:")
    print(learned_adj.numpy())
    
    # Show soft adjacency
    learner.dag_model.eval()
    with torch.no_grad():
        soft_adj = learner.dag_model.get_adjacency_matrix(hard=False)
    
    print(f"\nSoft adjacency matrix:")
    print(soft_adj.numpy())

if __name__ == "__main__":
    test_sparsity_progression()
    test_no_regularization()
    test_reconstruction_only() 