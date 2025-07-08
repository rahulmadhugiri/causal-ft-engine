import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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

def detailed_structure_analysis():
    """Detailed analysis of what edges are being learned"""
    print("=== DETAILED STRUCTURE LEARNING ANALYSIS ===")
    
    # Generate data
    X, true_adjacency = generate_simple_chain_data(1000, noise_std=0.01)
    
    print(f"True adjacency matrix:")
    print(true_adjacency.numpy())
    print(f"Expected edges: X1→X2, X2→X3 (2 edges total)")
    
    # Test multiple learning approaches
    approaches = [
        {"name": "Standard", "lr": 0.01, "lambda_sparse": 0.1, "lambda_acyclic": 0.1},
        {"name": "High Sparsity", "lr": 0.005, "lambda_sparse": 0.3, "lambda_acyclic": 0.2},
        {"name": "Ultra Sparsity", "lr": 0.002, "lambda_sparse": 0.5, "lambda_acyclic": 0.3},
        {"name": "Low Sparsity", "lr": 0.01, "lambda_sparse": 0.05, "lambda_acyclic": 0.05},
    ]
    
    for approach in approaches:
        print(f"\n--- {approach['name']} Approach ---")
        
        learner = StructureLearner(num_variables=3, hidden_dim=16)
        learned_adj, history = learner.learn_structure(
            X, num_epochs=1000, 
            lr=approach['lr'],
            lambda_acyclic=approach['lambda_acyclic'],
            lambda_sparse=approach['lambda_sparse'],
            verbose=False
        )
        
        print(f"Learned adjacency matrix:")
        print(learned_adj.numpy())
        
        # Analyze specific edges
        edges_learned = []
        for i in range(3):
            for j in range(3):
                if learned_adj[i, j] > 0.5:
                    edges_learned.append(f"X{i+1}→X{j+1}")
        
        edges_true = []
        for i in range(3):
            for j in range(3):
                if true_adjacency[i, j] > 0.5:
                    edges_true.append(f"X{i+1}→X{j+1}")
        
        print(f"True edges: {edges_true}")
        print(f"Learned edges: {edges_learned}")
        
        # Calculate metrics
        metrics = learner.evaluate_structure_recovery(learned_adj, true_adjacency)
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1: {metrics['f1_score']:.3f}")
        
        # Check if we have the right edges
        correct_edges = []
        missing_edges = []
        spurious_edges = []
        
        for edge in edges_true:
            if edge in edges_learned:
                correct_edges.append(edge)
            else:
                missing_edges.append(edge)
        
        for edge in edges_learned:
            if edge not in edges_true:
                spurious_edges.append(edge)
        
        print(f"✅ Correct edges: {correct_edges}")
        print(f"❌ Missing edges: {missing_edges}")
        print(f"⚠️ Spurious edges: {spurious_edges}")

def test_manual_threshold_tuning():
    """Test different thresholds for edge detection"""
    print("\n=== MANUAL THRESHOLD TUNING ===")
    
    X, true_adjacency = generate_simple_chain_data(1000, noise_std=0.01)
    
    learner = StructureLearner(num_variables=3, hidden_dim=16)
    
    # Learn with soft adjacency
    soft_adj, _ = learner.learn_structure(
        X, num_epochs=1000, 
        lr=0.005, lambda_acyclic=0.2, lambda_sparse=0.15,
        verbose=False
    )
    
    print(f"Soft adjacency matrix (before thresholding):")
    print(soft_adj.numpy())
    
    # Test different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        hard_adj = (soft_adj > threshold).float()
        metrics = learner.evaluate_structure_recovery(hard_adj, true_adjacency)
        
        print(f"Threshold {threshold}: P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold} (F1={best_f1:.3f})")
    
    # Show best result
    best_adj = (soft_adj > best_threshold).float()
    print(f"Best adjacency matrix (threshold={best_threshold}):")
    print(best_adj.numpy())

if __name__ == "__main__":
    detailed_structure_analysis()
    test_manual_threshold_tuning() 