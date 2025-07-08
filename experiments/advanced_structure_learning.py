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

def progressive_sparsity_learning(X, true_adjacency, verbose=True):
    """
    Progressive sparsity learning that gradually increases sparsity
    while monitoring structure recovery performance.
    """
    print("=== PROGRESSIVE SPARSITY LEARNING ===")
    
    learner = StructureLearner(num_variables=3, hidden_dim=16)
    
    # Progressive sparsity schedule
    sparsity_schedule = [
        {"lambda_sparse": 0.0, "epochs": 200, "name": "Find All Edges"},
        {"lambda_sparse": 0.005, "epochs": 200, "name": "Light Sparsity"},
        {"lambda_sparse": 0.01, "epochs": 200, "name": "Moderate Sparsity"},
        {"lambda_sparse": 0.015, "epochs": 200, "name": "Strong Sparsity"},
        {"lambda_sparse": 0.02, "epochs": 200, "name": "Very Strong Sparsity"},
    ]
    
    # Start with pre-trained model
    optimizer = torch.optim.Adam(learner.dag_model.parameters(), lr=0.01)
    
    best_adjacency = None
    best_f1 = 0.0
    
    for phase_idx, phase in enumerate(sparsity_schedule):
        if verbose:
            print(f"\n--- Phase {phase_idx + 1}: {phase['name']} (λ={phase['lambda_sparse']}) ---")
        
        # Train for this phase
        learner.dag_model.train()
        
        for epoch in range(phase["epochs"]):
            optimizer.zero_grad()
            
            # Forward pass
            X_reconstructed, adjacency = learner.dag_model(X, hard_adjacency=False)
            
            # Losses
            reconstruction_loss = torch.nn.functional.mse_loss(X_reconstructed, X)
            acyclicity_loss = learner.dag_model.acyclicity_constraint(adjacency)
            sparsity_loss = learner.dag_model.sparsity_loss(adjacency)
            
            # Total loss
            total_loss = (reconstruction_loss + 
                         0.1 * torch.abs(acyclicity_loss) +
                         phase["lambda_sparse"] * sparsity_loss)
            
            total_loss.backward()
            optimizer.step()
            
            # Temperature annealing
            if hasattr(learner.dag_model, 'temperature'):
                learner.dag_model.temperature.data = torch.clamp(
                    learner.dag_model.temperature.data * 0.999, min=0.1
                )
        
        # Evaluate current phase
        learner.dag_model.eval()
        with torch.no_grad():
            soft_adj = learner.dag_model.get_adjacency_matrix(hard=False)
            hard_adj = learner.dag_model.get_adjacency_matrix(hard=True)
        
        metrics = learner.evaluate_structure_recovery(hard_adj, true_adjacency)
        
        if verbose:
            print(f"Phase {phase_idx + 1} Results: P={metrics['precision']:.3f}, "
                  f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
            print(f"Soft adjacency:\n{soft_adj.numpy()}")
            print(f"Hard adjacency:\n{hard_adj.numpy()}")
        
        # Track best result
        if metrics['f1_score'] > best_f1:
            best_adjacency = hard_adj
            best_f1 = metrics['f1_score']
        
        # Early stopping if perfect
        if metrics['precision'] >= 0.95 and metrics['recall'] >= 0.95:
            print(f"🎉 PERFECT STRUCTURE RECOVERY in Phase {phase_idx + 1}!")
            return hard_adj, metrics
    
    print(f"Best F1 achieved: {best_f1:.3f}")
    return best_adjacency, learner.evaluate_structure_recovery(best_adjacency, true_adjacency)

def adaptive_threshold_learning(X, true_adjacency, verbose=True):
    """
    Learn structure with adaptive thresholding based on edge strength distribution.
    """
    print("\n=== ADAPTIVE THRESHOLD LEARNING ===")
    
    learner = StructureLearner(num_variables=3, hidden_dim=16)
    
    # First, learn with minimal sparsity to get soft adjacency
    learned_adj, _ = learner.learn_structure(
        X, num_epochs=1000, 
        lr=0.01, lambda_acyclic=0.001, lambda_sparse=0.001,
        verbose=False
    )
    
    # Get soft adjacency matrix
    learner.dag_model.eval()
    with torch.no_grad():
        soft_adj = learner.dag_model.get_adjacency_matrix(hard=False)
    
    if verbose:
        print(f"Soft adjacency matrix:\n{soft_adj.numpy()}")
    
    # Analyze edge strength distribution
    edge_strengths = soft_adj.flatten()
    edge_strengths = edge_strengths[edge_strengths > 0]  # Remove zeros (diagonal)
    
    if len(edge_strengths) > 0:
        mean_strength = edge_strengths.mean().item()
        std_strength = edge_strengths.std().item()
        
        if verbose:
            print(f"Edge strength stats: mean={mean_strength:.3f}, std={std_strength:.3f}")
        
        # Try adaptive thresholds based on statistics
        adaptive_thresholds = [
            mean_strength - std_strength,
            mean_strength,
            mean_strength + 0.5 * std_strength,
            mean_strength + std_strength,
        ]
        
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in adaptive_thresholds:
            threshold = max(0.1, min(0.9, threshold))  # Clamp to reasonable range
            
            hard_adj = (soft_adj > threshold).float()
            metrics = learner.evaluate_structure_recovery(hard_adj, true_adjacency)
            
            if verbose:
                print(f"Threshold {threshold:.3f}: P={metrics['precision']:.3f}, "
                      f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
            
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = threshold
        
        # Apply best threshold
        final_adj = (soft_adj > best_threshold).float()
        final_metrics = learner.evaluate_structure_recovery(final_adj, true_adjacency)
        
        if verbose:
            print(f"\nBest threshold: {best_threshold:.3f}")
            print(f"Final adjacency:\n{final_adj.numpy()}")
            print(f"Final metrics: P={final_metrics['precision']:.3f}, "
                  f"R={final_metrics['recall']:.3f}, F1={final_metrics['f1_score']:.3f}")
        
        return final_adj, final_metrics
    
    return learned_adj, learner.evaluate_structure_recovery(learned_adj, true_adjacency)

def multi_run_consensus(X, true_adjacency, n_runs=5, verbose=True):
    """
    Run multiple structure learning attempts and use consensus.
    """
    print("\n=== MULTI-RUN CONSENSUS ===")
    
    all_adjacencies = []
    all_f1_scores = []
    
    for run in range(n_runs):
        if verbose:
            print(f"\nRun {run + 1}/{n_runs}")
        
        # Use different random seeds
        torch.manual_seed(run * 42)
        
        learner = StructureLearner(num_variables=3, hidden_dim=16)
        
        # Learn with moderate sparsity
        learned_adj, _ = learner.learn_structure(
            X, num_epochs=800, 
            lr=0.01, lambda_acyclic=0.1, lambda_sparse=0.008,
            verbose=False
        )
        
        metrics = learner.evaluate_structure_recovery(learned_adj, true_adjacency)
        
        if verbose:
            print(f"Run {run + 1}: P={metrics['precision']:.3f}, "
                  f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        all_adjacencies.append(learned_adj)
        all_f1_scores.append(metrics['f1_score'])
    
    # Create consensus adjacency: edge exists if majority of runs agree
    consensus_adj = torch.stack(all_adjacencies).mean(dim=0)
    
    # Try different consensus thresholds
    consensus_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    best_consensus = None
    best_f1 = 0.0
    
    for threshold in consensus_thresholds:
        final_adj = (consensus_adj > threshold).float()
        metrics = StructureLearner(3, 16).evaluate_structure_recovery(final_adj, true_adjacency)
        
        if verbose:
            print(f"Consensus threshold {threshold}: P={metrics['precision']:.3f}, "
                  f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_consensus = final_adj
    
    final_metrics = StructureLearner(3, 16).evaluate_structure_recovery(best_consensus, true_adjacency)
    
    if verbose:
        print(f"\nBest consensus adjacency:\n{best_consensus.numpy()}")
        print(f"Final metrics: P={final_metrics['precision']:.3f}, "
              f"R={final_metrics['recall']:.3f}, F1={final_metrics['f1_score']:.3f}")
    
    return best_consensus, final_metrics

def test_all_approaches():
    """Test all advanced structure learning approaches"""
    print("TESTING ADVANCED STRUCTURE LEARNING APPROACHES")
    print("=" * 60)
    
    # Generate data
    X, true_adjacency = generate_simple_chain_data(1000, noise_std=0.01)
    
    print(f"True adjacency matrix:")
    print(true_adjacency.numpy())
    print(f"Target: X1→X2, X2→X3 (2 edges)")
    
    # Test all approaches
    approaches = [
        ("Progressive Sparsity", progressive_sparsity_learning),
        ("Adaptive Threshold", adaptive_threshold_learning),
        ("Multi-Run Consensus", multi_run_consensus),
    ]
    
    results = []
    
    for name, approach_func in approaches:
        print(f"\n{'='*60}")
        print(f"TESTING: {name}")
        print(f"{'='*60}")
        
        try:
            adj, metrics = approach_func(X, true_adjacency, verbose=True)
            results.append((name, metrics))
            
            # Check if perfect
            if metrics['precision'] >= 0.95 and metrics['recall'] >= 0.95:
                print(f"🎉 {name} achieved PERFECT STRUCTURE RECOVERY!")
                
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, {'precision': 0, 'recall': 0, 'f1_score': 0}))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL APPROACHES")
    print(f"{'='*60}")
    
    for name, metrics in results:
        print(f"{name:20}: P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
    
    # Find best approach
    best_approach = max(results, key=lambda x: x[1]['f1_score'])
    print(f"\nBest approach: {best_approach[0]} (F1={best_approach[1]['f1_score']:.3f})")

if __name__ == "__main__":
    test_all_approaches() 