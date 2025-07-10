#!/usr/bin/env python3
"""
Phase 4: Scaling and Robustness Benchmark

Goal: Test whether the causal approach has true "causal power" by evaluating:
1. Scaling to larger datasets (5k-10k samples)
2. Robustness to various noise conditions
3. Resistance to spurious correlations
4. Graceful degradation under adverse conditions

This will distinguish true causal generalization from clever regularization.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from phase4_soft_interventions import SoftInterventionNetwork
from experiments.utils import generate_synthetic_data


class VanillaMLP(nn.Module):
    """Standard MLP baseline for comparison."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, activation: str = 'relu'):
        super(VanillaMLP, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        
        # Hidden layers
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ScalingRobustnessBenchmark:
    """
    Comprehensive benchmark for scaling and robustness testing.
    """
    
    def __init__(self, 
                 input_dim: int = 3,
                 hidden_dims: List[int] = [32, 16],
                 output_dim: int = 1,
                 activation: str = 'relu',
                 num_seeds: int = 3,
                 num_epochs: int = 100,
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.num_seeds = num_seeds
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        # Scaling test sizes
        self.scaling_sizes = [1000, 2500, 5000, 7500, 10000]
        
        # Noise test configurations
        self.noise_configs = {
            'clean': {'label_noise': 0.0, 'feature_noise': 0.0, 'irrelevant_features': 0},
            'light_noise': {'label_noise': 0.05, 'feature_noise': 0.1, 'irrelevant_features': 2},
            'moderate_noise': {'label_noise': 0.10, 'feature_noise': 0.2, 'irrelevant_features': 5},
            'heavy_noise': {'label_noise': 0.20, 'feature_noise': 0.4, 'irrelevant_features': 10},
            'extreme_noise': {'label_noise': 0.35, 'feature_noise': 0.6, 'irrelevant_features': 15}
        }
        
        # Results storage
        self.results = {
            'scaling_test': {},
            'noise_robustness': {},
            'spurious_correlation_test': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'input_dim': input_dim,
                    'hidden_dims': hidden_dims,
                    'output_dim': output_dim,
                    'scaling_sizes': self.scaling_sizes,
                    'noise_configs': self.noise_configs,
                    'num_seeds': num_seeds,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate
                }
            }
        }
    
    def add_noise_to_data(self, x: torch.Tensor, y: torch.Tensor, 
                         label_noise: float, feature_noise: float, 
                         irrelevant_features: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add various types of noise to the data."""
        
        batch_size = x.shape[0]
        
        # Add irrelevant features (spurious correlations)
        if irrelevant_features > 0:
            # Create irrelevant features that are randomly correlated with the output
            irrelevant = torch.randn(batch_size, irrelevant_features, device=self.device)
            # Make some irrelevant features spuriously correlated
            correlation_strength = 0.3
            for i in range(min(3, irrelevant_features)):  # Correlate first 3 irrelevant features
                irrelevant[:, i] += correlation_strength * y.squeeze() + torch.randn(batch_size, device=self.device) * 0.1
            
            # Concatenate irrelevant features
            x = torch.cat([x, irrelevant], dim=1)
        
        # Add feature noise (Gaussian noise to inputs)
        if feature_noise > 0:
            feature_noise_tensor = torch.randn_like(x) * feature_noise
            x = x + feature_noise_tensor
        
        # Add label noise (flip some labels)
        if label_noise > 0:
            noise_mask = torch.rand(batch_size, device=self.device) < label_noise
            y_noisy = y.clone()
            if noise_mask.any():
                # Add random noise to noisy labels
                y_noisy[noise_mask] += torch.randn(noise_mask.sum(), 1, device=self.device) * y.std()
            y = y_noisy
        
        return x, y
    
    def create_spurious_correlation_data(self, size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create data with strong spurious correlations to test causal robustness."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate base causal data
        x_np, y_np, true_adjacency_np = generate_synthetic_data(
            size, n_nodes=self.input_dim + 1, graph_type='chain', noise_level=0.2
        )
        
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        true_adjacency = torch.tensor(true_adjacency_np, dtype=torch.float32, device=self.device)
        
        # Add spurious correlates - features that correlate with y but aren't causal
        num_spurious = 5
        spurious_features = torch.randn(size, num_spurious, device=self.device)
        
        # Make spurious features strongly correlated with output
        for i in range(num_spurious):
            correlation_strength = 0.6 + 0.3 * np.random.random()  # Strong correlation
            spurious_features[:, i] = (correlation_strength * y.squeeze() + 
                                     torch.randn(size, device=self.device) * 0.3)
        
        # Concatenate spurious features
        x_with_spurious = torch.cat([x, spurious_features], dim=1)
        
        return x_with_spurious, y, true_adjacency
    
    def train_model(self, model_type: str, x: torch.Tensor, y: torch.Tensor, seed: int) -> Dict:
        """Train a model and return metrics."""
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Adjust input dimension for model creation
        actual_input_dim = x.shape[1]
        
        # Create model
        if model_type == 'vanilla':
            model = VanillaMLP(
                input_dim=actual_input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                activation=self.activation
            ).to(self.device)
        else:  # soft_intervention
            model = SoftInterventionNetwork(
                input_dim=actual_input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                activation=self.activation,
                lambda_reg=0.01
            ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Split data: 80% train, 20% validation
        n_train = int(0.8 * len(x))
        indices = torch.randperm(len(x))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            if model_type == 'soft_intervention':
                # Create interventions for causal training
                interventions = self.create_interventions(x_train)
                pred = model(x_train, interventions=interventions)
            else:
                pred = model(x_train)
            
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                if model_type == 'soft_intervention':
                    val_pred = model(x_val)  # No interventions for validation
                else:
                    val_pred = model(x_val)
                val_loss = criterion(val_pred, y_val)
                val_losses.append(val_loss.item())
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'min_val_loss': min(val_losses),
            'model_params': sum(p.numel() for p in model.parameters())
        }
    
    def create_interventions(self, x: torch.Tensor, intervention_prob: float = 0.3) -> List[Dict]:
        """Create random interventions for causal network training."""
        batch_size = x.shape[0]
        actual_input_dim = min(self.input_dim, x.shape[1])  # Only intervene on original causal features
        interventions = []
        
        for i in range(batch_size):
            if np.random.random() < intervention_prob:
                mask = torch.zeros(x.shape[1])  # Full feature dimension
                values = torch.zeros(x.shape[1])
                
                # Only intervene on original causal features
                node_idx = np.random.randint(0, actual_input_dim)
                mask[node_idx] = 1.0
                values[node_idx] = np.random.normal(0, 1)
                
                interventions.append({'random': (mask, values)})
            else:
                interventions.append({})
        
        return interventions
    
    def run_scaling_test(self):
        """Test performance scaling with dataset size."""
        
        print("=== RUNNING SCALING TEST ===")
        
        for size in self.scaling_sizes:
            print(f"\nTesting with {size} samples...")
            
            self.results['scaling_test'][size] = {
                'vanilla': [],
                'soft_intervention': []
            }
            
            for seed in range(self.num_seeds):
                print(f"  Seed {seed + 1}/{self.num_seeds}")
                
                # Generate dataset
                x_np, y_np, _ = generate_synthetic_data(size, n_nodes=self.input_dim + 1, 
                                                     graph_type='chain', noise_level=0.3)
                x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
                y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
                
                # Train vanilla MLP
                vanilla_results = self.train_model('vanilla', x, y, seed)
                self.results['scaling_test'][size]['vanilla'].append(vanilla_results)
                
                # Train soft intervention network
                soft_results = self.train_model('soft_intervention', x, y, seed)
                self.results['scaling_test'][size]['soft_intervention'].append(soft_results)
                
                print(f"    Vanilla: {vanilla_results['final_val_loss']:.4f}, "
                      f"Soft: {soft_results['final_val_loss']:.4f}")
    
    def run_noise_robustness_test(self):
        """Test robustness to various noise conditions."""
        
        print("\n=== RUNNING NOISE ROBUSTNESS TEST ===")
        
        base_size = 2000  # Fixed size for noise testing
        
        for noise_name, noise_config in self.noise_configs.items():
            print(f"\nTesting {noise_name} condition...")
            print(f"  Label noise: {noise_config['label_noise']}")
            print(f"  Feature noise: {noise_config['feature_noise']}")
            print(f"  Irrelevant features: {noise_config['irrelevant_features']}")
            
            self.results['noise_robustness'][noise_name] = {
                'vanilla': [],
                'soft_intervention': []
            }
            
            for seed in range(self.num_seeds):
                print(f"  Seed {seed + 1}/{self.num_seeds}")
                
                # Generate base dataset
                x_np, y_np, _ = generate_synthetic_data(base_size, n_nodes=self.input_dim + 1,
                                                     graph_type='chain', noise_level=0.3)
                x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
                y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
                
                # Add noise
                x_noisy, y_noisy = self.add_noise_to_data(
                    x, y, 
                    noise_config['label_noise'],
                    noise_config['feature_noise'],
                    noise_config['irrelevant_features']
                )
                
                # Train vanilla MLP
                vanilla_results = self.train_model('vanilla', x_noisy, y_noisy, seed)
                self.results['noise_robustness'][noise_name]['vanilla'].append(vanilla_results)
                
                # Train soft intervention network
                soft_results = self.train_model('soft_intervention', x_noisy, y_noisy, seed)
                self.results['noise_robustness'][noise_name]['soft_intervention'].append(soft_results)
                
                print(f"    Vanilla: {vanilla_results['final_val_loss']:.4f}, "
                      f"Soft: {soft_results['final_val_loss']:.4f}")
    
    def run_spurious_correlation_test(self):
        """Test resistance to spurious correlations."""
        
        print("\n=== RUNNING SPURIOUS CORRELATION TEST ===")
        
        test_size = 2000
        
        self.results['spurious_correlation_test'] = {
            'vanilla': [],
            'soft_intervention': []
        }
        
        for seed in range(self.num_seeds):
            print(f"  Seed {seed + 1}/{self.num_seeds}")
            
            # Generate data with spurious correlations
            x_spurious, y, _ = self.create_spurious_correlation_data(test_size, seed)
            
            # Train vanilla MLP
            vanilla_results = self.train_model('vanilla', x_spurious, y, seed)
            self.results['spurious_correlation_test']['vanilla'].append(vanilla_results)
            
            # Train soft intervention network
            soft_results = self.train_model('soft_intervention', x_spurious, y, seed)
            self.results['spurious_correlation_test']['soft_intervention'].append(soft_results)
            
            print(f"    Vanilla: {vanilla_results['final_val_loss']:.4f}, "
                  f"Soft: {soft_results['final_val_loss']:.4f}")
    
    def compute_summary_statistics(self):
        """Compute summary statistics for all tests."""
        
        summary = {
            'scaling_test': {},
            'noise_robustness': {},
            'spurious_correlation_test': {}
        }
        
        # Scaling test summary
        for size in self.scaling_sizes:
            summary['scaling_test'][size] = {}
            for model_type in ['vanilla', 'soft_intervention']:
                results = self.results['scaling_test'][size][model_type]
                val_losses = [r['final_val_loss'] for r in results]
                min_val_losses = [r['min_val_loss'] for r in results]
                
                summary['scaling_test'][size][model_type] = {
                    'mean_val_loss': np.mean(val_losses),
                    'std_val_loss': np.std(val_losses),
                    'mean_min_val_loss': np.mean(min_val_losses),
                    'std_min_val_loss': np.std(min_val_losses)
                }
        
        # Noise robustness summary
        for noise_name in self.noise_configs.keys():
            summary['noise_robustness'][noise_name] = {}
            for model_type in ['vanilla', 'soft_intervention']:
                results = self.results['noise_robustness'][noise_name][model_type]
                val_losses = [r['final_val_loss'] for r in results]
                
                summary['noise_robustness'][noise_name][model_type] = {
                    'mean_val_loss': np.mean(val_losses),
                    'std_val_loss': np.std(val_losses)
                }
        
        # Spurious correlation summary
        for model_type in ['vanilla', 'soft_intervention']:
            results = self.results['spurious_correlation_test'][model_type]
            val_losses = [r['final_val_loss'] for r in results]
            
            summary['spurious_correlation_test'][model_type] = {
                'mean_val_loss': np.mean(val_losses),
                'std_val_loss': np.std(val_losses)
            }
        
        self.results['summary'] = summary
        
        # Print scaling summary
        print("\n=== SCALING TEST SUMMARY ===")
        print("Size\tVanilla MLP\t\tSoft Intervention\tImprovement")
        print("-" * 70)
        
        for size in self.scaling_sizes:
            vanilla_mean = summary['scaling_test'][size]['vanilla']['mean_val_loss']
            soft_mean = summary['scaling_test'][size]['soft_intervention']['mean_val_loss']
            improvement = ((vanilla_mean - soft_mean) / vanilla_mean) * 100
            
            print(f"{size}\t{vanilla_mean:.4f}\t\t{soft_mean:.4f}\t\t{improvement:+.1f}%")
        
        # Print noise robustness summary
        print("\n=== NOISE ROBUSTNESS SUMMARY ===")
        print("Condition\t\tVanilla MLP\t\tSoft Intervention\tImprovement")
        print("-" * 80)
        
        for noise_name in self.noise_configs.keys():
            vanilla_mean = summary['noise_robustness'][noise_name]['vanilla']['mean_val_loss']
            soft_mean = summary['noise_robustness'][noise_name]['soft_intervention']['mean_val_loss']
            improvement = ((vanilla_mean - soft_mean) / vanilla_mean) * 100
            
            print(f"{noise_name:15}\t{vanilla_mean:.4f}\t\t{soft_mean:.4f}\t\t{improvement:+.1f}%")
        
        # Print spurious correlation summary
        print("\n=== SPURIOUS CORRELATION SUMMARY ===")
        vanilla_mean = summary['spurious_correlation_test']['vanilla']['mean_val_loss']
        soft_mean = summary['spurious_correlation_test']['soft_intervention']['mean_val_loss']
        improvement = ((vanilla_mean - soft_mean) / vanilla_mean) * 100
        
        print(f"Vanilla MLP: {vanilla_mean:.4f}")
        print(f"Soft Intervention: {soft_mean:.4f}")
        print(f"Improvement: {improvement:+.1f}%")
    
    def run_complete_benchmark(self):
        """Run all benchmark tests."""
        
        print("=== COMPREHENSIVE SCALING AND ROBUSTNESS BENCHMARK ===")
        print(f"Scaling sizes: {self.scaling_sizes}")
        print(f"Noise conditions: {list(self.noise_configs.keys())}")
        print(f"Seeds per test: {self.num_seeds}")
        print(f"Epochs per model: {self.num_epochs}")
        
        # Run all tests
        self.run_scaling_test()
        self.run_noise_robustness_test()
        self.run_spurious_correlation_test()
        
        # Compute and display summary
        self.compute_summary_statistics()
        
        return self.results
    
    def save_results(self, filename: str = "scaling_robustness_results.json"):
        """Save benchmark results to file."""
        
        os.makedirs("results/phase4_scaling_robustness", exist_ok=True)
        filepath = os.path.join("results/phase4_scaling_robustness", filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
    
    def plot_results(self):
        """Create visualization plots for the results."""
        
        summary = self.results['summary']
        
        # Create plots directory
        os.makedirs("results/phase4_scaling_robustness", exist_ok=True)
        
        # Plot 1: Scaling performance
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Scaling plot
        sizes = self.scaling_sizes
        vanilla_means = [summary['scaling_test'][size]['vanilla']['mean_val_loss'] for size in sizes]
        vanilla_stds = [summary['scaling_test'][size]['vanilla']['std_val_loss'] for size in sizes]
        soft_means = [summary['scaling_test'][size]['soft_intervention']['mean_val_loss'] for size in sizes]
        soft_stds = [summary['scaling_test'][size]['soft_intervention']['std_val_loss'] for size in sizes]
        
        ax1.errorbar(sizes, vanilla_means, yerr=vanilla_stds, marker='o', label='Vanilla MLP', capsize=5)
        ax1.errorbar(sizes, soft_means, yerr=soft_stds, marker='s', label='Soft Intervention', capsize=5)
        ax1.set_xlabel('Dataset Size')
        ax1.set_ylabel('Validation Loss')
        ax1.set_title('Scaling Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Noise robustness plot
        noise_names = list(self.noise_configs.keys())
        vanilla_noise = [summary['noise_robustness'][name]['vanilla']['mean_val_loss'] for name in noise_names]
        soft_noise = [summary['noise_robustness'][name]['soft_intervention']['mean_val_loss'] for name in noise_names]
        
        x_pos = np.arange(len(noise_names))
        width = 0.35
        
        ax2.bar(x_pos - width/2, vanilla_noise, width, label='Vanilla MLP', alpha=0.8)
        ax2.bar(x_pos + width/2, soft_noise, width, label='Soft Intervention', alpha=0.8)
        ax2.set_xlabel('Noise Condition')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Noise Robustness')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(noise_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Improvement percentages
        scaling_improvements = []
        for size in sizes:
            vanilla_mean = summary['scaling_test'][size]['vanilla']['mean_val_loss']
            soft_mean = summary['scaling_test'][size]['soft_intervention']['mean_val_loss']
            improvement = ((vanilla_mean - soft_mean) / vanilla_mean) * 100
            scaling_improvements.append(improvement)
        
        ax3.plot(sizes, scaling_improvements, 'o-', linewidth=2, markersize=8)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Dataset Size')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Performance Improvement vs Dataset Size')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig('results/phase4_scaling_robustness/scaling_robustness_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plots saved to: results/phase4_scaling_robustness/scaling_robustness_plots.png")


def main():
    """Run the comprehensive scaling and robustness benchmark."""
    
    # Create benchmark
    benchmark = ScalingRobustnessBenchmark(
        input_dim=3,
        hidden_dims=[32, 16],
        output_dim=1,
        activation='relu',
        num_seeds=3,
        num_epochs=75,  # Reduced for faster execution
        learning_rate=0.001,
        device='cpu'
    )
    
    # Run benchmark
    results = benchmark.run_complete_benchmark()
    
    # Save results
    benchmark.save_results()
    
    # Create plots
    benchmark.plot_results()
    
    print("\n=== COMPREHENSIVE BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main() 