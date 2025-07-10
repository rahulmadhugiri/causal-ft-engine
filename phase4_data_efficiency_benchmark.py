#!/usr/bin/env python3
"""
Phase 4: Data Efficiency Benchmark

Goal: Demonstrate that the soft intervention causal network achieves comparable
performance to vanilla MLP with significantly less training data.

Test Setup:
- Vanilla MLP baseline
- Soft Intervention Causal Network
- Data sizes: 5%, 10%, 25%, 50%, 100% of full dataset
- Multiple seeds for statistical significance
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


class DataEfficiencyBenchmark:
    """
    Benchmark for comparing data efficiency between vanilla MLP and soft intervention causal networks.
    """
    
    def __init__(self, 
                 input_dim: int = 4,
                 hidden_dims: List[int] = [16, 8],
                 output_dim: int = 1,
                 activation: str = 'relu',
                 num_seeds: int = 3,
                 base_dataset_size: int = 1000,
                 num_epochs: int = 100,
                 learning_rate: float = 0.01,
                 device: str = 'cpu'):
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.num_seeds = num_seeds
        self.base_dataset_size = base_dataset_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device(device)
        
        # Data fractions to test
        self.data_fractions = [0.05, 0.1, 0.25, 0.5, 1.0]
        
        # Results storage
        self.results = {
            'vanilla_mlp': {},
            'soft_intervention_network': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'input_dim': input_dim,
                    'hidden_dims': hidden_dims,
                    'output_dim': output_dim,
                    'activation': activation,
                    'num_seeds': num_seeds,
                    'base_dataset_size': base_dataset_size,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate,
                    'data_fractions': self.data_fractions
                }
            }
        }
    
    def generate_dataset(self, size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic dataset with specified size and seed."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate data with chain structure
        x_np, y_np, true_adjacency_np = generate_synthetic_data(
            size, 
            n_nodes=self.input_dim + 1,  # +1 because one node becomes the output
            graph_type='chain', 
            noise_level=0.3
        )
        
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
        true_adjacency = torch.tensor(true_adjacency_np, dtype=torch.float32, device=self.device)
        
        return x, y, true_adjacency
    
    def create_interventions(self, x: torch.Tensor, intervention_prob: float = 0.3) -> List[Dict]:
        """Create random interventions for causal network training."""
        batch_size = x.shape[0]
        interventions = []
        
        for i in range(batch_size):
            if np.random.random() < intervention_prob:
                # Random intervention
                mask = torch.zeros(self.input_dim)
                values = torch.zeros(self.input_dim)
                
                # Intervene on random node
                node_idx = np.random.randint(0, self.input_dim)
                mask[node_idx] = 1.0
                values[node_idx] = np.random.normal(0, 1)  # Random intervention value
                
                interventions.append({'random': (mask, values)})
            else:
                interventions.append({})
        
        return interventions
    
    def train_vanilla_mlp(self, x: torch.Tensor, y: torch.Tensor, seed: int) -> Dict:
        """Train vanilla MLP and return training metrics."""
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create model
        model = VanillaMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation
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
            
            pred = model(x_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = criterion(val_pred, y_val)
                val_losses.append(val_loss.item())
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_params': sum(p.numel() for p in model.parameters())
        }
    
    def train_soft_intervention_network(self, x: torch.Tensor, y: torch.Tensor, seed: int) -> Dict:
        """Train soft intervention causal network and return training metrics."""
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create model
        model = SoftInterventionNetwork(
            input_dim=self.input_dim,
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
        alpha_histories = []
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            # Create interventions for training
            interventions = self.create_interventions(x_train)
            
            pred = model(x_train, interventions=interventions)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Validation phase (no interventions)
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = criterion(val_pred, y_val)
                val_losses.append(val_loss.item())
            
            # Track alpha parameters
            alpha_summary = model.get_alpha_summary()
            alpha_histories.append({
                unit_name: info['mean_alpha'] 
                for unit_name, info in alpha_summary.items()
            })
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'train_losses': train_losses,
            'val_losses': val_losses,
            'alpha_histories': alpha_histories,
            'final_alpha_summary': model.get_alpha_summary(),
            'model_params': sum(p.numel() for p in model.parameters())
        }
    
    def run_benchmark(self) -> Dict:
        """Run the complete data efficiency benchmark."""
        
        print(f"=== RUNNING DATA EFFICIENCY BENCHMARK ===")
        print(f"Base dataset size: {self.base_dataset_size}")
        print(f"Data fractions: {self.data_fractions}")
        print(f"Number of seeds: {self.num_seeds}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Device: {self.device}")
        
        for fraction in self.data_fractions:
            data_size = int(self.base_dataset_size * fraction)
            print(f"\n--- Testing with {data_size} samples ({fraction*100:.0f}% of full dataset) ---")
            
            # Initialize results for this fraction
            self.results['vanilla_mlp'][fraction] = []
            self.results['soft_intervention_network'][fraction] = []
            
            for seed in range(self.num_seeds):
                print(f"  Seed {seed + 1}/{self.num_seeds}")
                
                # Generate dataset
                x, y, true_adjacency = self.generate_dataset(data_size, seed)
                
                # Train vanilla MLP
                print(f"    Training vanilla MLP...")
                vanilla_results = self.train_vanilla_mlp(x, y, seed)
                self.results['vanilla_mlp'][fraction].append(vanilla_results)
                
                # Train soft intervention network
                print(f"    Training soft intervention network...")
                soft_results = self.train_soft_intervention_network(x, y, seed)
                self.results['soft_intervention_network'][fraction].append(soft_results)
                
                # Print results
                print(f"      Vanilla MLP - Val Loss: {vanilla_results['final_val_loss']:.4f}")
                print(f"      Soft Intervention - Val Loss: {soft_results['final_val_loss']:.4f}")
        
        # Compute summary statistics
        self.compute_summary_statistics()
        
        return self.results
    
    def compute_summary_statistics(self):
        """Compute mean and std statistics across seeds."""
        
        summary = {'vanilla_mlp': {}, 'soft_intervention_network': {}}
        
        for model_type in ['vanilla_mlp', 'soft_intervention_network']:
            for fraction in self.data_fractions:
                results = self.results[model_type][fraction]
                
                train_losses = [r['final_train_loss'] for r in results]
                val_losses = [r['final_val_loss'] for r in results]
                
                summary[model_type][fraction] = {
                    'mean_train_loss': np.mean(train_losses),
                    'std_train_loss': np.std(train_losses),
                    'mean_val_loss': np.mean(val_losses),
                    'std_val_loss': np.std(val_losses),
                    'sample_size': int(self.base_dataset_size * fraction)
                }
        
        self.results['summary'] = summary
        
        # Print summary
        print(f"\n=== BENCHMARK SUMMARY ===")
        print(f"Data Size\tVanilla MLP\t\tSoft Intervention")
        print(f"(samples)\tVal Loss ± Std\t\tVal Loss ± Std")
        print(f"-" * 60)
        
        for fraction in self.data_fractions:
            sample_size = int(self.base_dataset_size * fraction)
            
            vanilla_mean = summary['vanilla_mlp'][fraction]['mean_val_loss']
            vanilla_std = summary['vanilla_mlp'][fraction]['std_val_loss']
            
            soft_mean = summary['soft_intervention_network'][fraction]['mean_val_loss']
            soft_std = summary['soft_intervention_network'][fraction]['std_val_loss']
            
            print(f"{sample_size:>8}\t{vanilla_mean:.4f} ± {vanilla_std:.4f}\t{soft_mean:.4f} ± {soft_std:.4f}")
    
    def save_results(self, filename: str = "phase4_data_efficiency_results.json"):
        """Save benchmark results to file."""
        
        os.makedirs("results/phase4_data_efficiency", exist_ok=True)
        filepath = os.path.join("results/phase4_data_efficiency", filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filepath}")
    
    def plot_results(self, save_path: str = "results/phase4_data_efficiency/data_efficiency_plot.png"):
        """Plot the data efficiency comparison."""
        
        plt.figure(figsize=(12, 8))
        
        # Extract data for plotting
        sample_sizes = [int(self.base_dataset_size * f) for f in self.data_fractions]
        summary = self.results['summary']
        
        vanilla_means = [summary['vanilla_mlp'][f]['mean_val_loss'] for f in self.data_fractions]
        vanilla_stds = [summary['vanilla_mlp'][f]['std_val_loss'] for f in self.data_fractions]
        
        soft_means = [summary['soft_intervention_network'][f]['mean_val_loss'] for f in self.data_fractions]
        soft_stds = [summary['soft_intervention_network'][f]['std_val_loss'] for f in self.data_fractions]
        
        # Plot with error bars
        plt.errorbar(sample_sizes, vanilla_means, yerr=vanilla_stds, 
                    marker='o', linewidth=2, capsize=5, label='Vanilla MLP', color='blue')
        plt.errorbar(sample_sizes, soft_means, yerr=soft_stds, 
                    marker='s', linewidth=2, capsize=5, label='Soft Intervention Network', color='red')
        
        plt.xlabel('Number of Training Samples')
        plt.ylabel('Validation Loss')
        plt.title('Data Efficiency Comparison: Vanilla MLP vs Soft Intervention Network')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {save_path}")


def main():
    """Run the data efficiency benchmark."""
    
    # Create benchmark
    benchmark = DataEfficiencyBenchmark(
        input_dim=3,  # Fixed: generate_synthetic_data with n_nodes=4 creates 3 input features + 1 output
        hidden_dims=[16, 8],
        output_dim=1,
        activation='relu',
        num_seeds=3,
        base_dataset_size=1000,
        num_epochs=50,  # Reduced for faster testing
        learning_rate=0.01,
        device='cpu'
    )
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    # Save results
    benchmark.save_results()
    
    # Plot results
    benchmark.plot_results()
    
    print(f"\n=== BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main() 