import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from causal_unit_network import CausalUnitNetwork
from train_causalunit import CausalUnitTrainer
from engine.loss_functions import CausalLosses, CausalMetrics
from experiments.utils import generate_synthetic_data


def convert_for_json(obj):
    """Convert numpy/torch types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


class CausalUnitEvaluator:
    """
    Phase 3 CausalUnit Evaluator: Comprehensive evaluation framework with automated benchmarking,
    ablation studies, robustness testing, and novel experiments.
    
    Key Features:
    1. Automated benchmarking with standard metrics
    2. Comprehensive ablation studies
    3. Multi-seed robustness testing
    4. Out-of-distribution intervention testing
    5. Novel experiments (multiple interventions, latent variables, scheduling)
    6. Gradient flow visualization and analysis
    """
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 random_seed: int = 42):
        
        self.device = device
        self.random_seed = random_seed
        self.set_random_seeds(random_seed)
        
        # Evaluation configurations
        self.graph_configs = [
            {'name': 'chain', 'type': 'chain', 'n_nodes': 4},
            {'name': 'fork', 'type': 'fork', 'n_nodes': 4},
            {'name': 'v_structure', 'type': 'v_structure', 'n_nodes': 4},
            {'name': 'confounder', 'type': 'confounder', 'n_nodes': 4},
            {'name': 'complex', 'type': 'complex', 'n_nodes': 6}
        ]
        
        self.noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
        self.sample_sizes = [500, 1000, 2000, 5000]
        self.intervention_probs = [0.1, 0.3, 0.5, 0.7]
        
        # Results storage
        self.evaluation_results = {}
        self.benchmark_results = {}
        self.ablation_results = {}
        self.robustness_results = {}
        self.novel_experiment_results = {}
        
    def set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def create_test_datasets(self, 
                           graph_type: str = 'chain',
                           n_nodes: int = 4,
                           n_samples: int = 1000,
                           noise_level: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create test datasets for evaluation.
        
        Args:
            graph_type: Type of causal graph
            n_nodes: Number of nodes
            n_samples: Number of samples
            noise_level: Noise level in data
            
        Returns:
            x: Input data
            y: Target data
            true_adjacency: True causal structure
        """
        # Generate synthetic data
        x, y, true_adjacency = generate_synthetic_data(
            n_samples=n_samples,
            n_nodes=n_nodes,
            graph_type=graph_type,
            noise_level=noise_level
        )
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(true_adjacency, dtype=torch.float32)
    
    def evaluate_structure_learning_comprehensive(self, 
                                                learned_adjacency: np.ndarray,
                                                true_adjacency: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive structure learning evaluation.
        
        Args:
            learned_adjacency: Learned adjacency matrix
            true_adjacency: True adjacency matrix
            
        Returns:
            Dictionary with structure learning metrics
        """
        # Flatten matrices for binary classification metrics
        true_flat = true_adjacency.flatten()
        learned_flat = learned_adjacency.flatten()
        
        # Convert to binary
        learned_binary = (learned_flat > 0.5).astype(int)
        true_binary = (true_flat > 0.5).astype(int)
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_binary, learned_binary, average='binary', zero_division=0
        )
        
        # Compute AUC if possible
        try:
            auc = roc_auc_score(true_binary, learned_flat)
        except:
            auc = 0.0
        
        # Structural Hamming Distance
        shd = np.sum(np.abs(learned_binary - true_binary))
        
        # Edge accuracy
        edge_accuracy = np.mean(learned_binary == true_binary)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'shd': shd,
            'edge_accuracy': edge_accuracy
        }
    
    def evaluate_counterfactual_performance_comprehensive(self, 
                                                        network: CausalUnitNetwork,
                                                        x: torch.Tensor,
                                                        y: torch.Tensor,
                                                        true_adjacency: Optional[torch.Tensor] = None,
                                                        n_interventions: int = 20) -> Dict[str, float]:
        """
        Comprehensive counterfactual performance evaluation.
        
        Args:
            network: CausalUnit network
            x: Input data
            y: Target data
            true_adjacency: True causal structure
            n_interventions: Number of interventions to test
            
        Returns:
            Dictionary with counterfactual metrics
        """
        network.eval()
        
        original_predictions = []
        counterfactual_predictions = []
        intervention_effects = []
        
        with torch.no_grad():
            # Original predictions
            y_original = network(x, interventions=None)
            
            for _ in range(n_interventions):
                # Random intervention
                intervention_node = np.random.randint(0, x.shape[1])
                intervention_value = torch.randn(1, device=self.device) * 2.0
                
                # Create intervention
                intervention_mask = torch.zeros(x.shape[1], device=self.device)
                intervention_values = torch.zeros(x.shape[1], device=self.device)
                intervention_mask[intervention_node] = 1.0
                intervention_values[intervention_node] = intervention_value
                
                # Apply intervention to all samples
                interventions = []
                for i in range(x.shape[0]):
                    interventions.append({'test': (intervention_mask, intervention_values)})
                
                # Counterfactual predictions
                y_counterfactual = network(x, interventions=interventions)
                
                # Compute intervention effect
                effect = torch.mean(torch.abs(y_counterfactual - y_original))
                intervention_effects.append(effect.item())
                
                original_predictions.append(y_original.cpu().numpy())
                counterfactual_predictions.append(y_counterfactual.cpu().numpy())
        
        # Compute metrics
        intervention_effects = np.array(intervention_effects)
        
        # Correlation between original and counterfactual predictions
        correlations = []
        for i in range(len(original_predictions)):
            try:
                corr, _ = pearsonr(
                    original_predictions[i].flatten(),
                    counterfactual_predictions[i].flatten()
                )
                # Only add valid correlations (not NaN)
                if not np.isnan(corr):
                    correlations.append(corr)
                else:
                    # For NaN correlations, use 0.0 as default
                    correlations.append(0.0)
            except:
                # If correlation computation fails, use 0.0
                correlations.append(0.0)
        
        # Handle case where all correlations are invalid
        if len(correlations) == 0:
            correlations = [0.0]
        
        return {
            'mean_intervention_effect': np.mean(intervention_effects),
            'std_intervention_effect': np.std(intervention_effects),
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'min_effect': np.min(intervention_effects),
            'max_effect': np.max(intervention_effects)
        }
    
    def run_benchmark_evaluation(self, 
                               network: CausalUnitNetwork,
                               graph_configs: Optional[List[Dict]] = None,
                               save_path: str = 'results/phase3_benchmark') -> Dict[str, Any]:
        """
        Run comprehensive benchmark evaluation.
        
        Args:
            network: CausalUnit network to evaluate
            graph_configs: Optional graph configurations
            save_path: Path to save results
            
        Returns:
            Benchmark evaluation results
        """
        print("Starting Phase 3 Benchmark Evaluation...")
        
        if graph_configs is None:
            graph_configs = self.graph_configs
        
        benchmark_results = {}
        
        for config in graph_configs:
            print(f"\nEvaluating on {config['name']} graph...")
            
            # Create test data
            x, y, true_adjacency = self.create_test_datasets(
                graph_type=config['type'],
                n_nodes=config['n_nodes'],
                n_samples=1000,
                noise_level=0.3
            )
            
            x, y, true_adjacency = x.to(self.device), y.to(self.device), true_adjacency.to(self.device)
            
            # Evaluate structure learning
            learned_adjacency = network.get_adjacency_matrix(hard=True)
            structure_metrics = self.evaluate_structure_learning_comprehensive(
                learned_adjacency.cpu().numpy(),
                true_adjacency.cpu().numpy()
            )
            
            # Evaluate counterfactual performance
            counterfactual_metrics = self.evaluate_counterfactual_performance_comprehensive(
                network, x, y, true_adjacency
            )
            
            # Evaluate standard prediction performance
            network.eval()
            with torch.no_grad():
                y_pred = network(x, interventions=None)
                prediction_mse = torch.mean((y_pred - y) ** 2).item()
                prediction_mae = torch.mean(torch.abs(y_pred - y)).item()
            
            benchmark_results[config['name']] = {
                'structure': structure_metrics,
                'counterfactual': counterfactual_metrics,
                'prediction': {
                    'mse': prediction_mse,
                    'mae': prediction_mae
                },
                'config': config
            }
        
        # Save results
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'benchmark_results.json'), 'w') as f:
            json.dump(convert_for_json(benchmark_results), f, indent=2)
        
        self.benchmark_results = benchmark_results
        
        print(f"Benchmark evaluation completed. Results saved to {save_path}")
        
        return benchmark_results
    
    def run_ablation_study(self, 
                          base_config: Dict,
                          train_loader: torch.utils.data.DataLoader,
                          test_loader: torch.utils.data.DataLoader,
                          true_adjacency: torch.Tensor,
                          save_path: str = 'results/phase3_ablation') -> Dict[str, Any]:
        """
        Run comprehensive ablation study.
        
        Args:
            base_config: Base network configuration
            train_loader: Training data loader
            test_loader: Test data loader
            true_adjacency: True causal structure
            save_path: Path to save results
            
        Returns:
            Ablation study results
        """
        print("Starting Phase 3 Ablation Study...")
        
        ablation_configs = [
            {'name': 'full_model', 'config': {}},
            {'name': 'no_interventions', 'config': {'enable_interventions': False}},
            {'name': 'no_gradient_blocking', 'config': {'enable_gradient_blocking': False}},
            {'name': 'no_dynamic_rewiring', 'config': {'enable_dynamic_rewiring': False}},
            {'name': 'no_structure_learning', 'config': {'enable_structure_learning': False}},
            {'name': 'no_gradient_surgery', 'config': {'enable_gradient_surgery': False}},
            {'name': 'vanilla_mlp', 'config': {
                'enable_interventions': False,
                'enable_gradient_blocking': False,
                'enable_dynamic_rewiring': False,
                'enable_structure_learning': False,
                'enable_gradient_surgery': False
            }}
        ]
        
        ablation_results = {}
        
        for ablation in ablation_configs:
            print(f"\nRunning ablation: {ablation['name']}")
            
            # Create network
            network = CausalUnitNetwork(**base_config)
            trainer = CausalUnitTrainer(network, device=self.device)
            
            # Set ablation config
            trainer.set_ablation_config(**ablation['config'])
            
            # Train
            training_results = trainer.train(
                train_loader, test_loader, num_epochs=50,
                true_adjacency=true_adjacency,
                save_path=os.path.join(save_path, ablation['name'])
            )
            
            # Evaluate
            benchmark_results = self.run_benchmark_evaluation(
                network, 
                save_path=os.path.join(save_path, ablation['name'], 'benchmark')
            )
            
            ablation_results[ablation['name']] = {
                'training_results': training_results['final_metrics'],
                'benchmark_results': benchmark_results,
                'config': ablation['config']
            }
        
        # Save ablation results
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'ablation_summary.json'), 'w') as f:
            json.dump(convert_for_json(ablation_results), f, indent=2)
        
        # Create ablation comparison plots
        self.plot_ablation_results(ablation_results, save_path)
        
        self.ablation_results = ablation_results
        
        print(f"Ablation study completed. Results saved to {save_path}")
        
        return ablation_results
    
    def run_robustness_testing(self, 
                             base_config: Dict,
                             n_seeds: int = 5,
                             save_path: str = 'results/phase3_robustness') -> Dict[str, Any]:
        """
        Run multi-seed robustness testing.
        
        Args:
            base_config: Base network configuration
            n_seeds: Number of random seeds to test
            save_path: Path to save results
            
        Returns:
            Robustness testing results
        """
        print("Starting Phase 3 Robustness Testing...")
        
        robustness_results = {}
        
        for seed in range(n_seeds):
            print(f"\nTesting seed {seed + 1}/{n_seeds}")
            
            # Set random seed
            self.set_random_seeds(seed)
            
            # Test different noise levels
            for noise_level in self.noise_levels:
                print(f"  Testing noise level {noise_level}")
                
                # Create network
                network = CausalUnitNetwork(**base_config)
                
                # Create test data
                x, y, true_adjacency = self.create_test_datasets(
                    graph_type='chain',
                    n_nodes=4,
                    n_samples=1000,
                    noise_level=noise_level
                )
                
                x, y, true_adjacency = x.to(self.device), y.to(self.device), true_adjacency.to(self.device)
                
                # Create data loaders
                dataset = torch.utils.data.TensorDataset(x, y)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
                test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
                
                # Train
                trainer = CausalUnitTrainer(network, device=self.device)
                training_results = trainer.train(
                    train_loader, test_loader, num_epochs=30,
                    true_adjacency=true_adjacency,
                    save_path=os.path.join(save_path, f'seed_{seed}_noise_{noise_level}')
                )
                
                # Store results
                key = f'seed_{seed}_noise_{noise_level}'
                robustness_results[key] = {
                    'seed': seed,
                    'noise_level': noise_level,
                    'final_metrics': training_results['final_metrics']
                }
        
        # Analyze robustness
        robustness_analysis = self.analyze_robustness(robustness_results)
        
        # Save results
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'robustness_results.json'), 'w') as f:
            json.dump(convert_for_json(robustness_results), f, indent=2)
        
        with open(os.path.join(save_path, 'robustness_analysis.json'), 'w') as f:
            json.dump(convert_for_json(robustness_analysis), f, indent=2)
        
        # Plot robustness results
        self.plot_robustness_results(robustness_results, save_path)
        
        self.robustness_results = robustness_results
        
        print(f"Robustness testing completed. Results saved to {save_path}")
        
        return robustness_results
    
    def run_novel_experiments(self, 
                            base_config: Dict,
                            save_path: str = 'results/phase3_novel') -> Dict[str, Any]:
        """
        Run novel experiments specific to Phase 3.
        
        Args:
            base_config: Base network configuration
            save_path: Path to save results
            
        Returns:
            Novel experiment results
        """
        print("Starting Phase 3 Novel Experiments...")
        
        novel_results = {}
        
        # Experiment 1: Multiple simultaneous interventions
        print("\n1. Testing multiple simultaneous interventions...")
        novel_results['multiple_interventions'] = self.test_multiple_interventions(base_config)
        
        # Experiment 2: Intervention scheduling
        print("\n2. Testing intervention scheduling...")
        novel_results['intervention_scheduling'] = self.test_intervention_scheduling(base_config)
        
        # Experiment 3: Out-of-distribution interventions
        print("\n3. Testing out-of-distribution interventions...")
        novel_results['ood_interventions'] = self.test_ood_interventions(base_config)
        
        # Experiment 4: Gradient flow analysis
        print("\n4. Analyzing gradient flow patterns...")
        novel_results['gradient_flow_analysis'] = self.analyze_gradient_flow(base_config)
        
        # Save results
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'novel_experiments.json'), 'w') as f:
            json.dump(convert_for_json(novel_results), f, indent=2)
        
        self.novel_experiment_results = novel_results
        
        print(f"Novel experiments completed. Results saved to {save_path}")
        
        return novel_results
    
    def test_multiple_interventions(self, base_config: Dict) -> Dict[str, Any]:
        """Test multiple simultaneous interventions."""
        network = CausalUnitNetwork(**base_config)
        
        # Create test data
        x, y, true_adjacency = self.create_test_datasets()
        x, y = x.to(self.device), y.to(self.device)
        
        results = {}
        
        # Test different numbers of simultaneous interventions
        for n_interventions in [1, 2, 3, 4]:
            intervention_effects = []
            
            for _ in range(20):  # 20 trials
                # Create multiple interventions
                interventions = []
                for i in range(x.shape[0]):
                    sample_interventions = {}
                    
                    for j in range(n_interventions):
                        node_idx = np.random.randint(0, x.shape[1])
                        value = torch.randn(1, device=self.device) * 2.0
                        
                        mask = torch.zeros(x.shape[1], device=self.device)
                        values = torch.zeros(x.shape[1], device=self.device)
                        mask[node_idx] = 1.0
                        values[node_idx] = value
                        
                        sample_interventions[f'intervention_{j}'] = (mask, values)
                    
                    interventions.append(sample_interventions)
                
                # Test intervention
                with torch.no_grad():
                    y_original = network(x, interventions=None)
                    y_intervened = network(x, interventions=interventions)
                    effect = torch.mean(torch.abs(y_intervened - y_original)).item()
                    intervention_effects.append(effect)
            
            results[f'{n_interventions}_interventions'] = {
                'mean_effect': np.mean(intervention_effects),
                'std_effect': np.std(intervention_effects),
                'effects': intervention_effects
            }
        
        return results
    
    def test_intervention_scheduling(self, base_config: Dict) -> Dict[str, Any]:
        """Test different intervention scheduling strategies."""
        network = CausalUnitNetwork(**base_config)
        
        # Create test data
        x, y, true_adjacency = self.create_test_datasets()
        x, y = x.to(self.device), y.to(self.device)
        
        results = {}
        
        # Test different intervention probabilities
        for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            effects = []
            
            for _ in range(10):  # 10 trials
                # Create intervention schedule
                interventions = []
                for i in range(x.shape[0]):
                    if np.random.random() < prob:
                        node_idx = np.random.randint(0, x.shape[1])
                        value = torch.randn(1, device=self.device) * 2.0
                        
                        mask = torch.zeros(x.shape[1], device=self.device)
                        values = torch.zeros(x.shape[1], device=self.device)
                        mask[node_idx] = 1.0
                        values[node_idx] = value
                        
                        interventions.append({'scheduled': (mask, values)})
                    else:
                        interventions.append({})
                
                # Test intervention
                with torch.no_grad():
                    y_original = network(x, interventions=None)
                    y_intervened = network(x, interventions=interventions)
                    effect = torch.mean(torch.abs(y_intervened - y_original)).item()
                    effects.append(effect)
            
            results[f'prob_{prob}'] = {
                'mean_effect': np.mean(effects),
                'std_effect': np.std(effects),
                'effects': effects
            }
        
        return results
    
    def test_ood_interventions(self, base_config: Dict) -> Dict[str, Any]:
        """Test out-of-distribution interventions."""
        network = CausalUnitNetwork(**base_config)
        
        # Create test data
        x, y, true_adjacency = self.create_test_datasets()
        x, y = x.to(self.device), y.to(self.device)
        
        results = {}
        
        # Test different intervention magnitudes
        for magnitude in [0.5, 1.0, 2.0, 5.0, 10.0]:
            effects = []
            
            for _ in range(10):  # 10 trials
                # Create OOD intervention
                node_idx = np.random.randint(0, x.shape[1])
                value = torch.randn(1, device=self.device) * magnitude
                
                mask = torch.zeros(x.shape[1], device=self.device)
                values = torch.zeros(x.shape[1], device=self.device)
                mask[node_idx] = 1.0
                values[node_idx] = value
                
                # Apply to all samples
                interventions = []
                for i in range(x.shape[0]):
                    interventions.append({'ood': (mask, values)})
                
                # Test intervention
                with torch.no_grad():
                    y_original = network(x, interventions=None)
                    y_intervened = network(x, interventions=interventions)
                    effect = torch.mean(torch.abs(y_intervened - y_original)).item()
                    effects.append(effect)
            
            results[f'magnitude_{magnitude}'] = {
                'mean_effect': np.mean(effects),
                'std_effect': np.std(effects),
                'effects': effects
            }
        
        return results
    
    def analyze_gradient_flow(self, base_config: Dict) -> Dict[str, Any]:
        """Analyze gradient flow patterns during training."""
        network = CausalUnitNetwork(**base_config)
        
        # Create test data
        x, y, true_adjacency = self.create_test_datasets()
        x, y = x.to(self.device), y.to(self.device)
        
        # Create trainer with debug mode
        trainer = CausalUnitTrainer(network, device=self.device)
        trainer.enable_debug_mode(True)
        
        # Short training to collect gradient flow
        dataset = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Train for a few epochs
        trainer.train(train_loader, test_loader, num_epochs=10)
        
        # Analyze gradient flow history
        gradient_flow_analysis = {
            'num_checkpoints': len(trainer.gradient_flow_history),
            'gradient_patterns': self.analyze_gradient_patterns(trainer.gradient_flow_history),
            'blocking_effectiveness': self.analyze_blocking_effectiveness(trainer.gradient_flow_history)
        }
        
        return gradient_flow_analysis
    
    def analyze_gradient_patterns(self, gradient_flow_history: List[Dict]) -> Dict[str, Any]:
        """Analyze gradient flow patterns."""
        if not gradient_flow_history:
            return {'error': 'No gradient flow history available'}
        
        # Extract gradient norms over time
        gradient_norms = []
        for checkpoint in gradient_flow_history:
            norms = []
            for unit_name, unit_info in checkpoint.items():
                if 'weights_grad' in unit_info and unit_info['weights_grad'] is not None:
                    norm = np.linalg.norm(unit_info['weights_grad'])
                    norms.append(norm)
            gradient_norms.append(norms)
        
        return {
            'mean_gradient_norms': [np.mean(norms) if norms else 0 for norms in gradient_norms],
            'std_gradient_norms': [np.std(norms) if norms else 0 for norms in gradient_norms],
            'gradient_stability': np.std([np.mean(norms) if norms else 0 for norms in gradient_norms])
        }
    
    def analyze_blocking_effectiveness(self, gradient_flow_history: List[Dict]) -> Dict[str, Any]:
        """Analyze how effectively gradients are being blocked."""
        if not gradient_flow_history:
            return {'error': 'No gradient flow history available'}
        
        # This is a simplified analysis - in practice, we'd need more detailed tracking
        blocking_effectiveness = {
            'intervention_checkpoints': len(gradient_flow_history),
            'average_gradient_reduction': 0.0,  # Placeholder
            'blocking_consistency': 0.0  # Placeholder
        }
        
        return blocking_effectiveness
    
    def analyze_robustness(self, robustness_results: Dict) -> Dict[str, Any]:
        """Analyze robustness results across seeds and noise levels."""
        # Group results by noise level
        noise_analysis = {}
        
        for key, result in robustness_results.items():
            noise_level = result['noise_level']
            
            if noise_level not in noise_analysis:
                noise_analysis[noise_level] = {
                    'test_losses': [],
                    'structure_accuracies': [],
                    'counterfactual_accuracies': []
                }
            
            noise_analysis[noise_level]['test_losses'].append(result['final_metrics']['test_loss'])
            noise_analysis[noise_level]['structure_accuracies'].append(result['final_metrics']['structure_accuracy'])
            noise_analysis[noise_level]['counterfactual_accuracies'].append(result['final_metrics']['counterfactual_accuracy'])
        
        # Compute statistics
        analysis = {}
        for noise_level, metrics in noise_analysis.items():
            analysis[f'noise_{noise_level}'] = {
                'test_loss': {
                    'mean': np.mean(metrics['test_losses']),
                    'std': np.std(metrics['test_losses']),
                    'min': np.min(metrics['test_losses']),
                    'max': np.max(metrics['test_losses'])
                },
                'structure_accuracy': {
                    'mean': np.mean(metrics['structure_accuracies']),
                    'std': np.std(metrics['structure_accuracies']),
                    'min': np.min(metrics['structure_accuracies']),
                    'max': np.max(metrics['structure_accuracies'])
                },
                'counterfactual_accuracy': {
                    'mean': np.mean(metrics['counterfactual_accuracies']),
                    'std': np.std(metrics['counterfactual_accuracies']),
                    'min': np.min(metrics['counterfactual_accuracies']),
                    'max': np.max(metrics['counterfactual_accuracies'])
                }
            }
        
        return analysis
    
    def plot_ablation_results(self, ablation_results: Dict, save_path: str):
        """Plot ablation study results."""
        # Extract metrics for plotting
        methods = list(ablation_results.keys())
        
        # Structure accuracy
        structure_f1 = []
        counterfactual_corr = []
        test_loss = []
        
        for method in methods:
            # Get average structure F1 from benchmark results
            benchmark = ablation_results[method]['benchmark_results']
            avg_f1 = np.mean([v['structure']['f1'] for v in benchmark.values()])
            structure_f1.append(avg_f1)
            
            # Get average counterfactual correlation
            avg_corr = np.mean([v['counterfactual']['mean_correlation'] for v in benchmark.values()])
            counterfactual_corr.append(avg_corr)
            
            # Get test loss
            test_loss.append(ablation_results[method]['training_results']['test_loss'])
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Structure learning performance
        axes[0].bar(methods, structure_f1)
        axes[0].set_title('Structure Learning Performance')
        axes[0].set_ylabel('F1 Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Counterfactual performance
        axes[1].bar(methods, counterfactual_corr)
        axes[1].set_title('Counterfactual Performance')
        axes[1].set_ylabel('Correlation')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Test loss
        axes[2].bar(methods, test_loss)
        axes[2].set_title('Test Loss')
        axes[2].set_ylabel('MSE Loss')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'ablation_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_robustness_results(self, robustness_results: Dict, save_path: str):
        """Plot robustness testing results."""
        # Extract data for plotting
        noise_levels = []
        test_losses = []
        structure_accuracies = []
        counterfactual_accuracies = []
        
        for key, result in robustness_results.items():
            noise_levels.append(result['noise_level'])
            test_losses.append(result['final_metrics']['test_loss'])
            structure_accuracies.append(result['final_metrics']['structure_accuracy'])
            counterfactual_accuracies.append(result['final_metrics']['counterfactual_accuracy'])
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'noise_level': noise_levels,
            'test_loss': test_losses,
            'structure_accuracy': structure_accuracies,
            'counterfactual_accuracy': counterfactual_accuracies
        })
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Group by noise level and plot
        noise_groups = df.groupby('noise_level')
        
        for noise_level, group in noise_groups:
            axes[0].scatter([noise_level] * len(group), group['test_loss'], alpha=0.7)
            axes[1].scatter([noise_level] * len(group), group['structure_accuracy'], alpha=0.7)
            axes[2].scatter([noise_level] * len(group), group['counterfactual_accuracy'], alpha=0.7)
        
        axes[0].set_title('Test Loss vs Noise Level')
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('Test Loss')
        
        axes[1].set_title('Structure Accuracy vs Noise Level')
        axes[1].set_xlabel('Noise Level')
        axes[1].set_ylabel('Structure Accuracy')
        
        axes[2].set_title('Counterfactual Accuracy vs Noise Level')
        axes[2].set_xlabel('Noise Level')
        axes[2].set_ylabel('Counterfactual Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'robustness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, save_path: str = 'results/phase3_report'):
        """Generate a comprehensive evaluation report."""
        print("Generating comprehensive Phase 3 evaluation report...")
        
        # Create report directory
        os.makedirs(save_path, exist_ok=True)
        
        # Generate report
        report = {
            'evaluation_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_experiments': len(self.benchmark_results) + len(self.ablation_results) + len(self.robustness_results),
                'device_used': self.device,
                'random_seed': self.random_seed
            },
            'benchmark_results': self.benchmark_results,
            'ablation_results': self.ablation_results,
            'robustness_results': self.robustness_results,
            'novel_experiment_results': self.novel_experiment_results
        }
        
        # Save comprehensive report
        with open(os.path.join(save_path, 'comprehensive_report.json'), 'w') as f:
            json.dump(convert_for_json(report), f, indent=2)
        
        # Create summary plots
        self.create_summary_plots(save_path)
        
        print(f"Comprehensive report generated and saved to {save_path}")
        
        return report
    
    def create_summary_plots(self, save_path: str):
        """Create summary plots for the comprehensive report."""
        # This is a placeholder for summary visualization
        # In practice, this would create comprehensive plots summarizing all results
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Phase 3 CausalUnit Evaluation\nComprehensive Summary', 
                ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.savefig(os.path.join(save_path, 'summary_placeholder.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main evaluation function."""
    print("Phase 3 CausalUnit Evaluation Framework")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = CausalUnitEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Base network configuration
    base_config = {
        'input_dim': 3,  # Fixed to match synthetic data (3 features)
        'hidden_dims': [8, 8],
        'output_dim': 1,
        'activation': 'relu',
        'enable_structure_learning': True,
        'enable_gradient_surgery': True,
        'use_low_rank_adjacency': True,  # Enable parameter efficiency
        'adjacency_rank': 2  # Rank for low-rank factorization
    }
    
    # Run comprehensive evaluation
    
    # 1. Benchmark evaluation
    network = CausalUnitNetwork(**base_config)
    benchmark_results = evaluator.run_benchmark_evaluation(network)
    
    # 2. Ablation study
    x, y, true_adjacency = evaluator.create_test_datasets()
    dataset = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    ablation_results = evaluator.run_ablation_study(
        base_config, train_loader, test_loader, true_adjacency
    )
    
    # 3. Robustness testing
    robustness_results = evaluator.run_robustness_testing(base_config, n_seeds=3)
    
    # 4. Novel experiments
    novel_results = evaluator.run_novel_experiments(base_config)
    
    # 5. Generate comprehensive report
    report = evaluator.generate_comprehensive_report()
    
    print("\nPhase 3 Evaluation completed successfully!")
    print("Check the results/ directory for detailed findings.")


if __name__ == "__main__":
    main() 