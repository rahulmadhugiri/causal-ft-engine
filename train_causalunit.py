import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from causal_unit_network import CausalUnitNetwork
from engine.loss_functions import CausalLosses, CausalMetrics
from experiments.utils import (
    generate_synthetic_data,
    create_dag_from_edges,
    evaluate_structure_learning
)


class CausalUnitTrainer:
    """
    Phase 3 CausalUnit Trainer: Training loop with interventions, gradient blocking, and ablation studies.
    
    Key Features:
    1. Probability-based interventions during training
    2. Gradient blocking for causal interventions
    3. Joint structure learning and prediction training
    4. Comprehensive ablation studies
    5. Counterfactual loss integration
    6. Gradient flow visualization and debugging
    """
    
    def __init__(self, 
                 network: CausalUnitNetwork,
                 intervention_prob: float = 0.3,
                 multi_intervention_prob: float = 0.1,
                 counterfactual_weight: float = 1.0,
                 structure_weight: float = 0.1,
                 learning_rate: float = 0.001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.network = network.to(device)
        self.intervention_prob = intervention_prob
        self.multi_intervention_prob = multi_intervention_prob
        self.counterfactual_weight = counterfactual_weight
        self.structure_weight = structure_weight
        self.device = device
        
        # Optimizers
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=50, factor=0.5)
        
        # Loss functions
        self.prediction_loss_fn = nn.MSELoss()
        self.causal_losses = CausalLosses(
            counterfactual_weight=counterfactual_weight,
            structure_weight=structure_weight
        )
        
        # Training state
        self.training_history = {
            'epoch': [],
            'prediction_loss': [],
            'counterfactual_loss': [],
            'structure_loss': [],
            'total_loss': [],
            'intervention_rate': [],
            'gradient_norms': [],
            'structure_accuracy': [],
            'counterfactual_accuracy': []
        }
        
        # Ablation settings
        self.ablation_config = {
            'enable_interventions': True,
            'enable_gradient_blocking': True,
            'enable_dynamic_rewiring': True,
            'enable_structure_learning': True,
            'enable_gradient_surgery': True
        }
        
        # Debugging
        self.debug_mode = False
        self.gradient_flow_history = []
    
    def set_ablation_config(self, **config):
        """Set ablation study configuration."""
        self.ablation_config.update(config)
        
        # Update network settings
        self.network.set_training_mode(
            structure_learning=self.ablation_config['enable_structure_learning'],
            intervention_training=self.ablation_config['enable_interventions']
        )
        
        self.network.enable_gradient_surgery_all(
            self.ablation_config['enable_gradient_surgery']
        )
    
    def compute_counterfactual_loss(self, 
                                   x: torch.Tensor, 
                                   y: torch.Tensor,
                                   true_adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute counterfactual loss for causal learning.
        
        Args:
            x: Input data (batch_size, input_dim)
            y: Target data (batch_size, output_dim)
            true_adjacency: Optional true causal structure
            
        Returns:
            Counterfactual loss tensor
        """
        if not self.ablation_config['enable_interventions']:
            return torch.tensor(0.0, device=self.device)
        
        batch_size = x.shape[0]
        counterfactual_losses = []
        
        # Generate counterfactual samples
        for _ in range(min(5, batch_size)):  # Limit for efficiency
            # Random intervention
            intervention_node = np.random.randint(0, x.shape[1])
            intervention_value = torch.randn(1, device=self.device) * 2.0
            
            # Create intervention
            intervention_mask = torch.zeros(x.shape[1], device=self.device)
            intervention_values = torch.zeros(x.shape[1], device=self.device)
            intervention_mask[intervention_node] = 1.0
            intervention_values[intervention_node] = intervention_value
            
            # Apply intervention to a random subset of batch
            intervention_indices = np.random.choice(
                batch_size, size=min(4, batch_size), replace=False
            )
            
            interventions = []
            for i in range(batch_size):
                if i in intervention_indices:
                    interventions.append({'counterfactual': (intervention_mask, intervention_values)})
                else:
                    interventions.append({})
            
            # Forward pass with intervention
            y_counterfactual = self.network(x, interventions=interventions)
            
            # Compute counterfactual loss
            cf_loss = self.causal_losses.counterfactual_loss(
                y, y_counterfactual, y, y_counterfactual
            )
            counterfactual_losses.append(cf_loss)
        
        return torch.mean(torch.stack(counterfactual_losses)) if counterfactual_losses else torch.tensor(0.0, device=self.device)
    
    def train_epoch(self, 
                    train_loader: torch.utils.data.DataLoader,
                    true_adjacency: Optional[torch.Tensor] = None,
                    epoch: int = 0) -> Dict[str, float]:
        """
        Train for one epoch with interventions and gradient blocking.
        
        Args:
            train_loader: Training data loader
            true_adjacency: Optional true causal structure for evaluation
            epoch: Current epoch number
            
        Returns:
            Dictionary with loss metrics
        """
        self.network.train()
        
        epoch_metrics = {
            'prediction_loss': 0.0,
            'counterfactual_loss': 0.0,
            'structure_loss': 0.0,
            'total_loss': 0.0,
            'intervention_rate': 0.0,
            'gradient_norm': 0.0,
            'num_batches': 0
        }
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.shape[0]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with interventions
            if self.ablation_config['enable_interventions']:
                output, structure_info = self.network(
                    x, 
                    intervention_prob=self.intervention_prob,
                    return_structure=True
                )
            else:
                output, structure_info = self.network(
                    x,
                    interventions=None,
                    return_structure=True
                )
            
            # Compute losses
            prediction_loss = self.prediction_loss_fn(output, y)
            
            # Counterfactual loss
            counterfactual_loss = self.compute_counterfactual_loss(x, y, true_adjacency)
            
            # Structure learning loss
            structure_loss = self.network.get_structure_learning_loss(x)
            
            # Total loss
            total_loss = (prediction_loss + 
                         self.counterfactual_weight * counterfactual_loss + 
                         self.structure_weight * structure_loss)
            
            # Backward pass with gradient blocking
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Compute gradient norm
            grad_norm = 0.0
            for param in self.network.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Update metrics
            epoch_metrics['prediction_loss'] += prediction_loss.item()
            epoch_metrics['counterfactual_loss'] += counterfactual_loss.item()
            epoch_metrics['structure_loss'] += structure_loss.item()
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['gradient_norm'] += grad_norm
            epoch_metrics['num_batches'] += 1
            
            # Track intervention rate
            if self.network.intervention_schedule is not None:
                intervention_rate = len([i for i in self.network.intervention_schedule if i]) / batch_size
            else:
                intervention_rate = 0.0
            epoch_metrics['intervention_rate'] += intervention_rate
            
            # Debug information
            if self.debug_mode and batch_idx % 100 == 0:
                self.gradient_flow_history.append(
                    self.network.visualize_gradient_flow()
                )
                
                print(f"Batch {batch_idx}: "
                      f"Pred Loss: {prediction_loss.item():.4f}, "
                      f"CF Loss: {counterfactual_loss.item():.4f}, "
                      f"Struct Loss: {structure_loss.item():.4f}, "
                      f"Intervention Rate: {intervention_rate:.2f}")
        
        # Average metrics
        for key in epoch_metrics:
            if key != 'num_batches':
                epoch_metrics[key] /= epoch_metrics['num_batches']
        
        return epoch_metrics
    
    def evaluate_epoch(self, 
                       test_loader: torch.utils.data.DataLoader,
                       true_adjacency: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluate the network on test data.
        
        Args:
            test_loader: Test data loader
            true_adjacency: Optional true causal structure for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.network.eval()
        
        eval_metrics = {
            'test_loss': 0.0,
            'structure_accuracy': 0.0,
            'counterfactual_accuracy': 0.0,
            'num_batches': 0
        }
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass without interventions
                output = self.network(x, interventions=None)
                test_loss = self.prediction_loss_fn(output, y)
                
                eval_metrics['test_loss'] += test_loss.item()
                eval_metrics['num_batches'] += 1
                
                # Evaluate structure learning
                if true_adjacency is not None:
                    learned_adjacency = self.network.get_adjacency_matrix(hard=True)
                    structure_metrics = evaluate_structure_learning(
                        learned_adjacency.cpu().numpy(),
                        true_adjacency.cpu().numpy()
                    )
                    eval_metrics['structure_accuracy'] += structure_metrics['f1']
                
                # Evaluate counterfactual performance
                if self.ablation_config['enable_interventions']:
                    # Simple counterfactual test - compute intervention effects
                    with torch.no_grad():
                        y_original = self.network(x, interventions=None)
                        
                        # Random intervention
                        intervention_node = torch.randint(0, x.shape[1], (1,)).item()
                        intervention_value = torch.randn(1, device=x.device) * 2.0
                        
                        # Create intervention
                        intervention_mask = torch.zeros(x.shape[1], device=x.device)
                        intervention_values = torch.zeros(x.shape[1], device=x.device)
                        intervention_mask[intervention_node] = 1.0
                        intervention_values[intervention_node] = intervention_value
                        
                        # Apply intervention
                        interventions = []
                        for i in range(x.shape[0]):
                            interventions.append({'test': (intervention_mask, intervention_values)})
                        
                        y_counterfactual = self.network(x, interventions=interventions)
                        intervention_effect = torch.mean(torch.abs(y_counterfactual - y_original))
                        
                    eval_metrics['counterfactual_accuracy'] += intervention_effect.item()
        
        # Average metrics
        for key in eval_metrics:
            if key != 'num_batches':
                eval_metrics[key] /= eval_metrics['num_batches']
        
        return eval_metrics
    
    def train(self, 
              train_loader: torch.utils.data.DataLoader,
              test_loader: torch.utils.data.DataLoader,
              num_epochs: int = 100,
              true_adjacency: Optional[torch.Tensor] = None,
              save_path: str = 'results/phase3_training',
              early_stopping_patience: int = 20) -> Dict[str, Any]:
        """
        Full training loop with interventions, gradient blocking, and evaluation.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of training epochs
            true_adjacency: Optional true causal structure
            save_path: Path to save results
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training results dictionary
        """
        print(f"Starting Phase 3 CausalUnit Training")
        print(f"Ablation Config: {self.ablation_config}")
        print(f"Device: {self.device}")
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, true_adjacency, epoch)
            
            # Evaluate epoch
            eval_metrics = self.evaluate_epoch(test_loader, true_adjacency)
            
            # Update learning rate
            self.scheduler.step(eval_metrics['test_loss'])
            
            # Update training history
            self.training_history['epoch'].append(epoch)
            self.training_history['prediction_loss'].append(train_metrics['prediction_loss'])
            self.training_history['counterfactual_loss'].append(train_metrics['counterfactual_loss'])
            self.training_history['structure_loss'].append(train_metrics['structure_loss'])
            self.training_history['total_loss'].append(train_metrics['total_loss'])
            self.training_history['intervention_rate'].append(train_metrics['intervention_rate'])
            self.training_history['gradient_norms'].append(train_metrics['gradient_norm'])
            self.training_history['structure_accuracy'].append(eval_metrics['structure_accuracy'])
            self.training_history['counterfactual_accuracy'].append(eval_metrics['counterfactual_accuracy'])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}: "
                      f"Train Loss: {train_metrics['total_loss']:.4f}, "
                      f"Test Loss: {eval_metrics['test_loss']:.4f}, "
                      f"Struct Acc: {eval_metrics['structure_accuracy']:.3f}, "
                      f"CF Acc: {eval_metrics['counterfactual_accuracy']:.3f}, "
                      f"Intervention Rate: {train_metrics['intervention_rate']:.2f}")
            
            # Early stopping
            if eval_metrics['test_loss'] < best_loss:
                best_loss = eval_metrics['test_loss']
                patience_counter = 0
                
                # Save best model
                torch.save(self.network.state_dict(), 
                          os.path.join(save_path, 'best_model.pth'))
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Save final results
        results = {
            'training_history': self.training_history,
            'ablation_config': self.ablation_config,
            'final_metrics': {
                'train_loss': train_metrics['total_loss'],
                'test_loss': eval_metrics['test_loss'],
                'structure_accuracy': eval_metrics['structure_accuracy'],
                'counterfactual_accuracy': eval_metrics['counterfactual_accuracy']
            },
            'network_info': self.network.get_network_info()
        }
        
        # Save results
        with open(os.path.join(save_path, 'training_results.json'), 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'network_info'}, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves(save_path)
        
        print(f"Training completed. Results saved to {save_path}")
        
        return results
    
    def plot_training_curves(self, save_path: str):
        """Plot training curves and save to file."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['prediction_loss'], label='Prediction')
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['counterfactual_loss'], label='Counterfactual')
        axes[0, 0].plot(self.training_history['epoch'], self.training_history['structure_loss'], label='Structure')
        axes[0, 0].set_title('Loss Components')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Total loss
        axes[0, 1].plot(self.training_history['epoch'], self.training_history['total_loss'])
        axes[0, 1].set_title('Total Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Intervention rate
        axes[0, 2].plot(self.training_history['epoch'], self.training_history['intervention_rate'])
        axes[0, 2].set_title('Intervention Rate')
        axes[0, 2].set_ylabel('Rate')
        axes[0, 2].grid(True)
        
        # Gradient norms
        axes[1, 0].plot(self.training_history['epoch'], self.training_history['gradient_norms'])
        axes[1, 0].set_title('Gradient Norms')
        axes[1, 0].set_ylabel('Norm')
        axes[1, 0].grid(True)
        
        # Structure accuracy
        axes[1, 1].plot(self.training_history['epoch'], self.training_history['structure_accuracy'])
        axes[1, 1].set_title('Structure Learning Accuracy')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].grid(True)
        
        # Counterfactual accuracy
        axes[1, 2].plot(self.training_history['epoch'], self.training_history['counterfactual_accuracy'])
        axes[1, 2].set_title('Counterfactual Accuracy')
        axes[1, 2].set_ylabel('Correlation')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_ablation_study(self, 
                          train_loader: torch.utils.data.DataLoader,
                          test_loader: torch.utils.data.DataLoader,
                          num_epochs: int = 50,
                          true_adjacency: Optional[torch.Tensor] = None,
                          save_path: str = 'results/phase3_ablation') -> Dict[str, Any]:
        """
        Run comprehensive ablation study.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of epochs per ablation
            true_adjacency: Optional true causal structure
            save_path: Path to save ablation results
            
        Returns:
            Ablation study results
        """
        print("Starting Phase 3 Ablation Study...")
        
        ablation_configs = [
            {'name': 'full', 'config': {}},  # Full model
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
            
            # Reset network
            self.network.reset_network_state()
            
            # Set ablation config
            self.set_ablation_config(**ablation['config'])
            
            # Train
            results = self.train(
                train_loader, test_loader, num_epochs=num_epochs,
                true_adjacency=true_adjacency,
                save_path=os.path.join(save_path, ablation['name'])
            )
            
            ablation_results[ablation['name']] = results['final_metrics']
        
        # Save ablation results
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'ablation_results.json'), 'w') as f:
            json.dump(ablation_results, f, indent=2)
        
        print(f"\nAblation study completed. Results saved to {save_path}")
        
        return ablation_results
    
    def enable_debug_mode(self, enable: bool = True):
        """Enable/disable debug mode for detailed gradient flow tracking."""
        self.debug_mode = enable 