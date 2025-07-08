import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class CounterfactualSimulator:
    """
    Handles counterfactual reasoning for causal models.
    
    Implements Pearl's three-level hierarchy:
    1. Association (P(Y|X)) - standard prediction
    2. Intervention (P(Y|do(X))) - causal intervention
    3. Counterfactual (P(Y_x|X',Y')) - what would have happened
    """
    
    def __init__(self, model):
        """
        Initialize counterfactual simulator.
        
        Args:
            model: The causal model (CausalMLP or CausalUnit)
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode for consistent results
    
    def simulate_counterfactual(self, factual_x, do_mask, do_values, 
                              original_outcome=None):
        """
        Simulate counterfactual outcome: "What would Y have been if X had been different?"
        
        This implements the counterfactual query P(Y_{do(X=x')} | X=x, Y=y)
        
        Args:
            factual_x: Original input tensor (batch_size, input_dim)
            do_mask: Binary mask indicating which variables to intervene on
            do_values: Values to set for intervened variables
            original_outcome: Optional original outcome for reference
            
        Returns:
            counterfactual_outcome: Predicted outcome under intervention
        """
        with torch.no_grad():
            # Create intervention specification
            if isinstance(self.model, type(self.model)) and hasattr(self.model, 'forward'):
                # For CausalMLP with interventions parameter
                interventions = {0: (do_mask, do_values)}
                counterfactual_outcome = self.model(factual_x, interventions=interventions)
            else:
                # For basic CausalUnit
                counterfactual_outcome = self.model(factual_x, do_mask=do_mask, do_values=do_values)
        
        return counterfactual_outcome
    
    def compute_counterfactual_effect(self, factual_x, do_mask, do_values):
        """
        Compute the causal effect of an intervention as the difference between
        factual and counterfactual outcomes.
        
        Args:
            factual_x: Original input
            do_mask: Intervention mask
            do_values: Intervention values
            
        Returns:
            effect: Counterfactual effect (difference in outcomes)
            factual_outcome: Outcome without intervention
            counterfactual_outcome: Outcome with intervention
        """
        # Get factual outcome (no intervention)
        factual_outcome = self.model(factual_x)
        
        # Get counterfactual outcome (with intervention)
        counterfactual_outcome = self.simulate_counterfactual(
            factual_x, do_mask, do_values
        )
        
        # Compute effect as difference
        effect = counterfactual_outcome - factual_outcome
        
        return effect, factual_outcome, counterfactual_outcome
    
    def batch_counterfactual_analysis(self, X, interventions_list):
        """
        Perform counterfactual analysis on a batch of inputs with multiple interventions.
        
        Args:
            X: Batch of inputs (batch_size, input_dim)
            interventions_list: List of (do_mask, do_values) tuples
            
        Returns:
            results: Dictionary with outcomes for each intervention
        """
        results = {
            'factual': self.model(X),
            'counterfactuals': [],
            'effects': []
        }
        
        for i, (do_mask, do_values) in enumerate(interventions_list):
            effect, _, counterfactual = self.compute_counterfactual_effect(
                X, do_mask, do_values
            )
            
            results['counterfactuals'].append(counterfactual)
            results['effects'].append(effect)
        
        return results
    
    def evaluate_counterfactual_accuracy(self, X, true_effects, do_masks, do_values_list):
        """
        Evaluate how accurately the model predicts counterfactual effects.
        
        Args:
            X: Input data
            true_effects: Ground truth causal effects
            do_masks: List of intervention masks
            do_values_list: List of intervention values
            
        Returns:
            accuracy_metrics: Dictionary with various accuracy measures
        """
        predicted_effects = []
        
        for do_mask, do_values in zip(do_masks, do_values_list):
            effect, _, _ = self.compute_counterfactual_effect(X, do_mask, do_values)
            predicted_effects.append(effect)
        
        predicted_effects = torch.stack(predicted_effects, dim=0)
        
        # Compute accuracy metrics
        mse = torch.mean((predicted_effects - true_effects) ** 2)
        mae = torch.mean(torch.abs(predicted_effects - true_effects))
        
        # Correlation between predicted and true effects
        pred_flat = predicted_effects.flatten()
        true_flat = true_effects.flatten()
        correlation = torch.corrcoef(torch.stack([pred_flat, true_flat]))[0, 1]
        
        return {
            'counterfactual_mse': mse.item(),
            'counterfactual_mae': mae.item(),
            'effect_correlation': correlation.item() if not torch.isnan(correlation) else 0.0,
            'predicted_effects': predicted_effects,
            'true_effects': true_effects
        }


def generate_counterfactual_data(n_samples=500, input_dim=3, noise_std=0.1):
    """
    Generate synthetic data with known counterfactual relationships for testing.
    
    Uses the same causal model as Phase 1: y = x1 + 2*x2 - 3*x3 + noise
    But also generates counterfactual examples with known effects.
    
    Args:
        n_samples: Number of samples to generate
        input_dim: Number of input features
        noise_std: Standard deviation of noise
        
    Returns:
        X: Input features
        y: Factual outcomes
        counterfactual_data: Dictionary with counterfactual examples
    """
    # Generate input features
    X = torch.randn(n_samples, input_dim)
    
    # True causal coefficients
    true_coeffs = torch.tensor([1.0, 2.0, -3.0])
    
    # Generate factual outcomes
    y = torch.sum(X * true_coeffs, dim=1, keepdim=True) + noise_std * torch.randn(n_samples, 1)
    
    # Generate counterfactual examples
    counterfactual_data = {}
    
    # Example 1: do(x2 = 0.5)
    do_mask_x2 = torch.tensor([0.0, 1.0, 0.0])
    do_values_x2 = torch.tensor([0.0, 0.5, 0.0])
    
    X_cf_x2 = X.clone()
    X_cf_x2[:, 1] = 0.5  # Set x2 = 0.5 for all samples
    y_cf_x2 = torch.sum(X_cf_x2 * true_coeffs, dim=1, keepdim=True) + noise_std * torch.randn(n_samples, 1)
    
    counterfactual_data['do_x2_0.5'] = {
        'X_counterfactual': X_cf_x2,
        'y_counterfactual': y_cf_x2,
        'do_mask': do_mask_x2,
        'do_values': do_values_x2,
        'true_effect': y_cf_x2 - y  # True counterfactual effect
    }
    
    # Example 2: do(x1 = 1.0)
    do_mask_x1 = torch.tensor([1.0, 0.0, 0.0])
    do_values_x1 = torch.tensor([1.0, 0.0, 0.0])
    
    X_cf_x1 = X.clone()
    X_cf_x1[:, 0] = 1.0  # Set x1 = 1.0 for all samples
    y_cf_x1 = torch.sum(X_cf_x1 * true_coeffs, dim=1, keepdim=True) + noise_std * torch.randn(n_samples, 1)
    
    counterfactual_data['do_x1_1.0'] = {
        'X_counterfactual': X_cf_x1,
        'y_counterfactual': y_cf_x1,
        'do_mask': do_mask_x1,
        'do_values': do_values_x1,
        'true_effect': y_cf_x1 - y
    }
    
    return X, y, counterfactual_data


class CounterfactualLoss(nn.Module):
    """
    Loss function for training models with counterfactual reasoning.
    """
    
    def __init__(self, factual_weight=1.0, counterfactual_weight=0.5):
        super(CounterfactualLoss, self).__init__()
        self.factual_weight = factual_weight
        self.counterfactual_weight = counterfactual_weight
    
    def forward(self, model, X, y_factual, counterfactual_data):
        """
        Compute combined factual and counterfactual loss.
        
        Args:
            model: The causal model
            X: Input features
            y_factual: Factual outcomes
            counterfactual_data: Dictionary with counterfactual examples
            
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary with individual loss components
        """
        # Factual loss (standard prediction)
        y_pred_factual = model(X)
        factual_loss = nn.MSELoss()(y_pred_factual, y_factual)
        
        # Counterfactual loss
        cf_loss = 0.0
        cf_count = 0
        
        for cf_name, cf_data in counterfactual_data.items():
            # Predict counterfactual outcome
            interventions = {0: (cf_data['do_mask'].unsqueeze(0).expand(X.shape[0], -1),
                               cf_data['do_values'].unsqueeze(0).expand(X.shape[0], -1))}
            y_pred_cf = model(X, interventions=interventions)
            
            # Compare with true counterfactual outcome
            cf_loss += nn.MSELoss()(y_pred_cf, cf_data['y_counterfactual'])
            cf_count += 1
        
        if cf_count > 0:
            cf_loss = cf_loss / cf_count
        
        # Combined loss
        total_loss = (self.factual_weight * factual_loss + 
                     self.counterfactual_weight * cf_loss)
        
        return total_loss, {
            'factual_loss': factual_loss.item(),
            'counterfactual_loss': cf_loss.item() if cf_count > 0 else 0.0,
            'total_loss': total_loss.item()
        } 