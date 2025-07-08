import torch
import torch.nn.functional as F

class CausalLosses:
    """
    Collection of causal-aware loss functions for training neural networks
    with do-operator interventions.
    """
    
    def __init__(self, intervention_weight=1.0, counterfactual_weight=0.5):
        """
        Initialize causal loss functions.
        
        Args:
            intervention_weight: Weight for intervention loss component
            counterfactual_weight: Weight for counterfactual loss component
        """
        self.intervention_weight = intervention_weight
        self.counterfactual_weight = counterfactual_weight
    
    def standard_loss(self, predicted_output, target_output, loss_type='mse'):
        """
        Standard supervised learning loss.
        
        Args:
            predicted_output: Model predictions
            target_output: True targets
            loss_type: Type of loss ('mse', 'cross_entropy', 'mae')
            
        Returns:
            Computed loss
        """
        if loss_type == 'mse':
            return F.mse_loss(predicted_output, target_output)
        elif loss_type == 'cross_entropy':
            return F.cross_entropy(predicted_output, target_output)
        elif loss_type == 'mae':
            return F.l1_loss(predicted_output, target_output)
        else:
            # Default to MSE
            return F.mse_loss(predicted_output, target_output)
    
    def causal_intervention_loss(self, predicted_output, target_output, do_mask=None, 
                               ignore_intervened=True, loss_type='mse'):
        """
        Causal intervention loss that can optionally ignore loss on intervened variables.
        
        When we intervene on variables, we might want to ignore prediction error
        on those variables since they're being externally controlled.
        
        Args:
            predicted_output: Model predictions
            target_output: True targets  
            do_mask: Binary mask indicating intervened variables
            ignore_intervened: Whether to ignore loss on intervened variables
            loss_type: Type of base loss function
            
        Returns:
            Causal intervention loss
        """
        if do_mask is None or not ignore_intervened:
            # No interventions or we want to include all variables
            return self.standard_loss(predicted_output, target_output, loss_type)
        
        # Create mask for non-intervened variables
        if do_mask.dim() == 1:
            # Expand to match batch dimension
            batch_size = predicted_output.shape[0]
            do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
        
        # Mask for variables we want to include in loss (non-intervened)
        loss_mask = (do_mask == 0).float()
        
        # Apply mask to predictions and targets
        if loss_type == 'mse':
            # Compute element-wise squared error
            se = (predicted_output - target_output) ** 2
            # Apply mask and compute mean only over non-masked elements
            masked_se = se * loss_mask
            return masked_se.sum() / loss_mask.sum().clamp(min=1)
        
        elif loss_type == 'mae':
            # Compute element-wise absolute error  
            ae = torch.abs(predicted_output - target_output)
            # Apply mask and compute mean
            masked_ae = ae * loss_mask
            return masked_ae.sum() / loss_mask.sum().clamp(min=1)
        
        else:
            # For other loss types, fall back to standard loss
            return self.standard_loss(predicted_output, target_output, loss_type)
    
    def counterfactual_loss(self, pred_factual, pred_counterfactual, 
                          target_factual, target_counterfactual):
        """
        Loss to encourage correct counterfactual predictions.
        
        This loss encourages the model to predict what would have happened
        under different interventions.
        
        Args:
            pred_factual: Predictions under original conditions
            pred_counterfactual: Predictions under counterfactual conditions
            target_factual: True outcomes under original conditions
            target_counterfactual: True outcomes under counterfactual conditions
            
        Returns:
            Counterfactual loss
        """
        factual_loss = F.mse_loss(pred_factual, target_factual)
        counterfactual_loss = F.mse_loss(pred_counterfactual, target_counterfactual)
        
        return factual_loss + counterfactual_loss
    
    def causal_consistency_loss(self, pred_no_intervention, pred_with_intervention, 
                              do_mask, do_values):
        """
        Loss to enforce causal consistency: when we intervene on X=v,
        the predicted value of X should equal v.
        
        Args:
            pred_no_intervention: Predictions without intervention
            pred_with_intervention: Predictions with intervention
            do_mask: Binary mask of intervened variables
            do_values: Values of interventions
            
        Returns:
            Causal consistency loss
        """
        if do_mask is None or do_values is None:
            return torch.tensor(0.0)
        
        # Ensure proper dimensions
        if do_mask.dim() == 1:
            batch_size = pred_with_intervention.shape[0]
            do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
            do_values = do_values.unsqueeze(0).expand(batch_size, -1)
        
        # For intervened variables, prediction should match intervention value
        intervened_mask = do_mask.bool()
        if intervened_mask.any():
            intervened_predictions = pred_with_intervention[intervened_mask]
            intervened_targets = do_values[intervened_mask]
            return F.mse_loss(intervened_predictions, intervened_targets)
        
        return torch.tensor(0.0)
    
    def total_causal_loss(self, pred_output, target_output, do_mask=None, 
                         do_values=None, pred_counterfactual=None, 
                         target_counterfactual=None, loss_type='mse'):
        """
        Total causal loss combining multiple causal objectives.
        
        Args:
            pred_output: Model predictions
            target_output: True targets
            do_mask: Binary mask of interventions
            do_values: Values of interventions
            pred_counterfactual: Counterfactual predictions (optional)
            target_counterfactual: Counterfactual targets (optional)
            loss_type: Base loss function type
            
        Returns:
            Combined causal loss
        """
        # Standard intervention loss
        intervention_loss = self.causal_intervention_loss(
            pred_output, target_output, do_mask, loss_type=loss_type
        )
        
        total_loss = self.intervention_weight * intervention_loss
        
        # Add counterfactual loss if available
        if pred_counterfactual is not None and target_counterfactual is not None:
            cf_loss = self.counterfactual_loss(
                pred_output, pred_counterfactual, 
                target_output, target_counterfactual
            )
            total_loss += self.counterfactual_weight * cf_loss
        
        return total_loss
    
    def causal_regularization_loss(self, model, lambda_reg=0.01):
        """
        Regularization loss to encourage sparse causal structures.
        
        This can help in learning interpretable causal relationships.
        
        Args:
            model: The neural network model
            lambda_reg: Regularization strength
            
        Returns:
            Regularization loss
        """
        reg_loss = 0.0
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                # L1 regularization to encourage sparsity
                reg_loss += torch.abs(param).sum()
        
        return lambda_reg * reg_loss


class CausalMetrics:
    """
    Metrics for evaluating causal models.
    """
    
    @staticmethod
    def intervention_accuracy(pred_intervention, target_intervention, do_mask):
        """
        Compute accuracy specifically on intervened variables.
        
        Args:
            pred_intervention: Predictions under intervention
            target_intervention: True values under intervention
            do_mask: Binary mask of intervened variables
            
        Returns:
            Mean squared error on intervened variables
        """
        if do_mask is None or not do_mask.any():
            return torch.tensor(0.0)
        
        if do_mask.dim() == 1:
            batch_size = pred_intervention.shape[0]
            do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
        
        intervened_mask = do_mask.bool()
        if intervened_mask.any():
            intervened_pred = pred_intervention[intervened_mask]
            intervened_target = target_intervention[intervened_mask]
            return F.mse_loss(intervened_pred, intervened_target)
        
        return torch.tensor(0.0)
    
    @staticmethod
    def causal_effect_estimation_error(true_effect, estimated_effect):
        """
        Measure error in causal effect estimation.
        
        Args:
            true_effect: True causal effect
            estimated_effect: Estimated causal effect
            
        Returns:
            Mean absolute error in effect estimation
        """
        return torch.abs(true_effect - estimated_effect).mean()
