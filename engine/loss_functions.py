import torch
import torch.nn.functional as F
import math

class CausalLosses:
    """
    Collection of causal-aware loss functions for training neural networks
    with do-operator interventions, structure learning, and counterfactual reasoning.
    """
    
    def __init__(self, intervention_weight=1.0, counterfactual_weight=0.5, 
                 structure_weight=0.1, sparsity_weight=0.01):
        """
        Initialize causal loss functions.
        
        Args:
            intervention_weight: Weight for intervention loss component
            counterfactual_weight: Weight for counterfactual loss component
            structure_weight: Weight for structure learning loss
            sparsity_weight: Weight for sparsity regularization
        """
        self.intervention_weight = intervention_weight
        self.counterfactual_weight = counterfactual_weight
        self.structure_weight = structure_weight
        self.sparsity_weight = sparsity_weight
    
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
    
    def structure_reconstruction_loss(self, model_output, target_input, adjacency_matrix):
        """
        Loss for structure learning: how well can we reconstruct input features
        based on learned causal relationships.
        
        Args:
            model_output: Reconstructed features from structure learner
            target_input: Original input features
            adjacency_matrix: Learned adjacency matrix
            
        Returns:
            Structure reconstruction loss
        """
        reconstruction_loss = F.mse_loss(model_output, target_input)
        return reconstruction_loss
    
    def acyclicity_loss(self, adjacency_matrix):
        """
        Acyclicity constraint loss to ensure learned structure is a DAG.
        
        Uses matrix exponential trace: trace(e^(A ⊙ A)) - d should be 0 for DAGs
        
        Args:
            adjacency_matrix: Learned adjacency matrix
            
        Returns:
            Acyclicity constraint loss
        """
        if adjacency_matrix is None:
            return torch.tensor(0.0)
        
        num_variables = adjacency_matrix.shape[0]
        A_squared = adjacency_matrix * adjacency_matrix
        
        # Matrix exponential using series expansion (truncated)
        exp_A = torch.eye(num_variables, device=adjacency_matrix.device)
        A_power = torch.eye(num_variables, device=adjacency_matrix.device)
        
        for i in range(1, 10):  # Truncate at 10 terms
            A_power = torch.matmul(A_power, A_squared)
            exp_A = exp_A + A_power / math.factorial(i)
        
        # Trace of exponential minus number of variables
        constraint = torch.trace(exp_A) - num_variables
        return torch.abs(constraint)
    
    def sparsity_loss(self, adjacency_matrix):
        """
        Sparsity regularization to encourage sparse causal graphs.
        
        Args:
            adjacency_matrix: Learned adjacency matrix
            
        Returns:
            Sparsity loss (L1 norm)
        """
        if adjacency_matrix is None:
            return torch.tensor(0.0)
        return torch.sum(torch.abs(adjacency_matrix))
    
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
    
    def phase2_total_loss(self, predictions, targets, counterfactual_data=None,
                         structure_info=None, interventions=None):
        """
        Total Phase 2 loss combining prediction, counterfactual, and structure learning.
        
        Args:
            predictions: Model predictions (can be None during structure-only training)
            targets: True targets
            counterfactual_data: Dictionary with counterfactual examples
            structure_info: Dictionary with structure learning outputs
            interventions: Applied interventions
            
        Returns:
            total_loss: Combined loss
            loss_components: Dictionary with individual loss components
        """
        loss_components = {}
        
        # Initialize total loss
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Standard prediction loss (only if predictions are provided)
        if predictions is not None:
            prediction_loss = self.standard_loss(predictions, targets)
            loss_components['prediction_loss'] = prediction_loss.item()
            total_loss = total_loss + prediction_loss
        else:
            loss_components['prediction_loss'] = 0.0
        
        # Counterfactual loss
        if counterfactual_data is not None:
            cf_loss = 0.0
            cf_count = 0
            
            for cf_name, cf_data in counterfactual_data.items():
                if 'y_counterfactual' in cf_data and 'predicted_counterfactual' in cf_data:
                    cf_loss += F.mse_loss(cf_data['predicted_counterfactual'], 
                                        cf_data['y_counterfactual'])
                    cf_count += 1
            
            if cf_count > 0:
                cf_loss = cf_loss / cf_count
                total_loss = total_loss + self.counterfactual_weight * cf_loss
                loss_components['counterfactual_loss'] = cf_loss.item()
        
        # Structure learning losses
        if structure_info is not None:
            # Reconstruction loss
            if 'reconstruction' in structure_info and 'original_input' in structure_info:
                structure_recon_loss = self.structure_reconstruction_loss(
                    structure_info['reconstruction'], structure_info['original_input'],
                    structure_info.get('adjacency')
                )
                total_loss = total_loss + self.structure_weight * structure_recon_loss
                loss_components['structure_reconstruction_loss'] = structure_recon_loss.item()
            
            # Acyclicity constraint
            if 'adjacency' in structure_info:
                acyc_loss = self.acyclicity_loss(structure_info['adjacency'])
                total_loss = total_loss + self.structure_weight * acyc_loss
                loss_components['acyclicity_loss'] = acyc_loss.item()
                
                # Sparsity regularization
                sparse_loss = self.sparsity_loss(structure_info['adjacency'])
                total_loss = total_loss + self.sparsity_weight * sparse_loss
                loss_components['sparsity_loss'] = sparse_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss, loss_components
    
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
    Metrics for evaluating causal models with Phase 2 enhancements.
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
    
    @staticmethod
    def structure_recovery_metrics(learned_adjacency, true_adjacency):
        """
        Evaluate structure recovery performance.
        
        Args:
            learned_adjacency: Learned adjacency matrix
            true_adjacency: True adjacency matrix
            
        Returns:
            Dictionary with structure recovery metrics
        """
        # Convert to binary matrices
        learned_binary = (learned_adjacency > 0.5).float()
        true_binary = true_adjacency.float()
        
        # True positives, false positives, false negatives
        tp = torch.sum(learned_binary * true_binary)
        fp = torch.sum(learned_binary * (1 - true_binary))
        fn = torch.sum((1 - learned_binary) * true_binary)
        tn = torch.sum((1 - learned_binary) * (1 - true_binary))
        
        # Metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Structural Hamming Distance (SHD)
        shd = torch.sum(torch.abs(learned_binary - true_binary))
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1_score.item(),
            'accuracy': accuracy.item(),
            'shd': shd.item(),
            'learned_edges': torch.sum(learned_binary).item(),
            'true_edges': torch.sum(true_binary).item()
        }
    
    @staticmethod
    def counterfactual_accuracy(predicted_effects, true_effects):
        """
        Evaluate counterfactual prediction accuracy.
        
        Args:
            predicted_effects: Predicted counterfactual effects
            true_effects: True counterfactual effects
            
        Returns:
            Dictionary with counterfactual accuracy metrics
        """
        mse = F.mse_loss(predicted_effects, true_effects)
        mae = F.l1_loss(predicted_effects, true_effects)
        
        # Correlation between predicted and true effects
        pred_flat = predicted_effects.flatten()
        true_flat = true_effects.flatten()
        
        if len(pred_flat) > 1:
            correlation = torch.corrcoef(torch.stack([pred_flat, true_flat]))[0, 1]
            correlation = correlation.item() if not torch.isnan(correlation) else 0.0
        else:
            correlation = 0.0
        
        return {
            'counterfactual_mse': mse.item(),
            'counterfactual_mae': mae.item(),
            'effect_correlation': correlation
        }
