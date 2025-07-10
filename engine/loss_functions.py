import torch
import torch.nn.functional as F
import math
from typing import Optional, List

class CausalLosses:
    """
    Collection of causal-aware loss functions for training neural networks
    with do-operator interventions, structure learning, and counterfactual reasoning.
    """
    
    def __init__(self, intervention_weight=1.0, counterfactual_weight=0.5, 
                 structure_weight=0.1, sparsity_weight=0.01, 
                 counterfactual_structure_weight=0.2):
        """
        Initialize causal loss functions.
        
        Args:
            intervention_weight: Weight for intervention loss component
            counterfactual_weight: Weight for counterfactual loss component
            structure_weight: Weight for structure learning loss
            sparsity_weight: Weight for sparsity regularization
            counterfactual_structure_weight: Weight for counterfactual structure feedback
        """
        self.intervention_weight = intervention_weight
        self.counterfactual_weight = counterfactual_weight
        self.structure_weight = structure_weight
        self.sparsity_weight = sparsity_weight
        self.counterfactual_structure_weight = counterfactual_structure_weight
    
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
    
    def counterfactual_structure_loss(self, 
                                    factual_output: torch.Tensor,
                                    counterfactual_output: torch.Tensor,
                                    intervention_mask: torch.Tensor,
                                    intervention_values: torch.Tensor,
                                    adjacency_matrix: torch.Tensor,
                                    expected_effect_direction: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Counterfactual structure loss: measures how well the current adjacency matrix
        predicts intervention outcomes and provides gradients to update structure.
        
        This implements the critical feedback loop:
        1. Predict expected counterfactual effect based on current structure
        2. Compare to actual observed counterfactual effect
        3. Penalize adjacency matrix when structure predictions are wrong
        
        Args:
            factual_output: Output without intervention (batch_size, output_dim)
            counterfactual_output: Output with intervention (batch_size, output_dim)
            intervention_mask: Binary mask of intervened variables (batch_size, input_dim)
            intervention_values: Values of interventions (batch_size, input_dim)
            adjacency_matrix: Current adjacency matrix (input_dim, input_dim)
            expected_effect_direction: Optional expected direction of effects
            
        Returns:
            Counterfactual structure loss that provides gradients to adjacency matrix
        """
        if adjacency_matrix is None:
            return torch.tensor(0.0, device=factual_output.device)
        
        batch_size = factual_output.shape[0]
        input_dim = adjacency_matrix.shape[0]
        
        # Compute actual counterfactual effect
        actual_effect = counterfactual_output - factual_output  # (batch_size, output_dim)
        
        # For each intervention, predict expected effect based on current structure
        total_structure_error = torch.tensor(0.0, device=factual_output.device)
        
        for batch_idx in range(batch_size):
            # Get intervention for this batch item
            intervened_nodes = intervention_mask[batch_idx].bool()  # (input_dim,)
            
            if not intervened_nodes.any():
                continue
                
            # Get intervention values for this batch
            intervention_vals = intervention_values[batch_idx]  # (input_dim,)
            
            # Predict effect based on current adjacency structure
            # Effect should propagate through causal paths from intervened nodes
            predicted_effect = torch.zeros_like(actual_effect[batch_idx])  # (output_dim,)
            
            for node_idx in torch.where(intervened_nodes)[0]:
                # Find all downstream nodes (causally influenced by this intervention)
                downstream_nodes = adjacency_matrix[node_idx, :]  # (input_dim,)
                
                # Intervention strength should propagate proportionally to edge weights
                intervention_strength = intervention_vals[node_idx]
                
                # Predict effect on downstream nodes
                for downstream_idx in range(input_dim):
                    if downstream_nodes[downstream_idx] > 0.1:  # Significant edge
                        # Effect should be proportional to edge weight and intervention strength
                        edge_weight = downstream_nodes[downstream_idx]
                        predicted_node_effect = edge_weight * intervention_strength
                        
                        # Map input node effect to output (simplified assumption)
                        if downstream_idx < predicted_effect.shape[0]:
                            predicted_effect[downstream_idx] += predicted_node_effect
            
            # Compute error between predicted and actual effect
            effect_error = torch.sum((predicted_effect - actual_effect[batch_idx]) ** 2)
            total_structure_error += effect_error
        
        # Normalize by batch size
        total_structure_error = total_structure_error / batch_size
        
        return total_structure_error

    def counterfactual_consistency_loss(self, 
                                      factual_output: torch.Tensor,
                                      counterfactual_outputs: List[torch.Tensor],
                                      intervention_masks: List[torch.Tensor],
                                      intervention_values: List[torch.Tensor],
                                      adjacency_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        ENHANCED Counterfactual consistency loss: comprehensive penalty for violations
        of causal logic in intervention effects.
        
        This implements multiple consistency checks:
        1. Proportionality: Similar interventions should have proportional effects
        2. Directional consistency: Effects should align with causal graph structure
        3. No spurious effects: Interventions on non-parent nodes shouldn't affect outputs
        4. Monotonicity: Increasing intervention values should produce monotonic effects
        
        Args:
            factual_output: Output without intervention
            counterfactual_outputs: List of outputs under different interventions
            intervention_masks: List of intervention masks
            intervention_values: List of intervention values
            adjacency_matrix: Current adjacency matrix for causal structure validation
            
        Returns:
            Comprehensive consistency loss
        """
        if len(counterfactual_outputs) == 0:
            return torch.tensor(0.0, device=factual_output.device)
        
        consistency_loss = torch.tensor(0.0, device=factual_output.device)
        num_violations = 0
        
        # 1. PROPORTIONALITY CHECK: Similar interventions should have proportional effects
        for i in range(len(counterfactual_outputs)):
            for j in range(i + 1, len(counterfactual_outputs)):
                mask_i = intervention_masks[i]
                mask_j = intervention_masks[j]
                values_i = intervention_values[i]
                values_j = intervention_values[j]
                
                # Check if interventions are on the same variables
                if torch.equal(mask_i, mask_j):
                    # Same intervention nodes
                    effect_i = counterfactual_outputs[i] - factual_output
                    effect_j = counterfactual_outputs[j] - factual_output
                    
                    # Effects should be proportional to intervention values
                    intervened_indices = mask_i.bool()
                    if intervened_indices.any():
                        val_i = values_i[intervened_indices].mean()
                        val_j = values_j[intervened_indices].mean()
                        
                        if abs(val_i) > 1e-6:  # Avoid division by zero
                            expected_ratio = val_j / val_i
                            actual_ratio = effect_j.mean() / (effect_i.mean() + 1e-6)
                            
                            # Penalize disproportionate effects
                            proportionality_error = (expected_ratio - actual_ratio) ** 2
                            consistency_loss += proportionality_error
                            num_violations += 1
        
        # 2. DIRECTIONAL CONSISTENCY CHECK: Effects should align with causal structure
        if adjacency_matrix is not None:
            for i, (cf_output, mask, values) in enumerate(zip(counterfactual_outputs, intervention_masks, intervention_values)):
                effect = cf_output - factual_output  # shape: (batch_size, output_dim)
                
                # Process each batch item
                for batch_idx in range(effect.shape[0]):
                    batch_mask = mask[batch_idx]  # shape: (input_dim,)
                    batch_values = values[batch_idx]  # shape: (input_dim,)
                    batch_effect = effect[batch_idx]  # shape: (output_dim,)
                    
                    intervened_indices = batch_mask.bool()
                    
                    if intervened_indices.any():
                        # Check each intervened node
                        for node_idx in torch.where(intervened_indices)[0]:
                            intervention_val = batch_values[node_idx]
                            
                            # Check effects on all downstream nodes
                            for output_idx in range(batch_effect.shape[0]):
                                observed_effect = batch_effect[output_idx]  # This is a scalar
                                
                                # Check if causal relationship exists from node_idx to output_idx
                                has_causal_path = adjacency_matrix[node_idx, output_idx] > 0.1
                                
                                # If no causal path, effect should be minimal
                                if not has_causal_path:
                                    spurious_effect_penalty = observed_effect ** 2
                                    consistency_loss += spurious_effect_penalty
                                    num_violations += 1
                                else:
                                    # Effect should be in same direction as intervention
                                    direction_mismatch = (intervention_val * observed_effect < 0).float()
                                    consistency_loss += direction_mismatch * torch.abs(observed_effect)
                                    num_violations += 1
        
        # 3. MONOTONICITY CHECK: Increasing intervention values should produce monotonic effects
        # Group interventions by the same node
        node_interventions = {}
        for i, (cf_output, mask, values) in enumerate(zip(counterfactual_outputs, intervention_masks, intervention_values)):
            # Process each batch item
            for batch_idx in range(mask.shape[0]):
                batch_mask = mask[batch_idx]
                batch_values = values[batch_idx]
                
                intervened_indices = torch.where(batch_mask.bool())[0]
                for node_idx in intervened_indices:
                    node_idx_int = node_idx.item()
                    if node_idx_int not in node_interventions:
                        node_interventions[node_idx_int] = []
                    node_interventions[node_idx_int].append((batch_values[node_idx].item(), cf_output))
        
        # Check monotonicity for each node
        for node_idx, interventions in node_interventions.items():
            if len(interventions) >= 2:
                # Sort by intervention value
                interventions.sort(key=lambda x: x[0])
                
                # Check if effects are monotonic
                for k in range(len(interventions) - 1):
                    val1, output1 = interventions[k]
                    val2, output2 = interventions[k + 1]
                    
                    if val1 != val2:  # Different intervention values
                        effect1 = output1 - factual_output
                        effect2 = output2 - factual_output
                        
                        # Check if effect direction matches intervention direction
                        intervention_direction = val2 - val1
                        
                        # Process each batch item for effect comparison
                        for batch_idx in range(factual_output.shape[0]):
                            effect1_batch = (output1 - factual_output)[batch_idx]  # shape: (output_dim,)
                            effect2_batch = (output2 - factual_output)[batch_idx]  # shape: (output_dim,)
                            
                            if node_idx < effect1_batch.shape[0]:
                                effect_direction = effect2_batch[node_idx] - effect1_batch[node_idx]  # Scalar
                                
                                # Effects should be monotonic
                                if intervention_direction * effect_direction < 0:
                                    monotonicity_violation = torch.abs(effect_direction) * 0.5
                                    consistency_loss += monotonicity_violation
                                    num_violations += 1
        
        # 4. EFFECT MAGNITUDE CONSISTENCY: Large interventions should have larger effects
        for i, (cf_output, mask, values) in enumerate(zip(counterfactual_outputs, intervention_masks, intervention_values)):
            effect = cf_output - factual_output  # shape: (batch_size, output_dim)
            
            # Process each batch item
            for batch_idx in range(effect.shape[0]):
                batch_mask = mask[batch_idx]  # shape: (input_dim,)
                batch_values = values[batch_idx]  # shape: (input_dim,)
                batch_effect = effect[batch_idx]  # shape: (output_dim,)
                
                intervened_indices = batch_mask.bool()
                
                if intervened_indices.any():
                    intervention_magnitude = torch.abs(batch_values[intervened_indices]).mean()
                    effect_magnitude = torch.abs(batch_effect).mean()
                    
                    # Very large interventions should produce non-trivial effects
                    if intervention_magnitude > 2.0 and effect_magnitude < 0.01:
                        weak_effect_penalty = (intervention_magnitude - effect_magnitude) ** 2 * 0.1
                        consistency_loss += weak_effect_penalty
                        num_violations += 1
        
        # Normalize by number of violations
        if num_violations > 0:
            consistency_loss = consistency_loss / num_violations
        
        return consistency_loss

    def directional_effect_loss(self, 
                               factual_output: torch.Tensor,
                               counterfactual_output: torch.Tensor,
                               intervention_mask: torch.Tensor,
                               intervention_values: torch.Tensor,
                               expected_directions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Directional effect loss: penalizes when intervention effects go in wrong direction.
        
        For example, if increasing X should increase Y, but intervention do(X=high)
        actually decreases Y, this is a violation of causal logic.
        
        Args:
            factual_output: Output without intervention
            counterfactual_output: Output with intervention
            intervention_mask: Binary mask of intervened variables
            intervention_values: Values of interventions
            expected_directions: Expected direction of effects (optional)
            
        Returns:
            Directional effect loss
        """
        if expected_directions is None:
            return torch.tensor(0.0, device=factual_output.device)
        
        # Compute actual effect
        actual_effect = counterfactual_output - factual_output
        
        # Compute expected effect direction
        intervention_direction = torch.sign(intervention_values)
        
        # Penalty for effects going in wrong direction
        direction_violation = torch.sum(
            torch.clamp(-actual_effect * intervention_direction.unsqueeze(-1), min=0) ** 2
        )
        
        return direction_violation
    
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
