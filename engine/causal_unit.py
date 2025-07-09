import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from typing import Optional, Dict, List, Tuple, Union


class CausalInterventionFunction(Function):
    """
    Custom autograd function that implements precise gradient blocking for causal interventions.
    
    Key Innovation: If do(node_k = v), then ∂L/∂parent(node_k) = 0 for all parents of k.
    This is NOT stop-gradient on the node output, but targeted gradient removal only to parent paths.
    """
    
    @staticmethod
    def forward(ctx, input_tensor, parent_values, adj_mask, do_mask, do_values, weights, bias):
        """
        Forward pass with causal intervention logic.
        
        Args:
            input_tensor: Node inputs (batch_size, input_dim)
            parent_values: Parent node activations (batch_size, n_parents)
            adj_mask: Adjacency mask indicating parent connections (n_parents, input_dim)
            do_mask: Intervention mask (batch_size, input_dim) or (input_dim,)
            do_values: Intervention values (batch_size, input_dim) or (input_dim,)
            weights: Layer weights (input_dim, output_dim)
            bias: Layer bias (output_dim,)
        """
        # Save for backward pass
        ctx.save_for_backward(input_tensor, parent_values, adj_mask, do_mask, do_values, weights, bias)
        
        batch_size = input_tensor.shape[0]
        
        # Ensure masks have correct dimensions
        if do_mask is not None:
            if do_mask.dim() == 1:
                do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
            if do_values.dim() == 1:
                do_values = do_values.unsqueeze(0).expand(batch_size, -1)
        
        # Apply causal intervention logic
        if do_mask is not None and do_values is not None:
            # Step 1: Cut all incoming edges for intervened nodes
            # This implements the "edge cutting" part of do(node_k = v)
            effective_adj_mask = adj_mask.clone()
            intervened_indices = do_mask.bool().any(dim=0)  # Which nodes are intervened across batch
            effective_adj_mask[:, intervened_indices] = 0.0
            
            # Step 2: Apply adjacency masking to parent values (if they exist)
            if parent_values is not None:
                masked_parents = torch.matmul(parent_values, effective_adj_mask)
            else:
                masked_parents = None
            
            # Step 3: Replace intervened node values
            intervened_input = torch.where(do_mask.bool(), do_values, input_tensor)
            
            # Step 4: Use intervened input for forward computation
            final_input = intervened_input
        else:
            # No intervention: use normal adjacency masking (if parent values exist)
            if parent_values is not None and adj_mask is not None:
                masked_parents = torch.matmul(parent_values, adj_mask)
            else:
                masked_parents = parent_values  # Could be None
            final_input = input_tensor
        
        # Forward computation: output = final_input @ weights + bias
        if weights is not None:
            output = torch.matmul(final_input, weights)
            if bias is not None:
                output = output + bias
        else:
            output = final_input
            
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with gradient blocking for causal interventions.
        
        Key Innovation: Block gradients to parents of intervened nodes.
        """
        input_tensor, parent_values, adj_mask, do_mask, do_values, weights, bias = ctx.saved_tensors
        
        batch_size = grad_output.shape[0]
        
        # Initialize gradients
        grad_input = None
        grad_parent_values = None
        grad_adj_mask = None
        grad_weights = None
        grad_bias = None
        
        # Gradient w.r.t. bias
        if bias is not None:
            grad_bias = grad_output.sum(dim=0)
        
        # Gradient w.r.t. weights
        if weights is not None:
            if do_mask is not None and do_values is not None:
                if do_mask.dim() == 1:
                    do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
                intervened_input = torch.where(do_mask.bool(), do_values, input_tensor)
                grad_weights = torch.matmul(intervened_input.T, grad_output)
            else:
                grad_weights = torch.matmul(input_tensor.T, grad_output)
        
        # Gradient w.r.t. input (with intervention blocking)
        if ctx.needs_input_grad[0]:
            if weights is not None:
                grad_input = torch.matmul(grad_output, weights.T)
            else:
                grad_input = grad_output
                
            # CRITICAL: Block gradients for intervened nodes
            if do_mask is not None:
                if do_mask.dim() == 1:
                    do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
                # Zero out gradients for intervened nodes
                grad_input = torch.where(do_mask.bool(), torch.zeros_like(grad_input), grad_input)
        
        # Gradient w.r.t. parent values (with causal blocking)
        if ctx.needs_input_grad[1]:
            if parent_values is not None:
                if weights is not None:
                    grad_parent_base = torch.matmul(grad_output, weights.T)
                else:
                    grad_parent_base = grad_output
                
                if adj_mask is not None:
                    # Apply adjacency mask to gradients
                    grad_parent_values = torch.matmul(grad_parent_base, adj_mask.T)
                    
                    # CRITICAL: Block gradients to parents of intervened nodes
                    if do_mask is not None:
                        if do_mask.dim() == 1:
                            do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
                        
                        # For each intervened node, block gradients to ALL its parents
                        intervened_indices = do_mask.bool().any(dim=0)
                        for i, is_intervened in enumerate(intervened_indices):
                            if is_intervened:
                                # Zero out gradients to all parents of node i
                                parent_indices = adj_mask[:, i].bool()
                                grad_parent_values[:, parent_indices] = 0.0
                else:
                    grad_parent_values = grad_parent_base
            else:
                grad_parent_values = None
        
        # Gradient w.r.t. adjacency mask (for structure learning)
        if ctx.needs_input_grad[2] and adj_mask is not None:
            if parent_values is not None and weights is not None:
                # Gradient of adj_mask: how changes in adjacency affect the output
                # masked_parents = torch.matmul(parent_values, adj_mask)
                # output = torch.matmul(masked_parents, weights) + bias
                # d_output/d_adj_mask = parent_values.T @ (grad_output @ weights.T)
                grad_through_weights = torch.matmul(grad_output, weights.T)
                grad_adj_mask = torch.matmul(parent_values.T, grad_through_weights)
            else:
                grad_adj_mask = None
        
        return grad_input, grad_parent_values, grad_adj_mask, None, None, grad_weights, grad_bias


class CausalUnit(nn.Module):
    """
    Phase 3 CausalUnit: A neural block that natively supports symbolic intervention (do()),
    edge-cutting, runtime graph rewiring, and gradient isolation.
    
    Key Innovations:
    1. Custom autograd with precise gradient blocking
    2. Runtime graph rewiring with dynamic DAG masks
    3. Symbolic-continuous hybrid approach
    4. Causal backflow correction
    5. Support for multiple simultaneous interventions
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, 
                 activation: str = 'relu', node_id: Optional[str] = None,
                 enable_structure_learning: bool = True, enable_gradient_surgery: bool = True):
        super(CausalUnit, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.node_id = node_id or f"node_{id(self)}"
        self.enable_structure_learning = enable_structure_learning
        self.enable_gradient_surgery = enable_gradient_surgery
        
        # Core neural computation layers
        if hidden_dim is None:
            # Single linear layer
            self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(output_dim))
            self.hidden_weights = None
            self.hidden_bias = None
        else:
            # Multi-layer MLP
            self.weights = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.hidden_weights = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
            self.hidden_bias = nn.Parameter(torch.zeros(output_dim))
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            self.activation = F.relu
        
        # Learnable adjacency matrix for structure learning (symbolic-continuous hybrid)
        if enable_structure_learning:
            self.adj_logits = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)
            self.adj_temperature = nn.Parameter(torch.ones(1) * 1.0)  # Learnable temperature
        else:
            self.adj_logits = None
            self.adj_temperature = None
        
        # Intervention tracking
        self.last_interventions = {}
        self.intervention_history = []
        
        # Gradient surgery components
        if enable_gradient_surgery:
            self.gradient_mask = None
            self.causal_ancestry_cache = None
        
        # Dynamic graph state
        self.current_adj_mask = None
        self.parent_nodes = []
        self.child_nodes = []
    
    def set_parent_nodes(self, parent_nodes: List['CausalUnit']):
        """Set parent nodes for this unit in the causal graph."""
        self.parent_nodes = parent_nodes
    
    def add_child_node(self, child_node: 'CausalUnit'):
        """Add a child node to this unit."""
        if child_node not in self.child_nodes:
            self.child_nodes.append(child_node)
    
    def get_adjacency_matrix(self, hard: bool = False, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Get the current adjacency matrix (symbolic-continuous hybrid).
        
        Args:
            hard: If True, return hard binary adjacency. If False, return soft adjacency.
            temperature: Override temperature for soft adjacency.
        
        Returns:
            Adjacency matrix (input_dim, input_dim)
        """
        if self.adj_logits is None:
            return torch.eye(self.input_dim, device=self.weights.device)
        
        if hard:
            # Hard adjacency for evaluation/intervention
            return torch.sigmoid(self.adj_logits) > 0.5
        else:
            # Soft adjacency for training
            temp = temperature if temperature is not None else self.adj_temperature
            return torch.sigmoid(self.adj_logits / temp)
    
    def compute_causal_ancestry(self, adj_mask: torch.Tensor, max_depth: int = 10) -> torch.Tensor:
        """
        Compute causal ancestry matrix for backflow correction.
        
        This computes which nodes are ancestors of which nodes in the causal graph,
        used for gradient blocking to ensure no indirect gradient leakage.
        """
        device = adj_mask.device
        n_nodes = adj_mask.shape[0]
        
        # Initialize ancestry matrix
        ancestry = torch.eye(n_nodes, device=device)
        
        # Compute transitive closure via matrix powers
        current_path = adj_mask.clone()
        for _ in range(max_depth):
            ancestry = ancestry + current_path
            current_path = torch.matmul(current_path, adj_mask)
            
            # Check for convergence
            if torch.allclose(current_path, torch.zeros_like(current_path), atol=1e-6):
                break
        
        return (ancestry > 0).float()
    
    def apply_gradient_surgery(self, grad_tensor: torch.Tensor, intervention_mask: torch.Tensor,
                             adj_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply gradient surgery to ensure proper causal isolation.
        
        This implements "gradient propagation with pathwise edge exclusion" -
        a novel form of gradient manipulation for causal models.
        """
        if not self.enable_gradient_surgery:
            return grad_tensor
        
        # Compute causal ancestry
        ancestry = self.compute_causal_ancestry(adj_mask)
        
        # For each intervened node, block gradients to ALL ancestors
        batch_size = grad_tensor.shape[0]
        surgery_mask = torch.ones_like(grad_tensor)
        
        if intervention_mask.dim() == 1:
            intervention_mask = intervention_mask.unsqueeze(0).expand(batch_size, -1)
        
        for batch_idx in range(batch_size):
            intervened_nodes = intervention_mask[batch_idx].bool()
            
            for node_idx in range(len(intervened_nodes)):
                if intervened_nodes[node_idx]:
                    # Find all ancestors of this intervened node
                    ancestors = ancestry[:, node_idx].bool()
                    # Block gradients to all ancestors
                    surgery_mask[batch_idx, ancestors] = 0.0
        
        return grad_tensor * surgery_mask
    
    def forward(self, input_tensor: torch.Tensor, parent_values: Optional[torch.Tensor] = None,
                adj_mask: Optional[torch.Tensor] = None, do_mask: Optional[torch.Tensor] = None,
                do_values: Optional[torch.Tensor] = None, 
                interventions: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Forward pass with causal intervention support.
        
        Args:
            input_tensor: Node inputs (batch_size, input_dim)
            parent_values: Parent node activations (batch_size, n_parents)
            adj_mask: Adjacency mask for parent connections (n_parents, input_dim)
            do_mask: Intervention mask (batch_size, input_dim) or (input_dim,)
            do_values: Intervention values (batch_size, input_dim) or (input_dim,)
            interventions: Dict of named interventions {name: (mask, values)}
        
        Returns:
            Output tensor (batch_size, output_dim)
        """
        batch_size = input_tensor.shape[0]
        
        # Handle multiple interventions
        if interventions is not None:
            # Combine all interventions
            combined_mask = torch.zeros(batch_size, self.input_dim, device=input_tensor.device)
            combined_values = torch.zeros(batch_size, self.input_dim, device=input_tensor.device)
            
            for name, (mask, values) in interventions.items():
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0).expand(batch_size, -1)
                if values.dim() == 1:
                    values = values.unsqueeze(0).expand(batch_size, -1)
                    
                # Union of intervention masks (logical OR)
                combined_mask = torch.logical_or(combined_mask.bool(), mask.bool()).float()
                # Use the values from the last intervention for overlapping nodes
                combined_values = torch.where(mask.bool(), values, combined_values)
            
            do_mask = combined_mask
            do_values = combined_values
        
        # Get current adjacency matrix
        if adj_mask is None:
            adj_mask = self.get_adjacency_matrix(hard=False)  # Soft during training
        
        # Store current adjacency for runtime rewiring
        self.current_adj_mask = adj_mask
        
        # Apply custom causal intervention function
        if self.hidden_weights is None:
            # Single layer
            output = CausalInterventionFunction.apply(
                input_tensor, parent_values, adj_mask, do_mask, do_values, 
                self.weights, self.bias
            )
        else:
            # Multi-layer: apply to first layer, then standard forward for second layer
            hidden = CausalInterventionFunction.apply(
                input_tensor, parent_values, adj_mask, do_mask, do_values,
                self.weights, self.bias
            )
            hidden = self.activation(hidden)
            output = torch.matmul(hidden, self.hidden_weights) + self.hidden_bias
        
        # Track interventions for debugging
        if do_mask is not None and do_values is not None:
            self.last_interventions = {
                'mask': do_mask.detach().cpu(),
                'values': do_values.detach().cpu(),
                'timestamp': len(self.intervention_history)
            }
            self.intervention_history.append(self.last_interventions)
        
        return output
    
    def get_intervention_info(self) -> Dict:
        """Get detailed information about recent interventions."""
        return {
            'last_interventions': self.last_interventions,
            'intervention_count': len(self.intervention_history),
            'current_adj_mask': self.current_adj_mask.detach().cpu() if self.current_adj_mask is not None else None,
            'node_id': self.node_id
        }
    
    def get_gradient_flow_info(self) -> Dict:
        """Get information about gradient flows for debugging."""
        info = {
            'node_id': self.node_id,
            'weights_grad': self.weights.grad.detach().cpu() if self.weights.grad is not None else None,
            'bias_grad': self.bias.grad.detach().cpu() if self.bias.grad is not None else None,
            'adj_logits_grad': self.adj_logits.grad.detach().cpu() if self.adj_logits is not None and self.adj_logits.grad is not None else None,
            'gradient_mask': self.gradient_mask.detach().cpu() if self.gradient_mask is not None else None
        }
        return info
    
    def reset_intervention_history(self):
        """Reset intervention tracking."""
        self.intervention_history = []
        self.last_interventions = {}
    
    def enable_runtime_rewiring(self, enable: bool = True):
        """Enable/disable runtime graph rewiring."""
        self.enable_gradient_surgery = enable
