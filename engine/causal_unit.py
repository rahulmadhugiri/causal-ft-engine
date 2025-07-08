import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalUnit(nn.Module):
    """
    A causal-aware MLP block that implements do-operator interventions.
    
    This module can:
    - Accept interventions via do_mask and do_values
    - Replace intervened inputs during forward pass
    - Block gradients to parents of intervened nodes
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=None, activation='relu'):
        super(CausalUnit, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # If no hidden dimension specified, use a simple linear layer
        if hidden_dim is None:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, output_dim)
            )
        else:
            # Multi-layer MLP
            if activation == 'relu':
                act_fn = nn.ReLU()
            elif activation == 'tanh':
                act_fn = nn.Tanh()
            else:
                act_fn = nn.ReLU()  # default
                
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act_fn,
                nn.Linear(hidden_dim, output_dim)
            )
        
        # Track intervention state for debugging
        self.last_intervention_mask = None
        self.last_intervention_values = None
    
    def forward(self, x, do_mask=None, do_values=None, structure_mask=None):
        """
        Forward pass with optional causal interventions and structure masking.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            do_mask: Binary tensor of shape (input_dim,) indicating which variables are intervened
            do_values: Tensor of shape (input_dim,) with intervention values
            structure_mask: Adjacency matrix for structural masking (input_dim, input_dim)
            
        Returns:
            Output tensor after applying interventions and forward pass
        """
        batch_size = x.shape[0]
        
        # Store intervention info for debugging
        self.last_intervention_mask = do_mask
        self.last_intervention_values = do_values
        
        # Apply structure masking first if provided
        if structure_mask is not None:
            # Apply structural constraints: x_masked = x * structure_mask
            # Each row of structure_mask indicates which variables can influence that variable
            x_structured = torch.matmul(x, structure_mask.T)  # (batch_size, input_dim)
        else:
            x_structured = x
        
        # If no interventions, proceed with structured input
        if do_mask is None or do_values is None:
            return self.layers(x_structured)
        
        # Ensure do_mask and do_values have the right shape
        if do_mask.dim() == 1:
            do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
        if do_values.dim() == 1:
            do_values = do_values.unsqueeze(0).expand(batch_size, -1)
            
        # Apply interventions: replace intervened inputs
        intervened_indices = do_mask.bool()
        
        # Detach intervened variables to block gradient flow
        x_detached = x_structured.detach()
        
        # Create the intervened input tensor
        # Use original values where not intervened, intervention values where intervened
        x_final = torch.where(intervened_indices, x_detached, x_structured)
        x_final = torch.where(intervened_indices, do_values, x_final)
        
        return self.layers(x_final)
    
    def get_intervention_info(self):
        """
        Get information about the last intervention applied.
        Useful for debugging and logging.
        """
        return {
            'mask': self.last_intervention_mask,
            'values': self.last_intervention_values
        }


class CausalMLP(nn.Module):
    """
    A multi-layer causal MLP that can have interventions at multiple layers
    and support learned causal structure.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super(CausalMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(CausalUnit(prev_dim, hidden_dim, activation=activation))
            prev_dim = hidden_dim
            
        # Output layer
        self.layers.append(CausalUnit(prev_dim, output_dim, activation=activation))
        
        # Optional learned structure (adjacency matrix)
        self.learned_structure = None
        self.use_learned_structure = False
    
    def set_learned_structure(self, adjacency_matrix, use_structure=True):
        """
        Set a learned causal structure to constrain the model.
        
        Args:
            adjacency_matrix: Learned adjacency matrix (input_dim, input_dim)
            use_structure: Whether to use this structure in forward pass
        """
        self.learned_structure = adjacency_matrix
        self.use_learned_structure = use_structure
    
    def forward(self, x, interventions=None, structure_mask=None):
        """
        Forward pass through causal MLP with optional layer-wise interventions
        and structural constraints.
        
        Args:
            x: Input tensor
            interventions: Dict with layer indices as keys and (do_mask, do_values) tuples as values
                          e.g., {0: (mask_tensor, values_tensor), 2: (mask_tensor, values_tensor)}
            structure_mask: Optional structure mask to override learned structure
        """
        current = x
        
        # Determine which structure mask to use
        if structure_mask is not None:
            effective_structure = structure_mask
        elif self.use_learned_structure and self.learned_structure is not None:
            effective_structure = self.learned_structure
        else:
            effective_structure = None
        
        for i, layer in enumerate(self.layers):
            # Apply structure mask only to first layer (input layer)
            layer_structure = effective_structure if i == 0 else None
            
            if interventions and i in interventions:
                do_mask, do_values = interventions[i]
                current = layer(current, do_mask=do_mask, do_values=do_values, 
                              structure_mask=layer_structure)
            else:
                current = layer(current, structure_mask=layer_structure)
                
        return current
    
    def get_structure_info(self):
        """
        Get information about the current learned structure.
        """
        return {
            'learned_structure': self.learned_structure,
            'use_learned_structure': self.use_learned_structure,
            'structure_shape': self.learned_structure.shape if self.learned_structure is not None else None
        }


class StructureAwareCausalMLP(nn.Module):
    """
    Enhanced CausalMLP that integrates structure learning and counterfactual reasoning.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu',
                 learn_structure=True, structure_hidden_dim=16):
        super(StructureAwareCausalMLP, self).__init__()
        
        self.input_dim = input_dim
        self.learn_structure = learn_structure
        
        # Main prediction network
        self.causal_mlp = CausalMLP(input_dim, hidden_dims, output_dim, activation)
        
        # Structure learning component
        if learn_structure:
            from .structure_learning import DifferentiableDAG
            self.structure_learner = DifferentiableDAG(input_dim, structure_hidden_dim)
        else:
            self.structure_learner = None
        
        # Training mode flags
        self.structure_learning_mode = False
        self.prediction_mode = True
    
    def set_training_mode(self, structure_learning=False, prediction=True):
        """
        Set training mode for different components.
        
        Args:
            structure_learning: Whether to train structure learning component
            prediction: Whether to train prediction component
        """
        self.structure_learning_mode = structure_learning
        self.prediction_mode = prediction
        
        if self.structure_learner is not None:
            if structure_learning:
                self.structure_learner.train()
            else:
                self.structure_learner.eval()
    
    def forward(self, x, interventions=None, return_structure=False):
        """
        Forward pass with integrated structure learning and prediction.
        
        Args:
            x: Input tensor
            interventions: Optional interventions
            return_structure: Whether to return learned structure
            
        Returns:
            predictions: Main predictions
            structure_info: Optional structure information
        """
        structure_info = {}
        
        # Learn or use structure
        if self.structure_learner is not None:
            if self.structure_learning_mode:
                # During structure learning phase
                x_reconstructed, adjacency = self.structure_learner(x)
                structure_info['adjacency'] = adjacency
                structure_info['reconstruction'] = x_reconstructed
                
                # Use learned structure for prediction
                self.causal_mlp.set_learned_structure(adjacency, use_structure=True)
            else:
                # Use current structure without updating
                with torch.no_grad():
                    adjacency = self.structure_learner.get_adjacency_matrix(hard=True)
                    self.causal_mlp.set_learned_structure(adjacency, use_structure=True)
                    structure_info['adjacency'] = adjacency
        
        # Main prediction
        if self.prediction_mode:
            predictions = self.causal_mlp(x, interventions=interventions)
        else:
            predictions = None
        
        if return_structure:
            return predictions, structure_info
        else:
            return predictions
    
    def get_learned_adjacency(self, hard=True):
        """
        Get the current learned adjacency matrix.
        """
        if self.structure_learner is not None:
            with torch.no_grad():
                return self.structure_learner.get_adjacency_matrix(hard=hard)
        return None
