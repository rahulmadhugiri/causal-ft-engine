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
    
    def forward(self, x, do_mask=None, do_values=None):
        """
        Forward pass with optional causal interventions.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            do_mask: Binary tensor of shape (input_dim,) indicating which variables are intervened
            do_values: Tensor of shape (input_dim,) with intervention values
            
        Returns:
            Output tensor after applying interventions and forward pass
        """
        batch_size = x.shape[0]
        
        # Store intervention info for debugging
        self.last_intervention_mask = do_mask
        self.last_intervention_values = do_values
        
        # If no interventions, proceed normally
        if do_mask is None or do_values is None:
            return self.layers(x)
        
        # Ensure do_mask and do_values have the right shape
        if do_mask.dim() == 1:
            do_mask = do_mask.unsqueeze(0).expand(batch_size, -1)
        if do_values.dim() == 1:
            do_values = do_values.unsqueeze(0).expand(batch_size, -1)
            
        # Apply interventions: replace intervened inputs
        x_intervened = x.clone()
        
        # For intervened variables, detach from computation graph to block gradients
        # and replace with intervention values
        intervened_indices = do_mask.bool()
        
        # Detach intervened variables to block gradient flow
        x_detached = x.detach()
        
        # Create the intervened input tensor
        # Use original values where not intervened, intervention values where intervened
        x_intervened = torch.where(intervened_indices, do_values, x)
        
        # For the intervened dimensions, we want to block gradients
        # So we use detached values for those dimensions
        x_final = torch.where(intervened_indices, x_detached, x)
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
    A multi-layer causal MLP that can have interventions at multiple layers.
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
    
    def forward(self, x, interventions=None):
        """
        Forward pass through causal MLP with optional layer-wise interventions.
        
        Args:
            x: Input tensor
            interventions: Dict with layer indices as keys and (do_mask, do_values) tuples as values
                          e.g., {0: (mask_tensor, values_tensor), 2: (mask_tensor, values_tensor)}
        """
        current = x
        
        for i, layer in enumerate(self.layers):
            if interventions and i in interventions:
                do_mask, do_values = interventions[i]
                current = layer(current, do_mask=do_mask, do_values=do_values)
            else:
                current = layer(current)
                
        return current
