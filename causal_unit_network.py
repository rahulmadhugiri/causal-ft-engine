import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from engine.causal_unit import CausalUnit
from engine.structure_learning import DifferentiableDAG


class CausalUnitNetwork(nn.Module):
    """
    Phase 3 CausalUnit Network: Dynamic network assembly with intervention support.
    
    Key Features:
    1. Stack N CausalUnits with learned or given DAG adjacency
    2. Batched forward/backward for scalable training/testing
    3. API for both fixed-structure and structure-learned networks
    4. Runtime graph rewiring and dynamic adjacency masks
    5. Support for multiple simultaneous interventions
    6. Pathwise intervention algebra
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 activation: str = 'relu',
                 enable_structure_learning: bool = True,
                 enable_gradient_surgery: bool = True,
                 structure_hidden_dim: int = 16,
                 max_graph_depth: int = 10,
                 intervention_dropout: float = 0.0,
                 lambda_reg: float = 0.01,
                 use_low_rank_adjacency: bool = False,
                 adjacency_rank: int = None):
        super(CausalUnitNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.enable_structure_learning = enable_structure_learning
        self.enable_gradient_surgery = enable_gradient_surgery
        self.max_graph_depth = max_graph_depth
        self.intervention_dropout = intervention_dropout
        self.lambda_reg = lambda_reg
        self.use_low_rank_adjacency = use_low_rank_adjacency
        self.adjacency_rank = adjacency_rank or min(input_dim // 2, 8)
        
        # Build network architecture
        self.units = nn.ModuleList()
        self.unit_names = []
        
        # Create units with proper dimensions
        all_dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(all_dims) - 1):
            unit_name = f"unit_{i}"
            unit = CausalUnit(
                input_dim=all_dims[i],
                output_dim=all_dims[i + 1],
                hidden_dim=None,  # Keep units simple for now
                activation=activation,
                node_id=unit_name,
                enable_structure_learning=enable_structure_learning,
                enable_gradient_surgery=enable_gradient_surgery,
                use_low_rank_adjacency=use_low_rank_adjacency,
                adjacency_rank=self.adjacency_rank
            )
            self.units.append(unit)
            self.unit_names.append(unit_name)
        
        # Structure learning component
        if enable_structure_learning:
            self.structure_learner = DifferentiableDAG(
                input_dim, 
                structure_hidden_dim, 
                enable_structure_transfer=True
            )
            self.learned_adjacency = None
            self.structure_temperature = nn.Parameter(torch.ones(1) * 1.0)
        else:
            self.structure_learner = None
            self.learned_adjacency = None
        
        # Dynamic graph state
        self.current_adjacency = None
        self.intervention_schedule = []
        self.gradient_flow_history = []
        
        # Training modes
        self.structure_learning_mode = False
        self.intervention_training_mode = True
        
        # CRITICAL FIX: Initialize proper chain adjacency matrix for causal structure
        self._initialize_chain_adjacency()
        
        # Initialize unit relationships
        self._setup_unit_relationships()
        
        # Log parameter efficiency gains
        if use_low_rank_adjacency:
            total_full_params = sum(dim**2 for dim in [input_dim] + hidden_dims + [output_dim])
            total_low_rank_params = sum(2 * dim * self.adjacency_rank for dim in [input_dim] + hidden_dims + [output_dim])
            print(f"Low-rank adjacency enabled: {total_full_params} → {total_low_rank_params} params ({100*total_low_rank_params/total_full_params:.1f}% of original)")
    
    def transfer_structure_from_previous(self, previous_network: 'CausalUnitNetwork', 
                                       adaptation_strength: float = 0.5):
        """
        Transfer structure knowledge from a previous network.
        
        Args:
            previous_network: Previous CausalUnitNetwork to transfer from
            adaptation_strength: How much to adapt towards the previous structure
        """
        if not self.enable_structure_learning or not previous_network.enable_structure_learning:
            print("Structure learning not enabled for transfer")
            return
        
        if previous_network.learned_adjacency is not None:
            # Transfer the learned adjacency matrix
            self.structure_learner.warm_start_from_previous(
                previous_network.learned_adjacency, 
                adaptation_strength=adaptation_strength
            )
            
            # Also add the previous structure to memory if it was successful
            if hasattr(previous_network, 'final_performance_score'):
                self.structure_learner.add_structure_to_memory(
                    previous_network.learned_adjacency, 
                    previous_network.final_performance_score
                )
            
            print(f"Structure transfer completed from previous network")
        else:
            print("No learned adjacency found in previous network")
    
    def save_structure_performance(self, performance_score: float):
        """
        Save the current structure's performance for future transfer.
        
        Args:
            performance_score: Performance score of current structure (higher is better)
        """
        if self.enable_structure_learning and self.learned_adjacency is not None:
            self.final_performance_score = performance_score
            self.structure_learner.add_structure_to_memory(
                self.learned_adjacency, 
                performance_score
            )
            print(f"Saved structure performance: {performance_score}")
    
    def get_structure_transfer_info(self) -> Dict:
        """
        Get information about structure transfer capabilities.
        
        Returns:
            Dictionary with transfer information
        """
        if not self.enable_structure_learning:
            return {'transfer_enabled': False}
        
        info = {
            'transfer_enabled': True,
            'memory_size': len(self.structure_learner.structure_memory) if self.structure_learner.structure_memory else 0,
            'transfer_weight': self.structure_learner.transfer_weight.item() if hasattr(self.structure_learner, 'transfer_weight') else 0.0,
            'similarity_threshold': self.structure_learner.similarity_threshold if hasattr(self.structure_learner, 'similarity_threshold') else 0.0,
            'has_learned_structure': self.learned_adjacency is not None
        }
        
        return info
    
    def _initialize_chain_adjacency(self):
        """Initialize proper chain adjacency matrix for causal structure."""
        # Create chain adjacency: 0->1->2->...->n-1
        chain_adjacency = torch.zeros(self.input_dim, self.input_dim)
        for i in range(self.input_dim - 1):
            chain_adjacency[i, i + 1] = 1.0
        
        # CRITICAL FIX: Make adjacency a learnable parameter for gradient flow
        self.learned_adjacency = nn.Parameter(chain_adjacency.clone())
        self.current_adjacency = self.learned_adjacency
        
        print(f"Initialized chain adjacency matrix:\n{chain_adjacency}")
    
    def _setup_unit_relationships(self):
        """Set up parent-child relationships between units."""
        for i in range(len(self.units) - 1):
            self.units[i].add_child_node(self.units[i + 1])
            self.units[i + 1].set_parent_nodes([self.units[i]])
    
    def set_fixed_structure(self, adjacency_matrix: torch.Tensor, hard: bool = True):
        """
        Set a fixed causal structure for the network.
        
        Args:
            adjacency_matrix: DAG adjacency matrix (input_dim, input_dim)
            hard: Whether to use hard or soft adjacency
        """
        self.learned_adjacency = adjacency_matrix
        self.current_adjacency = adjacency_matrix
        
        # Update structure learning if enabled
        if self.structure_learner is not None:
            self.structure_learner.set_adjacency(adjacency_matrix)
    
    def get_adjacency_matrix(self, hard: bool = False) -> torch.Tensor:
        """
        Get the current adjacency matrix for the network.
        
        Args:
            hard: Whether to return hard binary adjacency or soft adjacency
            
        Returns:
            Adjacency matrix (input_dim, input_dim)
        """
        if self.learned_adjacency is not None:
            if hard:
                return (torch.sigmoid(self.learned_adjacency) > 0.5).float()
            else:
                # Use raw parameter for better gradient flow, apply sigmoid for final output
                return torch.clamp(self.learned_adjacency, 0.0, 1.0)
        
        if self.structure_learner is not None:
            return self.structure_learner.get_adjacency_matrix(hard=hard)
        
        # CRITICAL FIX: Initialize with proper chain adjacency instead of identity
        # Identity matrix means no causal relationships exist!
        chain_adjacency = torch.zeros(self.input_dim, self.input_dim, device=next(self.parameters()).device)
        for i in range(self.input_dim - 1):
            chain_adjacency[i, i + 1] = 1.0
        return chain_adjacency
    
    def update_structure(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the learned structure using the structure learner.
        
        Args:
            x: Input data for structure learning
            
        Returns:
            reconstructed_x: Reconstructed input
            adjacency: Learned adjacency matrix
        """
        if self.structure_learner is None:
            return x, self.get_adjacency_matrix()
        
        reconstructed_x, adjacency = self.structure_learner(x)
        # CRITICAL FIX: Allow gradient flow to the adjacency parameter
        # Do not use torch.no_grad() here - we want gradients to flow back!
        # Properly update the parameter tensor data
        self.learned_adjacency.data = adjacency.data
        self.current_adjacency = self.learned_adjacency
        
        return reconstructed_x, adjacency
    
    def create_intervention_schedule(self, 
                                   batch_size: int,
                                   intervention_prob: float = 0.3,
                                   multi_intervention_prob: float = 0.1) -> List[Dict]:
        """
        Create a schedule of interventions for a batch.
        
        Args:
            batch_size: Number of samples in batch
            intervention_prob: Probability of intervention per sample
            multi_intervention_prob: Probability of multiple simultaneous interventions
            
        Returns:
            List of intervention dictionaries for each sample
        """
        schedule = []
        
        for batch_idx in range(batch_size):
            interventions = {}
            
            # Decide whether to intervene
            if np.random.random() < intervention_prob:
                # Single intervention
                node_idx = np.random.randint(0, self.input_dim)
                intervention_value = np.random.randn()
                
                mask = torch.zeros(self.input_dim)
                values = torch.zeros(self.input_dim)
                mask[node_idx] = 1.0
                values[node_idx] = intervention_value
                
                interventions['primary'] = (mask, values)
                
                # Maybe add multiple interventions
                if np.random.random() < multi_intervention_prob:
                    # Additional intervention
                    node_idx2 = np.random.randint(0, self.input_dim)
                    while node_idx2 == node_idx:
                        node_idx2 = np.random.randint(0, self.input_dim)
                    
                    intervention_value2 = np.random.randn()
                    mask2 = torch.zeros(self.input_dim)
                    values2 = torch.zeros(self.input_dim)
                    mask2[node_idx2] = 1.0
                    values2[node_idx2] = intervention_value2
                    
                    interventions['secondary'] = (mask2, values2)
            
            schedule.append(interventions)
        
        return schedule
    
    def apply_pathwise_intervention_algebra(self, 
                                          interventions_list: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply pathwise intervention algebra to combine multiple interventions.
        
        This implements algebraic composition of multiple simultaneous interventions
        using union/intersection of cut edges in the autograd path.
        """
        if not interventions_list:
            return None, None
        
        batch_size = len(interventions_list)
        device = next(self.parameters()).device
        
        # Initialize combined masks and values
        combined_mask = torch.zeros(batch_size, self.input_dim, device=device)
        combined_values = torch.zeros(batch_size, self.input_dim, device=device)
        
        for batch_idx, interventions in enumerate(interventions_list):
            if not interventions:
                continue
                
            # Process each intervention in the sample
            for intervention_name, (mask, values) in interventions.items():
                # Ensure proper dimensions
                if mask.dim() == 1:
                    mask = mask.to(device)
                if values.dim() == 1:
                    values = values.to(device)
                
                # Union of intervention masks (logical OR)
                combined_mask[batch_idx] = torch.logical_or(
                    combined_mask[batch_idx].bool(), mask.bool()
                ).float()
                
                # Use values from the last intervention for overlapping nodes
                combined_values[batch_idx] = torch.where(
                    mask.bool(), values, combined_values[batch_idx]
                )
        
        return combined_mask, combined_values
    
    def compute_dynamic_adjacency(self, x: torch.Tensor, interventions: Optional[List[Dict]] = None) -> torch.Tensor:
        """
        Compute dynamic adjacency matrix that changes based on interventions.
        
        This implements runtime graph rewiring where adjacencies change within
        a forward/backward pass based on the intervention schedule.
        """
        base_adjacency = self.get_adjacency_matrix(hard=False)
        
        if interventions is None:
            return base_adjacency
        
        # CRITICAL FIX: Add type checking to handle incorrect input types
        if not isinstance(interventions, list):
            print(f"WARNING: interventions expected to be List[Dict] but got {type(interventions)}: {interventions}")
            return base_adjacency
        
        # Apply intervention-based rewiring
        dynamic_adjacency = base_adjacency.clone()
        
        # For each intervention, cut edges to intervened nodes
        for intervention_dict in interventions:
            # Additional type checking for each intervention
            if not isinstance(intervention_dict, dict):
                print(f"WARNING: intervention_dict expected to be Dict but got {type(intervention_dict)}: {intervention_dict}")
                continue
                
            for intervention_name, (mask, values) in intervention_dict.items():
                intervened_nodes = mask.bool()
                
                # Cut all incoming edges to intervened nodes
                dynamic_adjacency[:, intervened_nodes] = 0.0
        
        return dynamic_adjacency
    
    def forward(self, 
                x: torch.Tensor,
                interventions: Optional[List[Dict]] = None,
                intervention_prob: float = 0.3,
                return_structure: bool = False,
                return_gradient_info: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass through the CausalUnit network.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            interventions: Optional list of interventions per sample
            intervention_prob: Probability of random interventions if interventions is None
            return_structure: Whether to return structure information
            return_gradient_info: Whether to return gradient flow information
            
        Returns:
            output: Network output (batch_size, output_dim)
            structure_info: Optional structure information
            gradient_info: Optional gradient flow information
        """
        batch_size = x.shape[0]
        
        # Handle structure learning
        structure_info = {}
        if self.structure_learning_mode and self.structure_learner is not None:
            x_reconstructed, adjacency = self.update_structure(x)
            structure_info['reconstruction'] = x_reconstructed
            structure_info['adjacency'] = adjacency
        
        # Create intervention schedule if not provided
        if interventions is None and self.intervention_training_mode:
            interventions = self.create_intervention_schedule(
                batch_size, intervention_prob=intervention_prob
            )
        
        # Apply pathwise intervention algebra
        combined_mask, combined_values = self.apply_pathwise_intervention_algebra(interventions)
        
        # Compute dynamic adjacency for forward pass
        dynamic_adjacency = self.compute_dynamic_adjacency(x, interventions)
        
        # CRITICAL FIX: Use original adjacency for violation penalty computation
        # Dynamic adjacency cuts edges which breaks violation penalty calculation!
        original_adjacency = self.get_adjacency_matrix(hard=False)
        
        # Forward pass through network
        current_activation = x
        gradient_info = {'unit_gradients': []}
        
        for i, unit in enumerate(self.units):
            # Only apply network-level adjacency to the first unit (input layer)
            # Other units use their own internal adjacency matrices
            if i == 0:
                # First unit: use network adjacency and interventions
                # For violation penalty: use input as parent_values to enable penalty computation
                parent_values = current_activation  # Input data for violation penalty computation
                
                # Ensure parent_values requires gradients for violation penalty computation
                if parent_values is not None and not parent_values.requires_grad:
                    parent_values = parent_values.requires_grad_(True)
                    print(f"DEBUG: Set parent_values.requires_grad = True")
                
                # Use original adjacency for violation penalty, dynamic for forward pass
                unit_adjacency = original_adjacency
                unit_do_mask = combined_mask
                unit_do_values = combined_values
            else:
                # Hidden/output units: use simple feedforward, no network-level interventions
                parent_values = None  # Not using parent-child relationships for hidden layers
                unit_adjacency = None  # Use unit's internal adjacency
                unit_do_mask = None    # No interventions for hidden layers
                unit_do_values = None
            
            # Forward through unit
            current_activation = unit(
                input_tensor=current_activation,
                parent_values=parent_values,
                adj_mask=unit_adjacency,
                do_mask=unit_do_mask,
                do_values=unit_do_values
            )
            
            # Debug: Verify adjacency matrix is being used
            if i == 0 and unit_adjacency is not None:
                print(f"DEBUG: Unit {i} using adjacency matrix:\n{unit_adjacency}")
                print(f"DEBUG: Adjacency matrix shape: {unit_adjacency.shape}")
                print(f"DEBUG: Non-zero elements: {torch.sum(unit_adjacency > 0).item()}")
            
            # Collect gradient info if requested
            if return_gradient_info:
                gradient_info['unit_gradients'].append(unit.get_gradient_flow_info())
        
        # Store intervention schedule for debugging
        self.intervention_schedule = interventions
        
        # Prepare return values
        returns = [current_activation]
        
        if return_structure:
            structure_info['dynamic_adjacency'] = dynamic_adjacency
            returns.append(structure_info)
        
        if return_gradient_info:
            returns.append(gradient_info)
        
        return returns[0] if len(returns) == 1 else tuple(returns)
    
    def set_training_mode(self, 
                         structure_learning: bool = False,
                         intervention_training: bool = True):
        """
        Set training modes for different components.
        
        Args:
            structure_learning: Whether to train structure learning
            intervention_training: Whether to use intervention training
        """
        self.structure_learning_mode = structure_learning
        self.intervention_training_mode = intervention_training
        
        # Update structure learner mode
        if self.structure_learner is not None:
            if structure_learning:
                self.structure_learner.train()
            else:
                self.structure_learner.eval()
    
    def get_network_info(self) -> Dict:
        """Get comprehensive information about the network state."""
        info = {
            'network_structure': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'output_dim': self.output_dim,
                'num_units': len(self.units),
                'unit_names': self.unit_names
            },
            'current_adjacency': self.current_adjacency.detach().cpu() if self.current_adjacency is not None else None,
            'learned_adjacency': self.learned_adjacency.detach().cpu() if self.learned_adjacency is not None else None,
            'intervention_schedule': self.intervention_schedule,
            'training_modes': {
                'structure_learning': self.structure_learning_mode,
                'intervention_training': self.intervention_training_mode
            },
            'unit_info': [unit.get_intervention_info() for unit in self.units]
        }
        
        return info
    
    def visualize_gradient_flow(self, save_path: Optional[str] = None) -> Dict:
        """
        Visualize gradient flow through the network for debugging.
        
        Args:
            save_path: Optional path to save visualization
            
        Returns:
            Dictionary with gradient flow information
        """
        gradient_flow = {}
        
        for i, unit in enumerate(self.units):
            unit_info = unit.get_gradient_flow_info()
            gradient_flow[f'unit_{i}'] = unit_info
        
        # Add network-level gradient information
        if self.learned_adjacency is not None and self.learned_adjacency.grad is not None:
            gradient_flow['adjacency_grad'] = self.learned_adjacency.grad.detach().cpu().numpy()
        
        # TODO: Add matplotlib visualization if save_path is provided
        
        return gradient_flow
    
    def reset_network_state(self):
        """Reset all network state for clean experiments."""
        self.intervention_schedule = []
        self.gradient_flow_history = []
        self.current_adjacency = None
        
        # Reset unit states
        for unit in self.units:
            unit.reset_intervention_history()
    
    def enable_gradient_surgery_all(self, enable: bool = True):
        """Enable/disable gradient surgery for all units."""
        for unit in self.units:
            unit.enable_runtime_rewiring(enable)
    
    def get_structure_learning_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Get structure learning loss for joint training."""
        if self.structure_learner is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Get structure loss with input data
        loss_dict = self.structure_learner.get_structure_loss(x)
        return loss_dict['total_loss'] 
    
    def get_causal_violation_penalty(self) -> torch.Tensor:
        """
        Compute the total causal violation penalty across all units.
        
        Returns:
            Total violation penalty as a tensor
        """
        total_penalty = 0.0
        device = next(self.parameters()).device
        
        for unit in self.units:
            unit_penalty = unit.get_causal_violation_penalty()
            total_penalty += unit_penalty
        
        return torch.tensor(total_penalty, device=device) * self.lambda_reg