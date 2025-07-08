import torch
import torch.nn as nn
import numpy as np

class DAGRewiring:
    """
    Handles DAG (Directed Acyclic Graph) rewiring logic for causal interventions.
    
    This class manages:
    - Edge masking based on causal interventions
    - Weight matrix modifications to simulate structural changes
    - Topological ordering for causal consistency
    """
    
    def __init__(self, num_variables):
        self.num_variables = num_variables
        # Adjacency matrix: adj_matrix[i][j] = 1 means edge from i to j
        self.adj_matrix = torch.zeros(num_variables, num_variables)
        self.original_adj_matrix = None
    
    def set_adjacency_matrix(self, adj_matrix):
        """
        Set the adjacency matrix representing the causal DAG.
        
        Args:
            adj_matrix: Tensor of shape (num_vars, num_vars) where entry (i,j) = 1
                       indicates a directed edge from variable i to variable j
        """
        self.adj_matrix = adj_matrix.clone()
        self.original_adj_matrix = adj_matrix.clone()
    
    def mask_edges(self, weight_matrix, do_mask):
        """
        Zero out edges (weights) that feed into intervened nodes.
        
        When we intervene on a variable (set do_mask[i] = 1), we should block
        all incoming edges to that variable to simulate the intervention.
        
        Args:
            weight_matrix: Tensor of shape (output_dim, input_dim) representing layer weights
            do_mask: Binary tensor of shape (input_dim,) indicating intervened variables
            
        Returns:
            Modified weight matrix with masked edges
        """
        if do_mask is None:
            return weight_matrix
            
        # Clone to avoid modifying original weights
        masked_weights = weight_matrix.clone()
        
        # For each intervened variable, zero out incoming weights
        intervened_vars = torch.where(do_mask == 1)[0]
        
        for var_idx in intervened_vars:
            # Zero out the column corresponding to the intervened variable
            # This prevents information from flowing TO the intervened variable
            if var_idx < masked_weights.shape[1]:
                masked_weights[:, var_idx] = 0.0
                
        return masked_weights
    
    def get_parents(self, node_idx):
        """
        Get parent nodes of a given node in the DAG.
        
        Args:
            node_idx: Index of the node
            
        Returns:
            List of parent node indices
        """
        parents = []
        for i in range(self.num_variables):
            if self.adj_matrix[i, node_idx] == 1:
                parents.append(i)
        return parents
    
    def get_children(self, node_idx):
        """
        Get child nodes of a given node in the DAG.
        
        Args:
            node_idx: Index of the node
            
        Returns:
            List of child node indices
        """
        children = []
        for j in range(self.num_variables):
            if self.adj_matrix[node_idx, j] == 1:
                children.append(j)
        return children
    
    def topological_sort(self):
        """
        Perform topological sorting of the DAG.
        
        Returns:
            List of node indices in topological order
        """
        # Kahn's algorithm for topological sorting
        in_degree = torch.sum(self.adj_matrix, dim=0)
        queue = []
        result = []
        
        # Find all nodes with no incoming edges
        for i in range(self.num_variables):
            if in_degree[i] == 0:
                queue.append(i)
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # For each child of current node
            children = self.get_children(node)
            for child in children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return result
    
    def apply_intervention_to_dag(self, do_mask):
        """
        Modify the DAG structure to reflect interventions.
        
        When we intervene on variables, we remove all incoming edges to those variables.
        
        Args:
            do_mask: Binary tensor indicating which variables are intervened
        """
        if do_mask is None:
            return
            
        # Reset to original structure
        if self.original_adj_matrix is not None:
            self.adj_matrix = self.original_adj_matrix.clone()
        
        # Remove incoming edges to intervened variables
        intervened_vars = torch.where(do_mask == 1)[0]
        
        for var_idx in intervened_vars:
            # Set all incoming edges to this variable to 0
            self.adj_matrix[:, var_idx] = 0
    
    def is_acyclic(self):
        """
        Check if the current graph structure is acyclic.
        
        Returns:
            Boolean indicating whether the graph is a DAG
        """
        try:
            topo_order = self.topological_sort()
            return len(topo_order) == self.num_variables
        except:
            return False


class CausalLayerRewiring(nn.Module):
    """
    A neural network layer that applies causal rewiring to its weights.
    
    This layer can dynamically modify its connectivity based on causal interventions.
    """
    
    def __init__(self, input_dim, output_dim, dag_rewiring=None):
        super(CausalLayerRewiring, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Standard linear layer
        self.linear = nn.Linear(input_dim, output_dim)
        
        # DAG rewiring manager
        self.dag_rewiring = dag_rewiring
        
        # Store original weights for reset
        self.original_weight = None
        
    def forward(self, x, do_mask=None):
        """
        Forward pass with optional edge masking based on interventions.
        
        Args:
            x: Input tensor
            do_mask: Binary mask indicating intervened variables
            
        Returns:
            Output tensor with causal rewiring applied
        """
        # Store original weights if not already stored
        if self.original_weight is None:
            self.original_weight = self.linear.weight.data.clone()
        
        # Apply edge masking if DAG rewiring is available and interventions are specified
        if self.dag_rewiring is not None and do_mask is not None:
            # Get masked weights
            masked_weights = self.dag_rewiring.mask_edges(self.linear.weight, do_mask)
            
            # Temporarily replace weights
            original_weight = self.linear.weight.data.clone()
            self.linear.weight.data = masked_weights
            
            # Forward pass
            output = self.linear(x)
            
            # Restore original weights
            self.linear.weight.data = original_weight
            
            return output
        else:
            # Normal forward pass
            return self.linear(x)
    
    def reset_weights(self):
        """Reset weights to original values."""
        if self.original_weight is not None:
            self.linear.weight.data = self.original_weight.clone()
