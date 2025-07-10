import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict, List
import warnings


class DifferentiableDAG(nn.Module):
    """
    Differentiable DAG structure learning module.
    
    Implements a simplified version of NOTEARS (Neural Structure Learning with 
    Smooth Acyclicity Constraint) for learning causal graph structure.
    """
    
    def __init__(self, num_variables: int, hidden_dim: int = 16, enable_structure_transfer: bool = True):
        """
        Initialize differentiable DAG learner.
        
        Args:
            num_variables: Number of variables in the causal graph
            hidden_dim: Hidden dimension for the neural network
            enable_structure_transfer: Whether to enable structure transfer mechanisms
        """
        super(DifferentiableDAG, self).__init__()
        
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        self.enable_structure_transfer = enable_structure_transfer
        
        # Learnable adjacency matrix (logits)
        self.adjacency_logits = nn.Parameter(
            torch.randn(num_variables, num_variables) * 0.1
        )
        
        # Neural network for modeling functional relationships
        self.functional_net = nn.ModuleDict()
        for i in range(num_variables):
            self.functional_net[f'var_{i}'] = nn.Sequential(
                nn.Linear(num_variables, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Temperature for Gumbel-Softmax (learnable)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Structure transfer components
        if enable_structure_transfer:
            # Memory bank for storing previous successful structures
            self.structure_memory = []
            self.structure_scores = []
            self.max_memory_size = 10
            
            # Transfer learning weights
            self.transfer_weight = nn.Parameter(torch.tensor(0.1))  # How much to rely on previous structures
            self.similarity_threshold = 0.7  # Threshold for structure similarity
            
            # Meta-learning component for structure adaptation
            self.meta_learner = nn.Sequential(
                nn.Linear(num_variables * num_variables, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_variables * num_variables),
                nn.Tanh()  # Output adjustment factors
            )
        
        print(f"DifferentiableDAG initialized with structure transfer: {enable_structure_transfer}")
    
    def add_structure_to_memory(self, adjacency: torch.Tensor, performance_score: float):
        """
        Add a successful structure to memory for future transfer.
        
        Args:
            adjacency: Adjacency matrix of successful structure
            performance_score: Performance score of this structure (higher is better)
        """
        if not self.enable_structure_transfer:
            return
        
        # Store structure and score
        self.structure_memory.append(adjacency.detach().clone())
        self.structure_scores.append(performance_score)
        
        # Keep only the best structures (up to max_memory_size)
        if len(self.structure_memory) > self.max_memory_size:
            # Sort by performance score and keep the best
            sorted_indices = sorted(range(len(self.structure_scores)), 
                                  key=lambda i: self.structure_scores[i], reverse=True)
            
            self.structure_memory = [self.structure_memory[i] for i in sorted_indices[:self.max_memory_size]]
            self.structure_scores = [self.structure_scores[i] for i in sorted_indices[:self.max_memory_size]]
        
        print(f"Added structure to memory. Memory size: {len(self.structure_memory)}")
    
    def find_similar_structures(self, current_adjacency: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """
        Find similar structures in memory based on structural similarity.
        
        Args:
            current_adjacency: Current adjacency matrix
            
        Returns:
            List of (similar_structure, similarity_score) tuples
        """
        if not self.enable_structure_transfer or not self.structure_memory:
            return []
        
        similar_structures = []
        
        for stored_structure in self.structure_memory:
            # Calculate structural similarity (normalized dot product)
            similarity = torch.sum(current_adjacency * stored_structure) / (
                torch.norm(current_adjacency) * torch.norm(stored_structure) + 1e-8
            )
            
            if similarity > self.similarity_threshold:
                similar_structures.append((stored_structure, similarity.item()))
        
        # Sort by similarity score
        similar_structures.sort(key=lambda x: x[1], reverse=True)
        
        return similar_structures
    
    def warm_start_from_previous(self, target_structure: torch.Tensor, adaptation_strength: float = 0.5):
        """
        Warm-start the adjacency matrix from a previous successful structure.
        
        Args:
            target_structure: Target structure to initialize from
            adaptation_strength: How much to adapt towards the target (0=no adaptation, 1=full copy)
        """
        if not self.enable_structure_transfer:
            return
        
        with torch.no_grad():
            # Get current adjacency logits
            current_logits = self.adjacency_logits.data
            
            # Convert target structure to logits (inverse sigmoid)
            target_logits = torch.log(target_structure + 1e-8) - torch.log(1 - target_structure + 1e-8)
            
            # Apply meta-learning adaptation
            target_flat = target_structure.flatten()
            adaptation_factors = self.meta_learner(target_flat)
            adapted_target_logits = target_logits * adaptation_factors.view_as(target_logits)
            
            # Blend current and target logits
            self.adjacency_logits.data = (
                (1 - adaptation_strength) * current_logits + 
                adaptation_strength * adapted_target_logits
            )
        
        print(f"Warm-started adjacency matrix with adaptation strength: {adaptation_strength}")
    
    def transfer_learning_step(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform a structure transfer learning step.
        
        Args:
            X: Input data
            
        Returns:
            Transfer loss to add to the main loss
        """
        if not self.enable_structure_transfer or not self.structure_memory:
            return torch.tensor(0.0)
        
        current_adjacency = self.get_adjacency_matrix(hard=False)
        
        # Find similar structures
        similar_structures = self.find_similar_structures(current_adjacency)
        
        if not similar_structures:
            return torch.tensor(0.0)
        
        # Compute transfer loss: encourage similarity to successful structures
        transfer_loss = torch.tensor(0.0)
        
        for similar_structure, similarity_score in similar_structures[:3]:  # Top 3 similar structures
            # Distance loss weighted by similarity
            distance = torch.norm(current_adjacency - similar_structure, p='fro')
            weight = similarity_score * self.transfer_weight
            transfer_loss += weight * distance
        
        return transfer_loss
    
    def get_adjacency_matrix(self, hard: bool = False):
        """
        Get the current adjacency matrix using Gumbel-Softmax.
        
        Args:
            hard: Whether to use hard (discrete) or soft (continuous) adjacency
            
        Returns:
            adjacency: Adjacency matrix (num_variables, num_variables)
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(self.adjacency_logits)
        
        if hard:
            # Hard thresholding
            adjacency = (probs > 0.5).float()
        else:
            # Use Gumbel-Softmax for differentiability
            # Create binary choice logits [no_edge, edge]
            binary_logits = torch.stack([
                -self.adjacency_logits,  # logit for no edge
                self.adjacency_logits    # logit for edge
            ], dim=-1)
            
            # Apply Gumbel-Softmax
            gumbel_out = F.gumbel_softmax(
                binary_logits, tau=self.temperature, hard=hard, dim=-1
            )
            adjacency = gumbel_out[..., 1]  # Take the "edge" probability
        
        # Zero out diagonal (no self-loops)
        adjacency = adjacency * (1 - torch.eye(self.num_variables))
        
        return adjacency
    
    def forward(self, X: torch.Tensor, hard_adjacency: bool = False):
        """
        Forward pass through the differentiable DAG.
        
        Args:
            X: Input data (batch_size, num_variables)
            hard_adjacency: Whether to use hard adjacency matrix
            
        Returns:
            reconstructed_X: Reconstructed input based on learned structure
            adjacency: Current adjacency matrix
        """
        adjacency = self.get_adjacency_matrix(hard=hard_adjacency)
        
        # Reconstruct each variable based on its parents
        reconstructed = []
        
        for i in range(self.num_variables):
            # Get parents of variable i
            parents_mask = adjacency[:, i]  # Column i gives parents of variable i
            
            # Create input with only parent variables (others set to 0)
            masked_input = X * parents_mask.unsqueeze(0)  # Broadcasting
            
            # Pass through functional network
            var_prediction = self.functional_net[f'var_{i}'](masked_input)
            reconstructed.append(var_prediction)
        
        reconstructed_X = torch.cat(reconstructed, dim=1)
        
        return reconstructed_X, adjacency
    
    def acyclicity_constraint(self, adjacency: torch.Tensor):
        """
        Compute the acyclicity constraint using matrix exponential trace.
        
        Based on: trace(e^(A ⊙ A)) - d = 0 for DAGs
        where A is adjacency matrix and d is number of variables.
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            constraint: Acyclicity constraint value (should be close to 0 for DAGs)
        """
        # Element-wise square
        A_squared = adjacency * adjacency
        
        # Matrix exponential using series expansion (truncated for efficiency)
        # exp(A) ≈ I + A + A²/2! + A³/3! + ...
        exp_A = torch.eye(self.num_variables, device=adjacency.device)
        A_power = torch.eye(self.num_variables, device=adjacency.device)
        
        for i in range(1, 10):  # Truncate at 10 terms
            A_power = torch.matmul(A_power, A_squared)
            exp_A = exp_A + A_power / math.factorial(i)
        
        # Trace of exponential minus number of variables
        constraint = torch.trace(exp_A) - self.num_variables
        
        return constraint
    
    def sparsity_loss(self, adjacency: torch.Tensor):
        """
        Encourage sparsity in the adjacency matrix.
        
        Args:
            adjacency: Adjacency matrix
            
        Returns:
            sparsity_loss: L1 norm of adjacency matrix
        """
        return torch.sum(torch.abs(adjacency))
    
    def get_structure_loss(self, X: torch.Tensor, lambda_acyclic: float = 1.0, 
                          lambda_sparse: float = 0.01, lambda_transfer: float = 0.1):
        """
        Compute total structure learning loss including transfer learning.
        
        Args:
            X: Input data
            lambda_acyclic: Weight for acyclicity constraint
            lambda_sparse: Weight for sparsity constraint
            lambda_transfer: Weight for transfer learning loss
            
        Returns:
            Dictionary with individual loss components and total loss
        """
        # Forward pass
        X_reconstructed, adjacency = self.forward(X, hard_adjacency=False)
        
        # Individual loss components
        reconstruction_loss = F.mse_loss(X_reconstructed, X)
        acyclicity_loss = self.acyclicity_constraint(adjacency)
        sparsity_loss = self.sparsity_loss(adjacency)
        transfer_loss = self.transfer_learning_step(X)
        
        # Total loss
        total_loss = (reconstruction_loss + 
                     lambda_acyclic * torch.abs(acyclicity_loss) +
                     lambda_sparse * sparsity_loss +
                     lambda_transfer * transfer_loss)
        
        return {
            'reconstruction_loss': reconstruction_loss,
            'acyclicity_loss': acyclicity_loss,
            'sparsity_loss': sparsity_loss,
            'transfer_loss': transfer_loss,
            'total_loss': total_loss,
            'adjacency': adjacency
        }


class StructureLearner:
    """
    High-level interface for learning causal graph structure.
    """
    
    def __init__(self, num_variables: int, hidden_dim: int = 16):
        """
        Initialize structure learner.
        
        Args:
            num_variables: Number of variables in the causal graph
            hidden_dim: Hidden dimension for neural networks
        """
        self.num_variables = num_variables
        self.dag_model = DifferentiableDAG(num_variables, hidden_dim)
        
    def learn_structure(self, X: torch.Tensor, num_epochs: int = 1000,
                       lr: float = 0.01, lambda_acyclic: float = 1.0,
                       lambda_sparse: float = 0.01, verbose: bool = True):
        """
        Learn causal structure from data.
        
        Args:
            X: Training data (batch_size, num_variables)
            num_epochs: Number of training epochs
            lr: Learning rate
            lambda_acyclic: Weight for acyclicity constraint
            lambda_sparse: Weight for sparsity constraint
            verbose: Whether to print training progress
            
        Returns:
            learned_adjacency: Final learned adjacency matrix
            training_history: Dictionary with training metrics
        """
        optimizer = torch.optim.Adam(self.dag_model.parameters(), lr=lr)
        
        training_history = {
            'reconstruction_loss': [],
            'acyclicity_constraint': [],
            'sparsity_loss': [],
            'total_loss': []
        }
        
        self.dag_model.train()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            X_reconstructed, adjacency = self.dag_model(X, hard_adjacency=False)
            
            # Reconstruction loss
            reconstruction_loss = F.mse_loss(X_reconstructed, X)
            
            # Acyclicity constraint
            acyclicity_loss = self.dag_model.acyclicity_constraint(adjacency)
            
            # Sparsity loss
            sparsity_loss = self.dag_model.sparsity_loss(adjacency)
            
            # Total loss
            total_loss = (reconstruction_loss + 
                         lambda_acyclic * torch.abs(acyclicity_loss) +
                         lambda_sparse * sparsity_loss)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Temperature annealing
            if hasattr(self.dag_model, 'temperature'):
                self.dag_model.temperature.data = torch.clamp(
                    self.dag_model.temperature.data * 0.999, min=0.1
                )
            
            # Record metrics
            training_history['reconstruction_loss'].append(reconstruction_loss.item())
            training_history['acyclicity_constraint'].append(acyclicity_loss.item())
            training_history['sparsity_loss'].append(sparsity_loss.item())
            training_history['total_loss'].append(total_loss.item())
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d}: "
                      f"Recon={reconstruction_loss.item():.4f}, "
                      f"Acyc={acyclicity_loss.item():.4f}, "
                      f"Sparse={sparsity_loss.item():.4f}, "
                      f"Total={total_loss.item():.4f}")
        
        # Get final adjacency matrix
        self.dag_model.eval()
        with torch.no_grad():
            learned_adjacency = self.dag_model.get_adjacency_matrix(hard=True)
        
        return learned_adjacency, training_history
    
    def evaluate_structure_recovery(self, learned_adjacency: torch.Tensor,
                                   true_adjacency: torch.Tensor):
        """
        Evaluate how well the learned structure matches the true structure.
        
        Args:
            learned_adjacency: Learned adjacency matrix
            true_adjacency: True adjacency matrix
            
        Returns:
            metrics: Dictionary with evaluation metrics
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


def create_true_dag(num_variables: int, sparsity: float = 0.3):
    """
    Create a true DAG for testing structure learning.
    
    Args:
        num_variables: Number of variables
        sparsity: Probability of having an edge
        
    Returns:
        adjacency: True adjacency matrix
    """
    # Create random DAG by generating upper triangular matrix
    adjacency = torch.zeros(num_variables, num_variables)
    
    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            if torch.rand(1).item() < sparsity:
                adjacency[i, j] = 1.0
    
    return adjacency


def generate_dag_data(adjacency: torch.Tensor, n_samples: int = 1000,
                     noise_std: float = 0.1):
    """
    Generate data from a given DAG structure.
    
    Args:
        adjacency: True adjacency matrix
        n_samples: Number of samples to generate
        noise_std: Standard deviation of noise
        
    Returns:
        X: Generated data
        coefficients: True causal coefficients
    """
    num_variables = adjacency.shape[0]
    X = torch.zeros(n_samples, num_variables)
    
    # Random coefficients for edges
    coefficients = torch.randn_like(adjacency) * adjacency
    
    # Generate data in topological order
    for j in range(num_variables):
        # Parents of variable j
        parents = torch.where(adjacency[:, j] > 0)[0]
        
        if len(parents) > 0:
            # Linear combination of parents
            parent_contribution = torch.sum(
                X[:, parents] * coefficients[parents, j], dim=1
            )
            X[:, j] = parent_contribution + noise_std * torch.randn(n_samples)
        else:
            # Root node (no parents)
            X[:, j] = torch.randn(n_samples)
    
    return X, coefficients 