#!/usr/bin/env python3
"""
CausalTransformer: Applying Causal Fine-Tuning to Transformer Architecture

This implementation adapts all our sophisticated causal mechanisms to work with
transformer models, including:

1. CausalAttention with gradient blocking
2. Soft interventions with alpha parameters
3. Dynamic adjacency matrices for attention patterns
4. Counterfactual reasoning for transformers
5. Active intervention sampling for attention heads
6. Intervention planning for transformer blocks
7. Violation penalty for attention mechanisms
8. Structure learning for attention patterns
9. All supporting causal systems

Key Innovation: Apply do-operator interventions to attention mechanisms while
maintaining the full transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
from transformers import GPT2LMHeadModel, GPT2Config
from dataclasses import dataclass

# Import all our causal components
from engine.causal_unit import CausalInterventionFunction
from engine.counterfactuals import CounterfactualSimulator
from engine.counterfactual_contrastive_learning import CounterfactualContrastiveLearner
from engine.intervention_planning import InterventionPlanner
from engine.active_intervention_sampling import ActiveInterventionSampler
from engine.structure_learning import DifferentiableDAG
from engine.loss_functions import CausalLosses
from engine.rewiring import DAGRewiring

@dataclass
class CausalTransformerConfig:
    """Configuration for CausalTransformer."""
    base_model_name: str = "gpt2"
    enable_causal_attention: bool = True
    enable_soft_interventions: bool = True
    enable_structure_learning: bool = True
    enable_gradient_surgery: bool = True
    intervention_prob: float = 0.3
    lambda_reg: float = 0.01
    use_low_rank_adjacency: bool = True
    adjacency_rank: int = 8
    max_intervention_strength: float = 0.8


class CausalAttentionFunction(Function):
    """
    Custom autograd function for causal attention with gradient blocking.
    
    Adapts our CausalInterventionFunction to work with transformer attention.
    """
    
    last_violation_penalty = 0.0
    
    @staticmethod
    def forward(ctx, query, key, value, attention_mask, adj_mask, do_mask, do_values, attention_weights):
        """
        Forward pass for causal attention with interventions.
        
        Args:
            query: Query tensor (batch_size, n_heads, seq_len, head_dim)
            key: Key tensor (batch_size, n_heads, seq_len, head_dim)
            value: Value tensor (batch_size, n_heads, seq_len, head_dim)
            attention_mask: Standard attention mask
            adj_mask: Causal adjacency mask for attention patterns
            do_mask: Intervention mask for attention
            do_values: Intervention values for attention
            attention_weights: Computed attention weights
        """
        # Save for backward pass
        ctx.save_for_backward(query, key, value, attention_mask, adj_mask, do_mask, do_values, attention_weights)
        
        # Apply causal interventions to attention weights
        if do_mask is not None and do_values is not None:
            # Cut edges in attention based on interventions
            effective_adj_mask = adj_mask.clone() if adj_mask is not None else torch.ones_like(attention_weights)
            intervened_positions = do_mask.bool()
            
            # Apply intervention to attention weights
            intervened_attention = torch.where(intervened_positions, do_values, attention_weights)
            
            # Apply causal masking
            masked_attention = intervened_attention * effective_adj_mask
            
            # Normalize attention weights
            masked_attention = F.softmax(masked_attention, dim=-1)
        else:
            # No intervention: apply normal attention
            masked_attention = F.softmax(attention_weights, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(masked_attention, value)
        
        return output, masked_attention
    
    @staticmethod
    def backward(ctx, grad_output, grad_attention):
        """
        Backward pass with gradient blocking for causal attention.
        """
        query, key, value, attention_mask, adj_mask, do_mask, do_values, attention_weights = ctx.saved_tensors
        
        # Initialize gradients
        grad_query = grad_key = grad_value = grad_attention_mask = None
        grad_adj_mask = grad_do_mask = grad_do_values = grad_attention_weights = None
        
        # Compute violation penalty
        violation_penalty = 0.0
        if do_mask is not None and adj_mask is not None:
            # Compute penalty for attention patterns that violate causal structure
            intervened_positions = do_mask.bool()
            
            # Find positions that should have zero attention due to causal structure
            causal_violations = adj_mask == 0
            
            # Compute penalty for attention to causally invalid positions
            attention_probs = F.softmax(attention_weights, dim=-1)
            violation_penalty = torch.sum(attention_probs * causal_violations.float())
        
        CausalAttentionFunction.last_violation_penalty = violation_penalty.item()
        
        # Block gradients for intervened attention positions
        if do_mask is not None and grad_attention_weights is not None:
            grad_attention_weights = grad_attention_weights * (~do_mask.bool()).float()
        
        # Standard gradient computation for transformer attention
        if ctx.needs_input_grad[0]:  # query
            grad_query = torch.matmul(grad_attention.transpose(-2, -1), key)
        if ctx.needs_input_grad[1]:  # key  
            grad_key = torch.matmul(grad_attention, query)
        if ctx.needs_input_grad[2]:  # value
            grad_value = torch.matmul(grad_attention.transpose(-2, -1), grad_output)
        
        return grad_query, grad_key, grad_value, grad_attention_mask, grad_adj_mask, grad_do_mask, grad_do_values, grad_attention_weights


class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with causal intervention support.
    
    Integrates all our causal mechanisms into transformer attention.
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 num_heads: int,
                 enable_structure_learning: bool = True,
                 enable_gradient_surgery: bool = True,
                 use_low_rank_adjacency: bool = True,
                 adjacency_rank: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.enable_structure_learning = enable_structure_learning
        self.enable_gradient_surgery = enable_gradient_surgery
        self.use_low_rank_adjacency = use_low_rank_adjacency
        self.adjacency_rank = adjacency_rank
        
        # Standard attention components
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Causal components adapted for attention
        
        # Soft intervention parameters for each attention head
        self.alpha = nn.Parameter(torch.zeros(num_heads))
        
        # Adjacency matrix for attention patterns (learned)
        if use_low_rank_adjacency:
            # Low-rank factorization for attention adjacency
            self.adj_matrix_U = nn.Parameter(torch.randn(num_heads, adjacency_rank) * 0.1)
            self.adj_matrix_V = nn.Parameter(torch.randn(adjacency_rank, num_heads) * 0.1)
        else:
            # Full adjacency matrix for attention heads
            self.adj_matrix = nn.Parameter(torch.randn(num_heads, num_heads) * 0.1)
        
        # Intervention history
        self.intervention_history = []
        self.attention_pattern_history = []
        
        # Temperature for attention adjacency
        self.attention_temperature = nn.Parameter(torch.ones(1))
        
    def get_attention_adjacency_matrix(self, hard: bool = False) -> torch.Tensor:
        """Get adjacency matrix for attention patterns."""
        if self.use_low_rank_adjacency:
            adj_matrix = torch.matmul(self.adj_matrix_U, self.adj_matrix_V)
        else:
            adj_matrix = self.adj_matrix
            
        if hard:
            return (torch.sigmoid(adj_matrix) > 0.5).float()
        else:
            return torch.sigmoid(adj_matrix / self.attention_temperature)
    
    def get_intervention_strength(self) -> torch.Tensor:
        """Get intervention strength for each attention head."""
        return torch.sigmoid(self.alpha)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                interventions: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Forward pass with causal attention.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_dim)
            attention_mask: Standard attention mask
            interventions: Dict of attention interventions
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute query, key, value
        query = self.query_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply standard attention mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Get causal adjacency matrix for attention
        adj_mask = self.get_attention_adjacency_matrix(hard=False)
        
        # Process interventions
        do_mask = None
        do_values = None
        if interventions is not None:
            # Combine interventions for attention patterns
            combined_mask = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=hidden_states.device)
            combined_values = torch.zeros(batch_size, self.num_heads, seq_len, seq_len, device=hidden_states.device)
            
            for name, (mask, values) in interventions.items():
                if mask.dim() == 2:  # Expand for batch and heads
                    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
                    values = values.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
                
                combined_mask = torch.logical_or(combined_mask.bool(), mask.bool()).float()
                combined_values = torch.where(mask.bool(), values, combined_values)
            
            do_mask = combined_mask
            do_values = combined_values
        
        # Apply soft interventions using alpha parameters
        if do_mask is not None and do_values is not None:
            alpha = self.get_intervention_strength()  # (num_heads,)
            alpha = alpha.view(1, -1, 1, 1).expand(batch_size, -1, seq_len, seq_len)
            
            # Soft intervention: attention = (1 - alpha) * original + alpha * intervention
            soft_intervention = (1 - alpha) * attention_scores + alpha * do_values
            attention_scores = torch.where(do_mask.bool(), soft_intervention, attention_scores)
        
        # Apply causal attention function
        attention_output, attention_weights = CausalAttentionFunction.apply(
            query, key, value, attention_mask, adj_mask, do_mask, do_values, attention_scores
        )
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(attention_output)
        
        # Store intervention history
        if interventions is not None:
            self.intervention_history.append({
                'interventions': interventions,
                'attention_weights': attention_weights.detach().cpu(),
                'timestamp': len(self.intervention_history)
            })
        
        return output
    
    def get_causal_violation_penalty(self) -> float:
        """Get the last computed causal violation penalty."""
        return CausalAttentionFunction.last_violation_penalty


class CausalTransformerBlock(nn.Module):
    """
    Transformer block with causal mechanisms.
    
    Integrates causal attention with feedforward layers and all our causal systems.
    """
    
    def __init__(self, config: CausalTransformerConfig):
        super().__init__()
        
        self.config = config
        self.hidden_dim = 768  # GPT-2 hidden dimension
        self.num_heads = 12    # GPT-2 attention heads
        
        # Causal attention layer
        self.causal_attention = CausalMultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            enable_structure_learning=config.enable_structure_learning,
            enable_gradient_surgery=config.enable_gradient_surgery,
            use_low_rank_adjacency=config.use_low_rank_adjacency,
            adjacency_rank=config.adjacency_rank
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        
        # Feedforward with causal mechanisms
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
        # Causal components for this block
        self.block_alpha = nn.Parameter(torch.zeros(self.hidden_dim))
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                interventions: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None) -> torch.Tensor:
        """
        Forward pass through causal transformer block.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, hidden_dim)
            attention_mask: Standard attention mask
            interventions: Dict of interventions for this block
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_dim)
        """
        # Causal attention with residual connection
        attention_output = self.causal_attention(
            self.ln1(hidden_states), 
            attention_mask=attention_mask,
            interventions=interventions
        )
        
        # Apply block-level soft interventions if specified
        if interventions is not None and 'block_intervention' in interventions:
            mask, values = interventions['block_intervention']
            alpha = torch.sigmoid(self.block_alpha)
            
            # Soft intervention on attention output
            soft_intervention = (1 - alpha) * attention_output + alpha * values
            attention_output = torch.where(mask.bool(), soft_intervention, attention_output)
        
        hidden_states = hidden_states + self.dropout(attention_output)
        
        # Feedforward with residual connection
        feedforward_output = self.feedforward(self.ln2(hidden_states))
        hidden_states = hidden_states + self.dropout(feedforward_output)
        
        return hidden_states


class CausalTransformer(nn.Module):
    """
    Complete Causal Transformer with all our sophisticated causal mechanisms.
    
    This is the main model that integrates:
    1. CausalTransformerBlocks with causal attention
    2. CounterfactualSimulator for transformer outputs
    3. ActiveInterventionSampler for strategic attention interventions
    4. InterventionPlanner for transformer block interventions
    5. Structure learning for attention patterns
    6. All supporting causal systems
    """
    
    def __init__(self, config: CausalTransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Load base transformer model
        self.base_config = GPT2Config.from_pretrained(config.base_model_name)
        self.base_model = GPT2LMHeadModel.from_pretrained(config.base_model_name)
        
        # Replace transformer blocks with causal versions
        self.causal_blocks = nn.ModuleList([
            CausalTransformerBlock(config) for _ in range(self.base_config.n_layer)
        ])
        
        # Causal systems integration
        self.counterfactual_simulator = CounterfactualSimulator(self)
        
        self.active_sampler = ActiveInterventionSampler(
            n_nodes=self.base_config.n_head,  # Use attention heads as nodes
            uncertainty_threshold=0.1,
            exploration_rate=0.3,
            temperature=1.0
        )
        
        self.intervention_planner = InterventionPlanner(
            n_nodes=self.base_config.n_head,
            max_worlds=8,
            min_world_probability=0.05,
            evidence_decay=0.9
        )
        
        self.contrastive_learner = CounterfactualContrastiveLearner(
            n_nodes=self.base_config.n_head,
            embedding_dim=64,
            temperature=0.1,
            learning_rate=0.01
        )
        
        # Structure learning for attention patterns
        self.structure_learner = DifferentiableDAG(
            num_variables=self.base_config.n_head,
            hidden_dim=32,
            enable_structure_transfer=True
        )
        
        # Causal loss functions
        self.causal_losses = CausalLosses(
            intervention_weight=1.0,
            counterfactual_weight=0.5,
            structure_weight=0.1,
            sparsity_weight=0.01
        )
        
        # Training state
        self.intervention_schedule = []
        self.learning_history = []
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                interventions: Optional[List[Dict]] = None,
                return_causal_info: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass through causal transformer.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Standard attention mask
            interventions: List of interventions per layer
            return_causal_info: Whether to return causal information
            
        Returns:
            Output logits or tuple with causal information
        """
        # Get embeddings from base model
        hidden_states = self.base_model.transformer.wte(input_ids)
        
        # Apply positional embeddings
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        position_embeddings = self.base_model.transformer.wpe(position_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Track causal information
        causal_info = {
            'intervention_history': [],
            'attention_patterns': [],
            'violation_penalties': []
        }
        
        # Forward through causal transformer blocks
        for i, block in enumerate(self.causal_blocks):
            # Get interventions for this block
            block_interventions = None
            if interventions is not None and i < len(interventions):
                block_interventions = interventions[i]
            
            # Apply strategic interventions if in training mode
            if self.training and block_interventions is None:
                # Sample interventions based on uncertainty
                if np.random.random() < self.config.intervention_prob:
                    # Create simple intervention for this block
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    
                    # Random attention intervention
                    mask = torch.zeros(batch_size, 12, seq_len, seq_len)  # 12 attention heads
                    values = torch.randn(batch_size, 12, seq_len, seq_len) * 0.1
                    
                    # Randomly intervene on some attention patterns
                    for b in range(batch_size):
                        n_heads = np.random.randint(1, 4)
                        head_indices = np.random.choice(12, n_heads, replace=False)
                        mask[b, head_indices, :, :] = 1.0
                    
                    block_interventions = {
                        'attention_intervention': (mask, values)
                    }
            
            # Forward through causal block
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                interventions=block_interventions
            )
            
            # Track causal information
            if return_causal_info:
                causal_info['intervention_history'].append(block_interventions)
                causal_info['violation_penalties'].append(block.causal_attention.get_causal_violation_penalty())
        
        # Final layer norm
        hidden_states = self.base_model.transformer.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.base_model.lm_head(hidden_states)
        
        if return_causal_info:
            return logits, causal_info
        else:
            return logits
    
    def compute_causal_loss(self, 
                          logits: torch.Tensor,
                          labels: torch.Tensor,
                          interventions: Optional[List[Dict]] = None) -> torch.Tensor:
        """
        Compute total causal loss including all causal components.
        
        Args:
            logits: Model output logits
            labels: Target labels
            interventions: Applied interventions
            
        Returns:
            Total causal loss
        """
        # Standard language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        prediction_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Causal violation penalty
        violation_penalty = sum(
            block.causal_attention.get_causal_violation_penalty()
            for block in self.causal_blocks
        )
        
        # Structure learning loss
        structure_loss = self.structure_learner.get_structure_loss()
        
        # Counterfactual consistency loss
        counterfactual_loss = 0.0
        if interventions is not None:
            # Compute counterfactual predictions
            counterfactual_logits = self.counterfactual_simulator.simulate_counterfactual(
                logits, interventions
            )
            counterfactual_loss = F.mse_loss(counterfactual_logits, logits)
        
        # Total loss
        total_loss = (prediction_loss + 
                     self.config.lambda_reg * violation_penalty +
                     0.1 * structure_loss +
                     0.5 * counterfactual_loss)
        
        return total_loss
    
    def update_causal_beliefs(self, 
                            intervention_outcomes: List[Dict],
                            prediction_accuracy: float):
        """
        Update causal beliefs based on intervention outcomes.
        
        Args:
            intervention_outcomes: Results of interventions
            prediction_accuracy: Current prediction accuracy
        """
        # Update intervention planner
        self.intervention_planner.update_beliefs(intervention_outcomes, prediction_accuracy)
        
        # Update active sampler
        self.active_sampler.update_uncertainty(intervention_outcomes)
        
        # Update contrastive learner
        self.contrastive_learner.update_representations(intervention_outcomes)
        
        # Track learning history
        self.learning_history.append({
            'outcomes': intervention_outcomes,
            'accuracy': prediction_accuracy,
            'timestamp': len(self.learning_history)
        })
    
    def get_causal_summary(self) -> Dict:
        """Get comprehensive summary of causal system state."""
        return {
            'model_config': self.config.__dict__,
            'intervention_count': len(self.intervention_schedule),
            'learning_history_length': len(self.learning_history),
            'active_sampler_state': self.active_sampler.get_state(),
            'intervention_planner_state': self.intervention_planner.get_state(),
            'structure_learner_state': self.structure_learner.get_structure_summary(),
            'causal_blocks_count': len(self.causal_blocks)
        }


def create_causal_gpt2(model_name: str = "gpt2") -> CausalTransformer:
    """
    Create a CausalTransformer based on GPT-2 architecture.
    
    Args:
        model_name: Name of the base GPT-2 model
        
    Returns:
        CausalTransformer instance
    """
    config = CausalTransformerConfig(
        base_model_name=model_name,
        enable_causal_attention=True,
        enable_soft_interventions=True,
        enable_structure_learning=True,
        enable_gradient_surgery=True,
        intervention_prob=0.3,
        lambda_reg=0.01,
        use_low_rank_adjacency=True,
        adjacency_rank=8
    )
    
    return CausalTransformer(config)


if __name__ == "__main__":
    # Test the CausalTransformer
    print("Creating CausalTransformer...")
    
    model = create_causal_gpt2("gpt2")
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    attention_mask = torch.ones(2, 10)
    
    print("Testing forward pass...")
    logits = model(input_ids, attention_mask)
    print(f"Output shape: {logits.shape}")
    
    # Test with interventions
    interventions = [
        {'attention_intervention': (
            torch.zeros(2, 12, 10, 10),  # mask
            torch.randn(2, 12, 10, 10)   # values
        )}
    ]
    
    print("Testing with interventions...")
    logits_with_interventions = model(input_ids, attention_mask, interventions)
    print(f"Output with interventions shape: {logits_with_interventions.shape}")
    
    # Test causal summary
    summary = model.get_causal_summary()
    print(f"Causal summary: {summary}")
    
    print("CausalTransformer test completed successfully!") 