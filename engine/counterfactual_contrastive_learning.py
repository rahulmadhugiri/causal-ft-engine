import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import math
from collections import defaultdict

@dataclass
class ContrastivePair:
    """Represents a contrastive pair of interventions."""
    intervention_a: Dict
    intervention_b: Dict
    outcome_a: Dict
    outcome_b: Dict
    similarity_score: float
    contrast_score: float

class CounterfactualContrastiveLearner:
    """
    Counterfactual contrastive learning system that learns which causal relationships
    matter most by comparing intervention outcomes.
    
    Key Features:
    1. Creates contrastive pairs from intervention outcomes
    2. Learns representations that separate different causal effects
    3. Identifies most important causal relationships
    4. Updates structure learning priorities
    """
    
    def __init__(self, 
                 n_nodes: int,
                 embedding_dim: int = 64,
                 temperature: float = 0.1,
                 negative_sampling_ratio: float = 2.0,
                 similarity_threshold: float = 0.1,
                 learning_rate: float = 0.001):
        """
        Initialize counterfactual contrastive learner.
        
        Args:
            n_nodes: Number of nodes in causal graph
            embedding_dim: Dimension of learned embeddings
            temperature: Temperature for contrastive loss
            negative_sampling_ratio: Ratio of negative to positive samples
            similarity_threshold: Threshold for considering interventions similar
            learning_rate: Learning rate for embedding updates
        """
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.negative_sampling_ratio = negative_sampling_ratio
        self.similarity_threshold = similarity_threshold
        self.learning_rate = learning_rate
        
        # Learned representations
        self.intervention_encoder = nn.Linear(n_nodes + 1, embedding_dim)  # +1 for intervention value
        self.outcome_encoder = nn.Linear(n_nodes, embedding_dim)
        self.causal_relation_embeddings = nn.Parameter(torch.randn(n_nodes, n_nodes, embedding_dim))
        
        # Optimizer for embeddings
        self.optimizer = torch.optim.Adam(
            list(self.intervention_encoder.parameters()) + 
            list(self.outcome_encoder.parameters()) + 
            [self.causal_relation_embeddings],
            lr=learning_rate
        )
        
        # Storage for contrastive pairs
        self.contrastive_pairs = []
        self.intervention_history = []
        self.outcome_history = []
        
        # Learning statistics
        self.contrastive_losses = []
        self.relationship_importance = torch.zeros(n_nodes, n_nodes)
        self.learning_progress = []
        
    def encode_intervention(self, intervention: Dict) -> torch.Tensor:
        """
        Encode an intervention into a learned representation.
        
        Args:
            intervention: Intervention dictionary
            
        Returns:
            Intervention embedding
        """
        # Create intervention vector
        intervention_vector = torch.zeros(self.n_nodes + 1)
        intervention_vector[intervention['node']] = 1.0
        intervention_vector[-1] = intervention['value']
        
        # Encode to embedding space
        embedding = self.intervention_encoder(intervention_vector)
        
        return embedding
    
    def encode_outcome(self, outcome: Dict) -> torch.Tensor:
        """
        Encode an outcome into a learned representation.
        
        Args:
            outcome: Outcome dictionary
            
        Returns:
            Outcome embedding
        """
        # Create outcome vector
        outcome_vector = torch.zeros(self.n_nodes)
        
        # Fill in effect magnitudes
        for node, effect in outcome.get('node_effects', {}).items():
            if node < self.n_nodes:
                outcome_vector[node] = effect
        
        # Encode to embedding space
        embedding = self.outcome_encoder(outcome_vector)
        
        return embedding
    
    def create_contrastive_pairs(self, 
                               min_pairs: int = 5,
                               max_pairs: int = 50) -> List[ContrastivePair]:
        """
        Create contrastive pairs from intervention history.
        
        Args:
            min_pairs: Minimum number of pairs to create
            max_pairs: Maximum number of pairs to create
            
        Returns:
            List of contrastive pairs
        """
        if len(self.intervention_history) < 2:
            return []
        
        pairs = []
        
        # Create all possible pairs
        for i in range(len(self.intervention_history)):
            for j in range(i + 1, len(self.intervention_history)):
                intervention_a = self.intervention_history[i]
                intervention_b = self.intervention_history[j]
                outcome_a = self.outcome_history[i]
                outcome_b = self.outcome_history[j]
                
                # Compute similarity and contrast scores
                similarity = self._compute_intervention_similarity(
                    intervention_a, intervention_b
                )
                contrast = self._compute_outcome_contrast(
                    outcome_a, outcome_b
                )
                
                pair = ContrastivePair(
                    intervention_a=intervention_a,
                    intervention_b=intervention_b,
                    outcome_a=outcome_a,
                    outcome_b=outcome_b,
                    similarity_score=similarity,
                    contrast_score=contrast
                )
                
                pairs.append(pair)
        
        # Sort by informativeness (high contrast, varied similarity)
        pairs.sort(key=lambda p: p.contrast_score, reverse=True)
        
        # Select diverse pairs
        selected_pairs = self._select_diverse_pairs(pairs, min_pairs, max_pairs)
        
        return selected_pairs
    
    def _compute_intervention_similarity(self, 
                                       intervention_a: Dict,
                                       intervention_b: Dict) -> float:
        """Compute similarity between two interventions."""
        # Same node intervention
        if intervention_a['node'] == intervention_b['node']:
            value_diff = abs(intervention_a['value'] - intervention_b['value'])
            return 1.0 - min(value_diff / 4.0, 1.0)  # Normalize by max expected difference
        else:
            # Different nodes - consider structural similarity
            return 0.1  # Low baseline similarity
    
    def _compute_outcome_contrast(self, 
                                outcome_a: Dict,
                                outcome_b: Dict) -> float:
        """Compute contrast between two outcomes."""
        # Get affected nodes
        nodes_a = set(outcome_a.get('affected_nodes', []))
        nodes_b = set(outcome_b.get('affected_nodes', []))
        
        # Compute difference in affected nodes
        union = nodes_a | nodes_b
        intersection = nodes_a & nodes_b
        
        if len(union) == 0:
            return 0.0
        
        # Jaccard distance as contrast
        jaccard_similarity = len(intersection) / len(union)
        contrast = 1.0 - jaccard_similarity
        
        # Also consider magnitude differences
        total_effect_a = outcome_a.get('total_effect', 0)
        total_effect_b = outcome_b.get('total_effect', 0)
        
        magnitude_contrast = abs(total_effect_a - total_effect_b)
        
        # Combine contrasts
        total_contrast = 0.7 * contrast + 0.3 * min(magnitude_contrast, 1.0)
        
        return total_contrast
    
    def _select_diverse_pairs(self, 
                            pairs: List[ContrastivePair],
                            min_pairs: int,
                            max_pairs: int) -> List[ContrastivePair]:
        """Select diverse pairs for contrastive learning."""
        if len(pairs) <= max_pairs:
            return pairs
        
        # Select pairs with high contrast and diverse similarity levels
        selected = []
        
        # Always include highest contrast pairs
        selected.extend(pairs[:min_pairs])
        
        # Add diverse pairs
        remaining = pairs[min_pairs:]
        similarity_bins = [0.0, 0.3, 0.6, 1.0]
        
        for i in range(len(similarity_bins) - 1):
            bin_low = similarity_bins[i]
            bin_high = similarity_bins[i + 1]
            
            # Find pairs in this similarity bin
            bin_pairs = [
                p for p in remaining 
                if bin_low <= p.similarity_score < bin_high
            ]
            
            # Add top pairs from this bin
            n_from_bin = min(len(bin_pairs), (max_pairs - len(selected)) // 2)
            selected.extend(bin_pairs[:n_from_bin])
            
            if len(selected) >= max_pairs:
                break
        
        return selected[:max_pairs]
    
    def compute_contrastive_loss(self, pairs: List[ContrastivePair]) -> torch.Tensor:
        """
        Compute contrastive loss for learning causal relationship embeddings.
        
        Args:
            pairs: List of contrastive pairs
            
        Returns:
            Contrastive loss
        """
        if not pairs:
            return torch.tensor(0.0)
        
        total_loss = 0.0
        n_pairs = 0
        
        for pair in pairs:
            # Encode interventions and outcomes
            int_a_emb = self.encode_intervention(pair.intervention_a)
            int_b_emb = self.encode_intervention(pair.intervention_b)
            out_a_emb = self.encode_outcome(pair.outcome_a)
            out_b_emb = self.encode_outcome(pair.outcome_b)
            
            # Compute similarity in embedding space
            int_similarity = F.cosine_similarity(int_a_emb, int_b_emb, dim=0)
            out_similarity = F.cosine_similarity(out_a_emb, out_b_emb, dim=0)
            
            # Contrastive loss: similar interventions should have similar outcomes
            # if they involve the same causal mechanism
            target_similarity = torch.tensor(pair.similarity_score)
            
            # Loss for intervention-outcome consistency
            consistency_loss = F.mse_loss(
                int_similarity * out_similarity,
                target_similarity
            )
            
            # Loss for outcome discrimination
            contrast_target = torch.tensor(pair.contrast_score)
            discrimination_loss = F.mse_loss(
                1.0 - out_similarity,
                contrast_target
            )
            
            # Combine losses
            pair_loss = consistency_loss + 0.5 * discrimination_loss
            total_loss += pair_loss
            n_pairs += 1
        
        return total_loss / (n_pairs + 1e-8)
    
    def learn_causal_relationships(self, 
                                 pairs: List[ContrastivePair],
                                 n_epochs: int = 5) -> Dict:
        """
        Learn causal relationship importance through contrastive learning.
        
        Args:
            pairs: Contrastive pairs to learn from
            n_epochs: Number of training epochs
            
        Returns:
            Learning statistics
        """
        if not pairs:
            return {'loss': 0.0, 'n_pairs': 0}
        
        total_loss = 0.0
        
        for epoch in range(n_epochs):
            self.optimizer.zero_grad()
            
            # Compute contrastive loss
            loss = self.compute_contrastive_loss(pairs)
            
            # Backward pass
            if loss.requires_grad:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
        
        # Update relationship importance based on learned embeddings
        self._update_relationship_importance()
        
        # Record learning progress
        avg_loss = total_loss / n_epochs
        self.contrastive_losses.append(avg_loss)
        
        return {
            'loss': avg_loss,
            'n_pairs': len(pairs),
            'n_epochs': n_epochs
        }
    
    def _update_relationship_importance(self):
        """Update importance scores for causal relationships."""
        # Compute importance based on embedding magnitudes
        with torch.no_grad():
            # L2 norm of relationship embeddings indicates importance
            importance = torch.norm(self.causal_relation_embeddings, dim=2)
            
            # Normalize to [0, 1] range
            importance = importance / (importance.max() + 1e-8)
            
            # Exponential moving average update
            alpha = 0.1
            self.relationship_importance = (
                (1 - alpha) * self.relationship_importance + 
                alpha * importance
            )
    
    def get_most_important_relationships(self, top_k: int = 5) -> List[Tuple[int, int, float]]:
        """
        Get the most important causal relationships learned.
        
        Args:
            top_k: Number of top relationships to return
            
        Returns:
            List of (parent, child, importance) tuples
        """
        # Flatten importance matrix
        flat_importance = self.relationship_importance.flatten()
        
        # Get top indices
        top_indices = torch.topk(flat_importance, min(top_k, len(flat_importance)))[1]
        
        # Convert to (parent, child) pairs
        relationships = []
        for idx in top_indices:
            parent = idx // self.n_nodes
            child = idx % self.n_nodes
            importance = self.relationship_importance[parent, child].item()
            
            relationships.append((parent.item(), child.item(), importance))
        
        return relationships
    
    def update_from_intervention(self, 
                               intervention: Dict,
                               outcome: Dict):
        """
        Update contrastive learner with new intervention outcome.
        
        Args:
            intervention: Intervention that was performed
            outcome: Observed outcome
        """
        # Store intervention and outcome
        self.intervention_history.append(intervention)
        self.outcome_history.append(outcome)
        
        # Keep only recent history
        max_history = 100
        if len(self.intervention_history) > max_history:
            self.intervention_history = self.intervention_history[-max_history:]
            self.outcome_history = self.outcome_history[-max_history:]
        
        # Create and learn from contrastive pairs
        if len(self.intervention_history) >= 2:
            pairs = self.create_contrastive_pairs(min_pairs=2, max_pairs=10)
            if pairs:
                learning_stats = self.learn_causal_relationships(pairs, n_epochs=3)
                self.learning_progress.append(learning_stats)
    
    def get_learning_priorities(self) -> Dict[Tuple[int, int], float]:
        """
        Get learning priorities for structure learning based on contrastive learning.
        
        Returns:
            Dictionary mapping (parent, child) to priority weight
        """
        priorities = {}
        
        # Higher importance relationships get higher priority
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    importance = self.relationship_importance[i, j].item()
                    
                    # Convert importance to priority
                    # High importance = high priority for refinement
                    priority = importance
                    
                    # Boost priority for relationships we're uncertain about
                    if hasattr(self, 'relationship_uncertainty'):
                        uncertainty = self.relationship_uncertainty.get((i, j), 0.5)
                        priority *= (1 + uncertainty)
                    
                    priorities[(i, j)] = priority
        
        return priorities
    
    def get_contrastive_statistics(self) -> Dict:
        """Get statistics about contrastive learning."""
        return {
            'n_interventions': len(self.intervention_history),
            'n_contrastive_pairs': len(self.contrastive_pairs),
            'avg_contrastive_loss': np.mean(self.contrastive_losses) if self.contrastive_losses else 0.0,
            'most_important_relationships': self.get_most_important_relationships(top_k=5),
            'relationship_importance_entropy': self._compute_importance_entropy(),
            'learning_progress': len(self.learning_progress)
        }
    
    def _compute_importance_entropy(self) -> float:
        """Compute entropy of relationship importance distribution."""
        # Flatten importance matrix
        flat_importance = self.relationship_importance.flatten()
        
        # Add small epsilon and normalize
        probs = flat_importance + 1e-8
        probs = probs / probs.sum()
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs)).item()
        
        return entropy
    
    def visualize_learned_relationships(self) -> Dict:
        """
        Create visualization data for learned causal relationships.
        
        Returns:
            Dictionary with visualization data
        """
        # Get top relationships
        top_relationships = self.get_most_important_relationships(top_k=10)
        
        # Create adjacency matrix for visualization
        vis_adjacency = torch.zeros(self.n_nodes, self.n_nodes)
        
        for parent, child, importance in top_relationships:
            vis_adjacency[parent, child] = importance
        
        return {
            'learned_adjacency': vis_adjacency.tolist(),
            'top_relationships': top_relationships,
            'importance_matrix': self.relationship_importance.tolist(),
            'n_learned_relationships': len([r for r in top_relationships if r[2] > 0.1])
        }
    
    def get_relationship_embeddings(self) -> torch.Tensor:
        """Get learned relationship embeddings."""
        return self.causal_relation_embeddings.detach().clone()
    
    def compute_relationship_similarity(self, 
                                      parent_a: int, child_a: int,
                                      parent_b: int, child_b: int) -> float:
        """
        Compute similarity between two causal relationships.
        
        Args:
            parent_a, child_a: First relationship
            parent_b, child_b: Second relationship
            
        Returns:
            Similarity score
        """
        emb_a = self.causal_relation_embeddings[parent_a, child_a]
        emb_b = self.causal_relation_embeddings[parent_b, child_b]
        
        similarity = F.cosine_similarity(emb_a, emb_b, dim=0)
        
        return similarity.item() 