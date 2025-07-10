import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.stats import entropy
import math

class ActiveInterventionSampler:
    """
    Active intervention sampling system that strategically chooses where to intervene
    based on uncertainty and expected information gain.
    
    Key Features:
    1. Uncertainty estimation for causal structure
    2. Information gain calculation for potential interventions
    3. Strategic intervention selection
    4. Adaptive sampling based on learning progress
    """
    
    def __init__(self, 
                 n_nodes: int,
                 uncertainty_threshold: float = 0.1,
                 exploration_rate: float = 0.2,
                 temperature: float = 1.0):
        """
        Initialize active intervention sampler.
        
        Args:
            n_nodes: Number of nodes in the causal graph
            uncertainty_threshold: Threshold for high uncertainty
            exploration_rate: Rate of random exploration vs exploitation
            temperature: Temperature for softmax sampling
        """
        self.n_nodes = n_nodes
        self.uncertainty_threshold = uncertainty_threshold
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        
        # Track intervention history and outcomes
        self.intervention_history = []
        self.outcome_history = []
        self.uncertainty_history = []
        
        # Track adjacency matrix uncertainty
        self.adjacency_uncertainty = torch.zeros(n_nodes, n_nodes)
        self.adjacency_samples = []
        
    def estimate_structure_uncertainty(self, 
                                     adjacency_samples: List[torch.Tensor]) -> torch.Tensor:
        """
        Estimate uncertainty in causal structure using adjacency matrix samples.
        
        Args:
            adjacency_samples: List of adjacency matrix samples
            
        Returns:
            Uncertainty matrix (higher values = more uncertain)
        """
        if len(adjacency_samples) < 2:
            return torch.ones(self.n_nodes, self.n_nodes) * 0.5
        
        # Stack samples and compute statistics
        samples = torch.stack(adjacency_samples)
        
        # Compute variance as uncertainty measure
        uncertainty = torch.var(samples, dim=0)
        
        # Normalize to [0, 1] range
        uncertainty = uncertainty / (uncertainty.max() + 1e-8)
        
        return uncertainty
    
    def compute_information_gain(self, 
                               intervention_node: int,
                               intervention_value: float,
                               current_adjacency: torch.Tensor,
                               uncertainty: torch.Tensor) -> float:
        """
        Compute expected information gain from an intervention.
        
        Args:
            intervention_node: Node to intervene on
            intervention_value: Value to set
            current_adjacency: Current adjacency matrix
            uncertainty: Current uncertainty matrix
            
        Returns:
            Expected information gain
        """
        # Information gain is higher for:
        # 1. High uncertainty nodes (we learn more)
        # 2. Nodes with many potential children (affects more structure)
        # 3. Nodes we haven't intervened on recently
        
        # Uncertainty-based gain
        uncertainty_gain = uncertainty[intervention_node, :].mean()
        
        # Structural importance (potential children)
        structural_gain = current_adjacency[intervention_node, :].sum()
        
        # Novelty gain (less explored interventions)
        novelty_gain = self._compute_novelty_gain(intervention_node, intervention_value)
        
        # Expected change in predictions
        prediction_gain = self._compute_prediction_gain(intervention_node, intervention_value)
        
        # Combine gains
        total_gain = (
            0.4 * uncertainty_gain +
            0.3 * structural_gain +
            0.2 * novelty_gain +
            0.1 * prediction_gain
        )
        
        return total_gain.item()
    
    def _compute_novelty_gain(self, node: int, value: float) -> float:
        """Compute novelty gain for intervention."""
        if not self.intervention_history:
            return 1.0
        
        # Count recent interventions on this node
        recent_interventions = [
            h for h in self.intervention_history[-10:]  # Last 10 interventions
            if h['node'] == node
        ]
        
        # Higher gain for less explored nodes
        novelty = 1.0 / (len(recent_interventions) + 1)
        
        # Consider value similarity
        if recent_interventions:
            value_distances = [
                abs(value - h['value']) for h in recent_interventions
            ]
            avg_distance = np.mean(value_distances)
            novelty *= (1.0 + avg_distance)
        
        return novelty
    
    def _compute_prediction_gain(self, node: int, value: float) -> float:
        """Compute expected prediction change from intervention."""
        if not self.outcome_history:
            return 0.5
        
        # Estimate how much this intervention might change predictions
        # Based on historical outcomes
        similar_interventions = [
            h for h in self.outcome_history[-20:]  # Last 20 outcomes
            if h['node'] == node
        ]
        
        if not similar_interventions:
            return 0.5
        
        # Compute average effect size
        effects = [h['effect_size'] for h in similar_interventions]
        avg_effect = np.mean(effects)
        
        return min(avg_effect, 1.0)
    
    def sample_intervention(self, 
                          current_adjacency: torch.Tensor,
                          adjacency_samples: List[torch.Tensor],
                          available_nodes: Optional[List[int]] = None,
                          intervention_budget: int = 1) -> List[Dict]:
        """
        Sample strategic interventions based on information gain.
        
        Args:
            current_adjacency: Current adjacency matrix
            adjacency_samples: Recent adjacency samples for uncertainty
            available_nodes: Nodes available for intervention
            intervention_budget: Number of interventions to sample
            
        Returns:
            List of intervention dictionaries
        """
        if available_nodes is None:
            available_nodes = list(range(self.n_nodes))
        
        # Estimate uncertainty
        uncertainty = self.estimate_structure_uncertainty(adjacency_samples)
        self.adjacency_uncertainty = uncertainty
        
        # Compute information gain for all possible interventions
        intervention_gains = []
        
        for node in available_nodes:
            # Try different intervention values
            for value in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                gain = self.compute_information_gain(
                    node, value, current_adjacency, uncertainty
                )
                intervention_gains.append({
                    'node': node,
                    'value': value,
                    'gain': gain,
                    'uncertainty': uncertainty[node, :].mean().item()
                })
        
        # Sort by information gain
        intervention_gains.sort(key=lambda x: x['gain'], reverse=True)
        
        # Sample interventions with temperature-based selection
        selected_interventions = []
        
        for _ in range(intervention_budget):
            if np.random.random() < self.exploration_rate:
                # Random exploration
                intervention = np.random.choice(intervention_gains)
            else:
                # Exploitation with temperature
                gains = np.array([ig['gain'] for ig in intervention_gains])
                
                # Apply temperature
                probabilities = torch.softmax(
                    torch.tensor(gains) / self.temperature, dim=0
                ).numpy()
                
                # Sample based on probabilities
                idx = np.random.choice(len(intervention_gains), p=probabilities)
                intervention = intervention_gains[idx]
            
            selected_interventions.append({
                'node': intervention['node'],
                'value': intervention['value'],
                'expected_gain': intervention['gain'],
                'uncertainty': intervention['uncertainty']
            })
            
            # Remove selected intervention to avoid duplicates
            intervention_gains = [
                ig for ig in intervention_gains 
                if not (ig['node'] == intervention['node'] and 
                       abs(ig['value'] - intervention['value']) < 1e-6)
            ]
        
        return selected_interventions
    
    def update_from_intervention(self, 
                               intervention: Dict,
                               outcome: Dict,
                               adjacency_before: torch.Tensor,
                               adjacency_after: torch.Tensor):
        """
        Update sampler based on intervention outcome.
        
        Args:
            intervention: Intervention that was performed
            outcome: Outcome of the intervention
            adjacency_before: Adjacency matrix before intervention
            adjacency_after: Adjacency matrix after intervention
        """
        # Record intervention
        self.intervention_history.append({
            'node': intervention['node'],
            'value': intervention['value'],
            'expected_gain': intervention.get('expected_gain', 0.0)
        })
        
        # Record outcome
        effect_size = torch.norm(adjacency_after - adjacency_before).item()
        
        self.outcome_history.append({
            'node': intervention['node'],
            'value': intervention['value'],
            'effect_size': effect_size,
            'prediction_change': outcome.get('prediction_change', 0.0),
            'structure_change': outcome.get('structure_change', 0.0)
        })
        
        # Update uncertainty tracking
        self.adjacency_samples.append(adjacency_after.clone())
        
        # Keep only recent samples
        if len(self.adjacency_samples) > 50:
            self.adjacency_samples = self.adjacency_samples[-50:]
    
    def get_sampling_statistics(self) -> Dict:
        """Get statistics about sampling performance."""
        if not self.intervention_history or not self.outcome_history:
            return {
                'total_interventions': 0,
                'avg_information_gain': 0.0,
                'uncertainty_reduction': 0.0,
                'exploration_rate': self.exploration_rate
            }
        
        # Compute statistics
        total_interventions = len(self.intervention_history)
        
        # Average realized information gain
        if len(self.outcome_history) >= 2:
            recent_effects = [h['effect_size'] for h in self.outcome_history[-10:]]
            avg_gain = np.mean(recent_effects)
        else:
            avg_gain = 0.0
        
        # Uncertainty reduction over time
        if len(self.uncertainty_history) >= 2:
            initial_uncertainty = self.uncertainty_history[0]
            current_uncertainty = self.uncertainty_history[-1]
            uncertainty_reduction = initial_uncertainty - current_uncertainty
        else:
            uncertainty_reduction = 0.0
        
        return {
            'total_interventions': total_interventions,
            'avg_information_gain': avg_gain,
            'uncertainty_reduction': uncertainty_reduction,
            'exploration_rate': self.exploration_rate,
            'current_uncertainty': self.adjacency_uncertainty.mean().item()
        }
    
    def adapt_sampling_strategy(self, performance_metrics: Dict):
        """
        Adapt sampling strategy based on performance.
        
        Args:
            performance_metrics: Recent performance metrics
        """
        # Increase exploration if not learning much
        if performance_metrics.get('structure_change', 0) < 0.01:
            self.exploration_rate = min(0.5, self.exploration_rate + 0.1)
        else:
            self.exploration_rate = max(0.1, self.exploration_rate - 0.05)
        
        # Adjust temperature based on uncertainty
        avg_uncertainty = self.adjacency_uncertainty.mean().item()
        if avg_uncertainty > 0.3:
            self.temperature = 2.0  # More exploration
        elif avg_uncertainty < 0.1:
            self.temperature = 0.5  # More exploitation
        else:
            self.temperature = 1.0  # Balanced 