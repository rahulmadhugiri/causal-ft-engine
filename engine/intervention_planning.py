import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import math
from collections import defaultdict

@dataclass
class World:
    """Represents a possible causal world/structure."""
    adjacency: torch.Tensor
    probability: float
    evidence_score: float
    intervention_history: List[Dict]
    prediction_accuracy: float
    
    def __post_init__(self):
        self.id = hash(tuple(self.adjacency.flatten().tolist()))

class InterventionPlanner:
    """
    Intervention planning system that maintains beliefs about different possible causal worlds
    and plans interventions to discriminate between them.
    
    Key Features:
    1. Maintains multiple world hypotheses
    2. Scores worlds based on intervention outcomes
    3. Plans interventions to maximize discrimination
    4. Updates beliefs based on evidence
    """
    
    def __init__(self, 
                 n_nodes: int,
                 max_worlds: int = 10,
                 min_world_probability: float = 0.01,
                 evidence_decay: float = 0.95,
                 discrimination_threshold: float = 0.1):
        """
        Initialize intervention planner.
        
        Args:
            n_nodes: Number of nodes in causal graph
            max_worlds: Maximum number of world hypotheses to maintain
            min_world_probability: Minimum probability to keep a world
            evidence_decay: Decay rate for old evidence
            discrimination_threshold: Threshold for world discrimination
        """
        self.n_nodes = n_nodes
        self.max_worlds = max_worlds
        self.min_world_probability = min_world_probability
        self.evidence_decay = evidence_decay
        self.discrimination_threshold = discrimination_threshold
        
        # World hypotheses and beliefs
        self.worlds = []
        self.world_beliefs = {}
        self.intervention_plans = []
        
        # Evidence tracking
        self.evidence_history = []
        self.discrimination_history = []
        
        # Initialize with uniform prior over structures
        self._initialize_world_hypotheses()
    
    def _initialize_world_hypotheses(self):
        """Initialize world hypotheses with diverse causal structures."""
        # Start with a few key structural hypotheses
        hypotheses = [
            self._create_chain_structure(),
            self._create_fork_structure(),
            self._create_collider_structure(),
            self._create_empty_structure(),
            self._create_dense_structure()
        ]
        
        # Add some random structures
        for _ in range(self.max_worlds - len(hypotheses)):
            hypotheses.append(self._create_random_structure())
        
        # Create World objects
        for i, adj in enumerate(hypotheses):
            world = World(
                adjacency=adj,
                probability=1.0 / len(hypotheses),
                evidence_score=0.0,
                intervention_history=[],
                prediction_accuracy=0.5
            )
            self.worlds.append(world)
            self.world_beliefs[world.id] = world.probability
    
    def _create_chain_structure(self) -> torch.Tensor:
        """Create a chain causal structure."""
        adj = torch.zeros(self.n_nodes, self.n_nodes)
        for i in range(self.n_nodes - 1):
            adj[i, i + 1] = 1.0
        return adj
    
    def _create_fork_structure(self) -> torch.Tensor:
        """Create a fork causal structure."""
        adj = torch.zeros(self.n_nodes, self.n_nodes)
        if self.n_nodes >= 3:
            adj[0, 1] = 1.0
            adj[0, 2] = 1.0
        return adj
    
    def _create_collider_structure(self) -> torch.Tensor:
        """Create a collider causal structure."""
        adj = torch.zeros(self.n_nodes, self.n_nodes)
        if self.n_nodes >= 3:
            adj[0, 2] = 1.0
            adj[1, 2] = 1.0
        return adj
    
    def _create_empty_structure(self) -> torch.Tensor:
        """Create an empty causal structure."""
        return torch.zeros(self.n_nodes, self.n_nodes)
    
    def _create_dense_structure(self) -> torch.Tensor:
        """Create a dense causal structure."""
        adj = torch.rand(self.n_nodes, self.n_nodes)
        adj = (adj > 0.5).float()
        # Remove self-loops
        adj.fill_diagonal_(0)
        return adj
    
    def _create_random_structure(self) -> torch.Tensor:
        """Create a random causal structure."""
        adj = torch.rand(self.n_nodes, self.n_nodes)
        adj = (adj > 0.7).float()  # Sparse structure
        adj.fill_diagonal_(0)
        return adj
    
    def score_world_given_evidence(self, 
                                  world: World,
                                  intervention: Dict,
                                  observed_outcome: Dict) -> float:
        """
        Score a world hypothesis given new evidence.
        
        Args:
            world: World hypothesis to score
            intervention: Intervention that was performed
            observed_outcome: Observed outcome
            
        Returns:
            Evidence score for this world
        """
        # Predict what this world would expect from the intervention
        predicted_outcome = self._predict_intervention_outcome(
            world, intervention
        )
        
        # Compare prediction to observation
        prediction_error = self._compute_prediction_error(
            predicted_outcome, observed_outcome
        )
        
        # Convert error to evidence score (lower error = higher score)
        evidence_score = math.exp(-prediction_error)
        
        # Consider structural consistency
        structure_score = self._compute_structure_consistency(
            world, intervention, observed_outcome
        )
        
        # Combine scores
        total_score = 0.7 * evidence_score + 0.3 * structure_score
        
        return total_score
    
    def _predict_intervention_outcome(self, 
                                    world: World,
                                    intervention: Dict) -> Dict:
        """Predict intervention outcome for a given world."""
        # Simple prediction based on causal structure
        intervention_node = intervention['node']
        intervention_value = intervention['value']
        
        # Find which nodes should be affected
        affected_nodes = []
        for i in range(self.n_nodes):
            if world.adjacency[intervention_node, i] > 0.5:
                affected_nodes.append(i)
        
        # Predict effect sizes
        predicted_effects = {}
        for node in affected_nodes:
            # Effect size depends on edge strength
            effect_size = world.adjacency[intervention_node, node].item()
            predicted_effects[node] = effect_size * abs(intervention_value)
        
        return {
            'affected_nodes': affected_nodes,
            'predicted_effects': predicted_effects,
            'total_effect': sum(predicted_effects.values())
        }
    
    def _compute_prediction_error(self, 
                                predicted: Dict,
                                observed: Dict) -> float:
        """Compute prediction error between predicted and observed outcomes."""
        # Compare affected nodes
        pred_nodes = set(predicted.get('affected_nodes', []))
        obs_nodes = set(observed.get('affected_nodes', []))
        
        # Jaccard similarity for affected nodes
        intersection = len(pred_nodes & obs_nodes)
        union = len(pred_nodes | obs_nodes)
        node_similarity = intersection / (union + 1e-8)
        
        # Compare effect magnitudes
        pred_total = predicted.get('total_effect', 0)
        obs_total = observed.get('total_effect', 0)
        
        effect_error = abs(pred_total - obs_total)
        
        # Combine errors
        total_error = (1 - node_similarity) + effect_error
        
        return total_error
    
    def _compute_structure_consistency(self, 
                                     world: World,
                                     intervention: Dict,
                                     outcome: Dict) -> float:
        """Compute how consistent the outcome is with the world's structure."""
        # Check if intervention effects follow causal paths
        intervention_node = intervention['node']
        
        consistency_score = 0.0
        total_checks = 0
        
        for target_node in range(self.n_nodes):
            if target_node == intervention_node:
                continue
                
            # Check if there's a causal path
            has_path = world.adjacency[intervention_node, target_node] > 0.5
            
            # Check if effect was observed
            has_effect = target_node in outcome.get('affected_nodes', [])
            
            # Consistency: path exists <-> effect observed
            if has_path == has_effect:
                consistency_score += 1.0
            
            total_checks += 1
        
        return consistency_score / (total_checks + 1e-8)
    
    def update_world_beliefs(self, 
                           intervention: Dict,
                           outcome: Dict):
        """
        Update beliefs about world hypotheses based on new evidence.
        
        Args:
            intervention: Intervention that was performed
            outcome: Observed outcome
        """
        # Score all worlds
        new_scores = {}
        for world in self.worlds:
            score = self.score_world_given_evidence(world, intervention, outcome)
            new_scores[world.id] = score
            
            # Update world's evidence score
            world.evidence_score = (
                world.evidence_score * self.evidence_decay + 
                score * (1 - self.evidence_decay)
            )
            
            # Update intervention history
            world.intervention_history.append({
                'intervention': intervention,
                'outcome': outcome,
                'score': score
            })
        
        # Update probabilities using Bayes' rule
        total_likelihood = sum(
            self.world_beliefs[world_id] * score 
            for world_id, score in new_scores.items()
        )
        
        if total_likelihood > 1e-8:
            for world_id, score in new_scores.items():
                prior = self.world_beliefs[world_id]
                likelihood = score
                posterior = (prior * likelihood) / total_likelihood
                self.world_beliefs[world_id] = posterior
                
                # Update world probability
                for world in self.worlds:
                    if world.id == world_id:
                        world.probability = posterior
                        break
        
        # Prune low-probability worlds
        self._prune_worlds()
        
        # Add new world hypotheses if needed
        self._add_new_worlds_if_needed()
    
    def _prune_worlds(self):
        """Remove worlds with very low probability."""
        # Keep worlds above threshold
        surviving_worlds = [
            world for world in self.worlds
            if world.probability >= self.min_world_probability
        ]
        
        # Always keep at least 2 worlds
        if len(surviving_worlds) < 2:
            # Keep top 2 worlds
            sorted_worlds = sorted(
                self.worlds, 
                key=lambda w: w.probability, 
                reverse=True
            )
            surviving_worlds = sorted_worlds[:2]
        
        # Update world list and beliefs
        self.worlds = surviving_worlds
        self.world_beliefs = {
            world.id: world.probability for world in self.worlds
        }
    
    def _add_new_worlds_if_needed(self):
        """Add new world hypotheses if current ones are too similar."""
        if len(self.worlds) >= self.max_worlds:
            return
        
        # Check diversity of current worlds
        diversity = self._compute_world_diversity()
        
        if diversity < 0.5:  # Low diversity
            # Add new random worlds
            n_new = min(3, self.max_worlds - len(self.worlds))
            for _ in range(n_new):
                new_adj = self._create_random_structure()
                new_world = World(
                    adjacency=new_adj,
                    probability=0.1 / n_new,
                    evidence_score=0.0,
                    intervention_history=[],
                    prediction_accuracy=0.5
                )
                
                self.worlds.append(new_world)
                self.world_beliefs[new_world.id] = new_world.probability
            
            # Renormalize probabilities
            total_prob = sum(self.world_beliefs.values())
            for world_id in self.world_beliefs:
                self.world_beliefs[world_id] /= total_prob
    
    def _compute_world_diversity(self) -> float:
        """Compute diversity of current world hypotheses."""
        if len(self.worlds) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = []
        for i in range(len(self.worlds)):
            for j in range(i + 1, len(self.worlds)):
                dist = torch.norm(
                    self.worlds[i].adjacency - self.worlds[j].adjacency
                ).item()
                distances.append(dist)
        
        # Return average distance
        return np.mean(distances)
    
    def plan_discriminative_intervention(self, 
                                       current_adjacency: torch.Tensor,
                                       available_nodes: Optional[List[int]] = None) -> Dict:
        """
        Plan an intervention that maximally discriminates between world hypotheses.
        
        Args:
            current_adjacency: Current best estimate of adjacency
            available_nodes: Nodes available for intervention
            
        Returns:
            Planned intervention
        """
        if available_nodes is None:
            available_nodes = list(range(self.n_nodes))
        
        # Generate candidate interventions
        candidates = []
        for node in available_nodes:
            for value in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                candidates.append({'node': node, 'value': value})
        
        # Score each candidate by discrimination power
        best_intervention = None
        best_score = -float('inf')
        
        for candidate in candidates:
            score = self._compute_discrimination_score(candidate)
            if score > best_score:
                best_score = score
                best_intervention = candidate
        
        # Add discrimination metadata
        if best_intervention:
            best_intervention['discrimination_score'] = best_score
            best_intervention['planning_method'] = 'discriminative'
        
        return best_intervention
    
    def _compute_discrimination_score(self, intervention: Dict) -> float:
        """Compute how well an intervention discriminates between worlds."""
        # Predict outcomes for all worlds
        predictions = {}
        for world in self.worlds:
            pred = self._predict_intervention_outcome(world, intervention)
            predictions[world.id] = pred
        
        # Compute discrimination power
        # Higher variance in predictions = better discrimination
        total_effects = [
            pred['total_effect'] for pred in predictions.values()
        ]
        
        if len(total_effects) < 2:
            return 0.0
        
        # Weighted variance (weight by world probability)
        weights = [
            world.probability for world in self.worlds
        ]
        
        weighted_mean = np.average(total_effects, weights=weights)
        weighted_var = np.average(
            (total_effects - weighted_mean)**2, 
            weights=weights
        )
        
        return weighted_var
    
    def get_world_summary(self) -> Dict:
        """Get summary of current world beliefs."""
        # Sort worlds by probability
        sorted_worlds = sorted(
            self.worlds, 
            key=lambda w: w.probability, 
            reverse=True
        )
        
        summary = {
            'n_worlds': len(self.worlds),
            'entropy': self._compute_belief_entropy(),
            'top_world_prob': sorted_worlds[0].probability if sorted_worlds else 0.0,
            'avg_evidence_score': np.mean([w.evidence_score for w in self.worlds]),
            'worlds': []
        }
        
        # Add details for top worlds
        for world in sorted_worlds[:5]:  # Top 5 worlds
            summary['worlds'].append({
                'id': world.id,
                'probability': world.probability,
                'evidence_score': world.evidence_score,
                'n_interventions': len(world.intervention_history),
                'adjacency': world.adjacency.tolist()
            })
        
        return summary
    
    def _compute_belief_entropy(self) -> float:
        """Compute entropy of belief distribution."""
        probs = [world.probability for world in self.worlds]
        probs = np.array(probs)
        probs = probs[probs > 1e-8]  # Remove zero probabilities
        
        if len(probs) == 0:
            return 0.0
        
        return -np.sum(probs * np.log(probs))
    
    def get_planning_statistics(self) -> Dict:
        """Get statistics about planning performance."""
        return {
            'total_interventions': len(self.evidence_history),
            'belief_entropy': self._compute_belief_entropy(),
            'world_diversity': self._compute_world_diversity(),
            'discrimination_power': np.mean(self.discrimination_history) if self.discrimination_history else 0.0,
            'top_world_confidence': max(w.probability for w in self.worlds) if self.worlds else 0.0
        } 