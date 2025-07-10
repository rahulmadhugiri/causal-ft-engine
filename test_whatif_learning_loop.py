#!/usr/bin/env python3
"""
Complete "What-If" Learning Loop Integration Test

This test validates the complete causal learning system with all components working together:
1. Counterfactual structure backpropagation
2. Counterfactual consistency loss 
3. Active intervention sampling
4. Intervention planning
5. Counterfactual contrastive learning

The test demonstrates the full counterfactual → structure → intervention → learning cycle.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from causal_unit_network import CausalUnitNetwork
from engine.active_intervention_sampling import ActiveInterventionSampler
from engine.intervention_planning import InterventionPlanner
from engine.counterfactual_contrastive_learning import CounterfactualContrastiveLearner
from engine.counterfactuals import CounterfactualSimulator
from experiments.utils import generate_synthetic_data

class WhatIfLearningLoop:
    """
    Complete "What-If" Learning Loop that integrates all components
    for intelligent causal discovery through strategic interventions.
    """
    
    def __init__(self, 
                 n_nodes: int = 3,
                 device: str = 'cpu',
                 random_seed: int = 42):
        """
        Initialize the complete what-if learning loop.
        
        Args:
            n_nodes: Number of nodes in causal graph
            device: Device to run on
            random_seed: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.device = device
        self.random_seed = random_seed
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize core causal system
        self.causal_network = CausalUnitNetwork(
            input_dim=n_nodes,
            hidden_dims=[8, 8],
            output_dim=n_nodes,  # FIXED: Should output values for all nodes
            activation='relu',
            enable_structure_learning=True,
            enable_gradient_surgery=True
        )
        
        # Initialize what-if learning components
        self.active_sampler = ActiveInterventionSampler(
            n_nodes=n_nodes,
            uncertainty_threshold=0.1,
            exploration_rate=0.3,
            temperature=1.0
        )
        
        self.intervention_planner = InterventionPlanner(
            n_nodes=n_nodes,
            max_worlds=8,
            min_world_probability=0.05,
            evidence_decay=0.9
        )
        
        self.contrastive_learner = CounterfactualContrastiveLearner(
            n_nodes=n_nodes,
            embedding_dim=32,
            temperature=0.1,
            learning_rate=0.01
        )
        
        # Initialize counterfactual simulator
        self.cf_simulator = CounterfactualSimulator(self.causal_network)
        
        # Learning history
        self.learning_history = []
        self.intervention_history = []
        self.structure_history = []
        self.performance_history = []
        
    def run_learning_cycle(self, 
                          x_train: torch.Tensor,
                          y_train: torch.Tensor,
                          true_adjacency: torch.Tensor,
                          n_cycles: int = 20,
                          interventions_per_cycle: int = 3) -> Dict:
        """
        Run the complete what-if learning cycle.
        
        Args:
            x_train: Training input data
            y_train: Training target data
            true_adjacency: True causal adjacency matrix
            n_cycles: Number of learning cycles
            interventions_per_cycle: Interventions per cycle
            
        Returns:
            Learning results dictionary
        """
        print(f"Starting What-If Learning Loop: {n_cycles} cycles")
        print("=" * 60)
        
        # Initial training
        self._train_initial_model(x_train, y_train)
        
        # Main learning loop
        for cycle in range(n_cycles):
            print(f"\n--- Cycle {cycle + 1}/{n_cycles} ---")
            
            # 1. Analyze current structure and uncertainty
            current_adjacency = self.causal_network.get_adjacency_matrix()
            adjacency_samples = self._collect_adjacency_samples(x_train, n_samples=10)
            
            # 2. Plan strategic interventions
            planned_interventions = self._plan_interventions(
                current_adjacency, 
                adjacency_samples,
                interventions_per_cycle
            )
            
            # 3. Execute interventions and collect outcomes
            intervention_outcomes = self._execute_interventions(
                planned_interventions,
                x_train,
                y_train
            )
            
            # 4. Update all learning components
            self._update_learning_components(
                planned_interventions,
                intervention_outcomes,
                current_adjacency
            )
            
            # 5. Refine causal structure
            self._refine_causal_structure(x_train, y_train)
            
            # 6. Evaluate progress
            cycle_metrics = self._evaluate_cycle_performance(
                true_adjacency,
                current_adjacency,
                intervention_outcomes
            )
            
            # Store results
            self.learning_history.append({
                'cycle': cycle + 1,
                'interventions': planned_interventions,
                'outcomes': intervention_outcomes,
                'metrics': cycle_metrics,
                'adjacency': current_adjacency.tolist()
            })
            
            print(f"Cycle {cycle + 1} Results:")
            print(f"  Structure F1: {cycle_metrics['structure_f1']:.3f}")
            print(f"  Intervention Effectiveness: {cycle_metrics['intervention_effectiveness']:.3f}")
            print(f"  Learning Progress: {cycle_metrics['learning_progress']:.3f}")
        
        # Final evaluation
        final_results = self._generate_final_results(true_adjacency)
        
        print("\n" + "=" * 60)
        print("What-If Learning Loop Complete!")
        print(f"Final Structure F1: {final_results['final_structure_f1']:.3f}")
        print(f"Total Interventions: {final_results['total_interventions']}")
        print(f"Learning Efficiency: {final_results['learning_efficiency']:.3f}")
        
        return final_results
    
    def _train_initial_model(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """Train initial causal model."""
        print("Training initial causal model...")
        
        # Simple training loop
        optimizer = torch.optim.Adam(self.causal_network.parameters(), lr=0.01)
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.causal_network(x_train)
            
            # Basic loss
            loss = nn.MSELoss()(predictions, y_train)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def _collect_adjacency_samples(self, x_train: torch.Tensor, n_samples: int) -> List[torch.Tensor]:
        """Collect adjacency matrix samples for uncertainty estimation."""
        samples = []
        
        for _ in range(n_samples):
            # Add noise to training data
            x_noisy = x_train + torch.randn_like(x_train) * 0.1
            
            # Forward pass to get adjacency
            self.causal_network(x_noisy)
            adjacency = self.causal_network.get_adjacency_matrix()
            samples.append(adjacency.clone())
        
        return samples
    
    def _plan_interventions(self, 
                          current_adjacency: torch.Tensor,
                          adjacency_samples: List[torch.Tensor],
                          n_interventions: int) -> List[Dict]:
        """Plan strategic interventions using all planning components."""
        planned_interventions = []
        
        # Use different planning strategies
        for i in range(n_interventions):
            if i % 3 == 0:
                # Active sampling based on uncertainty
                intervention = self.active_sampler.sample_intervention(
                    current_adjacency,
                    adjacency_samples,
                    intervention_budget=1
                )[0]
                intervention['strategy'] = 'active_sampling'
                
            elif i % 3 == 1:
                # Discriminative planning
                intervention = self.intervention_planner.plan_discriminative_intervention(
                    current_adjacency
                )
                intervention['strategy'] = 'discriminative_planning'
                
            else:
                # Contrastive learning priorities
                priorities = self.contrastive_learner.get_learning_priorities()
                
                # Choose intervention based on priorities
                if priorities:
                    # Find highest priority relationship
                    best_relation = max(priorities.items(), key=lambda x: x[1])
                    parent, child = best_relation[0]
                    
                    intervention = {
                        'node': parent,
                        'value': np.random.choice([-1.0, 1.0]),
                        'strategy': 'contrastive_priority',
                        'target_relation': (parent, child),
                        'priority': best_relation[1]
                    }
                else:
                    # Fallback to random
                    intervention = {
                        'node': np.random.randint(0, self.n_nodes),
                        'value': np.random.choice([-1.0, 1.0]),
                        'strategy': 'random_fallback'
                    }
            
            planned_interventions.append(intervention)
        
        return planned_interventions
    
    def _execute_interventions(self, 
                             interventions: List[Dict],
                             x_train: torch.Tensor,
                             y_train: torch.Tensor) -> List[Dict]:
        """Execute interventions and collect outcomes."""
        outcomes = []
        
        for intervention in interventions:
            # Create intervention mask and values
            intervention_mask = torch.zeros(self.n_nodes)
            intervention_values = torch.zeros(self.n_nodes)
            
            intervention_mask[intervention['node']] = 1.0
            intervention_values[intervention['node']] = intervention['value']
            
            # Execute intervention
            effect, factual_outcome, counterfactual_outcome = self.cf_simulator.compute_counterfactual_effect(
                x_train,
                intervention_mask,
                intervention_values
            )
            
            # Measure outcome
            outcome = {
                'effect_magnitude': torch.mean(torch.abs(effect)).item(),
                'total_effect': torch.sum(torch.abs(effect)).item(),
                'affected_nodes': [i for i in range(self.n_nodes) if torch.abs(effect[:, i]).mean() > 0.01],
                'node_effects': {i: torch.abs(effect[:, i]).mean().item() for i in range(self.n_nodes)}
            }
            
            outcomes.append(outcome)
        
        return outcomes
    
    def _update_learning_components(self, 
                                  interventions: List[Dict],
                                  outcomes: List[Dict],
                                  current_adjacency: torch.Tensor):
        """Update all learning components with new evidence."""
        
        for intervention, outcome in zip(interventions, outcomes):
            # Update active sampler
            self.active_sampler.update_from_intervention(
                intervention,
                outcome,
                current_adjacency,
                current_adjacency  # Simplified - would track actual changes
            )
            
            # Update intervention planner
            self.intervention_planner.update_world_beliefs(
                intervention,
                outcome
            )
            
            # Update contrastive learner
            self.contrastive_learner.update_from_intervention(
                intervention,
                outcome
            )
    
    def _refine_causal_structure(self, x_train: torch.Tensor, y_train: torch.Tensor):
        """Refine causal structure based on accumulated evidence."""
        # Get learning priorities from contrastive learner
        priorities = self.contrastive_learner.get_learning_priorities()
        
        # Fine-tune network with priority weighting
        optimizer = torch.optim.Adam(self.causal_network.parameters(), lr=0.005)
        
        for epoch in range(10):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.causal_network(x_train)
            
            # Basic prediction loss
            pred_loss = nn.MSELoss()(predictions, y_train)
            
            # Structure refinement loss (simplified)
            structure_loss = self._compute_structure_refinement_loss(priorities)
            
            # Total loss
            total_loss = pred_loss + 0.1 * structure_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
    
    def _compute_structure_refinement_loss(self, priorities: Dict) -> torch.Tensor:
        """Compute structure refinement loss based on learned priorities."""
        current_adjacency = self.causal_network.get_adjacency_matrix()
        
        # L2 regularization weighted by inverse priorities
        # High priority relationships get less regularization
        refinement_loss = 0.0
        
        for (parent, child), priority in priorities.items():
            if parent < self.n_nodes and child < self.n_nodes:
                edge_strength = current_adjacency[parent, child]
                
                # Inverse priority weighting
                weight = 1.0 / (priority + 1e-8)
                
                refinement_loss += weight * (edge_strength ** 2)
        
        return torch.tensor(refinement_loss, requires_grad=True)
    
    def _evaluate_cycle_performance(self, 
                                  true_adjacency: torch.Tensor,
                                  current_adjacency: torch.Tensor,
                                  intervention_outcomes: List[Dict]) -> Dict:
        """Evaluate performance of current cycle."""
        # Structure learning metrics
        structure_f1 = self._compute_structure_f1(true_adjacency, current_adjacency)
        
        # Intervention effectiveness
        avg_effect = np.mean([outcome['effect_magnitude'] for outcome in intervention_outcomes])
        intervention_effectiveness = min(avg_effect, 1.0)
        
        # Learning progress (decrease in uncertainty)
        learning_progress = self._compute_learning_progress()
        
        return {
            'structure_f1': structure_f1,
            'intervention_effectiveness': intervention_effectiveness,
            'learning_progress': learning_progress,
            'n_interventions': len(intervention_outcomes)
        }
    
    def _compute_structure_f1(self, true_adj: torch.Tensor, pred_adj: torch.Tensor) -> float:
        """Compute F1 score for structure learning."""
        # Convert to binary
        true_binary = (true_adj > 0.5).float().flatten()
        pred_binary = (pred_adj > 0.5).float().flatten()
        
        # Compute F1
        tp = torch.sum(true_binary * pred_binary)
        fp = torch.sum((1 - true_binary) * pred_binary)
        fn = torch.sum(true_binary * (1 - pred_binary))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return f1.item()
    
    def _compute_learning_progress(self) -> float:
        """Compute learning progress based on uncertainty reduction."""
        # Get current uncertainty from active sampler
        current_uncertainty = self.active_sampler.adjacency_uncertainty.mean().item()
        
        # Progress is reduction from initial uncertainty
        initial_uncertainty = 0.5  # Assume initial uncertainty
        progress = (initial_uncertainty - current_uncertainty) / initial_uncertainty
        
        return max(0.0, min(1.0, progress))
    
    def _generate_final_results(self, true_adjacency: torch.Tensor) -> Dict:
        """Generate final comprehensive results."""
        final_adjacency = self.causal_network.get_adjacency_matrix()
        
        # Structure learning performance
        final_structure_f1 = self._compute_structure_f1(true_adjacency, final_adjacency)
        
        # Total interventions
        total_interventions = sum(len(cycle['interventions']) for cycle in self.learning_history)
        
        # Learning efficiency (F1 per intervention)
        learning_efficiency = final_structure_f1 / (total_interventions + 1e-8)
        
        # Component statistics
        sampling_stats = self.active_sampler.get_sampling_statistics()
        planning_stats = self.intervention_planner.get_planning_statistics()
        contrastive_stats = self.contrastive_learner.get_contrastive_statistics()
        
        return {
            'final_structure_f1': final_structure_f1,
            'total_interventions': total_interventions,
            'learning_efficiency': learning_efficiency,
            'final_adjacency': final_adjacency.tolist(),
            'true_adjacency': true_adjacency.tolist(),
            'learning_history': self.learning_history,
            'component_stats': {
                'active_sampling': sampling_stats,
                'intervention_planning': planning_stats,
                'contrastive_learning': contrastive_stats
            }
        }


def test_complete_whatif_learning_loop():
    """Test the complete what-if learning loop."""
    print("Testing Complete What-If Learning Loop")
    print("=" * 50)
    
    # Generate synthetic data
    n_samples = 200
    n_nodes = 3
    
    x_data, y_data, true_adjacency = generate_synthetic_data(
        n_samples=n_samples,
        n_nodes=n_nodes,
        graph_type='chain',
        noise_level=0.3
    )
    
    x_train = torch.tensor(x_data, dtype=torch.float32)
    y_train = torch.tensor(y_data, dtype=torch.float32)
    true_adjacency = torch.tensor(true_adjacency, dtype=torch.float32)
    
    print(f"Generated data: {n_samples} samples, {n_nodes} nodes")
    print(f"True adjacency matrix:\n{true_adjacency}")
    
    # Initialize what-if learning loop
    whatif_loop = WhatIfLearningLoop(
        n_nodes=n_nodes,
        device='cpu',
        random_seed=42
    )
    
    # Run learning loop
    results = whatif_loop.run_learning_cycle(
        x_train=x_train,
        y_train=y_train,
        true_adjacency=true_adjacency,
        n_cycles=15,
        interventions_per_cycle=2
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"whatif_learning_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Validation checks
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    
    # Check if all components are working
    component_stats = results['component_stats']
    
    print("✓ Active Sampling Statistics:")
    print(f"  Total interventions: {component_stats['active_sampling']['total_interventions']}")
    print(f"  Avg information gain: {component_stats['active_sampling']['avg_information_gain']:.3f}")
    
    print("✓ Intervention Planning Statistics:")
    print(f"  Belief entropy: {component_stats['intervention_planning']['belief_entropy']:.3f}")
    print(f"  Top world confidence: {component_stats['intervention_planning']['top_world_confidence']:.3f}")
    
    print("✓ Contrastive Learning Statistics:")
    print(f"  Learning progress: {component_stats['contrastive_learning']['learning_progress']}")
    print(f"  Top relationships: {component_stats['contrastive_learning']['most_important_relationships']}")
    
    # Overall performance
    print(f"\n✓ Overall Performance:")
    print(f"  Final Structure F1: {results['final_structure_f1']:.3f}")
    print(f"  Learning Efficiency: {results['learning_efficiency']:.3f}")
    print(f"  Total Interventions: {results['total_interventions']}")
    
    # Success criteria
    success_criteria = {
        'structure_f1_threshold': 0.5,
        'learning_efficiency_threshold': 0.02,
        'component_integration': True
    }
    
    passed_tests = []
    
    if results['final_structure_f1'] >= success_criteria['structure_f1_threshold']:
        passed_tests.append("Structure Learning Performance")
    
    if results['learning_efficiency'] >= success_criteria['learning_efficiency_threshold']:
        passed_tests.append("Learning Efficiency")
    
    if all(stats['total_interventions'] > 0 for stats in component_stats.values()):
        passed_tests.append("Component Integration")
    
    print(f"\n✓ Passed Tests: {len(passed_tests)}/3")
    for test in passed_tests:
        print(f"  ✓ {test}")
    
    if len(passed_tests) == 3:
        print("\n🎉 ALL TESTS PASSED! What-If Learning Loop is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check component implementations.")
    
    return results


if __name__ == "__main__":
    # Run the complete test
    results = test_complete_whatif_learning_loop()
    
    print("\nWhat-If Learning Loop Integration Test Complete!") 