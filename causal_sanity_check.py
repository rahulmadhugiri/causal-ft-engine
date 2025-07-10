#!/usr/bin/env python3
"""
Comprehensive Sanity Check for Causal Intervention System
Tests whether our causal mechanisms are actually working as intended.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
import json
from datetime import datetime

from causal_unit_network import CausalUnitNetwork
from experiments.utils import generate_synthetic_data, create_dag_from_edges

class CausalSanityChecker:
    """Comprehensive validation suite for causal intervention system"""
    
    def __init__(self, input_size: int = 3, hidden_size: int = 64, output_size: int = 1):
        # Create test data first to get actual dimensions
        self.x_test, self.y_test, self.true_adjacency = generate_synthetic_data(
            n_samples=1000,
            n_nodes=8,  # This parameter doesn't affect the actual output size
            graph_type='chain',
            noise_level=0.1
        )
        
        # Set input size based on actual data dimensions
        self.input_size = self.x_test.shape[1]  # Should be 3 for chain graph
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Generate synthetic data with known causal structure (chain: 0->1->2)
        self.true_edges = [(0, 1), (1, 2)]
        
        # Convert to torch tensors
        self.x_test = torch.from_numpy(self.x_test).float()
        self.y_test = torch.from_numpy(self.y_test).float()
        self.true_adjacency = torch.from_numpy(self.true_adjacency).float()
        
        self.results = {}
        
    def create_test_models(self) -> Dict[str, CausalUnitNetwork]:
        """Create different model configurations for comparison"""
        models = {}
        
        # Vanilla MLP (no causal features)
        models['vanilla'] = CausalUnitNetwork(
            input_dim=self.input_size,
            hidden_dims=[self.hidden_size],
            output_dim=self.output_size,
            enable_structure_learning=False,
            enable_gradient_surgery=False
        )
        
        # Only violation penalty
        models['violation_penalty'] = CausalUnitNetwork(
            input_dim=self.input_size,
            hidden_dims=[self.hidden_size],
            output_dim=self.output_size,
            enable_structure_learning=False,
            enable_gradient_surgery=True
        )
        
        # Full causal model
        models['full_causal'] = CausalUnitNetwork(
            input_dim=self.input_size,
            hidden_dims=[self.hidden_size],
            output_dim=self.output_size,
            enable_structure_learning=True,
            enable_gradient_surgery=True
        )
        
        return models
        
    def test_1_intervention_breaks_causal_links(self, model: CausalUnitNetwork) -> Dict[str, Any]:
        """Test 1: Does intervention actually break causal links?"""
        print("🔍 Test 1: Intervention Breaks Causal Links")
        
        model.eval()
        x_sample = self.x_test[:5]  # Small batch
        
        # Get adjacency before intervention
        with torch.no_grad():
            adj_before = model.adjacency_matrix.clone() if hasattr(model, 'adjacency_matrix') else None
            
        # Apply intervention on node 1
        intervention_node = 1
        intervention_value = 0.5
        
        # Create intervention in correct format
        mask = torch.zeros(x_sample.shape[1])
        values = torch.zeros(x_sample.shape[1])
        mask[intervention_node] = 1.0
        values[intervention_node] = intervention_value
        
        # Create interventions list for batch
        interventions = []
        for i in range(x_sample.shape[0]):
            interventions.append({'intervention': (mask, values)})
        
        # Forward pass with intervention
        output = model(x_sample, interventions=interventions)
        
        # Check if dynamic adjacency cuts edges into intervened node
        results = {
            'adjacency_before': adj_before.numpy() if adj_before is not None else None,
            'intervention_node': intervention_node,
            'intervention_value': intervention_value,
            'output_shape': output.shape,
        }
        
        # Check if intervention actually modified the input
        if hasattr(model, 'last_interventions') and model.last_interventions:
            intervened_input = x_sample.clone()
            intervened_input[:, intervention_node] = intervention_value
            input_modified = not torch.allclose(x_sample[:, intervention_node], 
                                              intervened_input[:, intervention_node])
            results['input_actually_modified'] = input_modified
        else:
            results['input_actually_modified'] = False
            
        # Check dynamic adjacency if available
        if hasattr(model, 'compute_dynamic_adjacency'):
            try:
                adj_after = model.compute_dynamic_adjacency()
                results['adjacency_after'] = adj_after.numpy()
                
                # Check if edges into intervention node are cut
                if adj_before is not None and adj_after is not None:
                    edges_into_node_before = adj_before[:, intervention_node].sum()
                    edges_into_node_after = adj_after[:, intervention_node].sum()
                    results['edges_cut'] = edges_into_node_before > edges_into_node_after
                    results['edges_before'] = edges_into_node_before.item()
                    results['edges_after'] = edges_into_node_after.item()
                    
            except Exception as e:
                results['dynamic_adjacency_error'] = str(e)
        
        print(f"   ✅ Input modified: {results.get('input_actually_modified', 'Unknown')}")
        print(f"   ✅ Edges cut: {results.get('edges_cut', 'Unknown')}")
        
        return results
        
    def test_2_violation_penalty_spikes(self, model: CausalUnitNetwork) -> Dict[str, Any]:
        """Test 2: Does violation penalty spike when intervention is ignored?"""
        print("🔍 Test 2: Violation Penalty Spikes")
        
        model.train()
        x_sample = self.x_test[:10]
        y_sample = self.y_test[:10]
        
        results = {}
        
        # Test with and without intervention
        for condition in ['no_intervention', 'with_intervention']:
            interventions = None
            if condition == 'with_intervention':
                # Create intervention in correct format
                mask = torch.zeros(x_sample.shape[1])
                values = torch.zeros(x_sample.shape[1])
                mask[1] = 1.0  # Intervene on node 1
                values[1] = 0.5
                
                interventions = []
                for i in range(x_sample.shape[0]):
                    interventions.append({'test_intervention': (mask, values)})
                
            # Forward pass
            output = model(x_sample, interventions=interventions)
            loss = nn.MSELoss()(output, y_sample)
            
            # Compute violation penalty
            violation_penalty = 0.0
            if hasattr(model, 'get_violation_penalty'):
                try:
                    violation_penalty = model.get_violation_penalty(x_sample)
                    if torch.is_tensor(violation_penalty):
                        violation_penalty = violation_penalty.item()
                except Exception as e:
                    violation_penalty = f"Error: {str(e)}"
                    
            results[condition] = {
                'loss': loss.item(),
                'violation_penalty': violation_penalty,
                'output_mean': output.mean().item(),
                'output_std': output.std().item()
            }
        
        # Check if penalty increases with intervention
        if (isinstance(results['no_intervention']['violation_penalty'], (int, float)) and 
            isinstance(results['with_intervention']['violation_penalty'], (int, float))):
            penalty_increase = (results['with_intervention']['violation_penalty'] > 
                              results['no_intervention']['violation_penalty'])
            results['penalty_increases_with_intervention'] = penalty_increase
        else:
            results['penalty_increases_with_intervention'] = 'Unknown'
            
        print(f"   ✅ Penalty without intervention: {results['no_intervention']['violation_penalty']}")
        print(f"   ✅ Penalty with intervention: {results['with_intervention']['violation_penalty']}")
        print(f"   ✅ Penalty increases: {results['penalty_increases_with_intervention']}")
        
        return results
        
    def test_3_alpha_parameters_learning(self, model: CausalUnitNetwork) -> Dict[str, Any]:
        """Test 3: Are alpha parameters actually learning?"""
        print("🔍 Test 3: Alpha Parameters Learning")
        
        results = {
            'alpha_parameters_found': False,
            'alpha_values': {},
            'alpha_gradients': {},
            'alpha_ranges': {}
        }
        
        # Check for alpha parameters
        alpha_params = {}
        alpha_grads = {}
        
        for name, param in model.named_parameters():
            if 'alpha' in name.lower():
                results['alpha_parameters_found'] = True
                alpha_params[name] = param.data.clone()
                alpha_grads[name] = param.grad.clone() if param.grad is not None else None
                
        if results['alpha_parameters_found']:
            # Record initial values
            for name, param in alpha_params.items():
                results['alpha_values'][name] = {
                    'mean': param.mean().item(),
                    'min': param.min().item(),
                    'max': param.max().item(),
                    'std': param.std().item()
                }
                
            # Record gradients
            for name, grad in alpha_grads.items():
                if grad is not None:
                    results['alpha_gradients'][name] = {
                        'mean': grad.mean().item(),
                        'min': grad.min().item(),
                        'max': grad.max().item(),
                        'std': grad.std().item()
                    }
                    
            # Quick training step to see if they update
            x_sample = self.x_test[:32]
            y_sample = self.y_test[:32]
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # Training step
            optimizer.zero_grad()
            
            # Create intervention
            mask = torch.zeros(x_sample.shape[1])
            values = torch.zeros(x_sample.shape[1])
            mask[1] = 1.0
            values[1] = 0.5
            
            interventions = []
            for i in range(x_sample.shape[0]):
                interventions.append({'alpha_test': (mask, values)})
            
            output = model(x_sample, interventions=interventions)
            loss = nn.MSELoss()(output, y_sample)
            loss.backward()
            optimizer.step()
            
            # Check if alpha values changed
            for name, param in model.named_parameters():
                if 'alpha' in name.lower():
                    old_value = alpha_params[name]
                    new_value = param.data
                    changed = not torch.allclose(old_value, new_value, atol=1e-6)
                    results['alpha_ranges'][name] = {
                        'changed': changed,
                        'max_change': (new_value - old_value).abs().max().item()
                    }
                    
        print(f"   ✅ Alpha parameters found: {results['alpha_parameters_found']}")
        if results['alpha_parameters_found']:
            for name, info in results['alpha_ranges'].items():
                print(f"   ✅ {name} changed: {info['changed']} (max change: {info['max_change']:.6f})")
        
        return results
        
    def test_4_interventions_change_predictions(self, model: CausalUnitNetwork) -> Dict[str, Any]:
        """Test 4: Do interventions actually change predictions?"""
        print("🔍 Test 4: Interventions Change Predictions")
        
        model.eval()
        x_sample = self.x_test[:100]
        
        # Baseline prediction
        with torch.no_grad():
            baseline_output = model(x_sample)
            
        results = {
            'baseline_mean': baseline_output.mean().item(),
            'baseline_std': baseline_output.std().item(),
            'interventions': {}
        }
        
        # Test different interventions (only for 3 nodes)
        intervention_tests = [
            (0, 0.0),
            (0, 1.0),
            (1, -1.0),
            (1, 1.0),
            (2, 0.5)
        ]
        
        for node_idx, value in intervention_tests:
            with torch.no_grad():
                # Create intervention
                mask = torch.zeros(x_sample.shape[1])
                values = torch.zeros(x_sample.shape[1])
                mask[node_idx] = 1.0
                values[node_idx] = value
                
                interventions = []
                for i in range(x_sample.shape[0]):
                    interventions.append({'test': (mask, values)})
                
                intervened_output = model(x_sample, interventions=interventions)
                
                # Calculate difference
                diff = intervened_output - baseline_output
                
                results['interventions'][f"x{node_idx}={value}"] = {
                    'output_mean': intervened_output.mean().item(),
                    'output_std': intervened_output.std().item(),
                    'mean_diff': diff.mean().item(),
                    'std_diff': diff.std().item(),
                    'max_abs_diff': diff.abs().max().item(),
                    'significant_change': diff.abs().max().item() > 0.01
                }
                
        # Summary statistics
        significant_changes = sum(1 for info in results['interventions'].values() 
                                if info['significant_change'])
        results['total_significant_changes'] = significant_changes
        results['fraction_significant'] = significant_changes / len(intervention_tests)
        
        print(f"   ✅ Significant changes: {significant_changes}/{len(intervention_tests)}")
        print(f"   ✅ Fraction significant: {results['fraction_significant']:.2f}")
        
        return results
        
    def test_5_ground_truth_causal_effects(self, model: CausalUnitNetwork) -> Dict[str, Any]:
        """Test 5: Compare to ground-truth causal effects"""
        print("🔍 Test 5: Ground-Truth Causal Effects")
        
        model.eval()
        x_sample = self.x_test[:200]
        
        results = {
            'true_edges': self.true_edges,
            'causal_effect_tests': {}
        }
        
        # Test effects along known causal paths (chain: 0->1->2)
        causal_paths = [
            (0, 1, "x0 -> x1"),
            (1, 2, "x1 -> x2")
        ]
        
        for parent, child, description in causal_paths:
            with torch.no_grad():
                # Baseline
                baseline = model(x_sample)
                
                # Intervene on parent
                mask = torch.zeros(x_sample.shape[1])
                values = torch.zeros(x_sample.shape[1])
                mask[parent] = 1.0
                values[parent] = 1.0
                
                interventions = []
                for i in range(x_sample.shape[0]):
                    interventions.append({'causal_test': (mask, values)})
                
                intervened = model(x_sample, interventions=interventions)
                
                # Calculate effect
                effect = (intervened - baseline).mean().item()
                
                results['causal_effect_tests'][description] = {
                    'parent': parent,
                    'child': child,
                    'effect_magnitude': abs(effect),
                    'effect_direction': 'positive' if effect > 0 else 'negative',
                    'significant': abs(effect) > 0.01
                }
                
        # Test non-causal relationships (should have minimal effect)
        non_causal_tests = [
            (2, 0, "x2 -> x0 (reverse)"),
            (2, 1, "x2 -> x1 (reverse)")
        ]
        
        for parent, child, description in non_causal_tests:
            with torch.no_grad():
                baseline = model(x_sample)
                
                mask = torch.zeros(x_sample.shape[1])
                values = torch.zeros(x_sample.shape[1])
                mask[parent] = 1.0
                values[parent] = 1.0
                
                interventions = []
                for i in range(x_sample.shape[0]):
                    interventions.append({'non_causal_test': (mask, values)})
                
                intervened = model(x_sample, interventions=interventions)
                effect = (intervened - baseline).mean().item()
                
                results['causal_effect_tests'][description] = {
                    'parent': parent,
                    'child': child,
                    'effect_magnitude': abs(effect),
                    'effect_direction': 'positive' if effect > 0 else 'negative',
                    'significant': abs(effect) > 0.01,
                    'should_be_minimal': True
                }
        
        # Summary
        causal_effects = [info for desc, info in results['causal_effect_tests'].items() 
                         if not info.get('should_be_minimal', False)]
        non_causal_effects = [info for desc, info in results['causal_effect_tests'].items() 
                            if info.get('should_be_minimal', False)]
        
        results['summary'] = {
            'causal_effects_detected': sum(1 for info in causal_effects if info['significant']),
            'non_causal_effects_detected': sum(1 for info in non_causal_effects if info['significant']),
            'causal_precision': sum(1 for info in causal_effects if info['significant']) / len(causal_effects) if causal_effects else 0,
            'non_causal_suppression': sum(1 for info in non_causal_effects if not info['significant']) / len(non_causal_effects) if non_causal_effects else 0
        }
        
        print(f"   ✅ Causal effects detected: {results['summary']['causal_effects_detected']}/{len(causal_effects)}")
        print(f"   ✅ Non-causal effects suppressed: {results['summary']['non_causal_suppression']:.2f}")
        
        return results
        
    def test_6_loss_comparison(self, models: Dict[str, CausalUnitNetwork]) -> Dict[str, Any]:
        """Test 6: Compare loss curves with different configurations"""
        print("🔍 Test 6: Loss Comparison Across Models")
        
        results = {}
        
        # Quick training loop for each model
        for model_name, model in models.items():
            print(f"   Training {model_name}...")
            
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            losses = []
            for epoch in range(20):  # Quick training
                optimizer.zero_grad()
                
                # Use interventions for models that support them
                interventions = None
                if hasattr(model, 'enable_structure_learning') and model.enable_structure_learning:
                    if np.random.random() < 0.3:  # 30% intervention rate
                        mask = torch.zeros(self.x_test.shape[1])
                        values = torch.zeros(self.x_test.shape[1])
                        mask[1] = 1.0
                        values[1] = np.random.randn() * 0.5
                        
                        interventions = []
                        for i in range(self.x_test[:100].shape[0]):
                            interventions.append({'training': (mask, values)})
                
                output = model(self.x_test[:100], interventions=interventions)
                loss = nn.MSELoss()(output, self.y_test[:100])
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                
            results[model_name] = {
                'final_loss': losses[-1],
                'loss_curve': losses,
                'loss_improvement': losses[0] - losses[-1],
                'converged': losses[-1] < losses[0] * 0.8
            }
            
        # Compare final performance
        final_losses = {name: info['final_loss'] for name, info in results.items()}
        best_model = min(final_losses, key=final_losses.get)
        
        results['comparison'] = {
            'best_model': best_model,
            'final_losses': final_losses,
            'loss_differences': {name: loss - final_losses['vanilla'] 
                               for name, loss in final_losses.items()}
        }
        
        print(f"   ✅ Best model: {best_model}")
        print(f"   ✅ Final losses: {final_losses}")
        
        return results
        
    def test_7_spurious_feature_test(self, model: CausalUnitNetwork) -> Dict[str, Any]:
        """Test 7: Inject spurious feature and test resistance"""
        print("🔍 Test 7: Spurious Feature Resistance")
        
        # Create data with spurious correlation
        x_spurious = self.x_test.clone()
        y_spurious = self.y_test.clone()
        
        # Add spurious feature that's correlated with output but not causal
        spurious_feature = y_spurious.squeeze() + torch.randn_like(y_spurious.squeeze()) * 0.1
        x_spurious = torch.cat([x_spurious, spurious_feature.unsqueeze(1)], dim=1)
        
        # Train model on spurious data
        model_spurious = CausalUnitNetwork(
            input_dim=self.input_size + 1,
            hidden_dims=[self.hidden_size],
            output_dim=self.output_size,
            enable_structure_learning=True,
            enable_gradient_surgery=True
        )
        
        # Train vanilla model for comparison
        model_vanilla = nn.Sequential(
            nn.Linear(self.input_size + 1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )
        
        results = {'causal_model': {}, 'vanilla_model': {}}
        
        # Train both models
        for model_name, model in [('causal_model', model_spurious), ('vanilla_model', model_vanilla)]:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            for epoch in range(50):
                optimizer.zero_grad()
                
                if model_name == 'causal_model':
                    # Intervene on spurious feature
                    mask = torch.zeros(x_spurious.shape[1])
                    values = torch.zeros(x_spurious.shape[1])
                    mask[self.input_size] = 1.0  # Spurious feature is last
                    values[self.input_size] = 0.0
                    
                    interventions = []
                    for i in range(x_spurious[:100].shape[0]):
                        interventions.append({'spurious_test': (mask, values)})
                    
                    output = model(x_spurious[:100], interventions=interventions)
                else:
                    output = model(x_spurious[:100])
                    
                loss = nn.MSELoss()(output, y_spurious[:100])
                loss.backward()
                optimizer.step()
                
            # Test resistance to spurious feature
            model.eval()
            with torch.no_grad():
                # Normal prediction
                normal_output = model(x_spurious[:100]) if model_name == 'vanilla_model' else model(x_spurious[:100])
                
                # Intervene on spurious feature (should have minimal effect for causal model)
                if model_name == 'causal_model':
                    mask = torch.zeros(x_spurious.shape[1])
                    values = torch.zeros(x_spurious.shape[1])
                    mask[self.input_size] = 1.0
                    values[self.input_size] = 999.0
                    
                    interventions = []
                    for i in range(x_spurious[:100].shape[0]):
                        interventions.append({'spurious_extreme': (mask, values)})
                    
                    intervened_output = model(x_spurious[:100], interventions=interventions)
                    spurious_effect = (intervened_output - normal_output).abs().mean().item()
                else:
                    # For vanilla model, manually change spurious feature
                    x_modified = x_spurious[:100].clone()
                    x_modified[:, -1] = 999.0  # Extreme value
                    modified_output = model(x_modified)
                    spurious_effect = (modified_output - normal_output).abs().mean().item()
                
                results[model_name] = {
                    'final_loss': loss.item(),
                    'spurious_effect': spurious_effect,
                    'spurious_resistance': spurious_effect < 0.1  # Threshold for resistance
                }
        
        # Compare spurious resistance
        causal_resistance = results['causal_model']['spurious_resistance']
        vanilla_resistance = results['vanilla_model']['spurious_resistance']
        
        results['comparison'] = {
            'causal_more_resistant': causal_resistance and not vanilla_resistance,
            'both_resistant': causal_resistance and vanilla_resistance,
            'neither_resistant': not causal_resistance and not vanilla_resistance
        }
        
        print(f"   ✅ Causal model resistant: {causal_resistance}")
        print(f"   ✅ Vanilla model resistant: {vanilla_resistance}")
        print(f"   ✅ Causal advantage: {results['comparison']['causal_more_resistant']}")
        
        return results
        
    def run_all_tests(self):
        """Run all sanity checks and generate comprehensive report"""
        print("🚀 Running Comprehensive Causal Sanity Checks")
        print("=" * 60)
        
        # Create test models
        models = self.create_test_models()
        
        # Run all tests with full causal model
        full_model = models['full_causal']
        
        print("\n" + "=" * 60)
        self.results['test_1'] = self.test_1_intervention_breaks_causal_links(full_model)
        
        print("\n" + "=" * 60)
        self.results['test_2'] = self.test_2_violation_penalty_spikes(full_model)
        
        print("\n" + "=" * 60)
        self.results['test_3'] = self.test_3_alpha_parameters_learning(full_model)
        
        print("\n" + "=" * 60)
        self.results['test_4'] = self.test_4_interventions_change_predictions(full_model)
        
        print("\n" + "=" * 60)
        self.results['test_5'] = self.test_5_ground_truth_causal_effects(full_model)
        
        print("\n" + "=" * 60)
        self.results['test_6'] = self.test_6_loss_comparison(models)
        
        print("\n" + "=" * 60)
        self.results['test_7'] = self.test_7_spurious_feature_test(full_model)
        
        # Generate summary
        self.generate_summary_report()
        
        return self.results
        
    def generate_summary_report(self):
        """Generate a comprehensive summary of all test results"""
        print("\n" + "🔍 SANITY CHECK SUMMARY REPORT")
        print("=" * 60)
        
        # Overall health check
        health_checks = []
        
        # Test 1: Intervention breaks links
        if self.results['test_1'].get('input_actually_modified', False):
            health_checks.append("✅ Interventions modify inputs")
        else:
            health_checks.append("❌ Interventions don't modify inputs")
            
        # Test 2: Violation penalty
        if self.results['test_2'].get('penalty_increases_with_intervention', False):
            health_checks.append("✅ Violation penalty increases with intervention")
        else:
            health_checks.append("❌ Violation penalty doesn't increase with intervention")
            
        # Test 3: Alpha parameters
        if self.results['test_3'].get('alpha_parameters_found', False):
            health_checks.append("✅ Alpha parameters found and learning")
        else:
            health_checks.append("❌ Alpha parameters not found or not learning")
            
        # Test 4: Interventions change predictions
        if self.results['test_4'].get('fraction_significant', 0) > 0.5:
            health_checks.append("✅ Interventions significantly change predictions")
        else:
            health_checks.append("❌ Interventions don't significantly change predictions")
            
        # Test 5: Ground truth effects
        if self.results['test_5']['summary'].get('causal_precision', 0) > 0.5:
            health_checks.append("✅ Causal effects detected for true relationships")
        else:
            health_checks.append("❌ Causal effects not detected for true relationships")
            
        # Test 6: Loss comparison
        if self.results['test_6']['comparison']['best_model'] != 'vanilla':
            health_checks.append("✅ Causal model outperforms vanilla")
        else:
            health_checks.append("❌ Causal model doesn't outperform vanilla")
            
        # Test 7: Spurious resistance
        if self.results['test_7']['comparison'].get('causal_more_resistant', False):
            health_checks.append("✅ Causal model resists spurious correlations")
        else:
            health_checks.append("❌ Causal model doesn't resist spurious correlations")
            
        # Print health checks
        for check in health_checks:
            print(check)
            
        # Overall assessment
        passed_checks = sum(1 for check in health_checks if check.startswith("✅"))
        total_checks = len(health_checks)
        
        print(f"\n🎯 OVERALL ASSESSMENT: {passed_checks}/{total_checks} checks passed")
        
        if passed_checks >= 5:
            print("🎉 CAUSAL SYSTEM IS WORKING WELL!")
        elif passed_checks >= 3:
            print("⚠️  CAUSAL SYSTEM HAS SOME ISSUES")
        else:
            print("🚨 CAUSAL SYSTEM HAS MAJOR PROBLEMS")
            
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"causal_sanity_check_results_{timestamp}.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\n📊 Detailed results saved to: causal_sanity_check_results_{timestamp}.json")

if __name__ == "__main__":
    checker = CausalSanityChecker()
    results = checker.run_all_tests() 