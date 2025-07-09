import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from causal_unit_network import CausalUnitNetwork
from engine.causal_unit import CausalUnit, CausalInterventionFunction
import json
from datetime import datetime


class GradientValidationTest:
    """
    Manual gradient validation test for Phase 3 CausalUnit architecture.
    
    This test validates that our novel gradient blocking and intervention logic
    works correctly on simple toy examples before running the full evaluation suite.
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_results = {}
        
    def create_toy_dag_data(self, n_samples=100):
        """Create simple 3-node DAG: X1 -> X2 -> X3"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # X1 is exogenous
        x1 = torch.randn(n_samples, 1)
        
        # X2 depends on X1: X2 = 0.8 * X1 + noise
        x2 = 0.8 * x1 + 0.2 * torch.randn(n_samples, 1)
        
        # X3 depends on X2: X3 = 0.6 * X2 + noise  
        x3 = 0.6 * x2 + 0.2 * torch.randn(n_samples, 1)
        
        # Input is [X1, X2] and target is X3
        x = torch.cat([x1, x2], dim=1)
        y = x3
        
        # True adjacency: X1->X2, X2->X3 (but we only model X2->X3 in our network)
        true_adjacency = torch.tensor([[0.0, 1.0],  # X1 doesn't directly affect X3
                                      [0.0, 1.0]], dtype=torch.float32)  # X2 affects X3
        
        return x.to(self.device), y.to(self.device), true_adjacency.to(self.device)
    
    def test_basic_forward_pass(self):
        """Test 1: Basic forward pass without interventions"""
        print("=== Test 1: Basic Forward Pass ===")
        
        # Create simple network
        network = CausalUnitNetwork(
            input_dim=2,
            hidden_dims=[],
            output_dim=1,
            enable_structure_learning=False,
            enable_gradient_surgery=False
        ).to(self.device)
        
        # Create toy data
        x, y, true_adj = self.create_toy_dag_data(n_samples=10)
        
        # Forward pass
        output = network(x, interventions=None)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Forward pass successful: {output is not None}")
        
        self.test_results['basic_forward'] = {
            'success': output is not None,
            'input_shape': list(x.shape),
            'output_shape': list(output.shape)
        }
        
        return True
    
    def test_intervention_forward_pass(self):
        """Test 2: Forward pass with interventions"""
        print("\n=== Test 2: Intervention Forward Pass ===")
        
        # Create simple network
        network = CausalUnitNetwork(
            input_dim=2,
            hidden_dims=[],
            output_dim=1,
            enable_structure_learning=False,
            enable_gradient_surgery=False
        ).to(self.device)
        
        # Create toy data
        x, y, true_adj = self.create_toy_dag_data(n_samples=10)
        
        # Create intervention: set X2 (index 1) to fixed value
        intervention_mask = torch.zeros(2, device=self.device)
        intervention_values = torch.zeros(2, device=self.device)
        intervention_mask[1] = 1.0  # Intervene on X2
        intervention_values[1] = 2.0  # Set X2 = 2.0
        
        interventions = []
        for i in range(x.shape[0]):
            interventions.append({'do_x2': (intervention_mask, intervention_values)})
        
        # Forward pass with intervention
        output_original = network(x, interventions=None)
        output_intervened = network(x, interventions=interventions)
        
        # Check that intervention had an effect
        intervention_effect = torch.mean(torch.abs(output_intervened - output_original)).item()
        
        print(f"Original output mean: {torch.mean(output_original).item():.4f}")
        print(f"Intervened output mean: {torch.mean(output_intervened).item():.4f}")
        print(f"Intervention effect magnitude: {intervention_effect:.4f}")
        print(f"Intervention had effect: {intervention_effect > 0.01}")
        
        self.test_results['intervention_forward'] = {
            'success': True,
            'intervention_effect': intervention_effect,
            'effect_detected': intervention_effect > 0.01
        }
        
        return intervention_effect > 0.01
    
    def test_gradient_blocking_manual(self):
        """Test 3: Manual verification of gradient blocking"""
        print("\n=== Test 3: Manual Gradient Blocking Verification ===")
        
        # Create simple single CausalUnit for direct testing
        unit = CausalUnit(
            input_dim=2,
            output_dim=1,
            hidden_dim=None,
            enable_structure_learning=False,
            enable_gradient_surgery=False
        ).to(self.device)
        
        # Create toy data - single sample for clarity
        x = torch.tensor([[1.0, 2.0]], device=self.device, requires_grad=True)
        target = torch.tensor([[3.0]], device=self.device)
        
        # Test 1: Forward pass without intervention
        unit.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
            
        output1 = unit(x)
        loss1 = torch.mean((output1 - target) ** 2)
        loss1.backward()
        
        grad_without_intervention = x.grad.clone() if x.grad is not None else None
        
        print(f"Without intervention - X gradient: {grad_without_intervention}")
        
        # Test 2: Forward pass with intervention on X[1]
        x = torch.tensor([[1.0, 2.0]], device=self.device, requires_grad=True)
        unit.zero_grad()
        
        intervention_mask = torch.tensor([0.0, 1.0], device=self.device)  # Intervene on X[1]
        intervention_values = torch.tensor([5.0, 5.0], device=self.device)  # Set X[1] = 5.0
        
        output2 = unit(x, do_mask=intervention_mask, do_values=intervention_values)
        loss2 = torch.mean((output2 - target) ** 2)
        loss2.backward()
        
        grad_with_intervention = x.grad.clone() if x.grad is not None else None
        
        print(f"With intervention on X[1] - X gradient: {grad_with_intervention}")
        
        # Check if gradient to X[1] is blocked (should be close to 0)
        if grad_with_intervention is not None:
            grad_x1_blocked = abs(grad_with_intervention[0, 1].item()) < 1e-6
            grad_x0_preserved = abs(grad_with_intervention[0, 0].item()) > 1e-6
            
            print(f"Gradient to X[1] blocked: {grad_x1_blocked}")
            print(f"Gradient to X[0] preserved: {grad_x0_preserved}")
        else:
            grad_x1_blocked = False
            grad_x0_preserved = False
            print("No gradients computed!")
        
        self.test_results['gradient_blocking'] = {
            'grad_without_intervention': grad_without_intervention.tolist() if grad_without_intervention is not None else None,
            'grad_with_intervention': grad_with_intervention.tolist() if grad_with_intervention is not None else None,
            'x1_gradient_blocked': grad_x1_blocked,
            'x0_gradient_preserved': grad_x0_preserved,
            'blocking_successful': grad_x1_blocked and grad_x0_preserved
        }
        
        return grad_x1_blocked and grad_x0_preserved
    
    def test_network_gradient_blocking(self):
        """Test 4: Network-level gradient blocking verification"""
        print("\n=== Test 4: Network-Level Gradient Blocking ===")
        
        # Create network
        network = CausalUnitNetwork(
            input_dim=2,
            hidden_dims=[],
            output_dim=1,
            enable_structure_learning=False,
            enable_gradient_surgery=True
        ).to(self.device)
        
        # Create toy data
        x, y, true_adj = self.create_toy_dag_data(n_samples=5)
        x.requires_grad_(True)
        
        # Forward pass without intervention
        network.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
            
        output1 = network(x, interventions=None)
        loss1 = torch.mean((output1 - y) ** 2)
        loss1.backward()
        
        grad_without_intervention = x.grad.clone() if x.grad is not None else None
        
        # Forward pass with intervention
        x = x.detach().requires_grad_(True)
        network.zero_grad()
        
        intervention_mask = torch.zeros(2, device=self.device)
        intervention_values = torch.zeros(2, device=self.device)
        intervention_mask[1] = 1.0  # Intervene on X2
        intervention_values[1] = 3.0
        
        interventions = []
        for i in range(x.shape[0]):
            interventions.append({'do_x2': (intervention_mask, intervention_values)})
        
        output2 = network(x, interventions=interventions)
        loss2 = torch.mean((output2 - y) ** 2)
        loss2.backward()
        
        grad_with_intervention = x.grad.clone() if x.grad is not None else None
        
        print(f"Network gradients without intervention: {grad_without_intervention}")
        print(f"Network gradients with intervention: {grad_with_intervention}")
        
        # Analyze gradient blocking
        if grad_with_intervention is not None and grad_without_intervention is not None:
            # Check if gradients to intervened variable are reduced
            grad_reduction = torch.mean(torch.abs(grad_with_intervention[:, 1])) / torch.mean(torch.abs(grad_without_intervention[:, 1]))
            blocking_effective = grad_reduction < 0.1  # 90% reduction
            
            print(f"Gradient reduction ratio for intervened variable: {grad_reduction:.4f}")
            print(f"Blocking effective (>90% reduction): {blocking_effective}")
        else:
            blocking_effective = False
            grad_reduction = float('inf')
        
        self.test_results['network_gradient_blocking'] = {
            'grad_reduction_ratio': float(grad_reduction) if grad_reduction != float('inf') else None,
            'blocking_effective': blocking_effective
        }
        
        return blocking_effective
    
    def test_multiple_interventions(self):
        """Test 5: Multiple simultaneous interventions"""
        print("\n=== Test 5: Multiple Simultaneous Interventions ===")
        
        # Create network with 3 inputs for clearer testing
        network = CausalUnitNetwork(
            input_dim=3,
            hidden_dims=[],
            output_dim=1,
            enable_structure_learning=False,
            enable_gradient_surgery=True
        ).to(self.device)
        
        # Create toy data
        x = torch.randn(5, 3, device=self.device)
        y = torch.randn(5, 1, device=self.device)
        
        # Create multiple interventions
        intervention1_mask = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        intervention1_values = torch.tensor([10.0, 0.0, 0.0], device=self.device)
        
        intervention2_mask = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        intervention2_values = torch.tensor([0.0, 0.0, 20.0], device=self.device)
        
        interventions = []
        for i in range(x.shape[0]):
            interventions.append({
                'int1': (intervention1_mask, intervention1_values),
                'int2': (intervention2_mask, intervention2_values)
            })
        
        # Forward pass
        output_original = network(x, interventions=None)
        output_intervened = network(x, interventions=interventions)
        
        intervention_effect = torch.mean(torch.abs(output_intervened - output_original)).item()
        
        print(f"Multiple intervention effect: {intervention_effect:.4f}")
        print(f"Multiple interventions successful: {intervention_effect > 0.01}")
        
        self.test_results['multiple_interventions'] = {
            'intervention_effect': intervention_effect,
            'success': intervention_effect > 0.01
        }
        
        return intervention_effect > 0.01
    
    def test_adjacency_matrix_learning(self):
        """Test 6: Adjacency matrix learning and hybrid approach"""
        print("\n=== Test 6: Adjacency Matrix Learning ===")
        
        # Create network with structure learning
        network = CausalUnitNetwork(
            input_dim=2,
            hidden_dims=[],
            output_dim=1,
            enable_structure_learning=True,
            enable_gradient_surgery=False
        ).to(self.device)
        
        # Test soft vs hard adjacency
        soft_adj = network.get_adjacency_matrix(hard=False)
        hard_adj = network.get_adjacency_matrix(hard=True)
        
        print(f"Soft adjacency shape: {soft_adj.shape}")
        print(f"Hard adjacency shape: {hard_adj.shape}")
        print(f"Soft adjacency:\n{soft_adj}")
        print(f"Hard adjacency:\n{hard_adj}")
        
        # Check that hard adjacency is binary
        is_binary = torch.all((hard_adj == 0) | (hard_adj == 1)).item()
        
        # Check that soft adjacency is in [0,1]
        is_normalized = torch.all((soft_adj >= 0) & (soft_adj <= 1)).item()
        
        print(f"Hard adjacency is binary: {is_binary}")
        print(f"Soft adjacency is normalized: {is_normalized}")
        
        self.test_results['adjacency_learning'] = {
            'soft_adj_shape': list(soft_adj.shape),
            'hard_adj_shape': list(hard_adj.shape),
            'hard_is_binary': is_binary,
            'soft_is_normalized': is_normalized,
            'hybrid_approach_working': is_binary and is_normalized
        }
        
        return is_binary and is_normalized
    
    def run_all_tests(self):
        """Run all gradient validation tests"""
        print("🧪 PHASE 3 GRADIENT VALIDATION TEST SUITE")
        print("=" * 60)
        
        test_functions = [
            self.test_basic_forward_pass,
            self.test_intervention_forward_pass,
            self.test_gradient_blocking_manual,
            self.test_network_gradient_blocking,
            self.test_multiple_interventions,
            self.test_adjacency_matrix_learning
        ]
        
        results = []
        
        for test_func in test_functions:
            try:
                result = test_func()
                results.append(result)
                print(f"✅ {test_func.__name__}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                print(f"❌ {test_func.__name__}: ERROR - {str(e)}")
                results.append(False)
                self.test_results[test_func.__name__] = {'error': str(e)}
        
        # Overall assessment
        all_passed = all(results)
        
        print(f"\n📊 GRADIENT VALIDATION SUMMARY")
        print(f"Tests passed: {sum(results)}/{len(results)}")
        print(f"Overall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        self.test_results['summary'] = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'tests_passed': sum(results),
            'overall_success': all_passed
        }
        
        return all_passed
    
    def save_results(self, filepath='gradient_validation_results.json'):
        """Save test results to file"""
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n💾 Results saved to {filepath}")


def main():
    """Run gradient validation tests"""
    tester = GradientValidationTest()
    success = tester.run_all_tests()
    tester.save_results()
    
    if success:
        print(f"\n🎉 GRADIENT VALIDATION SUCCESSFUL!")
        print(f"✅ All core mathematical innovations are working correctly")
        print(f"✅ Ready to proceed with full evaluation suite")
    else:
        print(f"\n⚠️  GRADIENT VALIDATION ISSUES DETECTED!")
        print(f"❌ Review gradient_validation_results.json for details")
        print(f"❌ Fix issues before proceeding with full evaluation")
    
    return success


if __name__ == "__main__":
    main() 