import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from causal_unit_network import CausalUnitNetwork
from train_causalunit import CausalUnitTrainer
from eval_causalunit import CausalUnitEvaluator
from engine.loss_functions import CausalLosses, CausalMetrics


class Phase3CoreValidation:
    """
    Core validation test for Phase 3 go/no-go decision.
    
    This validates the key requirements:
    1. Structure learning performance
    2. Counterfactual reasoning accuracy
    3. Ablation studies showing innovation importance
    4. Robustness under noise
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results = {}
        self.timestamp = datetime.now().isoformat()
        
    def create_simple_data(self, n_samples=500, noise_level=0.3):
        """Create simple synthetic data for validation"""
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Simple chain: X1 -> X2 -> Y
        x1 = torch.randn(n_samples, 1)
        x2 = 0.8 * x1 + noise_level * torch.randn(n_samples, 1)
        y = 0.6 * x2 + noise_level * torch.randn(n_samples, 1)
        
        x = torch.cat([x1, x2], dim=1)  # Input features
        
        # True adjacency matrix (2x2 for X1, X2)
        true_adjacency = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
        
        return x, y, true_adjacency
    
    def test_structure_learning(self):
        """Test 1: Structure learning performance"""
        print("🔬 TEST 1: STRUCTURE LEARNING")
        print("=" * 50)
        
        # Create data
        x, y, true_adjacency = self.create_simple_data(n_samples=800)
        
        # Create network
        network = CausalUnitNetwork(
            input_dim=2,
            hidden_dims=[],
            output_dim=1,
            enable_structure_learning=True,
            enable_gradient_surgery=False  # Simplified
        ).to(self.device)
        
        # Train
        trainer = CausalUnitTrainer(network, device=self.device)
        
        dataset = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        results = trainer.train(
            train_loader, test_loader, num_epochs=25,
            true_adjacency=true_adjacency,
            early_stopping_patience=8
        )
        
        # Evaluate structure learning
        learned_adjacency = network.get_adjacency_matrix(hard=True)
        
        evaluator = CausalUnitEvaluator(device=self.device)
        structure_metrics = evaluator.evaluate_structure_learning_comprehensive(
            learned_adjacency.cpu().numpy(),
            true_adjacency.cpu().numpy()
        )
        
        print(f"✅ Structure Learning Results:")
        print(f"   F1 Score: {structure_metrics['f1']:.3f}")
        print(f"   Precision: {structure_metrics['precision']:.3f}")
        print(f"   Recall: {structure_metrics['recall']:.3f}")
        print(f"   Edge Accuracy: {structure_metrics['edge_accuracy']:.3f}")
        
        success = structure_metrics['f1'] > 0.7  # Reasonable threshold
        
        self.results['structure_learning'] = {
            'success': success,
            'metrics': structure_metrics,
            'learned_adjacency': learned_adjacency.cpu().tolist(),
            'true_adjacency': true_adjacency.cpu().tolist()
        }
        
        return success
    
    def test_counterfactual_reasoning(self):
        """Test 2: Counterfactual reasoning accuracy"""
        print("\n🎯 TEST 2: COUNTERFACTUAL REASONING")
        print("=" * 50)
        
        # Create data
        x, y, true_adjacency = self.create_simple_data(n_samples=400)
        
        # Create and train network
        network = CausalUnitNetwork(
            input_dim=2,
            hidden_dims=[],
            output_dim=1,
            enable_structure_learning=False,  # Focus on CF
            enable_gradient_surgery=True
        ).to(self.device)
        
        trainer = CausalUnitTrainer(network, device=self.device)
        
        dataset = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Quick training
        trainer.train(train_loader, test_loader, num_epochs=15)
        
        # Evaluate counterfactual performance
        evaluator = CausalUnitEvaluator(device=self.device)
        cf_metrics = evaluator.evaluate_counterfactual_performance_comprehensive(
            network, x.to(self.device), y.to(self.device), true_adjacency.to(self.device),
            n_interventions=15
        )
        
        print(f"✅ Counterfactual Reasoning Results:")
        print(f"   Mean Intervention Effect: {cf_metrics['mean_intervention_effect']:.4f}")
        print(f"   Effect Standard Deviation: {cf_metrics['std_intervention_effect']:.4f}")
        print(f"   Mean Correlation: {cf_metrics['mean_correlation']:.3f}")
        
        success = cf_metrics['mean_intervention_effect'] > 0.01  # Detectable effects
        
        self.results['counterfactual_reasoning'] = {
            'success': success,
            'metrics': cf_metrics
        }
        
        return success
    
    def test_ablation_studies(self):
        """Test 3: Ablation studies showing innovation importance"""
        print("\n🧪 TEST 3: ABLATION STUDIES")
        print("=" * 50)
        
        # Create data
        x, y, true_adjacency = self.create_simple_data(n_samples=600)
        
        dataset = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        ablation_configs = [
            {'name': 'full_model', 'config': {}},
            {'name': 'no_interventions', 'config': {'enable_interventions': False}},
            {'name': 'no_gradient_surgery', 'config': {'enable_gradient_surgery': False}},
            {'name': 'vanilla_baseline', 'config': {
                'enable_interventions': False,
                'enable_gradient_surgery': False,
                'enable_structure_learning': False
            }}
        ]
        
        ablation_results = {}
        
        for ablation in ablation_configs:
            print(f"\n   Testing: {ablation['name']}")
            
            # Create network
            network = CausalUnitNetwork(
                input_dim=2,
                hidden_dims=[],
                output_dim=1,
                enable_structure_learning=True,
                enable_gradient_surgery=True
            ).to(self.device)
            
            trainer = CausalUnitTrainer(network, device=self.device)
            trainer.set_ablation_config(**ablation['config'])
            
            # Train
            results = trainer.train(
                train_loader, test_loader, num_epochs=20,
                true_adjacency=true_adjacency,
                early_stopping_patience=5
            )
            
            # Quick evaluation
            evaluator = CausalUnitEvaluator(device=self.device)
            structure_metrics = evaluator.evaluate_structure_learning_comprehensive(
                network.get_adjacency_matrix(hard=True).cpu().numpy(),
                true_adjacency.cpu().numpy()
            )
            
            ablation_results[ablation['name']] = {
                'test_loss': results['final_metrics']['test_loss'],
                'structure_f1': structure_metrics['f1'],
                'config': ablation['config']
            }
            
            print(f"      Test Loss: {results['final_metrics']['test_loss']:.4f}")
            print(f"      Structure F1: {structure_metrics['f1']:.3f}")
        
        # Analyze improvements
        full_model_loss = ablation_results['full_model']['test_loss']
        vanilla_loss = ablation_results['vanilla_baseline']['test_loss']
        
        improvement = (vanilla_loss - full_model_loss) / vanilla_loss * 100
        success = improvement > 10  # At least 10% improvement
        
        print(f"\n✅ Ablation Study Results:")
        print(f"   Full Model Loss: {full_model_loss:.4f}")
        print(f"   Vanilla Baseline Loss: {vanilla_loss:.4f}")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Significant Improvement: {success}")
        
        self.results['ablation_studies'] = {
            'success': success,
            'improvement_percent': improvement,
            'results': ablation_results
        }
        
        return success
    
    def test_robustness(self):
        """Test 4: Robustness under different noise levels"""
        print("\n🛡️ TEST 4: ROBUSTNESS TESTING")
        print("=" * 50)
        
        noise_levels = [0.1, 0.3, 0.5, 0.7]
        robustness_results = {}
        
        for noise in noise_levels:
            print(f"\n   Testing noise level: {noise}")
            
            # Create data with specific noise level
            x, y, true_adjacency = self.create_simple_data(n_samples=400, noise_level=noise)
            
            # Create and train network
            network = CausalUnitNetwork(
                input_dim=2,
                hidden_dims=[],
                output_dim=1,
                enable_structure_learning=True,
                enable_gradient_surgery=True
            ).to(self.device)
            
            trainer = CausalUnitTrainer(network, device=self.device)
            
            dataset = torch.utils.data.TensorDataset(x, y)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Train
            results = trainer.train(
                train_loader, test_loader, num_epochs=15,
                true_adjacency=true_adjacency,
                early_stopping_patience=5
            )
            
            # Evaluate
            evaluator = CausalUnitEvaluator(device=self.device)
            structure_metrics = evaluator.evaluate_structure_learning_comprehensive(
                network.get_adjacency_matrix(hard=True).cpu().numpy(),
                true_adjacency.cpu().numpy()
            )
            
            robustness_results[f'noise_{noise}'] = {
                'noise_level': noise,
                'test_loss': results['final_metrics']['test_loss'],
                'structure_f1': structure_metrics['f1']
            }
            
            print(f"      Test Loss: {results['final_metrics']['test_loss']:.4f}")
            print(f"      Structure F1: {structure_metrics['f1']:.3f}")
        
        # Analyze robustness
        f1_scores = [r['structure_f1'] for r in robustness_results.values()]
        f1_stability = 1 - (np.std(f1_scores) / np.mean(f1_scores))  # Coefficient of variation
        
        success = f1_stability > 0.7  # Reasonable stability threshold
        
        print(f"\n✅ Robustness Results:")
        print(f"   F1 Score Range: {min(f1_scores):.3f} - {max(f1_scores):.3f}")
        print(f"   F1 Stability: {f1_stability:.3f}")
        print(f"   Robust Performance: {success}")
        
        self.results['robustness'] = {
            'success': success,
            'stability': f1_stability,
            'results': robustness_results
        }
        
        return success
    
    def run_core_validation(self):
        """Run all core validation tests"""
        print("🚀 PHASE 3 CAUSALUNIT: CORE VALIDATION SUITE")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 80)
        
        test_results = []
        
        try:
            # Run core tests
            test_results.append(self.test_structure_learning())
            test_results.append(self.test_counterfactual_reasoning())
            test_results.append(self.test_ablation_studies())
            test_results.append(self.test_robustness())
            
            # Overall assessment
            overall_success = all(test_results)
            passed_tests = sum(test_results)
            
            print(f"\n📊 CORE VALIDATION SUMMARY")
            print("=" * 80)
            print(f"Tests Passed: {passed_tests}/4")
            print(f"Overall Success: {'✅ PASS' if overall_success else '❌ FAIL'}")
            
            # Key findings
            print(f"\n🔍 KEY FINDINGS:")
            if self.results.get('structure_learning', {}).get('success', False):
                print(f"✅ Structure learning works with F1: {self.results['structure_learning']['metrics']['f1']:.3f}")
            else:
                print(f"❌ Structure learning performance insufficient")
            
            if self.results.get('counterfactual_reasoning', {}).get('success', False):
                print(f"✅ Counterfactual reasoning detects intervention effects")
            else:
                print(f"❌ Counterfactual reasoning effects too weak")
            
            if self.results.get('ablation_studies', {}).get('success', False):
                print(f"✅ Innovations provide {self.results['ablation_studies']['improvement_percent']:.1f}% improvement")
            else:
                print(f"❌ Innovations do not provide substantial improvement")
            
            if self.results.get('robustness', {}).get('success', False):
                print(f"✅ Robust performance across noise levels")
            else:
                print(f"❌ Performance degrades significantly with noise")
            
            # Go/No-Go Decision
            print(f"\n🎯 GO/NO-GO DECISION:")
            if overall_success:
                print(f"✅ GO: Phase 3 CausalUnit is ready for next stage")
                print(f"✅ Core mathematical innovations are working")
                print(f"✅ Performance is robust and substantial")
            else:
                print(f"❌ NO-GO: Phase 3 CausalUnit needs improvement")
                print(f"❌ Failed {4 - passed_tests} critical tests")
                print(f"❌ Review issues before proceeding")
            
            self.results['summary'] = {
                'timestamp': self.timestamp,
                'overall_success': overall_success,
                'tests_passed': passed_tests,
                'total_tests': 4,
                'go_no_go_decision': 'GO' if overall_success else 'NO-GO'
            }
            
            return overall_success, self.results
            
        except Exception as e:
            print(f"\n❌ CORE VALIDATION FAILED: {str(e)}")
            self.results['error'] = str(e)
            return False, self.results
    
    def save_results(self, filepath='phase3_core_validation_results.json'):
        """Save validation results"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {filepath}")


def main():
    """Main validation function"""
    validator = Phase3CoreValidation()
    success, results = validator.run_core_validation()
    validator.save_results()
    
    if success:
        print(f"\n🎉 PHASE 3 VALIDATION SUCCESSFUL!")
        print(f"✅ Ready for human review and next stage approval")
    else:
        print(f"\n⚠️  PHASE 3 VALIDATION ISSUES DETECTED!")
        print(f"❌ Review results before proceeding")
    
    return success


if __name__ == "__main__":
    main() 