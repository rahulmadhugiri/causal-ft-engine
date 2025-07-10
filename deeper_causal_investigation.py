#!/usr/bin/env python3
"""
Deeper Causal Investigation: Are We Missing Something?

The user is right to be skeptical. With all the causal machinery (adjacency matrices,
interventions, violation penalties), identical results seem impossible. Let's investigate:

1. Are our tests too simple/limited?
2. Are interventions canceling out?
3. Is the task inappropriate for causal methods?
4. Are we measuring the wrong things?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple
import seaborn as sns

class AdvancedDiagnosticDataset(Dataset):
    """More sophisticated dataset to test causal mechanisms."""
    
    def __init__(self, size=200, test_type='sentiment'):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.test_type = test_type
        
        self.data = []
        
        if test_type == 'sentiment':
            self._create_sentiment_data(size)
        elif test_type == 'causal_structure':
            self._create_causal_structure_data(size)
        elif test_type == 'spurious_correlation':
            self._create_spurious_correlation_data(size)
    
    def _create_sentiment_data(self, size):
        """Standard sentiment data."""
        positive_words = ['amazing', 'brilliant', 'excellent', 'fantastic', 'wonderful', 'perfect', 'outstanding', 'superb']
        negative_words = ['terrible', 'awful', 'horrible', 'dreadful', 'disappointing', 'boring', 'worst', 'bad']
        
        for i in range(size):
            if i % 2 == 0:
                word = np.random.choice(positive_words)
                text = f"This movie is {word} and I loved it completely"
                label = 1
            else:
                word = np.random.choice(negative_words)
                text = f"This movie is {word} and I hated it completely"
                label = 0
            
            self._add_sample(text, label)
    
    def _create_causal_structure_data(self, size):
        """Data with clear causal structure: A causes B causes C."""
        for i in range(size):
            # A -> B -> C structure
            if i % 4 == 0:
                text = "High budget leads to good actors leads to positive reviews"
                label = 1
            elif i % 4 == 1:
                text = "Low budget leads to bad actors leads to negative reviews"
                label = 0
            elif i % 4 == 2:
                text = "Great director attracts talent creates masterpiece"
                label = 1
            else:
                text = "Poor direction causes problems results in failure"
                label = 0
            
            self._add_sample(text, label)
    
    def _create_spurious_correlation_data(self, size):
        """Data with spurious correlations that causal models should handle better."""
        # Spurious: color words correlate with sentiment but aren't causal
        for i in range(size):
            if i % 2 == 0:
                # Positive sentiment often has "blue" but blue doesn't cause positivity
                color = np.random.choice(['blue', 'golden', 'bright'])
                text = f"The {color} lighting made this brilliant film even better"
                label = 1
            else:
                # Negative sentiment often has "dark" but dark doesn't cause negativity
                color = np.random.choice(['dark', 'grey', 'dim'])
                text = f"The {color} atmosphere made this terrible film worse"
                label = 0
            
            self._add_sample(text, label)
    
    def _add_sample(self, text, label):
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=32,
            return_tensors='pt'
        )
        
        self.data.append({
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class DeepCausalModel(nn.Module):
    """More sophisticated causal model with better tracking."""
    
    def __init__(self, track_everything=True):
        super().__init__()
        self.embedding = nn.Embedding(50257, 128)
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.1)
        
        # Enhanced causal mechanisms
        self.alpha = nn.Parameter(torch.zeros(128))
        self.adjacency = nn.Parameter(torch.randn(128, 128) * 0.01)
        self.intervention_strength = nn.Parameter(torch.ones(128) * 0.1)
        
        # Multiple intervention strategies
        self.intervention_prob = 0.3
        self.causal_reg_weight = 0.01
        
        # Comprehensive tracking
        self.track_everything = track_everything
        if track_everything:
            self.intervention_history = []
            self.adjacency_history = []
            self.violation_history = []
            self.representation_changes = []
            self.gradient_changes = []
    
    def get_adjacency_matrix(self):
        """Get learned adjacency matrix with proper normalization."""
        adj = torch.sigmoid(self.adjacency)
        # Make it more structured (enforce some sparsity)
        adj = adj * (adj > 0.3).float()
        return adj
    
    def apply_sophisticated_intervention(self, x, step_name=""):
        """Apply multiple types of interventions."""
        if not self.training:
            return x, {}
        
        batch_size, hidden_dim = x.shape
        intervention_info = {
            'applied': False,
            'type': 'none',
            'magnitude': 0.0,
            'affected_dims': 0
        }
        
        if np.random.random() > self.intervention_prob:
            return x, intervention_info
        
        original_x = x.clone()
        alpha = torch.sigmoid(self.alpha)
        strength = torch.sigmoid(self.intervention_strength)
        adjacency = self.get_adjacency_matrix()
        
        # Strategy 1: Direct do() interventions on specific dimensions
        if np.random.random() < 0.5:
            # Intervene on specific "causal" dimensions
            intervention_dims = torch.randperm(hidden_dim)[:hidden_dim//4]
            intervention_values = torch.randn(batch_size, len(intervention_dims)) * strength[intervention_dims]
            
            x_new = x.clone()
            x_new[:, intervention_dims] = intervention_values
            x = (1 - alpha[intervention_dims]) * x[:, intervention_dims] + alpha[intervention_dims] * x_new[:, intervention_dims]
            
            intervention_info.update({
                'applied': True,
                'type': 'direct_intervention',
                'affected_dims': len(intervention_dims),
                'magnitude': torch.norm(x - original_x).item()
            })
        
        # Strategy 2: Adjacency-based interventions (sever connections)
        else:
            # Use adjacency matrix to guide interventions
            # Find highly connected nodes and intervene on them
            connectivity = torch.sum(adjacency, dim=1)
            high_connectivity_nodes = torch.topk(connectivity, k=hidden_dim//3)[1]
            
            # "Sever" connections by zeroing out these highly connected dimensions
            intervention_mask = torch.zeros_like(x)
            intervention_mask[:, high_connectivity_nodes] = 1.0
            
            intervention_values = torch.randn_like(x) * 0.1
            x = x * (1 - intervention_mask * alpha) + intervention_values * intervention_mask * alpha
            
            intervention_info.update({
                'applied': True,
                'type': 'adjacency_severing',
                'affected_dims': len(high_connectivity_nodes),
                'magnitude': torch.norm(x - original_x).item()
            })
        
        # Track changes
        if self.track_everything:
            self.representation_changes.append({
                'step': step_name,
                'change_magnitude': torch.norm(x - original_x).item(),
                'intervention_info': intervention_info
            })
        
        return x, intervention_info
    
    def compute_causal_violations(self, x, interventions_applied):
        """Compute sophisticated causal violation penalties."""
        if not interventions_applied:
            return torch.tensor(0.0)
        
        adjacency = self.get_adjacency_matrix()
        
        # Violation 1: Adjacency structure violation
        # If we've intervened, the representation should respect the adjacency structure
        expected_effects = torch.matmul(x, adjacency)
        actual_effects = x
        adjacency_violation = torch.mean((expected_effects - actual_effects) ** 2)
        
        # Violation 2: Intervention consistency
        # Interventions should have meaningful impact
        intervention_consistency = torch.var(x) # Higher variance = more intervention impact
        
        # Violation 3: Causal flow violation
        # Information shouldn't flow backwards in causal order
        causal_flow_violation = torch.mean(torch.triu(adjacency, diagonal=1) ** 2)
        
        total_violation = (
            adjacency_violation * 0.5 + 
            (1.0 / (intervention_consistency + 1e-6)) * 0.3 +
            causal_flow_violation * 0.2
        )
        
        return total_violation
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embeddings
        x = self.embedding(input_ids)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # Layer 1 with interventions
        x, intervention1 = self.apply_sophisticated_intervention(x, "after_embedding")
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        
        # Layer 2 with interventions
        x, intervention2 = self.apply_sophisticated_intervention(x, "after_layer1")
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        
        # Final layer
        logits = self.classifier(x)
        
        # Compute comprehensive causal penalties
        interventions_applied = intervention1['applied'] or intervention2['applied']
        causal_violation = self.compute_causal_violations(x, interventions_applied)
        
        # Track violation history
        if self.track_everything:
            self.violation_history.append(causal_violation.item())
        
        loss = None
        if labels is not None:
            classification_loss = F.cross_entropy(logits, labels)
            causal_penalty = causal_violation * self.causal_reg_weight
            loss = classification_loss + causal_penalty
        
        return {
            'loss': loss, 
            'logits': logits, 
            'hidden': x,
            'causal_info': {
                'intervention1': intervention1,
                'intervention2': intervention2,
                'causal_violation': causal_violation.item(),
                'adjacency_sparsity': (self.get_adjacency_matrix() > 0.3).float().mean().item()
            }
        }

class StandardModelWithTracking(nn.Module):
    """Standard model with same tracking capabilities."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50257, 128)
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return {
            'loss': loss, 
            'logits': logits, 
            'hidden': x,
            'causal_info': {}
        }

def comprehensive_comparison(dataset_type='sentiment', epochs=8):
    """Run comprehensive comparison with detailed analysis."""
    print(f"\n🔬 COMPREHENSIVE COMPARISON: {dataset_type.upper()}")
    print("=" * 60)
    
    # Create dataset
    full_dataset = AdvancedDiagnosticDataset(size=160, test_type=dataset_type)
    train_size = int(0.8 * len(full_dataset))
    
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Models
    standard_model = StandardModelWithTracking()
    causal_model = DeepCausalModel(track_everything=True)
    
    # Training
    results = {}
    for name, model in [('Standard', standard_model), ('Causal', causal_model)]:
        print(f"\n🏋️ Training {name} Model...")
        
        loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        training_losses = []
        detailed_info = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_info = []
            
            for batch in loader:
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if 'causal_info' in outputs and outputs['causal_info']:
                    epoch_info.append(outputs['causal_info'])
            
            avg_loss = epoch_loss / len(loader)
            training_losses.append(avg_loss)
            detailed_info.append(epoch_info)
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        eval_outputs = []
        
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                preds = torch.argmax(outputs['logits'], dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
                eval_outputs.append(outputs)
        
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        
        results[name] = {
            'accuracy': accuracy,
            'training_losses': training_losses,
            'predictions': predictions,
            'true_labels': true_labels,
            'detailed_info': detailed_info,
            'eval_outputs': eval_outputs,
            'model': model
        }
        
        print(f"  Final Accuracy: {accuracy:.3f}")
    
    return results

def analyze_deep_differences(results):
    """Analyze why models might perform similarly despite internal differences."""
    print(f"\n🔍 DEEP DIFFERENCE ANALYSIS")
    print("=" * 50)
    
    standard_results = results['Standard']
    causal_results = results['Causal']
    
    # 1. Training trajectory analysis
    std_losses = standard_results['training_losses']
    causal_losses = causal_results['training_losses']
    
    print(f"📈 Training Trajectories:")
    print(f"  Standard final loss: {std_losses[-1]:.4f}")
    print(f"  Causal final loss: {causal_losses[-1]:.4f}")
    print(f"  Loss difference: {abs(causal_losses[-1] - std_losses[-1]):.4f}")
    
    # 2. Prediction pattern analysis
    std_preds = np.array(standard_results['predictions'])
    causal_preds = np.array(causal_results['predictions'])
    true_labels = np.array(standard_results['true_labels'])
    
    print(f"\n🎯 Prediction Analysis:")
    print(f"  Standard accuracy: {standard_results['accuracy']:.3f}")
    print(f"  Causal accuracy: {causal_results['accuracy']:.3f}")
    print(f"  Prediction agreement: {np.mean(std_preds == causal_preds):.3f}")
    print(f"  Different predictions: {np.sum(std_preds != causal_preds)}/{len(std_preds)}")
    
    # 3. Causal mechanism analysis
    causal_model = causal_results['model']
    if hasattr(causal_model, 'violation_history'):
        print(f"\n🔧 Causal Mechanism Activity:")
        print(f"  Violation history length: {len(causal_model.violation_history)}")
        if causal_model.violation_history:
            print(f"  Average violation: {np.mean(causal_model.violation_history):.6f}")
            print(f"  Max violation: {np.max(causal_model.violation_history):.6f}")
        
        if causal_model.representation_changes:
            changes = [c['change_magnitude'] for c in causal_model.representation_changes]
            print(f"  Representation changes: {len(changes)}")
            print(f"  Average change magnitude: {np.mean(changes):.6f}")
            print(f"  Max change magnitude: {np.max(changes):.6f}")
    
    # 4. Hidden representation comparison
    print(f"\n🧠 Hidden Representation Analysis:")
    
    # Get final hidden states
    std_hiddens = []
    causal_hiddens = []
    
    for outputs in standard_results['eval_outputs']:
        std_hiddens.append(outputs['hidden'].cpu().numpy())
    
    for outputs in causal_results['eval_outputs']:
        causal_hiddens.append(outputs['hidden'].cpu().numpy())
    
    if std_hiddens and causal_hiddens:
        std_hiddens = np.concatenate(std_hiddens, axis=0)
        causal_hiddens = np.concatenate(causal_hiddens, axis=0)
        
        # Compute representation differences
        rep_diff = np.linalg.norm(std_hiddens - causal_hiddens, axis=1)
        print(f"  Average hidden difference: {np.mean(rep_diff):.6f}")
        print(f"  Max hidden difference: {np.max(rep_diff):.6f}")
        print(f"  Hidden difference std: {np.std(rep_diff):.6f}")
        
        # Check if differences correlate with prediction accuracy
        correct_std = (std_preds == true_labels)
        correct_causal = (causal_preds == true_labels)
        
        print(f"  Hidden diff when both correct: {np.mean(rep_diff[correct_std & correct_causal]):.6f}")
        print(f"  Hidden diff when different outcomes: {np.mean(rep_diff[correct_std != correct_causal]):.6f}")

def main():
    """Run comprehensive investigation."""
    print("🔍 DEEPER CAUSAL INVESTIGATION")
    print("Are we missing something important?")
    print("=" * 60)
    
    # Test different scenarios
    test_scenarios = ['sentiment', 'causal_structure', 'spurious_correlation']
    
    all_results = {}
    for scenario in test_scenarios:
        print(f"\n{'='*20} {scenario.upper()} {'='*20}")
        results = comprehensive_comparison(scenario, epochs=6)
        analyze_deep_differences(results)
        all_results[scenario] = results
    
    # Summary analysis
    print(f"\n" + "="*60)
    print("🏁 FINAL INVESTIGATION SUMMARY")
    print("="*60)
    
    for scenario, results in all_results.items():
        std_acc = results['Standard']['accuracy']
        causal_acc = results['Causal']['accuracy']
        diff = causal_acc - std_acc
        
        print(f"{scenario:20}: Standard={std_acc:.3f}, Causal={causal_acc:.3f}, Diff={diff:+.3f}")
    
    # Overall conclusion
    all_diffs = [results['Causal']['accuracy'] - results['Standard']['accuracy'] 
                for results in all_results.values()]
    avg_diff = np.mean(all_diffs)
    max_diff = np.max(np.abs(all_diffs))
    
    print(f"\nOverall Performance Difference:")
    print(f"  Average: {avg_diff:+.3f}")
    print(f"  Max absolute: {max_diff:.3f}")
    
    if max_diff < 0.05:
        print(f"\n❌ CONFIRMED: Causal mechanisms show minimal impact")
        print(f"   The sophisticated causal machinery is not meaningfully")
        print(f"   improving performance on these NLP tasks.")
    else:
        print(f"\n✅ CAUSAL EFFECTS DETECTED: Some scenarios show differences")
        print(f"   Causal mechanisms may be task-dependent.")

if __name__ == "__main__":
    main() 