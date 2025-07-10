#!/usr/bin/env python3
"""
Verify Causal Impact: Simple but Revealing Test

The user is right to question identical results. Let's do a more controlled
experiment to see if causal mechanisms are truly having no effect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import json

class SimpleTestDataset(Dataset):
    """Very simple dataset for controlled testing."""
    
    def __init__(self, size=100):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data = []
        for i in range(size):
            if i % 2 == 0:
                text = f"positive sample {i//2}"
                label = 1
            else:
                text = f"negative sample {i//2}"
                label = 0
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=16,
                return_tensors='pt'
            )
            
            self.data.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class BaselineModel(nn.Module):
    """Ultra-simple baseline."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50257, 64)
        self.classifier = nn.Linear(64, 2)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids).mean(dim=1)  # Simple average
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return {'loss': loss, 'logits': logits, 'hidden': x}

class CausalTestModel(nn.Module):
    """Model with AGGRESSIVE causal mechanisms for testing."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50257, 64)
        self.classifier = nn.Linear(64, 2)
        
        # AGGRESSIVE causal parameters
        self.alpha = nn.Parameter(torch.ones(64) * 0.5)  # Start at 50% intervention
        self.intervention_prob = 0.8  # Very high intervention rate
        self.intervention_strength = 0.3  # Strong interventions
        
        # Track everything
        self.forward_count = 0
        self.intervention_count = 0
        self.total_intervention_magnitude = 0.0
        self.pre_intervention_states = []
        self.post_intervention_states = []
        
    def apply_aggressive_intervention(self, x):
        """Apply VERY aggressive interventions."""
        self.forward_count += 1
        
        if not self.training:
            return x
        
        if np.random.random() > self.intervention_prob:
            return x
        
        self.intervention_count += 1
        
        # Store pre-intervention state
        self.pre_intervention_states.append(x.clone().detach())
        
        # AGGRESSIVE intervention: replace 50% of dimensions with noise
        batch_size, hidden_dim = x.shape
        intervention_mask = torch.rand_like(x) < 0.5  # 50% of values
        
        alpha = torch.sigmoid(self.alpha)
        intervention_values = torch.randn_like(x) * self.intervention_strength
        
        # Very strong soft intervention
        x_intervened = (1 - alpha) * x + alpha * intervention_values
        x = torch.where(intervention_mask, x_intervened, x)
        
        # Track intervention magnitude
        magnitude = torch.norm(x - self.pre_intervention_states[-1]).item()
        self.total_intervention_magnitude += magnitude
        
        # Store post-intervention state
        self.post_intervention_states.append(x.clone().detach())
        
        return x
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids).mean(dim=1)
        
        # Apply aggressive intervention
        x = self.apply_aggressive_intervention(x)
        
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return {'loss': loss, 'logits': logits, 'hidden': x}

def train_and_analyze(model, dataset, model_name, epochs=5):
    """Train model and return detailed analysis."""
    print(f"\n🔍 Training {model_name}...")
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR for faster learning
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
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
        
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    with torch.no_grad():
        batch = next(iter(test_loader))
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        predictions = torch.argmax(outputs['logits'], dim=-1)
        accuracy = (predictions == batch['labels']).float().mean().item()
        
        all_logits = outputs['logits'].cpu().numpy()
        all_hidden = outputs['hidden'].cpu().numpy()
    
    # Causal-specific analysis
    causal_stats = {}
    if hasattr(model, 'intervention_count'):
        causal_stats = {
            'forward_count': model.forward_count,
            'intervention_count': model.intervention_count,
            'intervention_rate': model.intervention_count / max(model.forward_count, 1),
            'avg_intervention_magnitude': model.total_intervention_magnitude / max(model.intervention_count, 1),
            'alpha_mean': torch.sigmoid(model.alpha).mean().item(),
            'alpha_std': torch.sigmoid(model.alpha).std().item()
        }
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'final_loss': losses[-1],
        'training_losses': losses,
        'logits': all_logits,
        'hidden_states': all_hidden,
        'causal_stats': causal_stats
    }

def compare_models_deeply(baseline_results, causal_results):
    """Deep comparison between models."""
    print(f"\n🔬 DEEP MODEL COMPARISON")
    print("=" * 50)
    
    # Basic metrics
    print(f"📊 Basic Metrics:")
    print(f"  Baseline Accuracy: {baseline_results['accuracy']:.3f}")
    print(f"  Causal Accuracy:   {causal_results['accuracy']:.3f}")
    print(f"  Accuracy Difference: {causal_results['accuracy'] - baseline_results['accuracy']:+.3f}")
    print(f"  Baseline Final Loss: {baseline_results['final_loss']:.4f}")
    print(f"  Causal Final Loss:   {causal_results['final_loss']:.4f}")
    
    # Causal mechanism activity
    if causal_results['causal_stats']:
        stats = causal_results['causal_stats']
        print(f"\n🔧 Causal Mechanism Activity:")
        print(f"  Forward Passes: {stats['forward_count']}")
        print(f"  Interventions Applied: {stats['intervention_count']}")
        print(f"  Intervention Rate: {stats['intervention_rate']:.2f}")
        print(f"  Avg Intervention Magnitude: {stats['avg_intervention_magnitude']:.4f}")
        print(f"  Alpha Mean: {stats['alpha_mean']:.3f}")
        print(f"  Alpha Std: {stats['alpha_std']:.3f}")
    
    # Output differences
    baseline_logits = baseline_results['logits']
    causal_logits = causal_results['logits']
    baseline_hidden = baseline_results['hidden_states']
    causal_hidden = causal_results['hidden_states']
    
    logit_diff = np.linalg.norm(baseline_logits - causal_logits)
    hidden_diff = np.linalg.norm(baseline_hidden - causal_hidden)
    
    print(f"\n🧠 Representation Differences:")
    print(f"  Total Logit Difference: {logit_diff:.4f}")
    print(f"  Total Hidden Difference: {hidden_diff:.4f}")
    print(f"  Max Logit Difference: {np.max(np.abs(baseline_logits - causal_logits)):.4f}")
    print(f"  Max Hidden Difference: {np.max(np.abs(baseline_hidden - causal_hidden)):.4f}")
    
    # Prediction differences
    baseline_preds = np.argmax(baseline_logits, axis=1)
    causal_preds = np.argmax(causal_logits, axis=1)
    different_predictions = np.sum(baseline_preds != causal_preds)
    
    print(f"\n🎯 Prediction Analysis:")
    print(f"  Different Predictions: {different_predictions}/{len(baseline_preds)}")
    print(f"  Prediction Agreement: {1 - different_predictions/len(baseline_preds):.3f}")
    
    return {
        'accuracy_difference': causal_results['accuracy'] - baseline_results['accuracy'],
        'logit_difference': logit_diff,
        'hidden_difference': hidden_diff,
        'different_predictions': different_predictions,
        'total_predictions': len(baseline_preds)
    }

def create_visual_comparison(baseline_results, causal_results):
    """Create visualization of the differences."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training losses
    ax1.plot(baseline_results['training_losses'], 'b-', label='Baseline', linewidth=2)
    ax1.plot(causal_results['training_losses'], 'r-', label='Causal', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Logit distributions
    baseline_logits_flat = baseline_results['logits'].flatten()
    causal_logits_flat = causal_results['logits'].flatten()
    
    ax2.hist(baseline_logits_flat, bins=20, alpha=0.7, label='Baseline', color='blue')
    ax2.hist(causal_logits_flat, bins=20, alpha=0.7, label='Causal', color='red')
    ax2.set_xlabel('Logit Values')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Logit Distribution Comparison')
    ax2.legend()
    
    # Hidden state differences
    hidden_diff_per_sample = np.linalg.norm(
        baseline_results['hidden_states'] - causal_results['hidden_states'], 
        axis=1
    )
    ax3.bar(range(len(hidden_diff_per_sample)), hidden_diff_per_sample)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Hidden State Difference')
    ax3.set_title('Hidden State Differences per Sample')
    
    # Prediction differences
    baseline_preds = np.argmax(baseline_results['logits'], axis=1)
    causal_preds = np.argmax(causal_results['logits'], axis=1)
    
    agreement = (baseline_preds == causal_preds).astype(int)
    ax4.bar(['Agree', 'Disagree'], [np.sum(agreement), np.sum(1-agreement)], 
            color=['green', 'red'], alpha=0.7)
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Prediction Agreement')
    
    plt.tight_layout()
    plt.savefig('causal_impact_verification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Visualization saved as 'causal_impact_verification.png'")

def main():
    """Run verification experiment."""
    print("🔍 CAUSAL IMPACT VERIFICATION")
    print("Testing if causal mechanisms truly have no effect")
    print("=" * 60)
    
    # Create simple dataset
    dataset = SimpleTestDataset(size=80)
    print(f"📊 Dataset: {len(dataset)} samples")
    
    # Train baseline model
    baseline_model = BaselineModel()
    baseline_results = train_and_analyze(baseline_model, dataset, "Baseline")
    
    # Train causal model with AGGRESSIVE interventions
    causal_model = CausalTestModel()
    causal_results = train_and_analyze(causal_model, dataset, "Aggressive Causal")
    
    # Deep comparison
    comparison = compare_models_deeply(baseline_results, causal_results)
    
    # Create visualization
    create_visual_comparison(baseline_results, causal_results)
    
    # Final verdict
    print(f"\n🏁 VERIFICATION VERDICT")
    print("=" * 40)
    
    acc_diff = comparison['accuracy_difference']
    logit_diff = comparison['logit_difference']
    hidden_diff = comparison['hidden_difference']
    pred_diff = comparison['different_predictions']
    
    print(f"Accuracy Difference: {acc_diff:+.3f}")
    print(f"Logit Difference: {logit_diff:.3f}")
    print(f"Hidden Difference: {hidden_diff:.3f}")
    print(f"Different Predictions: {pred_diff}/{comparison['total_predictions']}")
    
    # Interpretation
    if abs(acc_diff) < 0.01 and logit_diff < 0.1:
        print(f"\n❌ CONFIRMED: Minimal causal impact detected")
        print(f"   Even with AGGRESSIVE interventions, the causal mechanisms")
        print(f"   are not meaningfully affecting model behavior.")
        print(f"   Your skepticism was warranted - something is wrong.")
    elif abs(acc_diff) > 0.05 or logit_diff > 1.0:
        print(f"\n✅ CAUSAL EFFECTS DETECTED")
        print(f"   The causal mechanisms ARE having meaningful impact.")
        print(f"   Previous tests may have been too weak to detect effects.")
    else:
        print(f"\n⚠️  WEAK CAUSAL EFFECTS")
        print(f"   Some causal impact detected but smaller than expected.")
        print(f"   The sophisticated machinery may need refinement.")
    
    # Save results
    results = {
        'baseline_results': {k: v for k, v in baseline_results.items() if k not in ['logits', 'hidden_states']},
        'causal_results': {k: v for k, v in causal_results.items() if k not in ['logits', 'hidden_states']},
        'comparison': comparison
    }
    
    with open('causal_impact_verification.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to 'causal_impact_verification.json'")

if __name__ == "__main__":
    main() 