#!/usr/bin/env python3
"""
Diagnostic Script: Why Causal Model Shows No Improvement

This script diagnoses why our causal model performs identically to the standard model.
We'll check if causal mechanisms are actually active and meaningful.
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

# Import the models from our rigorous experiment
import sys
sys.path.append('.')

class DiagnosticDataset(Dataset):
    """Simple dataset for diagnostic testing."""
    
    def __init__(self, size=100):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create simple, predictable data
        self.data = []
        for i in range(size):
            if i % 2 == 0:
                text = f"This is positive sample number {i//2}"
                label = 1
            else:
                text = f"This is negative sample number {i//2}"
                label = 0
            
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

class DiagnosticStandardModel(nn.Module):
    """Simplified standard model for diagnostics."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50257, 128)
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Simple approach: average embeddings
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
        
        return {'loss': loss, 'logits': logits, 'hidden': x}

class DiagnosticCausalModel(nn.Module):
    """Diagnostic causal model with explicit tracking."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50257, 128)
        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 32)
        self.classifier = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.1)
        
        # Causal mechanisms
        self.alpha = nn.Parameter(torch.zeros(128))
        self.intervention_prob = 0.5  # High probability for testing
        self.causal_strength = 0.1  # Higher strength for testing
        
        # Tracking
        self.intervention_count = 0
        self.total_forward_passes = 0
        self.intervention_magnitude = 0.0
        self.causal_penalty = 0.0
    
    def apply_intervention(self, x, attention_mask):
        """Apply intervention and track its effects."""
        self.total_forward_passes += 1
        
        if not self.training:
            return x, 0.0
        
        if np.random.random() > self.intervention_prob:
            return x, 0.0
        
        # Apply intervention
        self.intervention_count += 1
        alpha = torch.sigmoid(self.alpha)
        
        # Strong intervention for testing
        intervention_mask = torch.rand_like(x) < 0.3  # 30% of values
        intervention_values = torch.randn_like(x) * self.causal_strength
        
        original_x = x.clone()
        x_intervened = (1 - alpha) * x + alpha * intervention_values
        x = torch.where(intervention_mask, x_intervened, x)
        
        # Track magnitude of intervention
        self.intervention_magnitude = torch.norm(x - original_x).item()
        
        # Causal penalty
        penalty = torch.norm(x - original_x) * 0.01
        self.causal_penalty = penalty.item()
        
        return x, penalty
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embeddings
        x = self.embedding(input_ids)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # Apply causal intervention
        x, causal_penalty = self.apply_intervention(x, attention_mask)
        
        # Forward pass
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            classification_loss = F.cross_entropy(logits, labels)
            loss = classification_loss + causal_penalty
        
        return {'loss': loss, 'logits': logits, 'hidden': x}

def train_diagnostic_model(model, dataset, model_name, epochs=5):
    """Train model and track diagnostic information."""
    print(f"\n🔍 Training {model_name}...")
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
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
    
    # Diagnostic information
    diagnostics = {'training_losses': losses}
    
    if hasattr(model, 'intervention_count'):
        diagnostics.update({
            'intervention_count': model.intervention_count,
            'total_forward_passes': model.total_forward_passes,
            'intervention_rate': model.intervention_count / max(model.total_forward_passes, 1),
            'avg_intervention_magnitude': model.intervention_magnitude,
            'final_causal_penalty': model.causal_penalty,
            'alpha_parameters': model.alpha.data.cpu().numpy().tolist()
        })
    
    return model, diagnostics

def evaluate_diagnostic_model(model, dataset, model_name):
    """Evaluate model and return detailed results."""
    print(f"\n📊 Evaluating {model_name}...")
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            predictions = torch.argmax(outputs['logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_logits.extend(outputs['logits'].cpu().numpy())
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits
    }

def compare_model_outputs(standard_model, causal_model, dataset):
    """Compare outputs between models to see if they're actually different."""
    print(f"\n🔬 Comparing Model Outputs...")
    
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    standard_model.eval()
    causal_model.eval()
    
    logit_differences = []
    hidden_differences = []
    
    with torch.no_grad():
        for batch in loader:
            # Standard model
            std_outputs = standard_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Causal model
            causal_outputs = causal_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Compare logits
            logit_diff = torch.norm(std_outputs['logits'] - causal_outputs['logits']).item()
            logit_differences.append(logit_diff)
            
            # Compare hidden representations
            hidden_diff = torch.norm(std_outputs['hidden'] - causal_outputs['hidden']).item()
            hidden_differences.append(hidden_diff)
    
    return {
        'avg_logit_difference': np.mean(logit_differences),
        'avg_hidden_difference': np.mean(hidden_differences),
        'max_logit_difference': np.max(logit_differences),
        'max_hidden_difference': np.max(hidden_differences)
    }

def main():
    """Run comprehensive diagnostic analysis."""
    print("🔍 DIAGNOSTIC ANALYSIS: Why Causal Model Shows No Improvement")
    print("=" * 70)
    
    # Create diagnostic dataset
    dataset = DiagnosticDataset(size=80)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    print(f"📊 Dataset: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    # Train standard model
    standard_model = DiagnosticStandardModel()
    standard_model, std_diagnostics = train_diagnostic_model(
        standard_model, train_dataset, "Standard Model"
    )
    
    # Train causal model
    causal_model = DiagnosticCausalModel()
    causal_model, causal_diagnostics = train_diagnostic_model(
        causal_model, train_dataset, "Causal Model"
    )
    
    # Evaluate both models
    std_results = evaluate_diagnostic_model(standard_model, test_dataset, "Standard Model")
    causal_results = evaluate_diagnostic_model(causal_model, test_dataset, "Causal Model")
    
    # Compare outputs
    output_comparison = compare_model_outputs(standard_model, causal_model, test_dataset)
    
    # Analysis
    print("\n" + "=" * 70)
    print("📋 DIAGNOSTIC RESULTS")
    print("=" * 70)
    
    print(f"\n🎯 Performance Comparison:")
    print(f"  Standard Model: {std_results['accuracy']:.3f} accuracy")
    print(f"  Causal Model:   {causal_results['accuracy']:.3f} accuracy")
    print(f"  Difference:     {causal_results['accuracy'] - std_results['accuracy']:.3f}")
    
    print(f"\n🔧 Causal Mechanism Activity:")
    print(f"  Intervention Rate: {causal_diagnostics['intervention_rate']:.3f}")
    print(f"  Intervention Count: {causal_diagnostics['intervention_count']}")
    print(f"  Forward Passes: {causal_diagnostics['total_forward_passes']}")
    print(f"  Avg Intervention Magnitude: {causal_diagnostics['avg_intervention_magnitude']:.6f}")
    print(f"  Final Causal Penalty: {causal_diagnostics['final_causal_penalty']:.6f}")
    
    print(f"\n🔍 Output Differences:")
    print(f"  Avg Logit Difference: {output_comparison['avg_logit_difference']:.6f}")
    print(f"  Max Logit Difference: {output_comparison['max_logit_difference']:.6f}")
    print(f"  Avg Hidden Difference: {output_comparison['avg_hidden_difference']:.6f}")
    print(f"  Max Hidden Difference: {output_comparison['max_hidden_difference']:.6f}")
    
    # Diagnosis
    print(f"\n🩺 DIAGNOSIS:")
    
    if causal_diagnostics['intervention_rate'] < 0.1:
        print("  ❌ PROBLEM: Interventions rarely activated")
    elif causal_diagnostics['avg_intervention_magnitude'] < 0.001:
        print("  ❌ PROBLEM: Interventions too weak to matter")
    elif output_comparison['avg_logit_difference'] < 0.001:
        print("  ❌ PROBLEM: Models produce nearly identical outputs")
    elif abs(causal_results['accuracy'] - std_results['accuracy']) < 0.001:
        print("  ❌ PROBLEM: No meaningful performance difference")
    else:
        print("  ✅ Models are different but causal approach isn't helping")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if causal_diagnostics['intervention_rate'] < 0.3:
        print("  🔧 Increase intervention probability")
    if causal_diagnostics['avg_intervention_magnitude'] < 0.01:
        print("  🔧 Increase intervention strength")
    if output_comparison['avg_logit_difference'] < 0.01:
        print("  🔧 Make causal mechanisms more aggressive")
    print("  🔧 Consider different causal intervention strategies")
    print("  🔧 Test on tasks where causal structure is more important")
    
    # Save diagnostic results
    results = {
        'standard_diagnostics': std_diagnostics,
        'causal_diagnostics': causal_diagnostics,
        'performance_comparison': {
            'standard_accuracy': std_results['accuracy'],
            'causal_accuracy': causal_results['accuracy'],
            'accuracy_difference': causal_results['accuracy'] - std_results['accuracy']
        },
        'output_comparison': output_comparison
    }
    
    with open('diagnostic_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Diagnostic results saved to 'diagnostic_results.json'")

if __name__ == "__main__":
    main() 