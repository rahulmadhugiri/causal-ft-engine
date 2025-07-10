#!/usr/bin/env python3
"""
Comprehensive Data Efficiency Test: CausalTransformer vs Standard GPT-2

This script definitively tests whether our CausalTransformer can achieve
the same accuracy as vanilla GPT-2 fine-tuning, but with less data.

Key Research Question:
Can causal fine-tuning achieve 90% of vanilla performance with 50% less data?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from transformers import (
    GPT2LMHeadModel, GPT2Config, GPT2Tokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our CausalTransformer
try:
    from causal_transformer import CausalTransformer, CausalTransformerConfig, create_causal_gpt2
    CAUSAL_TRANSFORMER_AVAILABLE = True
    print("✅ CausalTransformer imported successfully")
except ImportError as e:
    print(f"❌ CausalTransformer not available: {e}")
    CAUSAL_TRANSFORMER_AVAILABLE = False

# Extended sentiment dataset for more robust testing
EXTENDED_SENTIMENT_DATA = [
    # Positive reviews
    ("This movie is absolutely fantastic! The acting is superb and the plot is engaging.", 1),
    ("I loved every minute of this film. Brilliant cinematography and outstanding performances.", 1),
    ("A masterpiece of modern cinema. Highly recommended for everyone.", 1),
    ("Exceptional storytelling with beautiful visuals. This film moved me to tears.", 1),
    ("Perfect blend of action and emotion. The best movie I've seen this year.", 1),
    ("Incredible character development and amazing special effects throughout.", 1),
    ("This film deserves all the awards. Absolutely stunning from start to finish.", 1),
    ("Captivating storyline with excellent direction. A true work of art.", 1),
    ("Outstanding performance by the lead actor. Compelling and thought-provoking.", 1),
    ("Brilliant script writing combined with perfect execution. Five stars!", 1),
    ("Visually stunning with an emotionally powerful narrative. Unforgettable experience.", 1),
    ("Creative and original approach to filmmaking. Exceeded all my expectations.", 1),
    ("Perfectly paced with incredible attention to detail. Cinematic excellence.", 1),
    ("Amazing soundtrack that perfectly complements the beautiful imagery.", 1),
    ("Sophisticated themes explored with remarkable depth and sensitivity.", 1),
    ("Engaging from the very first scene to the emotional conclusion.", 1),
    ("Superb ensemble cast delivering powerful and convincing performances.", 1),
    ("Innovative storytelling techniques make this a truly unique experience.", 1),
    ("Breathtaking visuals paired with an intelligent and moving script.", 1),
    ("This film sets a new standard for quality filmmaking and entertainment.", 1),
    
    # Negative reviews  
    ("This movie was terrible. Poor acting and a boring plot made it unwatchable.", 0),
    ("Complete waste of time. The story was confusing and the characters were flat.", 0),
    ("Poorly executed with bad special effects. I couldn't wait for it to end.", 0),
    ("Disappointing performances and a predictable storyline. Not worth watching.", 0),
    ("The worst film I've seen this year. Terrible direction and weak script.", 0),
    ("Boring and slow-paced. The dialogue was cringe-worthy throughout.", 0),
    ("Poor production values and unconvincing acting. Major disappointment.", 0),
    ("Confusing plot with no character development. Felt like a cheap production.", 0),
    ("Painfully dull with wooden performances from the entire cast.", 0),
    ("Terrible script with plot holes you could drive a truck through.", 0),
    ("Visually unappealing with poor cinematography. Completely forgettable.", 0),
    ("Generic and cliched approach with no original ideas whatsoever.", 0),
    ("Dragged on way too long with unnecessary scenes and poor pacing.", 0),
    ("Annoying soundtrack that didn't fit the mood of any scene.", 0),
    ("Shallow themes with no depth or meaningful content to explore.", 0),
    ("Lost my interest within the first ten minutes. Couldn't stay engaged.", 0),
    ("Weak ensemble cast with unconvincing and awkward performances.", 0),
    ("Outdated storytelling techniques that felt tired and overused.", 0),
    ("Ugly visuals combined with a nonsensical and poorly written script.", 0),
    ("This film represents everything wrong with modern entertainment industry.", 0),
    
    # More positive examples for balance
    ("Absolutely brilliant filmmaking with stunning visual effects.", 1),
    ("Heartwarming story with excellent character development throughout.", 1),
    ("Perfectly cast with each actor delivering memorable performances.", 1),
    ("Intelligent script that respects the audience's intelligence.", 1),
    ("Beautiful cinematography that enhances every scene.", 1),
    ("Compelling narrative that keeps you engaged until the very end.", 1),
    ("Exceptional direction with perfect attention to every detail.", 1),
    ("Moving and inspirational story that stays with you long after.", 1),
    ("Wonderful blend of humor and drama executed flawlessly.", 1),
    ("Remarkable achievement in storytelling and visual presentation.", 1),
    
    # More negative examples for balance
    ("Poorly written dialogue that felt forced and unnatural.", 0),
    ("Weak storyline with no real substance or meaning.", 0),
    ("Bad editing that made the pacing feel completely off.", 0),
    ("Unconvincing special effects that looked cheap and fake.", 0),
    ("Terrible casting choices that ruined the entire experience.", 0),
    ("Boring screenplay with no surprises or interesting moments.", 0),
    ("Poor direction that failed to bring out the best in anyone.", 0),
    ("Disappointing ending that made the whole journey feel pointless.", 0),
    ("Annoying characters that I couldn't care less about.", 0),
    ("Completely predictable plot with no twists or surprises.", 0),
]

class ExtendedSentimentDataset(Dataset):
    """Extended sentiment dataset for comprehensive testing."""
    
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Add classification token for GPT-2
        text_with_prompt = f"Sentiment: {text} = "
        
        encoding = self.tokenizer(
            text_with_prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class GPT2SentimentClassifier(nn.Module):
    """GPT-2 classifier for sentiment analysis."""
    
    def __init__(self, model_name="gpt2", num_classes=2, dropout=0.1):
        super().__init__()
        self.config = GPT2Config.from_pretrained(model_name)
        self.transformer = GPT2LMHeadModel.from_pretrained(model_name).transformer
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Freeze most of the transformer (for efficient fine-tuning)
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # Only fine-tune the last few layers
        for param in self.transformer.h[-2:].parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use the last token representation for classification
        hidden_states = outputs.last_hidden_state
        
        # Get the representation of the last non-padded token
        if attention_mask is not None:
            batch_size = hidden_states.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            hidden_states = hidden_states[range(batch_size), sequence_lengths]
        else:
            hidden_states = hidden_states[:, -1]
        
        # Apply dropout and classification
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

class CausalGPT2SentimentClassifier(nn.Module):
    """Causal GPT-2 classifier with our causal mechanisms."""
    
    def __init__(self, model_name="gpt2", num_classes=2, dropout=0.1):
        super().__init__()
        
        if CAUSAL_TRANSFORMER_AVAILABLE:
            # Use our CausalTransformer
            config = CausalTransformerConfig(
                base_model_name=model_name,
                enable_causal_attention=True,
                enable_soft_interventions=True,
                enable_structure_learning=True,
                intervention_prob=0.3,
                lambda_reg=0.01
            )
            self.causal_transformer = CausalTransformer(config)
            hidden_size = 768  # GPT-2 hidden size
        else:
            # Fallback to enhanced version of our simple classifier
            self.causal_transformer = None
            hidden_size = 128
            
        # Classification components
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        # Causal intervention parameters
        self.alpha = nn.Parameter(torch.zeros(hidden_size))
        self.intervention_probability = 0.3
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        
        if self.causal_transformer is not None:
            # Use full CausalTransformer
            
            # Apply strategic interventions during training
            interventions = None
            if self.training and np.random.random() < self.intervention_probability:
                # Create attention interventions for some layers
                n_layers = 12  # GPT-2 layers
                interventions = []
                
                for layer_idx in range(n_layers):
                    if np.random.random() < 0.2:  # 20% chance per layer
                        # Create attention pattern intervention
                        seq_len = input_ids.shape[1]
                        mask = torch.zeros(batch_size, 12, seq_len, seq_len)  # 12 attention heads
                        values = torch.randn(batch_size, 12, seq_len, seq_len) * 0.1
                        
                        # Randomly intervene on some attention patterns
                        for b in range(batch_size):
                            n_heads = np.random.randint(1, 4)
                            head_indices = np.random.choice(12, n_heads, replace=False)
                            mask[b, head_indices, :, :] = 1.0
                        
                        interventions.append({
                            'attention_intervention': (mask, values)
                        })
                    else:
                        interventions.append(None)
            
            # Forward through causal transformer
            logits, causal_info = self.causal_transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                interventions=interventions,
                return_causal_info=True
            )
            
            # Use the last token for classification
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                hidden_states = logits[range(batch_size), sequence_lengths]
            else:
                hidden_states = logits[:, -1]
                
            # Project to hidden size for classification
            hidden_states = hidden_states[:, :768]  # Take first 768 dims
            
        else:
            # Fallback implementation
            hidden_states = torch.randn(batch_size, 768)  # Placeholder
            
        # Apply soft interventions if in training
        if self.training and hasattr(self, 'alpha'):
            if np.random.random() < self.intervention_probability:
                alpha = torch.sigmoid(self.alpha)
                intervention_values = torch.randn_like(hidden_states) * 0.1
                
                # Random intervention mask
                intervention_mask = torch.rand_like(hidden_states) < 0.2
                
                # Soft intervention
                soft_intervention = (1 - alpha) * hidden_states + alpha * intervention_values
                hidden_states = torch.where(intervention_mask, soft_intervention, hidden_states)
        
        # Classification
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
            # Add causal violation penalty if available
            causal_penalty = 0.0
            if hasattr(self, 'causal_transformer') and self.causal_transformer is not None:
                # Sum violation penalties from all causal blocks
                causal_penalty = sum(
                    getattr(block.causal_attention, 'get_causal_violation_penalty', lambda: 0.0)()
                    for block in self.causal_transformer.causal_blocks
                )
            
            classification_loss = F.cross_entropy(logits, labels)
            loss = classification_loss + 0.01 * causal_penalty  # Small penalty weight
            
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

class ComprehensiveDataEfficiencyTester:
    """Comprehensive data efficiency comparison between causal and vanilla models."""
    
    def __init__(self, device='cpu', random_seed=42):
        self.device = device
        self.random_seed = random_seed
        self.set_random_seeds(random_seed)
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare data
        texts, labels = zip(*EXTENDED_SENTIMENT_DATA)
        self.texts = list(texts)
        self.labels = list(labels)
        
        print(f"📊 Total dataset size: {len(self.texts)} samples")
        print(f"📊 Positive samples: {sum(self.labels)}")
        print(f"📊 Negative samples: {len(self.labels) - sum(self.labels)}")
        
        # Data splits to test
        self.data_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
        self.results = {}
        
    def set_random_seeds(self, seed):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def create_balanced_datasets(self, fraction: float):
        """Create balanced train/test datasets."""
        # Separate positive and negative samples
        pos_samples = [(t, l) for t, l in zip(self.texts, self.labels) if l == 1]
        neg_samples = [(t, l) for t, l in zip(self.texts, self.labels) if l == 0]
        
        # Calculate split sizes
        n_pos_total = int(len(pos_samples) * fraction)
        n_neg_total = int(len(neg_samples) * fraction)
        
        # Train/test split (80/20)
        n_pos_train = int(n_pos_total * 0.8)
        n_neg_train = int(n_neg_total * 0.8)
        n_pos_test = n_pos_total - n_pos_train
        n_neg_test = n_neg_total - n_neg_train
        
        # Ensure minimum test size
        n_pos_test = max(n_pos_test, 2)
        n_neg_test = max(n_neg_test, 2)
        
        # Create train/test splits
        train_texts = [s[0] for s in pos_samples[:n_pos_train]] + [s[0] for s in neg_samples[:n_neg_train]]
        train_labels = [s[1] for s in pos_samples[:n_pos_train]] + [s[1] for s in neg_samples[:n_neg_train]]
        
        test_texts = [s[0] for s in pos_samples[n_pos_train:n_pos_train+n_pos_test]] + [s[0] for s in neg_samples[n_neg_train:n_neg_train+n_neg_test]]
        test_labels = [s[1] for s in pos_samples[n_pos_train:n_pos_train+n_pos_test]] + [s[1] for s in neg_samples[n_neg_train:n_neg_train+n_neg_test]]
        
        # Create datasets
        train_dataset = ExtendedSentimentDataset(train_texts, train_labels, self.tokenizer)
        test_dataset = ExtendedSentimentDataset(test_texts, test_labels, self.tokenizer)
        
        return train_dataset, test_dataset
    
    def train_and_evaluate_model(self, model, train_dataset, test_dataset, model_name="model", epochs=5):
        """Train and evaluate a model."""
        print(f"  🏋️ Training {model_name} with {len(train_dataset)} samples...")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        
        # Training loop
        model.train()
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            training_losses.append(avg_epoch_loss)
            print(f"    Epoch {epoch+1}/{epochs}: Loss = {avg_epoch_loss:.4f}")
        
        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        eval_loss = 0
        num_eval_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                eval_loss += outputs['loss'].item()
                num_eval_batches += 1
                
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        avg_eval_loss = eval_loss / num_eval_batches
        
        return {
            'model_name': model_name,
            'train_loss': training_losses[-1],
            'eval_loss': avg_eval_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'n_train_samples': len(train_dataset),
            'n_test_samples': len(test_dataset),
            'training_losses': training_losses
        }
    
    def run_comprehensive_comparison(self):
        """Run the definitive data efficiency comparison."""
        print("🚀 COMPREHENSIVE DATA EFFICIENCY TEST")
        print("=" * 60)
        print("🎯 Research Question: Can CausalTransformer achieve same accuracy with less data?")
        print("=" * 60)
        
        for fraction in self.data_fractions:
            print(f"\n📊 Testing with {fraction*100:.0f}% of data...")
            
            # Create datasets
            train_dataset, test_dataset = self.create_balanced_datasets(fraction)
            
            # Test standard GPT-2
            print(f"  Standard GPT-2...")
            vanilla_model = GPT2SentimentClassifier().to(self.device)
            vanilla_results = self.train_and_evaluate_model(
                vanilla_model, train_dataset, test_dataset, "Vanilla GPT-2"
            )
            
            # Test causal GPT-2
            print(f"  Causal GPT-2...")
            causal_model = CausalGPT2SentimentClassifier().to(self.device)
            causal_results = self.train_and_evaluate_model(
                causal_model, train_dataset, test_dataset, "Causal GPT-2"
            )
            
            # Store results
            self.results[fraction] = {
                'vanilla': vanilla_results,
                'causal': causal_results,
                'accuracy_improvement': causal_results['accuracy'] - vanilla_results['accuracy'],
                'f1_improvement': causal_results['f1_score'] - vanilla_results['f1_score']
            }
            
            # Print summary
            print(f"  📈 Vanilla:  {vanilla_results['accuracy']:.3f} accuracy, {vanilla_results['f1_score']:.3f} F1")
            print(f"  🎯 Causal:   {causal_results['accuracy']:.3f} accuracy, {causal_results['f1_score']:.3f} F1")
            print(f"  ⚡ Improvement: {self.results[fraction]['accuracy_improvement']:.3f} accuracy")
        
        # Analyze and visualize results
        self.analyze_results()
        self.create_visualizations()
        self.save_results()
        
    def analyze_results(self):
        """Comprehensive analysis of results."""
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE RESULTS ANALYSIS")
        print("=" * 60)
        
        # Summary table
        print(f"{'Data %':<8} {'Vanilla Acc':<12} {'Causal Acc':<12} {'Improvement':<12} {'Winner':<10}")
        print("-" * 60)
        
        causal_wins = 0
        total_improvements = []
        
        for fraction, results in self.results.items():
            vanilla_acc = results['vanilla']['accuracy']
            causal_acc = results['causal']['accuracy']
            improvement = results['accuracy_improvement']
            winner = "🏆 Causal" if improvement > 0 else "Vanilla"
            
            if improvement > 0:
                causal_wins += 1
            total_improvements.append(improvement)
            
            print(f"{fraction*100:>6.0f}% {vanilla_acc:>11.3f} {causal_acc:>11.3f} {improvement:>10.3f} {winner}")
        
        print("-" * 60)
        
        # Summary statistics
        avg_improvement = np.mean(total_improvements)
        max_improvement = np.max(total_improvements)
        min_improvement = np.min(total_improvements)
        
        print(f"🎯 Causal Wins: {causal_wins}/{len(self.results)} ({causal_wins/len(self.results)*100:.1f}%)")
        print(f"📊 Average Improvement: {avg_improvement:.3f} ({avg_improvement*100:.1f}%)")
        print(f"📈 Best Improvement: {max_improvement:.3f} ({max_improvement*100:.1f}%)")
        print(f"📉 Worst Result: {min_improvement:.3f} ({min_improvement*100:.1f}%)")
        
        # Data efficiency analysis
        print(f"\n🔍 DATA EFFICIENCY ANALYSIS:")
        
        # Find if causal achieves target accuracy with less data
        vanilla_100_acc = self.results[1.0]['vanilla']['accuracy']
        target_accuracy = vanilla_100_acc * 0.9  # 90% of full vanilla performance
        
        causal_achieves_target = None
        for fraction in sorted(self.results.keys()):
            if self.results[fraction]['causal']['accuracy'] >= target_accuracy:
                causal_achieves_target = fraction
                break
        
        if causal_achieves_target and causal_achieves_target < 1.0:
            data_savings = (1.0 - causal_achieves_target) * 100
            print(f"✅ Causal achieves 90% vanilla performance with {causal_achieves_target*100:.0f}% data")
            print(f"💰 Data savings: {data_savings:.0f}% less data needed")
        else:
            print(f"❌ Causal does not achieve significant data efficiency advantage")
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        # Set up the plot style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        fractions = list(self.results.keys())
        vanilla_accs = [self.results[f]['vanilla']['accuracy'] for f in fractions]
        causal_accs = [self.results[f]['causal']['accuracy'] for f in fractions]
        vanilla_f1s = [self.results[f]['vanilla']['f1_score'] for f in fractions]
        causal_f1s = [self.results[f]['causal']['f1_score'] for f in fractions]
        improvements = [self.results[f]['accuracy_improvement'] for f in fractions]
        
        # Plot 1: Accuracy comparison
        data_percentages = [f*100 for f in fractions]
        ax1.plot(data_percentages, vanilla_accs, 'o-', label='Vanilla GPT-2', linewidth=2, markersize=8)
        ax1.plot(data_percentages, causal_accs, 's-', label='Causal GPT-2', linewidth=2, markersize=8)
        ax1.set_xlabel('Training Data Percentage (%)')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Accuracy: Causal vs Vanilla GPT-2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1 Score comparison
        ax2.plot(data_percentages, vanilla_f1s, 'o-', label='Vanilla GPT-2', linewidth=2, markersize=8)
        ax2.plot(data_percentages, causal_f1s, 's-', label='Causal GPT-2', linewidth=2, markersize=8)
        ax2.set_xlabel('Training Data Percentage (%)')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score: Causal vs Vanilla GPT-2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Improvement over data fractions
        colors = ['red' if x < 0 else 'green' for x in improvements]
        ax3.bar(data_percentages, improvements, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Training Data Percentage (%)')
        ax3.set_ylabel('Accuracy Improvement (Causal - Vanilla)')
        ax3.set_title('Causal Advantage by Data Amount')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training efficiency
        sample_counts = [self.results[f]['vanilla']['n_train_samples'] for f in fractions]
        ax4.scatter(sample_counts, vanilla_accs, label='Vanilla GPT-2', s=100, alpha=0.7)
        ax4.scatter(sample_counts, causal_accs, label='Causal GPT-2', s=100, alpha=0.7)
        ax4.set_xlabel('Number of Training Samples')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_title('Sample Efficiency Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_data_efficiency_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Comprehensive visualizations saved as 'comprehensive_data_efficiency_results.png'")
        
    def save_results(self):
        """Save detailed results to JSON."""
        results_with_metadata = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_size': len(self.texts),
                'data_fractions_tested': self.data_fractions,
                'model_comparison': 'CausalTransformer vs Vanilla GPT-2',
                'task': 'Sentiment Analysis',
                'random_seed': self.random_seed
            },
            'results': self.results,
            'summary': {
                'causal_wins': sum(1 for r in self.results.values() if r['accuracy_improvement'] > 0),
                'total_tests': len(self.results),
                'average_improvement': np.mean([r['accuracy_improvement'] for r in self.results.values()]),
                'max_improvement': np.max([r['accuracy_improvement'] for r in self.results.values()]),
                'data_efficiency_achieved': self._check_data_efficiency()
            }
        }
        
        filename = f'comprehensive_causal_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"💾 Detailed results saved to {filename}")
        
    def _check_data_efficiency(self):
        """Check if causal model achieves data efficiency."""
        vanilla_100_acc = self.results[1.0]['vanilla']['accuracy']
        target_accuracy = vanilla_100_acc * 0.9
        
        for fraction in sorted(self.results.keys()):
            if self.results[fraction]['causal']['accuracy'] >= target_accuracy:
                if fraction < 1.0:
                    return {
                        'achieved': True,
                        'target_accuracy': target_accuracy,
                        'achieved_with_fraction': fraction,
                        'data_savings_percent': (1.0 - fraction) * 100
                    }
                break
        
        return {'achieved': False, 'target_accuracy': target_accuracy}

def main():
    """Main function to run the comprehensive data efficiency test."""
    print("🎯 COMPREHENSIVE DATA EFFICIENCY TEST")
    print("🔬 CausalTransformer vs Standard GPT-2")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    print(f"🤖 CausalTransformer Available: {CAUSAL_TRANSFORMER_AVAILABLE}")
    
    # Create and run tester
    tester = ComprehensiveDataEfficiencyTester(device=device)
    tester.run_comprehensive_comparison()
    
    print("\n✅ COMPREHENSIVE TEST COMPLETED!")
    print("📈 Key Research Findings:")
    print("   - Detailed comparison of data efficiency")
    print("   - Statistical analysis of causal advantages")
    print("   - Comprehensive visualizations generated")
    print("   - Results saved for further analysis")

if __name__ == "__main__":
    main() 