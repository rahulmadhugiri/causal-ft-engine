#!/usr/bin/env python3
"""
Simple but Comprehensive Data Efficiency Test

This test answers the core question: Can causal fine-tuning achieve 
the same accuracy as standard fine-tuning with less data?

Focus: Clean, reliable comparison between causal and standard approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Extended dataset for robust testing
COMPREHENSIVE_SENTIMENT_DATA = [
    # Strong positive examples
    ("This movie is absolutely brilliant with outstanding performances.", 1),
    ("Incredible storytelling and beautiful cinematography throughout.", 1),
    ("A masterpiece that exceeded all my expectations completely.", 1),
    ("Perfect blend of action and emotion with amazing special effects.", 1),
    ("Exceptional direction with compelling characters and great plot.", 1),
    ("Stunning visuals paired with an intelligent and moving script.", 1),
    ("Outstanding acting from the entire cast with perfect execution.", 1),
    ("Captivating from start to finish with remarkable attention to detail.", 1),
    ("Brilliant writing combined with superb direction and cinematography.", 1),
    ("Unforgettable experience that sets new standards for filmmaking.", 1),
    ("Creative and original approach with sophisticated themes explored.", 1),
    ("Emotionally powerful narrative with breathtaking visual presentation.", 1),
    ("Engaging storyline with excellent character development throughout.", 1),
    ("Amazing soundtrack that perfectly complements the beautiful imagery.", 1),
    ("Innovative techniques make this a truly unique cinematic experience.", 1),
    ("Sophisticated storytelling with remarkable depth and great sensitivity.", 1),
    ("Perfectly paced with incredible performances from all actors.", 1),
    ("Visually stunning with an emotionally resonant and intelligent plot.", 1),
    ("Exceptional quality filmmaking that deserves recognition and awards.", 1),
    ("Compelling narrative with outstanding production values throughout.", 1),
    
    # Strong negative examples
    ("This movie was terrible with poor acting and boring plot.", 0),
    ("Complete waste of time with confusing story and flat characters.", 0),
    ("Poorly executed with bad effects and I couldn't wait for it to end.", 0),
    ("Disappointing performances with predictable storyline not worth watching.", 0),
    ("The worst film with terrible direction and extremely weak script.", 0),
    ("Boring and slow with cringe-worthy dialogue throughout the entire film.", 0),
    ("Poor production values with unconvincing acting and major disappointment.", 0),
    ("Confusing plot with no development and felt like cheap production.", 0),
    ("Painfully dull with wooden performances from the entire cast.", 0),
    ("Terrible script with plot holes you could drive through.", 0),
    ("Visually unappealing with poor cinematography and completely forgettable.", 0),
    ("Generic and cliched with no original ideas whatsoever.", 0),
    ("Dragged on too long with unnecessary scenes and poor pacing.", 0),
    ("Annoying soundtrack that didn't fit any scene properly.", 0),
    ("Shallow themes with no depth or meaningful content.", 0),
    ("Lost interest immediately and couldn't stay engaged at all.", 0),
    ("Weak cast with unconvincing and awkward performances throughout.", 0),
    ("Outdated techniques that felt tired and completely overused.", 0),
    ("Ugly visuals with nonsensical and poorly written script.", 0),
    ("Represents everything wrong with modern entertainment industry today.", 0),
    
    # Additional examples for more robust testing
    ("Excellent film with great story and wonderful acting.", 1),
    ("Beautiful movie that moved me to tears with perfect direction.", 1),
    ("Amazing experience with incredible visuals and sound design.", 1),
    ("Fantastic performances with engaging plot and great cinematography.", 1),
    ("Wonderful storytelling with memorable characters and perfect pacing.", 1),
    ("Terrible movie with bad story and awful acting throughout.", 0),
    ("Horrible experience with poor visuals and terrible sound.", 0),
    ("Awful performances with boring plot and bad cinematography.", 0),
    ("Dreadful storytelling with forgettable characters and poor pacing.", 0),
    ("Disappointing film that failed to deliver on any level.", 0),
]

class SentimentDataset(Dataset):
    """Dataset for sentiment classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
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

class StandardGPTClassifier(nn.Module):
    """Standard GPT-2 based classifier without causal mechanisms."""
    
    def __init__(self, vocab_size=50257, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim*2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(6)
        ])
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embed and process
        x = self.embedding(input_ids)
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert to transformer mask format
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, src_key_padding_mask=src_key_padding_mask)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Pool to sentence representation (mean pooling)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            
        return {'loss': loss, 'logits': logits}

class CausalGPTClassifier(nn.Module):
    """GPT-2 classifier with causal mechanisms."""
    
    def __init__(self, vocab_size=50257, hidden_dim=128, num_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Standard transformer components
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim*2,
                dropout=0.1,
                batch_first=True
            ) for _ in range(6)
        ])
        self.ln_final = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
        # Causal mechanisms
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))  # Soft interventions
        self.adjacency = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)  # Structure learning
        self.intervention_probability = 0.3
        
        # Track causal metrics
        self.violation_penalty = 0.0
        self.intervention_count = 0
        
    def get_intervention_strength(self):
        """Get soft intervention strength."""
        return torch.sigmoid(self.alpha)
    
    def get_adjacency_matrix(self, hard=False):
        """Get learned adjacency matrix."""
        if hard:
            return (torch.sigmoid(self.adjacency) > 0.5).float()
        else:
            return torch.sigmoid(self.adjacency)
    
    def apply_causal_interventions(self, x, attention_mask):
        """Apply causal interventions during forward pass."""
        if not self.training:
            return x
            
        if np.random.random() < self.intervention_probability:
            batch_size, seq_len, hidden_dim = x.shape
            
            # Get intervention strength
            alpha = self.get_intervention_strength()
            
            # Create intervention mask and values
            intervention_mask = torch.rand(batch_size, seq_len, hidden_dim) < 0.2
            intervention_values = torch.randn(batch_size, seq_len, hidden_dim) * 0.1
            
            # Apply soft interventions
            soft_intervention = (1 - alpha) * x + alpha * intervention_values
            x = torch.where(intervention_mask, soft_intervention, x)
            
            # Apply causal adjacency constraints
            adjacency = self.get_adjacency_matrix()
            
            # Compute violation penalty (simplified)
            violation = torch.norm(torch.matmul(x.mean(dim=1), adjacency) - x.mean(dim=1)) ** 2
            self.violation_penalty = violation.item()
            self.intervention_count += 1
            
        return x
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embed
        x = self.embedding(input_ids)
        
        # Apply causal interventions
        x = self.apply_causal_interventions(x, attention_mask)
        
        # Apply attention mask for transformer
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer blocks with causal interventions between blocks
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, src_key_padding_mask=src_key_padding_mask)
            
            # Apply interventions between some blocks
            if i % 2 == 1:  # Every other block
                x = self.apply_causal_interventions(x, attention_mask)
        
        # Final processing
        x = self.ln_final(x)
        
        # Pool to sentence representation
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            classification_loss = F.cross_entropy(logits, labels)
            # Add causal violation penalty
            causal_penalty = self.violation_penalty * 0.01  # Small weight
            loss = classification_loss + causal_penalty
            
        return {'loss': loss, 'logits': logits}

class DataEfficiencyExperiment:
    """Clean data efficiency experiment."""
    
    def __init__(self, device='cpu', random_seed=42):
        self.device = device
        self.random_seed = random_seed
        self.set_random_seeds(random_seed)
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare data
        texts, labels = zip(*COMPREHENSIVE_SENTIMENT_DATA)
        self.texts = list(texts)
        self.labels = list(labels)
        
        print(f"📊 Dataset: {len(self.texts)} samples ({sum(self.labels)} positive, {len(self.labels)-sum(self.labels)} negative)")
        
        # Test different data amounts
        self.data_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.results = {}
        
    def set_random_seeds(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def create_datasets(self, fraction):
        """Create balanced train/test split."""
        # Separate by class
        pos_samples = [(t, l) for t, l in zip(self.texts, self.labels) if l == 1]
        neg_samples = [(t, l) for t, l in zip(self.texts, self.labels) if l == 0]
        
        # Sample fraction
        n_pos = max(int(len(pos_samples) * fraction), 2)
        n_neg = max(int(len(neg_samples) * fraction), 2)
        
        # Train/test split
        n_pos_train = max(int(n_pos * 0.8), 1)
        n_neg_train = max(int(n_neg * 0.8), 1) 
        n_pos_test = n_pos - n_pos_train
        n_neg_test = n_neg - n_neg_train
        
        # Create splits
        train_samples = pos_samples[:n_pos_train] + neg_samples[:n_neg_train]
        test_samples = pos_samples[n_pos_train:n_pos_train+n_pos_test] + neg_samples[n_neg_train:n_neg_train+n_neg_test]
        
        # Extract texts and labels
        train_texts, train_labels = zip(*train_samples)
        test_texts, test_labels = zip(*test_samples)
        
        # Create datasets
        train_dataset = SentimentDataset(train_texts, train_labels, self.tokenizer)
        test_dataset = SentimentDataset(test_texts, test_labels, self.tokenizer)
        
        return train_dataset, test_dataset
    
    def train_model(self, model, train_dataset, test_dataset, model_name="model", epochs=3):
        """Train and evaluate model."""
        print(f"  🏋️ Training {model_name} ({len(train_dataset)} samples)...")
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Training
        model.train()
        for epoch in range(epochs):
            total_loss = 0
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
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"    Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
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
                
                preds = torch.argmax(outputs['logits'], dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')
        avg_eval_loss = eval_loss / num_eval_batches
        
        # Causal metrics
        causal_info = {}
        if hasattr(model, 'violation_penalty'):
            causal_info['violation_penalty'] = model.violation_penalty
            causal_info['intervention_count'] = model.intervention_count
        
        return {
            'model_name': model_name,
            'train_samples': len(train_dataset),
            'test_samples': len(test_dataset),
            'accuracy': accuracy,
            'f1_score': f1,
            'eval_loss': avg_eval_loss,
            'causal_info': causal_info
        }
    
    def run_experiment(self):
        """Run comprehensive data efficiency experiment."""
        print("🚀 DATA EFFICIENCY EXPERIMENT")
        print("=" * 50)
        print("🎯 Question: Can causal models achieve same performance with less data?")
        print("=" * 50)
        
        for fraction in self.data_fractions:
            print(f"\n📊 Testing with {fraction*100:.0f}% of data...")
            
            # Create datasets
            train_dataset, test_dataset = self.create_datasets(fraction)
            
            # Test standard model
            print("  Standard Model...")
            standard_model = StandardGPTClassifier().to(self.device)
            standard_results = self.train_model(
                standard_model, train_dataset, test_dataset, "Standard"
            )
            
            # Test causal model
            print("  Causal Model...")
            causal_model = CausalGPTClassifier().to(self.device)
            causal_results = self.train_model(
                causal_model, train_dataset, test_dataset, "Causal"
            )
            
            # Store results
            self.results[fraction] = {
                'standard': standard_results,
                'causal': causal_results,
                'accuracy_diff': causal_results['accuracy'] - standard_results['accuracy'],
                'f1_diff': causal_results['f1_score'] - standard_results['f1_score']
            }
            
            print(f"  📈 Standard: {standard_results['accuracy']:.3f} accuracy")
            print(f"  🎯 Causal:   {causal_results['accuracy']:.3f} accuracy")
            print(f"  ⚡ Difference: {self.results[fraction]['accuracy_diff']:.3f}")
        
        self.analyze_results()
        self.create_visualization()
        self.save_results()
    
    def analyze_results(self):
        """Analyze experimental results."""
        print("\n" + "=" * 50)
        print("📋 EXPERIMENTAL RESULTS")
        print("=" * 50)
        
        print(f"{'Data %':<8} {'Standard':<10} {'Causal':<10} {'Difference':<11} {'Winner'}")
        print("-" * 50)
        
        causal_wins = 0
        improvements = []
        
        for fraction, results in self.results.items():
            std_acc = results['standard']['accuracy']
            causal_acc = results['causal']['accuracy']
            diff = results['accuracy_diff']
            winner = "🏆 Causal" if diff > 0 else "Standard"
            
            if diff > 0:
                causal_wins += 1
            improvements.append(diff)
            
            print(f"{fraction*100:>6.0f}% {std_acc:>9.3f} {causal_acc:>9.3f} {diff:>9.3f} {winner}")
        
        print("-" * 50)
        
        # Summary statistics
        avg_improvement = np.mean(improvements)
        print(f"🎯 Causal wins: {causal_wins}/{len(self.results)} ({causal_wins/len(self.results)*100:.1f}%)")
        print(f"📊 Average improvement: {avg_improvement:.3f} ({avg_improvement*100:.1f}%)")
        
        # Data efficiency analysis
        print(f"\n🔍 DATA EFFICIENCY ANALYSIS:")
        full_standard_acc = self.results[1.0]['standard']['accuracy']
        target_acc = full_standard_acc * 0.9  # 90% of full performance
        
        for fraction in sorted(self.results.keys()):
            causal_acc = self.results[fraction]['causal']['accuracy']
            if causal_acc >= target_acc:
                data_savings = (1.0 - fraction) * 100
                print(f"✅ Causal achieves 90% standard performance with {fraction*100:.0f}% data")
                print(f"💰 Potential data savings: {data_savings:.0f}%")
                break
        else:
            print(f"❌ Causal does not achieve 90% standard performance with less data")
    
    def create_visualization(self):
        """Create results visualization."""
        fractions = list(self.results.keys())
        standard_accs = [self.results[f]['standard']['accuracy'] for f in fractions]
        causal_accs = [self.results[f]['causal']['accuracy'] for f in fractions]
        
        plt.figure(figsize=(12, 5))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        data_percentages = [f*100 for f in fractions]
        plt.plot(data_percentages, standard_accs, 'o-', label='Standard Model', linewidth=2, markersize=8)
        plt.plot(data_percentages, causal_accs, 's-', label='Causal Model', linewidth=2, markersize=8)
        plt.xlabel('Training Data Percentage (%)')
        plt.ylabel('Test Accuracy')
        plt.title('Data Efficiency: Accuracy Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Improvement over data amounts
        plt.subplot(1, 2, 2)
        improvements = [self.results[f]['accuracy_diff'] for f in fractions]
        colors = ['red' if x < 0 else 'green' for x in improvements]
        plt.bar(data_percentages, improvements, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xlabel('Training Data Percentage (%)')
        plt.ylabel('Accuracy Improvement (Causal - Standard)')
        plt.title('Causal Model Advantage by Data Amount')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_efficiency_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Results visualization saved as 'data_efficiency_results.png'")
    
    def save_results(self):
        """Save results to file."""
        results_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'dataset_size': len(self.texts),
                'data_fractions': self.data_fractions,
                'random_seed': self.random_seed
            },
            'results': self.results
        }
        
        filename = f'data_efficiency_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to {filename}")

def main():
    """Run the data efficiency experiment."""
    print("🎯 CAUSAL vs STANDARD DATA EFFICIENCY TEST")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    experiment = DataEfficiencyExperiment(device=device)
    experiment.run_experiment()
    
    print("\n✅ EXPERIMENT COMPLETED!")
    print("📈 Key Findings:")
    print("   - Compared causal vs standard models across data amounts")
    print("   - Measured data efficiency and performance differences")
    print("   - Generated comprehensive analysis and visualizations")

if __name__ == "__main__":
    main() 