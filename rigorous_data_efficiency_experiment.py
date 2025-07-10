#!/usr/bin/env python3
"""
RIGOROUS DATA EFFICIENCY EXPERIMENT

This experiment will definitively prove or disprove the claim that causal 
fine-tuning provides 2x-5x data efficiency improvements over standard approaches.

Test Points: 5%, 10%, 25%, 50%, 100% of training data
Goal: Show causal model reaches similar performance with fewer examples
Revolutionary Threshold: 2x-5x improvement in data efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import GPT2Tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Comprehensive dataset for rigorous testing
RIGOROUS_SENTIMENT_DATASET = [
    # Very positive sentiment
    ("This movie is absolutely brilliant with outstanding performances and perfect direction.", 1),
    ("Incredible storytelling with beautiful cinematography that moved me to tears.", 1),
    ("A masterpiece that exceeded all expectations with amazing special effects.", 1),
    ("Perfect blend of action and emotion with compelling character development.", 1),
    ("Exceptional filmmaking with intelligent script and superb acting throughout.", 1),
    ("Stunning visuals paired with emotionally powerful and moving narrative.", 1),
    ("Outstanding cast delivering convincing performances in this remarkable film.", 1),
    ("Captivating from start to finish with incredible attention to detail.", 1),
    ("Brilliant writing combined with perfect execution and amazing cinematography.", 1),
    ("Unforgettable experience that sets new standards for quality entertainment.", 1),
    ("Creative and original with sophisticated themes explored beautifully.", 1),
    ("Breathtaking visual presentation with intelligent and thought-provoking plot.", 1),
    ("Engaging storyline with excellent character development and perfect pacing.", 1),
    ("Amazing soundtrack perfectly complementing the beautiful imagery throughout.", 1),
    ("Innovative storytelling techniques creating a truly unique cinematic experience.", 1),
    ("Sophisticated narrative with remarkable depth and emotional sensitivity.", 1),
    ("Perfectly paced with incredible performances from talented ensemble cast.", 1),
    ("Visually stunning with emotionally resonant and intellectually engaging content.", 1),
    ("Exceptional production values deserving recognition and critical acclaim.", 1),
    ("Compelling story with outstanding direction and memorable performances.", 1),
    ("Wonderful blend of humor and drama executed with perfect timing.", 1),
    ("Remarkable achievement in storytelling with beautiful visual composition.", 1),
    ("Excellent film with great story and wonderful acting throughout.", 1),
    ("Beautiful movie with perfect direction and incredible emotional depth.", 1),
    ("Amazing cinematic experience with stunning visuals and sound design.", 1),
    ("Fantastic performances with engaging plot and excellent cinematography.", 1),
    ("Wonderful storytelling with memorable characters and ideal pacing.", 1),
    ("Brilliant filmmaking with sophisticated approach to complex themes.", 1),
    ("Outstanding entertainment with perfect balance of action and emotion.", 1),
    ("Exceptional quality with remarkable attention to every single detail.", 1),
    
    # Very negative sentiment
    ("This movie was absolutely terrible with poor acting and boring plot.", 0),
    ("Complete waste of time with confusing story and flat character development.", 0),
    ("Poorly executed with bad special effects and I couldn't wait to leave.", 0),
    ("Disappointing performances with predictable storyline not worth watching.", 0),
    ("The worst film with terrible direction and extremely weak script.", 0),
    ("Boring and slow with cringe-worthy dialogue throughout the entire movie.", 0),
    ("Poor production values with unconvincing acting and major disappointment.", 0),
    ("Confusing plot with no development and felt like cheap production.", 0),
    ("Painfully dull with wooden performances from the entire cast.", 0),
    ("Terrible script with massive plot holes you could drive through.", 0),
    ("Visually unappealing with poor cinematography and completely forgettable.", 0),
    ("Generic and cliched with absolutely no original ideas whatsoever.", 0),
    ("Dragged on way too long with unnecessary scenes and awful pacing.", 0),
    ("Annoying soundtrack that didn't fit the mood of any scene.", 0),
    ("Shallow themes with no depth or meaningful content to explore.", 0),
    ("Lost my interest within minutes and couldn't stay engaged.", 0),
    ("Weak ensemble cast with unconvincing and awkward performances.", 0),
    ("Outdated storytelling techniques that felt tired and overused.", 0),
    ("Ugly visuals combined with nonsensical and poorly written script.", 0),
    ("Represents everything wrong with modern entertainment industry.", 0),
    ("Horrible experience with poor visuals and terrible sound quality.", 0),
    ("Awful performances with boring plot and bad cinematography.", 0),
    ("Dreadful storytelling with forgettable characters and poor pacing.", 0),
    ("Disappointing film that failed to deliver on any level.", 0),
    ("Terrible movie with bad story and awful acting performances.", 0),
    ("Poorly written dialogue that felt forced and completely unnatural.", 0),
    ("Weak storyline with no real substance or meaningful content.", 0),
    ("Bad editing that made the pacing feel completely off.", 0),
    ("Unconvincing special effects that looked cheap and fake.", 0),
    ("Terrible casting choices that ruined the entire viewing experience.", 0),
    
    # Additional samples for robustness
    ("Great movie with excellent story and fantastic acting.", 1),
    ("Wonderful film with beautiful visuals and perfect soundtrack.", 1),
    ("Amazing performances with compelling narrative and great direction.", 1),
    ("Brilliant cinematography with emotionally engaging storyline.", 1),
    ("Outstanding film with incredible attention to detail.", 1),
    ("Perfect entertainment with excellent character development.", 1),
    ("Remarkable storytelling with beautiful visual presentation.", 1),
    ("Exceptional quality with outstanding performances throughout.", 1),
    ("Incredible movie with perfect blend of action and emotion.", 1),
    ("Superb filmmaking with intelligent and moving script.", 1),
    ("Terrible film with poor story and bad acting.", 0),
    ("Awful movie with boring plot and terrible direction.", 0),
    ("Disappointing with weak performances and poor script.", 0),
    ("Bad film with unconvincing story and awful cinematography.", 0),
    ("Poor quality with terrible acting and boring storyline.", 0),
    ("Horrible movie with bad direction and weak characters.", 0),
    ("Terrible production with poor visuals and awful sound.", 0),
    ("Disappointing film with no redeeming qualities whatsoever.", 0),
    ("Bad movie with poor execution and terrible performances.", 0),
    ("Awful film with boring story and unconvincing acting.", 0),
]

class RigorousDataset(Dataset):
    """Dataset for rigorous testing with consistent preprocessing."""
    
    def __init__(self, texts, labels, tokenizer, max_length=48):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Consistent tokenization
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

class StandardModel(nn.Module):
    """Optimized standard baseline model."""
    
    def __init__(self, vocab_size=50257, hidden_dim=256, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(512, hidden_dim) * 0.02)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(8)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_len = input_ids.size(1)
        
        # Embeddings + positional encoding
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:seq_len, :]
        
        # Attention mask for transformer
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Final processing
        x = self.layer_norm(x)
        
        # Global average pooling with attention mask
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

class CausalModel(nn.Module):
    """Improved causal model with proper regularization."""
    
    def __init__(self, vocab_size=50257, hidden_dim=256, num_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(512, hidden_dim) * 0.02)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(8)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Causal mechanisms with proper regularization
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))
        self.adjacency_logits = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        
        # Causal hyperparameters
        self.intervention_prob = 0.2  # Reduced for stability
        self.causal_reg_strength = 0.001  # Reduced regularization
        
        # Track causal metrics
        self.current_violation_penalty = 0.0
        self.total_interventions = 0
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_intervention_strength(self):
        """Get soft intervention strength."""
        return torch.sigmoid(self.alpha * 0.5)  # Scale down for stability
    
    def get_adjacency_matrix(self):
        """Get learned adjacency matrix."""
        return torch.sigmoid(self.adjacency_logits)
    
    def apply_causal_intervention(self, x, attention_mask, layer_idx):
        """Apply causal interventions with proper regularization."""
        if not self.training:
            return x, 0.0
        
        # Apply interventions probabilistically
        if np.random.random() > self.intervention_prob:
            return x, 0.0
        
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get intervention parameters
        alpha = self.get_intervention_strength()
        adjacency = self.get_adjacency_matrix()
        
        # Create intervention mask (sparse for stability)
        intervention_mask = torch.rand(batch_size, seq_len, hidden_dim) < 0.1
        
        # Generate intervention values (small magnitude)
        intervention_values = torch.randn(batch_size, seq_len, hidden_dim) * 0.05
        
        # Apply soft intervention
        soft_intervention = (1 - alpha) * x + alpha * intervention_values
        x_intervened = torch.where(intervention_mask, soft_intervention, x)
        
        # Compute causal violation penalty
        # Simplified: measure how much the intervention affects the representation
        violation_penalty = torch.mean((x_intervened - x) ** 2)
        
        # Add adjacency constraint (simplified)
        if layer_idx % 2 == 0:  # Apply every other layer
            x_mean = x.mean(dim=1)  # [batch_size, hidden_dim]
            adjacency_effect = torch.matmul(x_mean, adjacency)
            adjacency_penalty = torch.mean((adjacency_effect - x_mean) ** 2)
            violation_penalty = violation_penalty + adjacency_penalty * 0.1
        
        self.current_violation_penalty = violation_penalty.item()
        self.total_interventions += 1
        
        return x_intervened, violation_penalty
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_len = input_ids.size(1)
        
        # Embeddings + positional encoding
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:seq_len, :]
        
        # Attention mask for transformer
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer layers with causal interventions
        total_causal_penalty = 0.0
        for layer_idx, layer in enumerate(self.transformer_layers):
            # Apply causal intervention before some layers
            if layer_idx % 3 == 0:  # Every 3rd layer
                x, penalty = self.apply_causal_intervention(x, attention_mask, layer_idx)
                total_causal_penalty += penalty
            
            # Transformer layer
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Final processing
        x = self.layer_norm(x)
        
        # Global average pooling with attention mask
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
            # Add causal penalty with proper weighting
            causal_penalty = total_causal_penalty * self.causal_reg_strength
            loss = classification_loss + causal_penalty
        
        return {'loss': loss, 'logits': logits}

class RigorousDataEfficiencyExperiment:
    """Rigorous experiment to prove/disprove data efficiency claims."""
    
    def __init__(self, device='cpu', random_seed=42):
        self.device = device
        self.random_seed = random_seed
        self.set_random_seeds(random_seed)
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Prepare comprehensive dataset
        texts, labels = zip(*RIGOROUS_SENTIMENT_DATASET)
        self.texts = list(texts)
        self.labels = list(labels)
        
        # Exact test points as specified
        self.data_percentages = [5, 10, 25, 50, 100]
        self.data_fractions = [p/100.0 for p in self.data_percentages]
        
        self.results = {}
        
        print(f"📊 Rigorous Dataset: {len(self.texts)} samples")
        print(f"🎯 Test Points: {self.data_percentages}% of data")
        print(f"🔍 Goal: Prove/disprove 2x-5x data efficiency improvement")
        
    def set_random_seeds(self, seed):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def create_balanced_split(self, fraction):
        """Create balanced train/test split for given fraction."""
        # Separate by class
        pos_samples = [(t, l) for t, l in zip(self.texts, self.labels) if l == 1]
        neg_samples = [(t, l) for t, l in zip(self.texts, self.labels) if l == 0]
        
        # Calculate sizes
        n_pos = max(int(len(pos_samples) * fraction), 1)
        n_neg = max(int(len(neg_samples) * fraction), 1)
        
        # Train/test split (80/20 but ensure minimum test size)
        n_pos_train = max(int(n_pos * 0.8), 1)
        n_neg_train = max(int(n_neg * 0.8), 1)
        n_pos_test = max(n_pos - n_pos_train, 1)
        n_neg_test = max(n_neg - n_neg_train, 1)
        
        # Create balanced splits
        train_samples = pos_samples[:n_pos_train] + neg_samples[:n_neg_train]
        test_samples = pos_samples[n_pos_train:n_pos_train+n_pos_test] + neg_samples[n_neg_train:n_neg_train+n_neg_test]
        
        # Shuffle
        np.random.shuffle(train_samples)
        np.random.shuffle(test_samples)
        
        # Extract texts and labels
        train_texts, train_labels = zip(*train_samples)
        test_texts, test_labels = zip(*test_samples)
        
        # Create datasets
        train_dataset = RigorousDataset(train_texts, train_labels, self.tokenizer)
        test_dataset = RigorousDataset(test_texts, test_labels, self.tokenizer)
        
        return train_dataset, test_dataset
    
    def train_and_evaluate(self, model_class, train_dataset, test_dataset, model_name, epochs=5):
        """Train and evaluate model with proper validation."""
        print(f"    🏋️ Training {model_name} ({len(train_dataset)} samples, {epochs} epochs)")
        
        # Create model
        model = model_class().to(self.device)
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Optimizer with proper learning rate
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, eps=1e-8)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
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
            
            scheduler.step()
            avg_loss = epoch_loss / num_batches
            training_losses.append(avg_loss)
            
            if epoch % 2 == 0:
                print(f"      Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
        
        # Comprehensive evaluation
        model.eval()
        all_predictions = []
        all_labels = []
        total_eval_loss = 0
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
                
                total_eval_loss += outputs['loss'].item()
                num_eval_batches += 1
                
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        avg_eval_loss = total_eval_loss / num_eval_batches
        
        # Causal-specific metrics
        causal_metrics = {}
        if hasattr(model, 'current_violation_penalty'):
            causal_metrics = {
                'violation_penalty': model.current_violation_penalty,
                'total_interventions': model.total_interventions,
                'avg_intervention_strength': torch.sigmoid(model.alpha * 0.5).mean().item() if hasattr(model, 'alpha') else 0.0
            }
        
        return {
            'model_name': model_name,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset),
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'eval_loss': avg_eval_loss,
            'final_train_loss': training_losses[-1],
            'training_losses': training_losses,
            'causal_metrics': causal_metrics
        }
    
    def run_rigorous_experiment(self):
        """Run the definitive data efficiency experiment."""
        print("🚀 RIGOROUS DATA EFFICIENCY EXPERIMENT")
        print("=" * 70)
        print("🎯 OBJECTIVE: Prove/disprove 2x-5x data efficiency improvement")
        print("🔍 METHOD: Performance vs Data Size Curve")
        print("📊 CHECKPOINTS: 5%, 10%, 25%, 50%, 100% of training data")
        print("=" * 70)
        
        for i, (percentage, fraction) in enumerate(zip(self.data_percentages, self.data_fractions)):
            print(f"\n📊 CHECKPOINT {i+1}/5: {percentage}% of training data")
            print("-" * 50)
            
            # Create datasets
            train_dataset, test_dataset = self.create_balanced_split(fraction)
            
            # Reset seeds for fair comparison
            self.set_random_seeds(self.random_seed + i)
            
            # Test standard model
            print("  🔷 Standard Model")
            standard_results = self.train_and_evaluate(
                StandardModel, train_dataset, test_dataset, "Standard", epochs=6
            )
            
            # Reset seeds again
            self.set_random_seeds(self.random_seed + i)
            
            # Test causal model
            print("  🔶 Causal Model")
            causal_results = self.train_and_evaluate(
                CausalModel, train_dataset, test_dataset, "Causal", epochs=6
            )
            
            # Calculate improvements
            accuracy_improvement = causal_results['accuracy'] - standard_results['accuracy']
            f1_improvement = causal_results['f1_score'] - standard_results['f1_score']
            
            # Store results
            self.results[percentage] = {
                'standard': standard_results,
                'causal': causal_results,
                'accuracy_improvement': accuracy_improvement,
                'f1_improvement': f1_improvement,
                'causal_winner': accuracy_improvement > 0
            }
            
            # Print summary
            print(f"  📈 Results:")
            print(f"    Standard: {standard_results['accuracy']:.3f} accuracy, {standard_results['f1_score']:.3f} F1")
            print(f"    Causal:   {causal_results['accuracy']:.3f} accuracy, {causal_results['f1_score']:.3f} F1")
            print(f"    Improvement: {accuracy_improvement:+.3f} accuracy ({accuracy_improvement*100:+.1f}%)")
            
            winner = "🏆 CAUSAL" if accuracy_improvement > 0 else "🏆 STANDARD"
            print(f"    Winner: {winner}")
        
        # Final analysis
        self.analyze_data_efficiency()
        self.create_performance_curve()
        self.save_rigorous_results()
        
    def analyze_data_efficiency(self):
        """Analyze if causal model achieves revolutionary data efficiency."""
        print("\n" + "=" * 70)
        print("🔬 DATA EFFICIENCY ANALYSIS")
        print("=" * 70)
        
        # Performance table
        print(f"{'Data %':<8} {'Standard':<10} {'Causal':<10} {'Improvement':<12} {'Winner':<10}")
        print("-" * 55)
        
        causal_wins = 0
        improvements = []
        
        for percentage in self.data_percentages:
            results = self.results[percentage]
            std_acc = results['standard']['accuracy']
            causal_acc = results['causal']['accuracy']
            improvement = results['accuracy_improvement']
            winner = "🏆 Causal" if improvement > 0 else "Standard"
            
            if improvement > 0:
                causal_wins += 1
            improvements.append(improvement)
            
            print(f"{percentage:>6}% {std_acc:>9.3f} {causal_acc:>9.3f} {improvement:>10.3f} {winner}")
        
        print("-" * 55)
        
        # Summary statistics
        avg_improvement = np.mean(improvements)
        max_improvement = np.max(improvements)
        min_improvement = np.min(improvements)
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Causal Wins: {causal_wins}/{len(self.data_percentages)} ({causal_wins/len(self.data_percentages)*100:.1f}%)")
        print(f"  Average Improvement: {avg_improvement:.3f} ({avg_improvement*100:.1f}%)")
        print(f"  Best Improvement: {max_improvement:.3f} ({max_improvement*100:.1f}%)")
        print(f"  Worst Performance: {min_improvement:.3f} ({min_improvement*100:.1f}%)")
        
        # Critical data efficiency analysis
        print(f"\n🎯 DATA EFFICIENCY ANALYSIS:")

        # Get baseline performance (100% standard)
        baseline_accuracy = self.results[100]['standard']['accuracy']
        target_accuracy = baseline_accuracy * 0.95  # 95% of baseline

        print(f"  Baseline (100% Standard): {baseline_accuracy:.3f} accuracy")
        print(f"  Target (95% of baseline): {target_accuracy:.3f} accuracy")

        # Find minimum data for standard model to reach target
        std_min_data = None
        for p in self.data_percentages:
            if self.results[p]['standard']['accuracy'] >= target_accuracy:
                std_min_data = p
                break
        
        if std_min_data is not None:
            print(f"  ✅ Standard model achieves target with {std_min_data}% data")
        else:
            print(f"  ❌ Standard model never reached target accuracy")

        # Find minimum data for causal model to reach target
        causal_min_data = None
        for p in self.data_percentages:
            if self.results[p]['causal']['accuracy'] >= target_accuracy:
                causal_min_data = p
                break

        if causal_min_data is not None:
            print(f"  ✅ Causal model achieves target with {causal_min_data}% data")
        else:
            print(f"  ❌ Causal model never reached target accuracy")

        # Analyze data efficiency improvement
        data_efficiency_achieved = False
        if causal_min_data is not None and std_min_data is not None:
            if causal_min_data < std_min_data:
                improvement_factor = std_min_data / causal_min_data
                print(f"  💰 Data savings: Causal model needs {std_min_data / causal_min_data:.1f}x less data.")
                if improvement_factor >= 2.0:
                    print(f"  🚀 REVOLUTIONARY: {improvement_factor:.1f}x data efficiency achieved!")
                    data_efficiency_achieved = True
                else:
                    print(f"  ⚠️  Sub-revolutionary: {improvement_factor:.1f}x < 2.0x threshold")
            elif causal_min_data == std_min_data:
                 print(f"  ⚖️ No data efficiency gain: Both models reach target at {causal_min_data}% data.")
            else: # causal_min_data > std_min_data
                print(f"  ❌ Causal model is LESS data efficient, needing {causal_min_data / std_min_data:.1f}x MORE data.")
        
        if not data_efficiency_achieved:
             if causal_min_data is None and std_min_data is None:
                print(f"  ❌ FAILED: Neither model reached target accuracy.")
             elif causal_min_data is None:
                print(f"  ❌ FAILED: Causal model never reached target accuracy.")
             elif std_min_data is not None and causal_min_data >= std_min_data:
                print(f"  ❌ FAILED: Causal model does not show data efficiency improvement.")


        # Final verdict
        print(f"\n🏁 FINAL VERDICT:")
        if data_efficiency_achieved:
            print(f"  ✅ REVOLUTIONARY DATA EFFICIENCY PROVEN")
            print(f"  🎯 Causal fine-tuning delivers on the promise")
        else:
            print(f"  ❌ REVOLUTIONARY DATA EFFICIENCY NOT PROVEN")
            print(f"  🔄 Architecture needs refinement for 2x-5x improvement")
    
    def create_performance_curve(self):
        """Create comprehensive performance vs data size visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        percentages = self.data_percentages
        standard_accs = [self.results[p]['standard']['accuracy'] for p in percentages]
        causal_accs = [self.results[p]['causal']['accuracy'] for p in percentages]
        standard_f1s = [self.results[p]['standard']['f1_score'] for p in percentages]
        causal_f1s = [self.results[p]['causal']['f1_score'] for p in percentages]
        improvements = [self.results[p]['accuracy_improvement'] for p in percentages]
        train_sizes = [self.results[p]['standard']['train_size'] for p in percentages]
        
        # Plot 1: Accuracy vs Data Percentage
        ax1.plot(percentages, standard_accs, 'o-', label='Standard Model', linewidth=2, markersize=8, color='blue')
        ax1.plot(percentages, causal_accs, 's-', label='Causal Model', linewidth=2, markersize=8, color='red')
        ax1.set_xlabel('Training Data Percentage (%)')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Performance vs Data Size: Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: F1 Score vs Data Percentage
        ax2.plot(percentages, standard_f1s, 'o-', label='Standard Model', linewidth=2, markersize=8, color='blue')
        ax2.plot(percentages, causal_f1s, 's-', label='Causal Model', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Training Data Percentage (%)')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Performance vs Data Size: F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Improvement by Data Amount
        colors = ['red' if x < 0 else 'green' for x in improvements]
        bars = ax3.bar(percentages, improvements, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Training Data Percentage (%)')
        ax3.set_ylabel('Accuracy Improvement (Causal - Standard)')
        ax3.set_title('Causal Model Advantage by Data Amount')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax3.annotate(f'{improvement:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Sample Efficiency
        ax4.plot(train_sizes, standard_accs, 'o-', label='Standard Model', linewidth=2, markersize=8, color='blue')
        ax4.plot(train_sizes, causal_accs, 's-', label='Causal Model', linewidth=2, markersize=8, color='red')
        ax4.set_xlabel('Number of Training Samples')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_title('Sample Efficiency Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('rigorous_data_efficiency_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Performance curve saved as 'rigorous_data_efficiency_curve.png'")
    
    def save_rigorous_results(self):
        """Save comprehensive experimental results."""
        # Calculate data efficiency metrics
        baseline_acc = self.results[100]['standard']['accuracy']
        data_efficiency_factor = None
        
        for percentage in self.data_percentages:
            causal_acc = self.results[percentage]['causal']['accuracy']
            if causal_acc >= baseline_acc * 0.95:  # 95% of baseline
                data_efficiency_factor = 100 / percentage
                break
        
        results_data = {
            'experiment_metadata': {
                'timestamp': datetime.now().isoformat(),
                'objective': 'Prove/disprove 2x-5x data efficiency improvement',
                'dataset_size': len(self.texts),
                'test_points': self.data_percentages,
                'random_seed': self.random_seed,
                'revolutionary_threshold': '2x-5x improvement'
            },
            'detailed_results': self.results,
            'summary_analysis': {
                'causal_wins': sum(1 for p in self.data_percentages if self.results[p]['causal_winner']),
                'total_tests': len(self.data_percentages),
                'win_rate': sum(1 for p in self.data_percentages if self.results[p]['causal_winner']) / len(self.data_percentages),
                'average_improvement': np.mean([self.results[p]['accuracy_improvement'] for p in self.data_percentages]),
                'max_improvement': np.max([self.results[p]['accuracy_improvement'] for p in self.data_percentages]),
                'min_improvement': np.min([self.results[p]['accuracy_improvement'] for p in self.data_percentages])
            },
            'data_efficiency_verdict': {
                'baseline_accuracy': baseline_acc,
                'data_efficiency_factor': data_efficiency_factor,
                'revolutionary_achieved': data_efficiency_factor is not None and data_efficiency_factor >= 2.0,
                'verdict': 'REVOLUTIONARY' if data_efficiency_factor is not None and data_efficiency_factor >= 2.0 else 'NOT_REVOLUTIONARY'
            }
        }
        
        filename = f'rigorous_data_efficiency_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Rigorous results saved to {filename}")

def main():
    """Run the rigorous data efficiency experiment."""
    print("🎯 RIGOROUS DATA EFFICIENCY EXPERIMENT")
    print("🔬 Definitive test of 2x-5x data efficiency claim")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Run experiment
    experiment = RigorousDataEfficiencyExperiment(device=device)
    experiment.run_rigorous_experiment()
    
    print("\n✅ RIGOROUS EXPERIMENT COMPLETED!")
    print("📊 Comprehensive analysis of data efficiency claims")
    print("🎯 Revolutionary threshold: 2x-5x improvement")
    print("📈 Results visualized in performance curve")

if __name__ == "__main__":
    main() 