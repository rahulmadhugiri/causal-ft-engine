import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.loss_functions import CausalLosses


def generate_synthetic_data(n_samples=1000, n_nodes=4, graph_type='chain', noise_level=0.3):
    """
    Generate synthetic data with known causal relationships.
    
    Args:
        n_samples: Number of samples to generate
        n_nodes: Number of nodes in the graph
        graph_type: Type of graph ('chain', 'fork', 'v_structure', 'confounder')
        noise_level: Standard deviation of noise
        
    Returns:
        X: Input features (n_samples, n_nodes-1)
        y: Target values (n_samples, 1)
        true_adjacency: True adjacency matrix (n_nodes-1, n_nodes-1)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    if graph_type == 'chain':
        # X1 -> X2 -> X3 -> Y
        x1 = np.random.randn(n_samples, 1)
        x2 = 0.8 * x1 + noise_level * np.random.randn(n_samples, 1)
        x3 = 0.6 * x2 + noise_level * np.random.randn(n_samples, 1)
        y = 0.5 * x3 + noise_level * np.random.randn(n_samples, 1)
        
        X = np.hstack([x1, x2, x3])
        true_adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        
    elif graph_type == 'fork':
        # X1 -> X2, X1 -> X3, X2 -> Y, X3 -> Y
        x1 = np.random.randn(n_samples, 1)
        x2 = 0.7 * x1 + noise_level * np.random.randn(n_samples, 1)
        x3 = 0.5 * x1 + noise_level * np.random.randn(n_samples, 1)
        y = 0.4 * x2 + 0.3 * x3 + noise_level * np.random.randn(n_samples, 1)
        
        X = np.hstack([x1, x2, x3])
        true_adjacency = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        
    elif graph_type == 'v_structure':
        # X1 -> X3, X2 -> X3, X3 -> Y
        x1 = np.random.randn(n_samples, 1)
        x2 = np.random.randn(n_samples, 1)
        x3 = 0.6 * x1 + 0.5 * x2 + noise_level * np.random.randn(n_samples, 1)
        y = 0.7 * x3 + noise_level * np.random.randn(n_samples, 1)
        
        X = np.hstack([x1, x2, x3])
        true_adjacency = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        
    else:  # confounder or default
        # X1 -> X2, X1 -> X3, X2 -> Y, X3 -> Y
        x1 = np.random.randn(n_samples, 1)
        x2 = 0.8 * x1 + noise_level * np.random.randn(n_samples, 1)
        x3 = 0.6 * x1 + noise_level * np.random.randn(n_samples, 1)
        y = 0.5 * x2 + 0.4 * x3 + noise_level * np.random.randn(n_samples, 1)
        
        X = np.hstack([x1, x2, x3])
        true_adjacency = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    
    return X, y, true_adjacency


def create_dag_from_edges(edges, n_nodes):
    """Create adjacency matrix from edge list."""
    adj = np.zeros((n_nodes, n_nodes))
    for edge in edges:
        adj[edge[0], edge[1]] = 1
    return adj


def evaluate_structure_learning(learned_adj, true_adj):
    """Evaluate structure learning performance."""
    # Flatten and binarize
    true_flat = (true_adj.flatten() > 0.5).astype(int)
    learned_flat = (learned_adj.flatten() > 0.5).astype(int)
    
    # Basic metrics
    tp = np.sum((true_flat == 1) & (learned_flat == 1))
    fp = np.sum((true_flat == 0) & (learned_flat == 1))
    fn = np.sum((true_flat == 1) & (learned_flat == 0))
    tn = np.sum((true_flat == 0) & (learned_flat == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_counterfactual_performance(model, x, y, true_adj):
    """Evaluate counterfactual performance."""
    model.eval()
    effects = []
    
    with torch.no_grad():
        # Original prediction
        y_original = model(x)
        
        # Test several random interventions
        for _ in range(10):
            intervention_node = np.random.randint(0, x.shape[1])
            intervention_value = torch.randn(1) * 2.0
            
            # Create intervention
            intervention_mask = torch.zeros(x.shape[1])
            intervention_values = torch.zeros(x.shape[1])
            intervention_mask[intervention_node] = 1.0
            intervention_values[intervention_node] = intervention_value
            
            # Apply intervention
            interventions = []
            for i in range(x.shape[0]):
                interventions.append({'test': (intervention_mask, intervention_values)})
            
            y_counterfactual = model(x, interventions=interventions)
            effect = torch.mean(torch.abs(y_counterfactual - y_original))
            effects.append(effect.item())
    
    return {'mean_effect': np.mean(effects), 'std_effect': np.std(effects)}


def generate_causal_data(n_samples=1000, noise_std=0.1):
    """
    Generate synthetic data with known causal relationships:
    y = x1 + 2*x2 - 3*x3 + noise
    
    Args:
        n_samples: Number of samples to generate
        noise_std: Standard deviation of noise
        
    Returns:
        X: Input features tensor (n_samples, 3)
        y: Target values tensor (n_samples, 1)
        true_coeffs: True causal coefficients tensor (3,)
    """
    # Generate input features
    X = torch.randn(n_samples, 3)  # [x1, x2, x3]
    
    # True causal relationships
    true_coeffs = torch.tensor([1.0, 2.0, -3.0])
    
    # Generate target with known causal structure
    y = torch.sum(X * true_coeffs, dim=1, keepdim=True) + noise_std * torch.randn(n_samples, 1)
    
    return X, y, true_coeffs

class BaselineMLP(nn.Module):
    """Standard MLP for comparison."""
    
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=1):
        super(BaselineMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(model, X_train, y_train, X_test, y_test, 
                is_causal=False, num_epochs=100, lr=0.01, verbose=True):
    """
    Train a model with optional causal interventions.
    
    Args:
        model: Neural network model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        is_causal: Whether to apply causal interventions
        num_epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print training progress
        
    Returns:
        train_losses: List of training losses
        test_losses: List of test losses
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    
    # Initialize causal loss function
    causal_losses = CausalLosses(intervention_weight=1.0)
    
    # Set up interventions for causal model
    if is_causal:
        # Intervention: set x2 = 0.5 for 50% of training samples
        intervention_prob = 0.5
        do_values = torch.tensor([0.0, 0.5, 0.0])  # Only intervene on x2
        do_mask = torch.tensor([0.0, 1.0, 0.0])    # Mask for x2
    
    for epoch in range(num_epochs):
        model.train()
        
        # Training
        optimizer.zero_grad()
        
        if is_causal:
            # Apply interventions to random subset of training data
            batch_size = X_train.shape[0]
            intervention_indices = torch.rand(batch_size) < intervention_prob
            
            # Create batch-level masks
            batch_do_mask = torch.zeros(batch_size, 3)
            batch_do_values = torch.zeros(batch_size, 3)
            
            if intervention_indices.any():
                batch_do_mask[intervention_indices] = do_mask
                batch_do_values[intervention_indices] = do_values
                
                # For causal model, use interventions
                pred_train = model(X_train, interventions={0: (batch_do_mask, batch_do_values)})
            else:
                pred_train = model(X_train)
            
            # Use causal loss
            train_loss = causal_losses.causal_intervention_loss(
                pred_train, y_train, batch_do_mask
            )
        else:
            # Standard training for baseline
            pred_train = model(X_train)
            train_loss = nn.MSELoss()(pred_train, y_train)
        
        train_loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            if is_causal:
                pred_test = model(X_test)
            else:
                pred_test = model(X_test)
            test_loss = nn.MSELoss()(pred_test, y_test)
        
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        if verbose and epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss.item():.4f}, Test Loss = {test_loss.item():.4f}")
    
    return train_losses, test_losses

def evaluate_models(causal_model, baseline_model, X_test, y_test, true_coeffs):
    """
    Evaluate and compare the trained models.
    
    Args:
        causal_model: Trained causal model
        baseline_model: Trained baseline model
        X_test, y_test: Test data
        true_coeffs: True causal coefficients
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Final test performance
    baseline_model.eval()
    causal_model.eval()

    with torch.no_grad():
        baseline_pred = baseline_model(X_test)
        causal_pred = causal_model(X_test)
        
        baseline_mse = nn.MSELoss()(baseline_pred, y_test)
        causal_mse = nn.MSELoss()(causal_pred, y_test)

    print(f"Final Test MSE:")
    print(f"  Baseline Model: {baseline_mse.item():.4f}")
    print(f"  Causal Model:   {causal_mse.item():.4f}")
    
    if causal_mse < baseline_mse:
        improvement = ((baseline_mse - causal_mse) / baseline_mse * 100).item()
        print(f"  Improvement:    {improvement:.2f}%")
    else:
        degradation = ((causal_mse - baseline_mse) / baseline_mse * 100).item()
        print(f"  Degradation:    {degradation:.2f}%")

    # Check if causal model learned the correct coefficients
    # Extract weights from first layer (should approximate true coefficients)
    first_layer_weights = causal_model.layers[0].layers[0].weight.data[0]  # First output neuron
    print(f"\nCoefficient Recovery:")
    print(f"  True coefficients:    {true_coeffs}")
    print(f"  Learned coefficients: {first_layer_weights}")
    print(f"  Coefficient error:    {torch.abs(true_coeffs - first_layer_weights)}")
    
    # Test causal interventions
    print(f"\n" + "="*50)
    print("INTERVENTION TESTING")
    print("="*50)
    
    # Test model's response to interventions
    test_input = torch.tensor([[1.0, 2.0, 3.0]])  # Single test sample
    print(f"Test input: {test_input[0]}")

    # Normal prediction
    normal_pred = causal_model(test_input)
    print(f"Normal prediction: {normal_pred.item():.3f}")

    # Intervention: do(x2 = 0.5)
    do_mask = torch.tensor([[0.0, 1.0, 0.0]])
    do_values = torch.tensor([[0.0, 0.5, 0.0]])

    intervention_pred = causal_model(test_input, interventions={0: (do_mask, do_values)})
    print(f"With do(x2=0.5):   {intervention_pred.item():.3f}")

    # Calculate expected change based on true coefficients
    # True model: y = 1*x1 + 2*x2 - 3*x3
    # Original: y = 1*1 + 2*2 - 3*3 = 1 + 4 - 9 = -4
    # Intervention: y = 1*1 + 2*0.5 - 3*3 = 1 + 1 - 9 = -7
    # Expected change: -7 - (-4) = -3

    original_expected = 1*1.0 + 2*2.0 - 3*3.0
    intervention_expected = 1*1.0 + 2*0.5 - 3*3.0
    expected_change = intervention_expected - original_expected

    actual_change = intervention_pred.item() - normal_pred.item()

    print(f"\nIntervention Analysis:")
    print(f"  Expected change: {expected_change:.3f}")
    print(f"  Actual change:   {actual_change:.3f}")
    print(f"  Intervention accuracy: {abs(expected_change - actual_change):.3f}")

def plot_training_curves(baseline_train_losses, baseline_test_losses,
                        causal_train_losses, causal_test_losses,
                        filename="phase1_training_summary.png"):
    """
    Plot and save training curves comparison.
    
    Args:
        baseline_train_losses, baseline_test_losses: Baseline model losses
        causal_train_losses, causal_test_losses: Causal model losses
        filename: Name of file to save plot
    """
    plt.figure(figsize=(15, 5))

    # Training loss
    plt.subplot(1, 3, 1)
    plt.plot(baseline_train_losses, label='Baseline', alpha=0.7)
    plt.plot(causal_train_losses, label='Causal', alpha=0.7)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Test loss
    plt.subplot(1, 3, 2)
    plt.plot(baseline_test_losses, label='Baseline', alpha=0.7)
    plt.plot(causal_test_losses, label='Causal', alpha=0.7)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Final comparison
    plt.subplot(1, 3, 3)
    categories = ['Final Train Loss', 'Final Test Loss']
    baseline_final = [baseline_train_losses[-1], baseline_test_losses[-1]]
    causal_final = [causal_train_losses[-1], causal_test_losses[-1]]

    x = np.arange(len(categories))
    width = 0.35

    plt.bar(x - width/2, baseline_final, width, label='Baseline', alpha=0.7)
    plt.bar(x + width/2, causal_final, width, label='Causal', alpha=0.7)
    plt.title('Final Performance Comparison')
    plt.ylabel('MSE Loss')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nTraining plots saved to: {filename}") 