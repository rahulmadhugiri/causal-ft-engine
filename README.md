# Causal Fine-Tuning Engine

A neural network framework that injects do-operator-based causal reasoning directly into neural architectures, reducing data requirements for model training through causal interventions.

## Project Goals

Rather than adding interventions at the data level, this engine modifies the neural architecture itself to:

- Inject `do(X=v)`-style logic into the forward pass
- Block gradients flowing to parents of intervened nodes  
- Optionally rewire a runtime DAG during forward/backward passes
- Enable causal consistency and counterfactual reasoning

## Architecture

```
causal-finetuning-engine/
├── engine/
│   ├── __init__.py            # Main imports and version info
│   ├── causal_unit.py         # CausalUnit & CausalMLP classes
│   ├── rewiring.py            # DAG rewiring and edge masking
│   └── loss_functions.py      # Causal-aware loss functions
├── experiments/
│   ├── run_phase1.py          # Main experiment runner
│   ├── utils.py               # Utility functions for experiments
│   └── phase1_training_summary.png  # Generated results plot
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## Quick Start

### Installation

```bash
git clone <repository>
cd causal-finetuning-engine
pip install -r requirements.txt
```

### Basic Usage

```python
import torch
from engine import CausalMLP, CausalLosses

# Create a causal neural network
model = CausalMLP(input_dim=3, hidden_dims=[16, 16], output_dim=1)

# Set up causal loss functions
causal_loss = CausalLosses(intervention_weight=1.0)

# Define an intervention: do(x2 = 0.5)
do_mask = torch.tensor([0.0, 1.0, 0.0])    # Intervene on x2
do_values = torch.tensor([0.0, 0.5, 0.0])   # Set x2 = 0.5

# Forward pass with intervention
x = torch.randn(10, 3)  # Batch of inputs
y_pred = model(x, interventions={0: (do_mask, do_values)})
```

## Key Components

### 1. CausalUnit
A neural network layer that implements do-operator interventions:
- **Gradient Blocking**: Detaches intervened variables from computation graph
- **Value Replacement**: Substitutes intervened variables with specified values
- **Debugging**: Tracks intervention history for analysis

### 2. DAGRewiring  
Manages causal graph structure and edge masking:
- **Edge Masking**: Zeros out weights feeding into intervened nodes
- **Topological Sorting**: Maintains causal ordering
- **Dynamic Rewiring**: Modifies graph structure based on interventions

### 3. CausalLosses
Collection of causal-aware loss functions:
- **Intervention Loss**: Optionally ignores loss on intervened variables
- **Counterfactual Loss**: Encourages correct "what if" predictions  
- **Causal Consistency**: Enforces that `do(X=v)` results in X=v
- **Regularization**: Promotes sparse, interpretable causal structures

## Phase 1 Results

Our toy example demonstrates the engine working on synthetic data where `y = x1 + 2*x2 - 3*x3 + noise`:

**Key Findings:**
- Causal models successfully respond to `do(x2=0.5)` interventions
- Gradient blocking prevents information flow to intervened variables
- Models learn to approximate true causal coefficients
- Training converges with causal-aware loss functions

**Experimental Setup:**
- **Dataset**: 1000 samples with known causal relationships
- **Intervention**: `do(x2 = 0.5)` applied to 50% of training data
- **Comparison**: Standard MLP vs CausalMLP
- **Metrics**: MSE loss, coefficient recovery, intervention accuracy

## Running the Demo

Execute the complete Phase 1 experiment:

```bash
python experiments/run_phase1.py
```

The experiment includes:
1. **Data Generation**: Synthetic dataset with known causal structure
2. **Model Comparison**: Baseline vs causal architectures  
3. **Training**: With and without causal interventions
4. **Evaluation**: Performance metrics and intervention testing
5. **Visualization**: Training curves saved as PNG files

This approach provides:
- **Cleaner dev cycle** (no notebook clutter)
- **Reproducible experiments** (consistent results)
- **Professional structure** (real engine, not just research demo)

## Technical Details

### Gradient Blocking Implementation
```python
# For intervened variables, detach from computation graph
x_detached = x.detach()
x_final = torch.where(intervened_indices, do_values, x)
x_final = torch.where(intervened_indices, x_detached, x_final)
```

### Edge Masking for DAG Rewiring
```python
# Zero out incoming weights to intervened variables
for var_idx in intervened_vars:
    masked_weights[:, var_idx] = 0.0
```

### Causal Loss Functions
```python
# Ignore loss on intervened variables
loss_mask = (do_mask == 0).float()
masked_loss = loss * loss_mask
return masked_loss.sum() / loss_mask.sum()
```

## Future Roadmap

### Phase 2: Advanced Causal Reasoning
- **Counterfactual Learning**: "What if X had been Y?"
- **Dynamic DAG Discovery**: Learn causal structure from data
- **Multi-variable Interventions**: Complex intervention patterns
- **Conditional Interventions**: Context-dependent do-operators

### Phase 3: Real-world Applications  
- **Medical Diagnosis**: Causal intervention modeling
- **Recommendation Systems**: Counterfactual personalization
- **Scientific Discovery**: Automated causal hypothesis testing
- **Policy Optimization**: Intervention effect prediction

## Dependencies

- `torch`: Core neural network framework
- `numpy`: Numerical computations
- `pandas`: Data manipulation  
- `scikit-learn`: Machine learning utilities
- `jupyter`: Interactive development
- `matplotlib`: Visualization

## Contributing

We welcome contributions! Areas of interest:
- Novel causal architectures
- Advanced intervention strategies  
- Real-world dataset applications
- Performance optimizations
- Documentation improvements

## License

[Add your license here]

## Contact

[Add contact information]
