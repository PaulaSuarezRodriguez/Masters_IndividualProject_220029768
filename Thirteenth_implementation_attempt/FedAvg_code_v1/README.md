# FedAvg Aggregation for LwF-Trained Models

A local implementation of Federated Averaging (FedAvg) to combine model weights from multiple Learning without Forgetting (LwF) trained models into a single aggregated model.

## Overview

**Goal**: Aggregate weights from multiple locally trained models without server-client communication

**Use Case**: Combine knowledge from multiple LwF-trained models into a single universal model

**Key Feature**: Data-driven weighting based on training dataset sizes

---

## File Structure

### 1. **fedavg_aggregator.py** (Core Implementation)
Main module containing the `FedAvgAggregator` class.

**Key Classes**:
- `FedAvgAggregator`: Main aggregator with uniform and data-driven weighting

**Key Methods**:
- `load_model_weights_from_files()`: Load weights from saved checkpoints
- `aggregate()`: Perform FedAvg aggregation
- `aggregate_into_model()`: Load aggregated weights into a model
- `save_aggregated_weights()`: Save aggregated model
- `get_aggregation_report()`: View aggregation details

**Example**:
```python
from fedavg_aggregator import FedAvgAggregator

aggregator = FedAvgAggregator(weighting_strategy='data_driven')
aggregator.load_model_weights_from_files(
    ['model1.pth', 'model2.pth', 'model3.pth'],
    dataset_sizes=[1000, 1500, 800]
)
aggregator.save_aggregated_weights('aggregated.pth')
```

---

### 2. **fedavg_example.py** (Practical Examples)
Three complete scenarios demonstrating FedAvg usage:

**Scenario 1: Uniform Aggregation**
- Use when models are equally important
- Simple averaging: w_final = (1/3)*(w_1 + w_2 + w_3)

**Scenario 2: Data-Driven Aggregation**
- Use when models trained on different dataset sizes
- Size-weighted averaging: w_final = Σ (n_i/n)*w_i

**Scenario 3: Domain Ensemble**
- Aggregate models trained on different domains
- Create universal feature extractor

**Run Examples**:
```bash
python fedavg_example.py
```

---

### 3. **FEDAVG_THEORY.md** (Mathematical Background)
Comprehensive theory document covering:

- Algorithm description and formulas
- Weighting strategies (uniform vs. data-driven)
- Mathematical properties and convergence
- Why FedAvg works for LwF
- Implementation details
- Common pitfalls and solutions
- Advanced aggregation strategies

**Key Sections**:
1. Basic FedAvg formula
2. Two weighting strategies
3. Why FedAvg works for LwF models
4. Mathematical properties
5. Step-by-step example
6. Implementation details
7. Common pitfalls and solutions
8. Practical recommendations

---

### 4. **QUICKSTART.md** (Quick Reference)
Fast-track guide with:

- 60-second overview
- 4 common usage patterns
- Complete workflow example
- Key concepts explained
- Common questions and answers
- Troubleshooting guide
- API reference
- Performance tips

**Best for**: Getting started quickly without deep theory

---

## Quick Start

### Installation
No external dependencies beyond PyTorch:
```bash
# Ensure PyTorch is installed
pip install torch
```

### Basic Usage

```python
from fedavg_aggregator import FedAvgAggregator
import torch.nn as nn

# Step 1: Create aggregator
aggregator = FedAvgAggregator(weighting_strategy='data_driven')

# Step 2: Load model weights
aggregator.load_model_weights_from_files(
    model_paths=['model_task1.pth', 'model_task2.pth', 'model_task3.pth'],
    dataset_sizes=[1000, 1500, 900]  # samples per task
)

# Step 3: View aggregation report
report = aggregator.get_aggregation_report()
print(f"Aggregating {report['num_models']} models")
print(f"Model weights: {report['model_weights']}")
print(f"Total samples: {report['total_samples']}")

# Step 4: Save aggregated model
aggregator.save_aggregated_weights('aggregated_model.pth')

# OR load into existing model instance
your_model = YourModelClass()
your_model = aggregator.aggregate_into_model(your_model)
```

---

## Algorithm Summary

### The FedAvg Formula

```
w_final = Σ(i=1 to K) [n_i / n_total] * w_i
```

Where:
- **w_i**: Weights from model i
- **n_i**: Training samples for model i
- **n_total**: Total training samples
- **K**: Number of models

### Two Weighting Strategies

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| **Uniform** | (1/K) × (w_1 + ... + w_K) | Models equally important, unknown data sizes |
| **Data-Driven** | Σ (n_i/n) × w_i | Models trained on different data sizes (recommended) |

---

## Features

✓ **Local Aggregation**: No server or distributed infrastructure needed

✓ **Flexible Weighting**: Both uniform and data-driven strategies

✓ **Clean API**: Simple, intuitive interface

✓ **State Dict Compatible**: Works with any PyTorch model

✓ **Detailed Reporting**: View aggregation weights and statistics

✓ **Production Ready**: Error handling and validation

✓ **Well Documented**: Theory, examples, and quick-start guides

---

## Workflow

### 1. Train Multiple Models with LwF
```python
for task in tasks:
    model = train_lwf(model, task)
    torch.save(model.state_dict(), f'models/task_{i}_lwf.pth')
```

### 2. Aggregate Weights
```python
aggregator = FedAvgAggregator(weighting_strategy='data_driven')
aggregator.load_model_weights_from_files(paths, sizes)
aggregator.save_aggregated_weights('aggregated.pth')
```

### 3. Evaluate Aggregated Model
```python
agg_model = load_model('aggregated.pth')
accuracy = evaluate(agg_model, test_data)
```

### 4. Deploy
```python
agg_model.eval()
predictions = agg_model(new_data)
```

---

## Key Concepts

### FedAvg vs. Traditional Federated Learning

| Aspect | Traditional FL | Your Use Case |
|--------|----------------|---------------|
| **Setup** | Distributed clients + server | Single local machine |
| **Training** | Iterative rounds | Pre-trained models |
| **Communication** | Round-trip server communication | None (local only) |
| **Use of FedAvg** | Iterative weight updates | One-shot aggregation |
| **Goal** | Collaborative learning | Local model combination |

### Why FedAvg Works for LwF

1. **Task Specialization**: Each LwF model specializes in different tasks
2. **Knowledge Retention**: Each model remembers previous tasks (via distillation)
3. **Smart Aggregation**: FedAvg intelligently combines these specializations
4. **Result**: Universal model with knowledge from all tasks

---

## Common Use Cases

### 1. Multi-Task Learning
Combine models trained on different tasks into one universal model

### 2. Domain Adaptation
Aggregate domain-specific models to create domain-general feature extractor

### 3. Sequential Learning
Combine models from different training phases

### 4. Ensemble Aggregation
Create a single model from multiple trained models

### 5. Knowledge Consolidation
Consolidate weights from multiple LwF-trained models

---

## Weighting Strategies

### Uniform Weighting (Simple)
```python
aggregator = FedAvgAggregator(weighting_strategy='uniform')
# Weight: [0.333, 0.333, 0.333] for 3 models
```

Use when:
- Models equally important
- Unknown dataset sizes
- Similar training amounts

### Data-Driven Weighting (Recommended)
```python
aggregator = FedAvgAggregator(weighting_strategy='data_driven')
# Weight: [0.303, 0.455, 0.242] for datasets [1000, 1500, 800]
```

Use when:
- Models trained on different data sizes
- Want to weight by data amount
- Dataset sizes are known

---

## Validation

Always validate the aggregated model:

```python
# Individual model accuracies
individual_accs = [evaluate(m, test) for m in models]

# Aggregated model accuracy
agg_model = load_model('aggregated.pth')
agg_acc = evaluate(agg_model, test)

print(f"Individual: {individual_accs}")
print(f"Aggregated: {agg_acc}")

if agg_acc > mean(individual_accs):
    print("✓ Aggregation improved performance!")
else:
    print("✗ Aggregation did not help (use best individual model)")
```

---

## Performance Considerations

### Time Complexity
- O(M × D) where M = number of models, D = model size
- Typically very fast (seconds for most models)

### Memory Usage
- Loads all model weights into memory
- Moderate memory footprint

### When to Use Each Strategy

| Scenario | Strategy | Reason |
|----------|----------|--------|
| Small uniform datasets | Uniform | Simple and effective |
| Large size variation | Data-driven | Prevents small-data bias |
| Unknown sizes | Uniform | Safe default |
| Many models (>10) | Data-driven | Better scaling |

---

## Troubleshooting

### Models have different architectures
**Problem**: KeyError when loading weights

**Solution**: Ensure all models use the same architecture class
```python
model1 = YourModel()  # Save: torch.save(model1.state_dict(), ...)
model2 = YourModel()  # Same class!
```

### Poor aggregation results
**Problem**: Aggregated model underperforms individuals

**Solution**: 
1. Try fine-tuning: `fine_tune(agg_model, diverse_data)`
2. Check weighting: `aggregator.get_aggregation_report()`
3. Verify validation: Test on held-out data

### Memory issues with large models
**Problem**: Out of memory

**Solution**: 
1. Aggregate to CPU (already default)
2. Process models sequentially if needed
3. Use smaller models or fewer of them

---

## API Reference

### FedAvgAggregator

```python
class FedAvgAggregator:
    def __init__(self, weighting_strategy='uniform')
    
    # Loading methods
    def add_model_weights(self, model_weights, dataset_size=None)
    def load_model_weights_from_files(self, weight_paths, dataset_sizes=None)
    
    # Aggregation methods
    def aggregate() → Dict[str, Tensor]
    def aggregate_into_model(self, model) → nn.Module
    def save_aggregated_weights(self, save_path)
    
    # Utility methods
    def get_aggregation_report() → Dict
    def reset()
```

---

## Examples

### Example 1: Simple Uniform Aggregation
```python
from fedavg_aggregator import FedAvgAggregator

aggregator = FedAvgAggregator(weighting_strategy='uniform')
aggregator.load_model_weights_from_files([
    'model1.pth',
    'model2.pth',
    'model3.pth'
])
aggregator.save_aggregated_weights('final.pth')
```

### Example 2: Size-Weighted Aggregation
```python
from fedavg_aggregator import FedAvgAggregator

aggregator = FedAvgAggregator(weighting_strategy='data_driven')
aggregator.load_model_weights_from_files(
    ['model1.pth', 'model2.pth', 'model3.pth'],
    dataset_sizes=[1000, 1500, 900]
)

report = aggregator.get_aggregation_report()
print(f"Model weights: {report['model_weights']}")

aggregator.save_aggregated_weights('final.pth')
```

### Example 3: Load into Model
```python
from fedavg_aggregator import FedAvgAggregator

aggregator = FedAvgAggregator()
aggregator.load_model_weights_from_files([...])

model = MyModel()
model = aggregator.aggregate_into_model(model)

# Use for inference
model.eval()
output = model(data)
```

---

## References

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Aguerri, B. A. (2017).
  "Communication-Efficient Learning of Deep Networks from Decentralized Data."
  *ICML 2017*

- Li, T., Sahu, A. K., Zaheer, M., Shlyan, M., Talwalkar, A., & Smith, V. (2020).
  "Federated Optimization in Heterogeneous Networks." *MLSys 2020*

---

## Summary

| File | Purpose |
|------|---------|
| `fedavg_aggregator.py` | Core FedAvg implementation |
| `fedavg_example.py` | Practical examples and scenarios |
| `FEDAVG_THEORY.md` | Mathematical theory and background |
| `QUICKSTART.md` | Quick reference and API |
| `README.md` | This file - overview and guide |

**Start Here**: 
- For quick start → `QUICKSTART.md`
- For examples → `fedavg_example.py`
- For theory → `FEDAVG_THEORY.md`
- For implementation → `fedavg_aggregator.py`

---

## Support

For issues or questions:
1. Check `QUICKSTART.md` troubleshooting section
2. Review theory in `FEDAVG_THEORY.md`
3. Check examples in `fedavg_example.py`
4. Validate setup with `get_aggregation_report()`

---

Good luck with your FedAvg aggregation! 🚀
