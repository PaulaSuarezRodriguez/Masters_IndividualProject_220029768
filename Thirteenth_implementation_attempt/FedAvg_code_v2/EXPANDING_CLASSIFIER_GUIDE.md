# FedAvg with Expanding Classifiers - Quick Reference

## Problem

Your models have **different architectures** because the classifier grows:
- Model 1: 10 classes
- Model 2: 15 classes (10 old + 5 new)
- Model 3: 22 classes (15 old + 7 new)

Standard FedAvg fails because:
```
Model 1 classifier.weight: [10, 256]
Model 2 classifier.weight: [15, 256]  ← Different shape!
Model 3 classifier.weight: [22, 256]

ERROR: stack expects each tensor to be equal size
```

## Solution

Aggregate **only the shared portion** (first 10 classes) and handle expanding dimensions intelligently.

---

## 3 Ways to Solve This

### Option 1: Use New Aggregator (Recommended) ⭐

```python
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier

# Initialize with expanding layers specified
aggregator = FedAvgAggregatorExpandingClassifier(
    weighting_strategy='data_driven',
    expanding_layers=['classifier.weight', 'classifier.bias']  # ← Specify which expand
)

# Load your models
aggregator.load_model_weights_from_files(
    ['task1.pth', 'task2.pth', 'task3.pth'],
    dataset_sizes=[1000, 1200, 900]
)

# View details
aggregator.print_aggregation_details()

# Aggregate (automatically handles expanding layers)
aggregator.save_aggregated_weights('aggregated.pth')
```

**Why this is best:**
- ✓ Automatic shape handling
- ✓ Element-wise aggregation included
- ✓ Clear and readable
- ✓ Production-ready error handling

---

### Option 2: Auto-Detect Expanding Layers

```python
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier

# Don't specify - let it auto-detect
aggregator = FedAvgAggregatorExpandingClassifier(weighting_strategy='data_driven')

# Load models
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# Auto-detect will find expanding layers
expanding = aggregator.detect_expanding_layers()
print(f"Expanding layers: {expanding}")

# Aggregate (auto-handles)
aggregator.save_aggregated_weights('aggregated.pth')
```

**When to use:**
- Unsure which layers expand
- More exploratory approach
- Good for debugging

---

### Option 3: Your Custom Element-Wise Code (Manual)

Adapted from your code snippet:

```python
import torch
import numpy as np

# Your setup
weights_task1 = torch.load('task1.pth')
weights_task2 = torch.load('task2.pth')
weights_task3 = torch.load('task3.pth')

list_load = [weights_task1, weights_task2, weights_task3]

# Group by layer (your code)
weights_by_layer = {}
for key in list_load[0]:
    if "weight" in key:
        weights_by_layer[key] = []
        for state_dict in list_load:
            if key in state_dict:
                weights_by_layer[key].append(state_dict[key])

# Aggregate with shape handling
aggregated = {}
agg_weights = np.array([0.303, 0.455, 0.242])  # Data-driven

for layer_name, tensors in weights_by_layer.items():
    shapes = [t.shape for t in tensors]
    is_expanding = len(set(shapes)) > 1
    
    if is_expanding:
        # Expanding layer: take minimum shape only
        common_shape = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))
        print(f"{layer_name}: aggregating shared portion {common_shape}")
        
        # Slice to common shape
        shared = [t[tuple(slice(0, size) for size in common_shape)] for t in tensors]
        
        # Element-wise weighted sum
        stacked = torch.stack(shared)
        agg_weights_t = torch.tensor(agg_weights).view(-1, *([1] * (stacked.dim() - 1)))
        aggregated[layer_name] = (stacked * agg_weights_t).sum(dim=0)
    
    else:
        # Fixed layer: normal aggregation
        stacked = torch.stack(tensors)
        agg_weights_t = torch.tensor(agg_weights).view(-1, *([1] * (stacked.dim() - 1)))
        aggregated[layer_name] = (stacked * agg_weights_t).sum(dim=0)

torch.save(aggregated, 'aggregated.pth')
```

**When to use:**
- Maximum control
- Custom logic needed
- Educational purposes

---

## Key Concept: Element-Wise Aggregation

**Element-wise** means: for each individual element in each tensor, compute the weighted average.

```python
# For a specific parameter (e.g., classifier.weight):

# Step 1: Stack weights from all models
stacked = torch.stack([
    model1['classifier.weight'],  # [10, 256]
    model2['classifier.weight'],  # [15, 256]
    model3['classifier.weight']   # [22, 256]
])  # Would fail here without handling expanding dims!

# Solution: Handle expanding dims BEFORE stacking
shared_shape = [10, 256]  # Common shape

# Slice each to shared shape
shared = [
    model1['classifier.weight'][:10, :],  # [10, 256]
    model2['classifier.weight'][:10, :],  # [10, 256]
    model3['classifier.weight'][:10, :],  # [10, 256]
]

# Now stack works
stacked = torch.stack(shared)  # [3, 10, 256]

# Step 2: Apply weights element-wise
weights = torch.tensor([0.303, 0.455, 0.242])  # Normalized: sum = 1
weights = weights.view(3, 1, 1)  # Reshape for broadcasting

# Element-wise multiplication and sum
result = (stacked * weights).sum(dim=0)  # [10, 256]

# Result: weighted average computed at each element level
# result[i, j] = 0.303*stacked[0,i,j] + 0.455*stacked[1,i,j] + 0.242*stacked[2,i,j]
```

---

## Using with Your Training

### Your Training Loop

```python
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier

# Training phase
for task_id, task_data in enumerate(tasks):
    # Initialize model with new classes added
    num_classes = 10 + 5*task_id  # Example: 10, 15, 20...
    model = YourModel(num_classes=num_classes)
    
    # Train with LwF
    model = lwf_train(model, task_data)
    
    # Save
    torch.save(model.state_dict(), f'models/task_{task_id}.pth')

# Aggregation phase
aggregator = FedAvgAggregatorExpandingClassifier(
    weighting_strategy='data_driven',
    expanding_layers=['classifier.weight', 'classifier.bias']
)

model_paths = [f'models/task_{i}.pth' for i in range(num_tasks)]
dataset_sizes = [len(task) for task in tasks]

aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
aggregator.print_aggregation_details()
aggregator.save_aggregated_weights('models/aggregated.pth')

# Evaluation
agg_model = YourModel(num_classes=10 + 5*(num_tasks-1))
agg_model.load_state_dict(torch.load('models/aggregated.pth'))
agg_model.eval()

accuracy = evaluate(agg_model, test_data)
```

---

## Specifying Expanding Layers

### Option A: Explicit List

If you know exactly which layers expand:

```python
aggregator = FedAvgAggregatorExpandingClassifier(
    expanding_layers=['classifier.weight', 'classifier.bias']
)
```

### Option B: Auto-Detect

Let the aggregator find them:

```python
aggregator = FedAvgAggregatorExpandingClassifier()
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

expanding = aggregator.detect_expanding_layers()
print(f"Expanding layers: {expanding}")
```

### Option C: Provide to aggregate()

Specify during aggregation:

```python
aggregated = aggregator.aggregate(
    expanding_layers=['classifier.weight', 'classifier.bias']
)
```

---

## What Gets Aggregated?

| Layer Type | Behavior | Example |
|------------|----------|---------|
| **Backbone** (fixed) | Full aggregation | feature_extractor weights |
| **Classifier** (expanding) | Shared only | First 10 classes' weights |
| **Biases** | Shared only | First 10 classes' biases |

---

## Verification

After aggregation, verify the results:

```python
# Load and check shapes
aggregated = torch.load('aggregated.pth')

for key, value in aggregated.items():
    print(f"{key}: {value.shape}")

# Example output:
# classifier.weight: torch.Size([10, 256])  ← Shared portion only
# classifier.bias: torch.Size([10])         ← Shared portion only
# feature_extractor.0.weight: torch.Size([256, 784])  ← Full aggregation
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named fedavg_aggregator_expanded_classifier"

**Solution**: Make sure the file is in your Python path:
```python
# Add to same directory as your script, or
import sys
sys.path.append('/path/to/aggregator')
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier
```

### Issue: "expanding_layers parameter not found"

**Solution**: Use the correct class:
```python
# ✓ CORRECT
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier

# ✗ WRONG (old version doesn't have this)
from fedavg_aggregator import FedAvgAggregator
```

### Issue: Aggregated model performs worse

**Possible causes**:
1. Task-specific weights being lost (new classes not aggregated)
2. Need to fine-tune aggregated model
3. Task interference in backbone features

**Solutions**:
```python
# Option 1: Fine-tune on diverse data
optimizer = torch.optim.Adam(agg_model.parameters(), lr=1e-5)
# Train for few epochs on representative data...

# Option 2: Use ensemble instead
ensemble_pred = mean([m(x) for m in [model1, model2, model3]])

# Option 3: Check if aggregation is actually improving
individual_accs = [evaluate(m, test) for m in models]
agg_acc = evaluate(agg_model, test)
if agg_acc < max(individual_accs):
    print("Use best individual model instead")
```

---

## Complete Example

```python
import torch
import torch.nn as nn
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier

# 1. Define model
class ContinualModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

# 2. Create models with different classes
configs = [(10, 1000), (15, 1200), (22, 900)]
for num_classes, n_samples in configs:
    model = ContinualModel(num_classes)
    # Train model...
    torch.save(model.state_dict(), f'model_{num_classes}cls.pth')

# 3. Aggregate
aggregator = FedAvgAggregatorExpandingClassifier(
    weighting_strategy='data_driven',
    expanding_layers=['classifier.weight', 'classifier.bias']
)

aggregator.load_model_weights_from_files(
    [f'model_{nc}cls.pth' for nc, _ in configs],
    [n for _, n in configs]
)

aggregator.print_aggregation_details()
aggregator.save_aggregated_weights('aggregated.pth')

# 4. Use aggregated model
agg_model = ContinualModel(22)
agg_model.load_state_dict(torch.load('aggregated.pth'))
agg_model.eval()

with torch.no_grad():
    predictions = agg_model(test_data)
```

---

## Summary

| Method | Complexity | Control | Recommended |
|--------|-----------|---------|------------|
| **Option 1: New Aggregator** | Low | High | ⭐⭐⭐ |
| **Option 2: Auto-Detect** | Low | Medium | ⭐⭐ |
| **Option 3: Manual** | High | Very High | ⭐ |

**Best Practice**: Use Option 1 (new aggregator) with explicit expanding layers.

---

## Key Points

✓ **Element-wise aggregation**: Process at individual tensor element level
✓ **Shared-only**: Aggregate only common classes
✓ **Task-specific preservation**: New class weights handled intelligently
✓ **Flexible**: Works with any expansion pattern
✓ **Automatic**: Can auto-detect expanding layers
