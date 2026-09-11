# FedAvg Backbone-Only Aggregation - Quick Reference

## Your Scenario

Models trained on **completely different, non-overlapping classes**:

```
Model 1: Trained on classes 0-9
  - Backbone: Shared feature extractor (same size)
  - Classifier: 10 output units (classes 0-9)

Model 2: Trained on classes 10-19
  - Backbone: Shared feature extractor (same size)
  - Classifier: 10 output units (classes 10-19, completely different!)

Model 3: Trained on classes 20-29
  - Backbone: Shared feature extractor (same size)
  - Classifier: 10 output units (classes 20-29, completely different!)
```

**Problem**: Can't aggregate classifiers (trained on different classes)

**Solution**: Aggregate only backbone layers using element-wise approach

---

## 60-Second Quick Start

```python
from fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly

# Create aggregator
aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',
    exclude_layers=['classifier']  # ← Skip classifier
)

# Load models
aggregator.load_model_weights_from_files(
    ['model1.pth', 'model2.pth', 'model3.pth'],
    dataset_sizes=[1000, 1200, 900]
)

# Aggregate
aggregator.save_aggregated_weights('aggregated_backbone.pth')
```

Done! ✓

---

## Two Methods

### Method 1: Aggregator Class (Recommended) ⭐

**Pros:**
- Clean API
- Built-in error checking
- Good reporting
- Production-ready

**Code:**
```python
from fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly

aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',
    exclude_layers=['classifier']
)

aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
aggregator.print_report()  # See aggregation details
aggregator.save_aggregated_weights('aggregated.pth')
```

---

### Method 2: Direct Element-Wise Function

**Pros:**
- Direct control
- Matches your code style
- Simple and explicit

**Code:**
```python
from fedavg_aggregator_backbone_only import fedavg_element_wise_aggregation
import torch

# Load weights
weights_list = [torch.load(p) for p in model_paths]

# Aggregate
aggregated = fedavg_element_wise_aggregation(
    weights_list=weights_list,
    dataset_sizes=[1000, 1200, 900],
    exclude_layers=['classifier'],
    weighting_strategy='data_driven'
)

torch.save(aggregated, 'aggregated.pth')
```

---

## Element-Wise Aggregation Explained

**What it means:** For each element in each tensor, compute weighted average

**Formula:**
```
result[i,j,k,...] = λ₁*model₁[i,j,k,...] + λ₂*model₂[i,j,k,...] + λ₃*model₃[i,j,k,...]

where λ = normalized weights (sum to 1.0)
```

**Example:**
```
Model 1 backbone.weight[0,0] = 0.5
Model 2 backbone.weight[0,0] = 0.7
Model 3 backbone.weight[0,0] = 0.9

Weights: λ₁=0.303, λ₂=0.455, λ₃=0.242

Result[0,0] = 0.303*0.5 + 0.455*0.7 + 0.242*0.9 = 0.6844
```

**Code:**
```python
# Stack tensors from all models
stacked = torch.stack([model1_backbone, model2_backbone, model3_backbone])

# Reshape weights for broadcasting
agg_w = torch.tensor([0.303, 0.455, 0.242]).view(3, 1, 1)

# Element-wise weighted sum
result = (stacked * agg_w).sum(dim=0)

# result[i,j] = 0.303*stacked[0,i,j] + 0.455*stacked[1,i,j] + 0.242*stacked[2,i,j]
```

---

## Weighting Strategies

### Uniform (Equal Weight)
```python
aggregator = FedAvgAggregatorBackboneOnly(weighting_strategy='uniform')

# λ₁ = 1/3 = 0.333
# λ₂ = 1/3 = 0.333
# λ₃ = 1/3 = 0.333
```

Use when: All models equally important or unknown data sizes

### Data-Driven (Weighted by Data Size)
```python
aggregator = FedAvgAggregatorBackboneOnly(weighting_strategy='data_driven')

# Model 1: 1000 samples → λ₁ = 1000/3100 = 0.323
# Model 2: 1200 samples → λ₂ = 1200/3100 = 0.387
# Model 3: 900 samples  → λ₃ = 900/3100  = 0.290
```

Use when: Models trained on different amounts of data (recommended)

---

## Layer Exclusion

### Default (Exclude Classifier Only)
```python
aggregator = FedAvgAggregatorBackboneOnly(
    exclude_layers=['classifier']
)

# Skips all layers containing 'classifier' in the name:
#   - classifier.weight ✗
#   - classifier.bias ✗
#   - classifier.0.weight ✗
#
# Aggregates:
#   - backbone.0.weight ✓
#   - backbone.1.weight ✓
#   - feature_extractor.weight ✓
```

### Custom Patterns
```python
aggregator = FedAvgAggregatorBackboneOnly(
    exclude_layers=['classifier', 'head', 'fc_final']
)

# Skips any layer with these patterns
```

### Your Use Case
Since you only have different classifiers:
```python
exclude_layers=['classifier']  # That's it!
```

---

## Complete Workflow

### Step 1: Train Models on Non-Overlapping Classes

```python
# Your continual learning training
for task_id in range(num_tasks):
    model = ContinualLearningModel(num_classes=10)  # Classes for this task
    
    # Train with LwF
    model = lwf_train(model, task_data)
    
    # Save
    torch.save(model.state_dict(), f'models/task_{task_id}.pth')
```

### Step 2: Aggregate Backbone Only

```python
from fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly

aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',
    exclude_layers=['classifier']
)

aggregator.load_model_weights_from_files(
    ['models/task_0.pth', 'models/task_1.pth', 'models/task_2.pth'],
    dataset_sizes=[1000, 1200, 900]
)

aggregator.save_aggregated_weights('aggregated_backbone.pth')
```

### Step 3: Use Aggregated Backbone for New Task

```python
# Create new model for new task
new_model = ContinualLearningModel(num_classes=10)

# Load aggregated backbone
agg_backbone = torch.load('aggregated_backbone.pth')
new_state = new_model.state_dict()

for key, value in agg_backbone.items():
    if 'classifier' not in key:  # Skip classifier
        new_state[key] = value

new_model.load_state_dict(new_state)

# Fine-tune classifier on new task
optimizer = torch.optim.Adam(new_model.classifier.parameters(), lr=1e-3)
# Train...
```

---

## Verification

After aggregation, verify what was included:

```python
aggregated = torch.load('aggregated_backbone.pth')

print("Aggregated layers:")
for key, value in aggregated.items():
    print(f"  ✓ {key}: {value.shape}")

print("\nSkipped layers:")
original = torch.load('models/task_0.pth')
for key in original.keys():
    if 'classifier' in key:
        print(f"  ✗ {key} (not in aggregated)")
```

Expected output:
```
Aggregated layers:
  ✓ backbone.0.weight: torch.Size([256, 784])
  ✓ backbone.0.bias: torch.Size([256])
  ✓ backbone.1.weight: torch.Size([256, 256])
  ✓ backbone.1.bias: torch.Size([256])
  ✓ backbone.2.weight: torch.Size([256, 256])
  ✓ backbone.2.bias: torch.Size([256])

Skipped layers:
  ✗ classifier.weight (not in aggregated)
  ✗ classifier.bias (not in aggregated)
```

---

## Troubleshooting

### "Module not found"
```python
# Make sure file is in path
import sys
sys.path.append('/path/to/aggregator')
from fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly
```

### "Shapes don't match in backbone"
```
ERROR: backbone layer shape mismatch
```

**Problem:** Backbone should be identical size for all models

**Solution:** Verify all models use same architecture:
```python
# ✓ Correct
model1 = ContinualLearningModel(num_classes=10)
model2 = ContinualLearningModel(num_classes=10)  # Same backbone!
model3 = ContinualLearningModel(num_classes=10)

# ✗ Wrong
model1 = ContinualLearningModel(input_size=28*28)
model2 = ContinualLearningModel(input_size=32*32)  # Different!
```

### "Aggregated model performs poorly"

**Possible causes:**
1. Task-specific backbone features might not aggregate well
2. Need fine-tuning on representative data
3. Backbone specialization per task

**Solutions:**
```python
# Option 1: Fine-tune on diverse data
optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-5)
for epoch in range(5):
    for batch_data, batch_labels in diverse_loader:
        outputs = new_model(batch_data)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Option 2: Use best individual model instead
individual_accs = [evaluate(m, test) for m in models]
agg_acc = evaluate(agg_model, test)

if max(individual_accs) > agg_acc:
    print("Use best individual model")
else:
    print("Use aggregated model")
```

---

## API Reference

### FedAvgAggregatorBackboneOnly

```python
# Initialize
aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='uniform' | 'data_driven',
    exclude_layers=['classifier']
)

# Load
aggregator.add_model_weights(state_dict, dataset_size)
aggregator.load_model_weights_from_files(paths, dataset_sizes)

# Aggregate
aggregated = aggregator.aggregate()

# Save/Load
aggregator.save_aggregated_weights(path)
model = aggregator.aggregate_into_model(model)

# Info
aggregator.print_report()
report = aggregator.get_aggregation_report()

# Reset
aggregator.reset()
```

### fedavg_element_wise_aggregation()

```python
aggregated = fedavg_element_wise_aggregation(
    weights_list=[dict1, dict2, dict3],
    dataset_sizes=[1000, 1200, 900],
    exclude_layers=['classifier'],
    weighting_strategy='data_driven'
)
```

---

## Complete Example

```python
import torch
from fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly

# Models trained on non-overlapping classes
model_paths = [
    'models/task1_classes_0to9.pth',
    'models/task2_classes_10to19.pth',
    'models/task3_classes_20to29.pth',
]
dataset_sizes = [1000, 1200, 900]

# Aggregate
aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',
    exclude_layers=['classifier']
)

aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
aggregator.print_report()
aggregator.save_aggregated_weights('aggregated_backbone.pth')

# Use for new task
new_model = YourModel(num_classes=10)
agg_backbone = torch.load('aggregated_backbone.pth')

new_state = new_model.state_dict()
for key in agg_backbone.keys():
    new_state[key] = agg_backbone[key]

new_model.load_state_dict(new_state)

# Fine-tune and evaluate
new_model = fine_tune(new_model, new_task_data)
accuracy = evaluate(new_model, test_data)
print(f"Accuracy: {accuracy:.2%}")
```

---

## Key Takeaways

| Aspect | Value |
|--------|-------|
| **Aggregation target** | Backbone only (feature extractor) |
| **Excluded layers** | Classifier (different classes) |
| **Aggregation method** | Element-wise weighted average |
| **Weighting strategy** | Data-driven (by dataset size) |
| **Formula** | result[i,j,...] = Σ λₙ * modelₙ[i,j,...] |
| **Use case** | Transfer learning to new classes |

✓ Element-wise: Each element averaged independently
✓ Backbone only: Skip classifier (non-overlapping classes)
✓ Data-driven: Weight by data size
✓ Result: Shared feature extractor for new tasks
