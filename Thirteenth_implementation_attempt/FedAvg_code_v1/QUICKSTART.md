# FedAvg Aggregation - Quick Start Guide

## 60-Second Overview

**What**: Combine weights from multiple LwF-trained models into one aggregated model using Federated Averaging.

**Why**: Create a universal model that captures knowledge from all training tasks.

**How**: Weighted average of model parameters.

```python
from fedavg_aggregator import FedAvgAggregator

# Create aggregator
aggregator = FedAvgAggregator(weighting_strategy='data_driven')

# Load trained models
aggregator.load_model_weights_from_files(
    ['model1.pth', 'model2.pth', 'model3.pth'],
    dataset_sizes=[1000, 1500, 800]
)

# Aggregate and save
aggregator.save_aggregated_weights('aggregated_model.pth')
```

Done! ✓

---

## Usage Patterns

### Pattern 1: Simple Averaging (All models equally important)

```python
from fedavg_aggregator import FedAvgAggregator

aggregator = FedAvgAggregator(weighting_strategy='uniform')
aggregator.load_model_weights_from_files([
    'task1_model.pth',
    'task2_model.pth', 
    'task3_model.pth'
])

aggregator.save_aggregated_weights('final_model.pth')
```

**When to use**: All models trained on similar data sizes, or you don't know dataset sizes.

---

### Pattern 2: Size-Weighted Averaging (Data-Driven)

```python
from fedavg_aggregator import FedAvgAggregator

aggregator = FedAvgAggregator(weighting_strategy='data_driven')
aggregator.load_model_weights_from_files(
    model_paths=['model1.pth', 'model2.pth', 'model3.pth'],
    dataset_sizes=[1000, 1500, 800]  # Training samples per model
)

aggregator.save_aggregated_weights('final_model.pth')
```

**When to use**: Models trained on different dataset sizes (recommended for LwF).

---

### Pattern 3: Load Into Existing Model

```python
from fedavg_aggregator import FedAvgAggregator

aggregator = FedAvgAggregator(weighting_strategy='data_driven')
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# Load directly into your model instance
your_model = YourModelClass()
your_model = aggregator.aggregate_into_model(your_model)

# Use for inference
your_model.eval()
predictions = your_model(test_data)
```

---

### Pattern 4: View Aggregation Details

```python
from fedavg_aggregator import FedAvgAggregator

aggregator = FedAvgAggregator(weighting_strategy='data_driven')
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# Get report
report = aggregator.get_aggregation_report()

print(f"Models aggregated: {report['num_models']}")
print(f"Total samples: {report['total_samples']}")
print(f"Model contributions: {report['model_weights']}")

# Expected output:
# Models aggregated: 3
# Total samples: 3300
# Model contributions: [0.303, 0.455, 0.242]
```

---

## Complete Workflow

### Your LwF Training Loop

```python
import torch
from fedavg_aggregator import FedAvgAggregator

# 1. Train multiple models with LwF
model_paths = []
dataset_sizes = []

for task_id, task_data in enumerate(tasks):
    model = initialize_model()
    model = lwf_train(model, task_data, previous_model)  # Your LwF training
    
    # Save weights
    path = f'models/task_{task_id}_lwf.pth'
    torch.save(model.state_dict(), path)
    model_paths.append(path)
    dataset_sizes.append(len(task_data))
    
    print(f"✓ Task {task_id} trained ({len(task_data)} samples)")

# 2. Aggregate all models
print("\nAggregating models with FedAvg...")
aggregator = FedAvgAggregator(weighting_strategy='data_driven')
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# 3. View aggregation report
report = aggregator.get_aggregation_report()
print(f"Aggregating {report['num_models']} models")
print(f"Model weights: {[f'{w:.1%}' for w in report['model_weights']]}")

# 4. Save aggregated model
aggregator.save_aggregated_weights('models/aggregated_fedavg.pth')

# 5. Test aggregated model
agg_model = YourModel()
agg_model = aggregator.aggregate_into_model(agg_model)
agg_model.eval()

test_accuracy = evaluate(agg_model, test_data)
print(f"\nAggregated model accuracy: {test_accuracy:.2%}")
```

---

## Key Concepts

### Uniform Weighting
```
aggregated_weights = (1/3) * model1 + (1/3) * model2 + (1/3) * model3
```
- Each model contributes equally
- Ignores dataset size differences

### Data-Driven Weighting
```
aggregated_weights = 0.30 * model1 + 0.45 * model2 + 0.25 * model3
```
- Weights proportional to dataset size
- Larger datasets → higher contribution
- Recommended for LwF

---

## Common Questions

### Q: Should I use uniform or data-driven weighting?
**A**: Use **data-driven** if you know dataset sizes (recommended). Use **uniform** if sizes are unknown or similar.

### Q: What if models have different architectures?
**A**: Aggregation only works if models have **identical architecture**. They must have the same state_dict keys.

### Q: Will aggregation always improve performance?
**A**: Not guaranteed. Always validate:
```python
individual_accs = [evaluate(m, test) for m in models]
agg_acc = evaluate(agg_model, test)

print(f"Individual: {individual_accs}")
print(f"Aggregated: {agg_acc}")
```

### Q: Should I fine-tune the aggregated model?
**A**: Optional. If performance is suboptimal, try fine-tuning on diverse data:
```python
agg_model = fine_tune(agg_model, diverse_data, epochs=5)
```

### Q: How many models should I aggregate?
**A**: Works with 2+ models. More models = more diverse knowledge (but also more computation).

---

## Troubleshooting

### Error: "No model weights to aggregate. Add weights first."
```python
# Check: Did you call load_model_weights_from_files()?
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
```

### Error: "dataset_size required for 'data_driven' weighting strategy"
```python
# Solution: Provide dataset sizes
aggregator.load_model_weights_from_files(model_paths, dataset_sizes=[1000, 1500])
```

### Error: "KeyError: some_parameter_name"
```python
# Problem: Models have different architectures
# Solution: Ensure all models are instances of the same class
model1 = MyModel()
model2 = MyModel()  # ← Same class
```

### Aggregated model performs worse than individuals
```python
# Possible causes:
# 1. Mode connectivity issue (models in different loss landscape regions)
# 2. Task interference (aggregating conflicting parameters)
# 3. Need fine-tuning

# Solution: Try fine-tuning on representative data
from torch.optim import Adam
optimizer = Adam(agg_model.parameters(), lr=1e-5)
# Fine-tune for a few epochs...
```

---

## Performance Tips

1. **Check Aggregation Weights**
   ```python
   report = aggregator.get_aggregation_report()
   print(report['model_weights'])  # Should be reasonable proportions
   ```

2. **Validate Before Deployment**
   ```python
   if agg_accuracy >= max(individual_accuracies):
       print("✓ Use aggregated model")
   else:
       print("✗ Use best individual model")
   ```

3. **Save Aggregation Report**
   ```python
   import json
   report = aggregator.get_aggregation_report()
   with open('aggregation_report.json', 'w') as f:
       json.dump(report, f, indent=2)
   ```

4. **Consider Ensemble Inference**
   ```python
   # If aggregation doesn't help, use ensemble inference instead:
   ensemble_pred = mean([m(x) for m in [model1, model2, model3]])
   ```

---

## API Reference

### FedAvgAggregator Class

```python
aggregator = FedAvgAggregator(weighting_strategy='uniform' | 'data_driven')

# Load methods
aggregator.add_model_weights(state_dict, dataset_size)
aggregator.load_model_weights_from_files(paths, dataset_sizes=None)

# Aggregation
aggregator.aggregate() → Dict[str, Tensor]
aggregator.aggregate_into_model(model) → nn.Module
aggregator.save_aggregated_weights(path)

# Utilities
aggregator.get_aggregation_report() → Dict
aggregator.reset()
```

---

## Next Steps

1. **Prepare your trained models**: Ensure all saved weight files exist
2. **Gather dataset sizes**: Count samples per training task
3. **Run aggregation**: Use `load_model_weights_from_files()`
4. **Validate results**: Test on held-out data
5. **Deploy**: Use aggregated model for inference

---

## Need Help?

- Check error messages (usually self-explanatory)
- Review examples in `fedavg_example.py`
- Read theory in `FEDAVG_THEORY.md`
- Validate your setup with `get_aggregation_report()`
