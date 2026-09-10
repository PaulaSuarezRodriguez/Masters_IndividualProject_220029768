# Federated Averaging (FedAvg) Algorithm - Theory & Application

## Overview

Federated Averaging (FedAvg) is an aggregation algorithm originally introduced for Federated Learning. While typically used to aggregate models trained across distributed clients, it can also be applied locally to combine weights from multiple independently trained models.

**Key difference from traditional FL**: In your case, there's no server-client communication or iterative training—you're simply aggregating the final weights from locally trained models into a single set of weights.

---

## Algorithm Description

### Basic FedAvg Formula

The core principle of FedAvg is a **weighted average** of model parameters:

```
w^(t+1) = Σ(i=1 to K) [n_i / n] * w_i^(t)
```

Where:
- **w^(t+1)**: Aggregated weights at time t+1
- **K**: Number of models/clients
- **n_i**: Number of training samples used for model i
- **n**: Total number of training samples across all models (n = Σ n_i)
- **w_i^(t)**: Weights of model i at time t

### In Your Context

Since you're not doing iterative federated learning:
- You already have **final trained weights** from each model
- You perform aggregation **once** to combine all weights
- The result is a single aggregated model

**Simplified formula for your use case:**
```
w_final = Σ(i=1 to K) [n_i / n] * w_i
```

---

## Two Weighting Strategies

### 1. Uniform Weighting

All models contribute equally regardless of training data size:

```
w_final = (1/K) * (w_1 + w_2 + ... + w_K)
```

**When to use:**
- Models are equally important
- All models have similar training data sizes
- No prior knowledge about data distribution
- Simple model combination

**Example:**
```python
aggregator = FedAvgAggregator(weighting_strategy='uniform')
# Weights: [0.333, 0.333, 0.333] for 3 models
```

### 2. Data-Driven Weighting

Models are weighted by their training dataset size:

```
w_final = Σ [n_i / n_total] * w_i
```

**When to use:**
- Models trained on different amounts of data
- Larger datasets → more reliable models
- Want to avoid bias from small-dataset models
- Dataset sizes are known

**Example:**
```python
aggregator = FedAvgAggregator(weighting_strategy='data_driven')
# Model 1: 1000 samples → 0.40 weight
# Model 2: 1500 samples → 0.60 weight
# Total:   2500 samples
```

---

## Why FedAvg Works for LwF-Trained Models

### Learning Without Forgetting (LwF) Context

LwF trains models sequentially on different tasks while preserving previous knowledge through knowledge distillation. When you have multiple LwF-trained models:

1. **Each model specializes** in certain tasks
2. **Each model retains knowledge** from previous tasks (via distillation)
3. **Aggregating them** creates a **universal representation** that combines all specializations

### Benefits of FedAvg for LwF

| Aspect | Benefit |
|--------|---------|
| **Combines Knowledge** | Merges insights from all tasks into single model |
| **No Forgetting** | Each model's learned knowledge is preserved |
| **Simple** | No iterative training needed |
| **Principled** | Well-established aggregation strategy |
| **Scalable** | Works with any number of models |
| **Local** | No communication overhead |

---

## Mathematical Properties

### 1. Weight Preservation

If all models have identical weights (unlikely but theoretically):
```
w_final = [n_1/n * w] + [n_2/n * w] + ... = [(n_1+n_2+...)/n * w] = w
```

### 2. Linear Interpolation

For two models, FedAvg creates a linear interpolation:
```
w_final = λ * w_1 + (1-λ) * w_2

where λ = n_1 / (n_1 + n_2)
```

### 3. Convex Combination

The aggregated weights are a **convex combination** of all model weights:
```
- All coefficients sum to 1: Σ(n_i/n) = 1
- All coefficients are non-negative: n_i/n ≥ 0
```

This means the aggregated model stays within the "geometric mean" of individual models.

---

## Convergence and Theoretical Guarantees

### Original FedAvg Theorem (McMahan et al., 2017)

For convex loss functions:
```
Aggregated model loss ≤ Weighted average of individual model losses + O(1/√T)
```

### For Your Use Case

Since you're aggregating **final weights** (not iterative training):
- **No convergence analysis needed** (one-shot aggregation)
- **Empirical evaluation is key**: Test aggregated model on held-out data
- **Expected benefit**: Aggregated model generalizes better due to ensemble effect

---

## Step-by-Step Example

### Scenario: 3 Models Trained with LwF

```
Model 1 (Task A):  w_1 = [w_1^1, w_1^2, ..., w_1^d], n_1 = 1000 samples
Model 2 (Task B):  w_2 = [w_2^1, w_2^2, ..., w_2^d], n_2 = 1500 samples  
Model 3 (Task C):  w_3 = [w_3^1, w_3^2, ..., w_3^d], n_3 = 800 samples
```

**Total samples:** n = 1000 + 1500 + 800 = 3300

**Aggregation weights:**
- λ_1 = 1000 / 3300 ≈ 0.303
- λ_2 = 1500 / 3300 ≈ 0.455
- λ_3 = 800 / 3300 ≈ 0.242

**Aggregated weights (element-wise):**
```
w_final^j = 0.303 * w_1^j + 0.455 * w_2^j + 0.242 * w_3^j

for all parameters j = 1, 2, ..., d
```

**Result:** Single model combining all three task specializations!

---

## Implementation Details

### Parameter-by-Parameter Aggregation

Each parameter (weight matrix, bias, etc.) is aggregated independently:

```python
for param_name in model.state_dict():
    # Stack parameter across all models
    params = [model_i[param_name] for model_i in all_models]
    
    # Compute weighted average
    aggregated = sum(λ_i * p_i for λ_i, p_i in zip(lambdas, params))
```

### Handling Different Layer Types

Works for any layer type:
- **Fully Connected**: Linear weights and biases
- **Convolutional**: Kernel and bias parameters
- **Batch Norm**: Running mean, running variance, weight, bias
- **Embeddings**: Embedding weights

All are aggregated using the same weighted averaging approach.

---

## Common Pitfalls and Solutions

### Pitfall 1: Mismatched Architectures

**Problem**: Models have different architectures

**Solution**: Ensure all models have identical architecture before aggregation
```python
model1 = MyModel()
model2 = MyModel()  # ← Same class, same initialization
# Both trained with LwF on different tasks
```

### Pitfall 2: Ignoring Scale Differences

**Problem**: Some models trained much longer (more iterations)

**Solution**: Use `data_driven` weighting if data size correlates with training quality
```python
aggregator = FedAvgAggregator(weighting_strategy='data_driven')
```

### Pitfall 3: Not Validating Result

**Problem**: Assuming aggregation always improves performance

**Solution**: Always evaluate the aggregated model
```python
test_acc_model1 = evaluate(model1, test_data)
test_acc_agg = evaluate(aggregated_model, test_data)

print(f"Individual models: {[test_acc_model1, ...]}")
print(f"Aggregated model: {test_acc_agg}")
```

### Pitfall 4: Mode Connectivity Issues

**Problem**: Models may be in different "modes" of the loss landscape

**Solution**: 
- Use reasonable initialization (same for all models)
- Consider fine-tuning aggregated model on diverse data
- Test whether aggregated model works better than individuals

---

## Advanced: Weighted Aggregation Strategies

Beyond uniform and data-driven, you can use custom weights based on:

1. **Model Performance**
   ```python
   # Weight by validation accuracy
   λ_i ∝ accuracy_i
   ```

2. **Ensemble Diversity**
   ```python
   # Weight by disagreement with other models
   λ_i ∝ disagreement_i
   ```

3. **Domain-Specific Importance**
   ```python
   # Weight by domain criticality
   λ_i = manual_importance_weight_i
   ```

---

## Practical Recommendations

### For Your LwF + FedAvg Use Case

1. **Data-Driven Weighting (Recommended)**
   ```python
   aggregator = FedAvgAggregator(weighting_strategy='data_driven')
   aggregator.load_model_weights_from_files(paths, dataset_sizes)
   ```
   
   Why: LwF models trained on different datasets benefit from size-weighted aggregation

2. **Always Validate**
   ```python
   agg_model = aggregator.aggregate_into_model(model)
   agg_acc = evaluate(agg_model, test_data)
   
   individual_accs = [evaluate(m, test_data) for m in individual_models]
   
   if agg_acc > mean(individual_accs):
       print("✓ Aggregation improved performance!")
   ```

3. **Fine-Tune if Needed**
   ```python
   # Optional: Fine-tune aggregated model on diverse data
   aggregated_model = fine_tune(aggregated_model, diverse_data)
   ```

4. **Save and Document**
   ```python
   aggregator.save_aggregated_weights('final_model.pth')
   report = aggregator.get_aggregation_report()
   # Save report for reproducibility
   ```

---

## References

- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Aguerri, B. A. (2017). 
  "Communication-Efficient Learning of Deep Networks from Decentralized Data." 
  *ICML 2017*

- Li, T., Sahu, A. K., Zaheer, M., Shlyan, M., Talwalkar, A., & Smith, V. (2020). 
  "Federated Optimization in Heterogeneous Networks." 
  *MLSys 2020*

---

## Summary

**FedAvg for Local Model Aggregation:**
- ✓ Simple weighted averaging of model weights
- ✓ Works well with LwF-trained models  
- ✓ No server/distributed infrastructure needed
- ✓ Single-shot aggregation (no iterative training)
- ✓ Data-driven weighting recommended for best results
- ✓ Always validate on test data
