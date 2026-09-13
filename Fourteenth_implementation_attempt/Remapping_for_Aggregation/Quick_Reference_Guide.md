# Quick Reference Guide: When to Map Labels Back (Non-Overlapping Classes)

## TL;DR Answer to Your Question

**Q: When must I map labels back to original CIFAR-100 labels?**

**A:** NOT during training. **AFTER** training, during the **aggregation phase**, you must remap each model's classifier weights from its local label space to CIFAR-100 space (0-99) BEFORE combining them together.

**Key insight:** Since your models train on **non-overlapping CIFAR classes**, aggregation is simple - just place each trained weight at its correct CIFAR-100 position. No averaging needed.

---

## Three Phases Explained

### 1️⃣ TRAINING PHASE (Current code - NO CHANGES)
```python
# Location: train_experience() line 774

targets = self.map_targets()  # Original labels → Mapped labels (0-N)
# Used in loss calculation

# ✅ CORRECT - Keep as is
```

**Why:** Each model only needs to learn its subset's classes

---

### 2️⃣ EVALUATION PHASE (Current code - NO CHANGES)
```python
# Location: evaluate_all_experiences() lines 1734-1735

predictions_mapped = torch.argmax(output, dim=1)  # Gives 0-N
predictions_original = self.map_targets_back_to_original(predictions_mapped)

targets_mapped = self.map_targets()              # Gives 0-N
targets_original = self.map_targets_back_to_original(targets_mapped)

# ✅ CORRECT - Keep as is (for evaluation metrics)
```

**Why:** You need to compute accuracy against original CIFAR-100 labels

---

### 3️⃣ AGGREGATION PHASE (ADD THIS - NEW CODE)
```python
# Location: After all models are trained

# For each trained specialized model:
remapped_weights, remapped_bias = remap_classifier_to_cifar100(
    model,
    new_to_original,  # {mapped_idx → original_cifar_label}
    num_classes=100
)

# Then aggregate:
aggregated_weights = (remapped_w1 + remapped_w2 + remapped_w3) / 3

# Create global model:
global_model.linear.weight = aggregated_weights
global_model.linear.bias = aggregated_bias

# ❌ MISSING in your current code - ADD THIS
```

**Why:** Models train in different label spaces, must align to common space before aggregation

---

## Quick Decision Tree

```
Question: Should I map labels back?

├─ During training forward pass?
│  └─→ NO. Use mapped labels.
│
├─ When computing training loss?
│  └─→ NO. Loss uses mapped labels.
│
├─ When computing evaluation metrics?
│  └─→ YES. Map predictions & targets back for accuracy computation.
│
├─ When aggregating models?
│  └─→ YES. Remap classifier weights to CIFAR-100 space BEFORE aggregation.
│
└─ During final inference?
   └─→ NO. Global model already works in CIFAR-100 space.
```

---

## The Core Issue (Visual)

### ❌ WRONG - What you're trying to do now:

```
Model 1: Predicts classes [0-4]
Model 2: Predicts classes [0-24]
Model 3: Predicts classes [0-25]
              ↓ Aggregate
Global Model: Predicts classes [0-??] 
              
              ❌ Incompatible spaces!
```

### ✅ RIGHT - What you should do:

```
Model 1: Weights for [5,25,42,63,78]  in 100-dim space
Model 2: Weights for [1,2,3,...,50]    in 100-dim space
Model 3: Weights for [51,52,...,75]    in 100-dim space
              ↓ Aggregate
Global Model: Weights for all [0-99]   in 100-dim space
              
              ✅ Compatible spaces!
```

---

## Minimal Code Example

### Remap one model's classifier:
```python
def remap_classifier(model, new_to_original, num_cifar=100):
    """Map classifier from subset space to CIFAR-100 space."""
    old_w = model.linear.weight  # Shape: [N_subset, 512]
    old_b = model.linear.bias    # Shape: [N_subset]
    
    # Create 100-class space (rest stays zero)
    new_w = torch.zeros(num_cifar, old_w.shape[1])
    new_b = torch.zeros(num_cifar)
    
    # Fill in trained classes at correct positions
    for mapped_idx, original_label in new_to_original.items():
        new_w[original_label] = old_w[mapped_idx]
        new_b[original_label] = old_b[mapped_idx]
    
    return new_w, new_b
```

### Aggregate multiple models:
```python
# Load all models and their mappings
models = [model1, model2, model3]
mappings = [new_to_original_1, new_to_original_2, new_to_original_3]

# Remap and aggregate
agg_w = torch.zeros(100, 512)
agg_b = torch.zeros(100)

for model, mapping in zip(models, mappings):
    w, b = remap_classifier(model, mapping, 100)
    agg_w += w / len(models)
    agg_b += b / len(models)

# Create global model
global_model = ResNet20(num_classes=100)
global_model.linear.weight = nn.Parameter(agg_w)
global_model.linear.bias = nn.Parameter(agg_b)
```

---

## Checklist for Success

- [ ] Models trained with mapped labels (0-N) during `train_experience()`
- [ ] Evaluation uses mapped targets in loss, but maps back for metrics
- [ ] After all training is done, **BEFORE aggregation**:
  - [ ] Load trained models
  - [ ] Load `new_to_original` mappings for each model
  - [ ] Remap each model's classifier to CIFAR-100 space
  - [ ] Aggregate remapped classifiers
  - [ ] Create global model with 100-class classifier
  - [ ] Test on original CIFAR-100 test data

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Only mapping back predictions, not weights
```python
# WRONG - This doesn't fix the problem
pred_mapped = argmax(output)
pred_original = map_targets_back_to_original(pred_mapped)
# Model still predicts in wrong space!
```

### ❌ Mistake 2: Trying to average classifiers with different sizes
```python
# WRONG - Dimension mismatch
avg_weights = (model1.linear.weight + model2.linear.weight) / 2
# Can't add [5, 512] + [20, 512]
```

### ❌ Mistake 3: Aggregating before remapping
```python
# WRONG - Lose information about which class is which
agg_w = (model1.linear.weight + model2.linear.weight) / 2
# Each row means different things in different models!
```

### ✅ Correct: Remap first, then aggregate
```python
# RIGHT
w1, b1 = remap_classifier(model1, mapping1, 100)  # Now [100, 512]
w2, b2 = remap_classifier(model2, mapping2, 100)  # Now [100, 512]
agg_w = (w1 + w2) / 2  # Both in same space, can aggregate!
```

---

## Why Your Current Evaluation Trick Doesn't Solve It

Your current code (lines 1734-1735):
```python
predictions = torch.argmax(output)
predictions = self.map_targets_back_to_original(predictions)
```

This is:
- ✅ Good for computing correct evaluation metrics
- ✅ Shows you what accuracy "would be" in CIFAR-100 space
- ❌ **Does NOT fix the model weights** - They're still in subset space
- ❌ **Doesn't help aggregation** - Weights still incompatible

It's like:
- The model predicts "3" (in subset space)
- You convert that to "63" (in CIFAR-100 space)
- But the model was trained to predict "3", not "63"!
- **The underlying weights are wrong** ← This is the real issue

For aggregation, you need to fix the weights themselves, not just convert the predictions.

---

## Summary Table

| When | What To Do | Why | Code Location |
|------|-----------|-----|---|
| Training loop | Use **mapped labels** | Each model trains on its subset only | Line 774 |
| Training loss | Use **mapped labels** | Model predicts subset classes | Line 793 |
| Eval metrics | Map **back to original** | Report accuracy in CIFAR-100 space | Lines 1734-1735 |
| Aggregation | **Remap weights** | Align label spaces before averaging | **ADD NEW STEP** |
| Final inference | **No mapping needed** | Global model in CIFAR-100 space | Test phase |

---

## Expected Outcome

After implementing the aggregation step:

```
Before:
  Test image with label 42
  Model predicts: [Garbage - wrong space]

After:
  Test image with label 42
  Model predicts: 42 (or nearby class)
  ✅ Predictions make sense in CIFAR-100 space!
```

---

## File References

Your code locations:
- `map_targets()` (Line 2005): Maps original → subset labels
- `map_targets_back_to_original()` (Line 2014): Maps subset → original labels  
- `train_experience()` (Line 727): Uses `map_targets()` ✅
- `evaluate_all_experiences()` (Line 1662): Uses both ✅
- **Missing:** Aggregation step using both functions

---

## Final Answer

**You DON'T need to change where you currently map labels back during evaluation.**

**You need to ADD a new aggregation step that:**
1. Remaps each model's weights from subset space to CIFAR-100 space
2. Averages the remapped weights
3. Creates a global model that works in CIFAR-100 space

This aggregation step happens **AFTER** all training is complete and should use the `new_to_original` mappings created during training.
