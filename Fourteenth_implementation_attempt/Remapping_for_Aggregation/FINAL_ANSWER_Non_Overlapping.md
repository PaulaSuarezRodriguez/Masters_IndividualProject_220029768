# Final Answer: When to Map Labels Back (Corrected for Non-Overlapping Classes)

## Your Original Question

> When in the training of the specialised models must I map the labels back to the original CIFAR-100 labels such that this issue is avoided?

## The Answer (Corrected)

**You should NOT map labels back during training.** However, you MUST map classifier weights back during the **aggregation phase** (after all training is complete).

The confusion arose because your models train on **non-overlapping CIFAR-100 subsets**, which makes this a specific variant of the label mapping problem.

---

## Your Actual Setup

You have 4 specialized models, each trained on disjoint CIFAR-100 classes:

| Model | Classes | Count |
|-------|---------|-------|
| Outdoors | {12, 17, 23, 33, 37, 49} | 6 |
| Nature | {0, 47, 51, 52, 53, 54, 56, 57, 59} | 9 |
| People | {2, 11, 35} | 3 |
| Objects/Vehicles | {5, 8, 9, 10, 13, 16, 20, 22, 25, 28, 39, 40, 41, 48, 58, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94} | 25 |
| **TOTAL** | **43 classes trained** | **43** |

---

## The Core Problem

### During Training (Current - Correct)

Your training code (line 774 in `train_experience()`):
```python
targets = self.map_targets()  # Maps CIFAR labels to local space (0-N)
```

**This is CORRECT.** Each model needs to map its subset to local label space:

- **Model 1 maps:** {12→0, 17→1, 23→2, 33→3, 37→4, 49→5}
- **Model 2 maps:** {0→0, 47→1, 51→2, 52→3, 53→4, 54→5, 56→6, 57→7, 59→8}
- **Model 3 maps:** {2→0, 11→1, 35→2}
- **Model 4 maps:** {5→0, 8→1, 9→2, 10→3, ...} (25 classes)

Each model's expanding classifier learns to predict only its subset in local space.

### After Training (Current Problem)

Your aggregated model produces predictions in the wrong label space:

```
Example:
  Model 1 outputs logits[0-5] meaning "I predict one of my 6 classes"
  But logits[0] means "I think it's CIFAR class 12" (trained on class 12)
  
  Your global model somehow combines these models without remapping
  Result: Predictions are meaningless in CIFAR-100 space
```

**The problem:** You never map the **classifier weights** back to CIFAR-100 space before aggregation.

---

## The Solution (Three Stages)

### Stage 1: Training (No Changes Needed)

Keep your current training code as-is:
- ✅ Use mapped labels (0-N for each model)
- ✅ Loss computed in local label space
- ✅ Each model learns its subset optimally

**Your current code is correct for this stage.**

### Stage 2: Saving (Minor Addition)

After training each model, **save the mappings**:

```python
# After line 2372 (where you save the model)
import pickle

new_to_original = new_to_original_creation()  # Your existing function
class_to_idx_new = class_to_idx_new  # Your existing variable

metadata = {
    'new_to_original': new_to_original,  # {0→12, 1→17, ...}
    'class_to_idx_new': class_to_idx_new,
    'num_classes': len(class_to_idx_new)
}

with open('./final_specialised_models/people_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
```

**Do this for each model (outdoors, nature, people, objects_vehicles)**

### Stage 3: Aggregation (New Step - After All Training)

This is where you fix the label space issue:

```python
import torch
import torch.nn as nn
import pickle

def remap_classifier_to_cifar100(model, new_to_original, num_classes=100):
    """
    Remap a specialized model's classifier from local space to CIFAR-100 space.
    
    new_to_original = {0→12, 1→17, 2→23, 3→33, 4→37, 5→49} for outdoors model
    
    Maps model's local classifier [6, feature_dim] to global [100, feature_dim]
    placing each row at its correct CIFAR-100 position.
    """
    classifier = model.linear
    feature_dim = classifier.weight.shape[1]
    
    # Initialize 100×feature_dim with zeros (untrained classes)
    remapped_weights = torch.zeros(num_classes, feature_dim)
    remapped_bias = torch.zeros(num_classes)
    
    # Place each trained class at its CIFAR-100 position
    for local_idx, cifar_class in new_to_original.items():
        # local_idx: 0-5 for outdoors (position in model's classifier)
        # cifar_class: 12, 17, 23, ... (actual CIFAR class)
        remapped_weights[cifar_class] = classifier.weight[local_idx]
        remapped_bias[cifar_class] = classifier.bias[local_idx]
    
    return remapped_weights, remapped_bias


# Load all trained models
models = {}
mappings = {}

for model_name in ['outdoors', 'nature', 'people', 'objects_vehicles']:
    # Load model
    model = ResNet20(num_classes=None)
    model.load_state_dict(torch.load(f'./final_specialised_models/final_{model_name}_model.pt'))
    models[model_name] = model
    
    # Load mapping
    with open(f'./final_specialised_models/{model_name}_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    mappings[model_name] = metadata['new_to_original']


# Remap and aggregate
global_weights = torch.zeros(100, feature_dim)
global_bias = torch.zeros(100)

for model_name in ['outdoors', 'nature', 'people', 'objects_vehicles']:
    model = models[model_name]
    mapping = mappings[model_name]
    
    # Remap this model's classifier
    w, b = remap_classifier_to_cifar100(model, mapping, num_classes=100)
    
    # Add to global classifier (no averaging - classes don't overlap!)
    global_weights += w
    global_bias += b
    
    print(f"✓ Aggregated {model_name}")


# Create global model
global_model = ResNet20(num_classes=100)
global_model.linear.weight = nn.Parameter(global_weights)
global_model.linear.bias = nn.Parameter(global_bias)

# Save
torch.save(global_model.state_dict(), './final_specialised_models/global_model.pt')
```

---

## Why This Solves the Problem

### Before Remapping

```
Outdoors model:        Predicts in space [0-5] meaning {12, 17, 23, 33, 37, 49}
Nature model:          Predicts in space [0-8] meaning {0, 47, 51, ...}
People model:          Predicts in space [0-2] meaning {2, 11, 35}
Objects/Vehicles:      Predicts in space [0-24] meaning {5, 8, 9, ...}

Try to aggregate:
  ❌ Can't combine incompatible spaces
  ❌ No way to know which row is which CIFAR class
  ❌ Predictions meaningless
```

### After Remapping

```
Outdoors classifier:   [100 × feature_dim]
  Row 0: zeros (not trained)
  Row 12: weight from model's row 0
  Row 17: weight from model's row 1
  ...
  Row 49: weight from model's row 5
  Rows 50-99: zeros

Nature classifier:     [100 × feature_dim]
  Row 0: weight from model's row 0
  Rows 1: zeros
  Row 47: weight from model's row 1
  ...

[etc. for other models]

Aggregate (combine):
  ✅ Each position has exactly one source (or zero if untrained)
  ✅ Global model predicts in CIFAR-100 space [0-99]
  ✅ Predictions correspond to correct CIFAR classes
```

---

## Practical Implications

### What Changes During Training
Nothing. Your current training code is correct.

### What Needs to Happen After Training
1. Save `new_to_original` mapping for each model
2. Remap each model's classifier weights to CIFAR-100 space
3. Combine the remapped classifiers into a 100-class model

### Test Behavior
```python
# Test on CIFAR-100 with original labels (0-99)
for image, label in test_loader:
    prediction = global_model(image)  # Outputs logits for classes 0-99
    predicted_class = torch.argmax(prediction)
    
    # Now predicted_class and label are in the same space!
    # ✓ This comparison makes sense
    if predicted_class == label:
        correct += 1
```

### Expected Accuracy
- **Trained classes (43):** Good predictions (depends on model quality)
- **Untrained classes (57):** ~50% error (random guessing from zero logits)
- **Overall:** Depends on test set distribution

---

## Why Your Current Evaluation Mapping Isn't Enough

Your code at lines 1734-1735:
```python
predictions = self.map_targets_back_to_original(predictions)
targets = self.map_targets_back_to_original(targets)
```

This is good for **evaluation metrics**, but it doesn't fix the underlying problem:

```
❌ What it does:
  - Converts predictions from local space (0-N) to CIFAR space (0-99)
  - Allows you to report accurate evaluation metrics
  
❌ What it doesn't do:
  - Fix the model's weights (still in local space)
  - Solve aggregation (weights still incompatible)
  - Enable standard CIFAR-100 inference
```

It's like translating the answer after the fact, rather than fixing the problem at its source.

---

## Complete Checklist

### During Training
- ✅ Use `map_targets()` to map CIFAR labels to local space (line 774)
- ✅ Compute loss in local label space
- ✅ Use `map_targets_back_to_original()` for evaluation metrics (lines 1734-1735)
- ✅ Train each model independently on its subset

### After Training (All Models)
- ✅ Save each model's weights
- ✅ Save each model's `new_to_original` mapping

### Aggregation Phase (NEW)
- ✅ Load all trained models
- ✅ Load all `new_to_original` mappings
- ✅ For each model: call `remap_classifier_to_cifar100(model, mapping)`
- ✅ Combine remapped classifiers (sum them, no averaging needed)
- ✅ Create global 100-class model

### Testing Phase
- ✅ Load global model
- ✅ Test on original CIFAR-100 dataset with original labels (0-99)
- ✅ No label mapping needed during inference

---

## Files You Should Create/Use

| File | Purpose | When |
|------|---------|------|
| `{model}_metadata.pkl` | Save `new_to_original` mappings | After each model trains |
| `remap_classifier_to_cifar100()` | Remap individual classifiers | During aggregation |
| `aggregate_specialized_models()` | Main aggregation function | After all training |
| `global_model.pt` | Final 100-class model | After aggregation |

---

## Summary

**Your question:** When must I map labels back?

**Answer:** 
1. **NOT** during training - keep labels mapped
2. **During aggregation** - remap classifier weights from local to CIFAR-100 space
3. **NOT** during inference - global model already in correct space

The reason your current aggregation fails: You're trying to combine models that each predict in their own label space without remapping them to a common space first.

The solution: Remap each model's classifier weights to CIFAR-100 space using the `new_to_original` mapping, then combine them.

With non-overlapping classes, this is simple - each trained weight goes to exactly one position with no conflicts.
