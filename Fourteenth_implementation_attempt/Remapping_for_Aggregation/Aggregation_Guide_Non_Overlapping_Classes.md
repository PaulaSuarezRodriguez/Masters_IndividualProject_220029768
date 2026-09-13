# Model Aggregation Guide: Non-Overlapping Specialized Classes

## Your Setup

You have **4 specialized models**, each trained on **disjoint CIFAR-100 subsets**:

```
Model 1 (Outdoors):         {12, 17, 23, 33, 37, 49}                    (6 classes)
Model 2 (Nature):           {0, 47, 51, 52, 53, 54, 56, 57, 59}         (9 classes)
Model 3 (People):           {2, 11, 35}                                  (3 classes)
Model 4 (Objects/Vehicles): {5, 8, 9, 10, 13, 16, 20, 22, 25, 28, ...}  (25 classes)
                                                        Total: 43 classes
```

**Key Feature:** NO OVERLAP. Each CIFAR-100 class belongs to exactly one model.

---

## The Problem

During training, each model maps its classes to a local label space:

**Model 1 (Outdoors):**
```
Training data comes with CIFAR-100 labels: {12, 17, 23, 33, 37, 49}
                            ↓ map_targets()
Map to local space:         {0, 1, 2, 3, 4, 5}
                            ↓
Classifier trained:         Linear(feature_dim, 6)
                            (expects input: 0-5, outputs logits for 6 classes)
```

**Problem during inference:**
- You feed original CIFAR-100 labels (0-99)
- But Model 1 only knows how to predict 0-5
- Model 1 has NO IDEA what class 12 is (it was trained as class 0)

---

## The Solution: Classifier Remapping

### Simple 3-Step Process

#### Step 1: Create the Remapping Function

```python
import torch
import torch.nn as nn

def remap_classifier_to_cifar100(model, new_to_original, num_classes=100):
    """
    Remap a specialized model's classifier from its local label space to CIFAR-100 space.
    
    Args:
        model: Trained model with classifier [num_subset_classes, feature_dim]
        new_to_original: Dict {local_idx → cifar_class}
                        Example: {0→12, 1→17, 2→23, 3→33, 4→37, 5→49}
        num_classes: Always 100 for CIFAR-100
    
    Returns:
        remapped_weights: [100, feature_dim] tensor positioned for CIFAR-100 space
        remapped_bias: [100] tensor positioned for CIFAR-100 space
    """
    # Get the trained classifier
    trained_classifier = model.linear  # Shape: [num_subset_classes, feature_dim]
    feature_dim = trained_classifier.weight.shape[1]
    
    # Create zero-initialized tensors for full CIFAR-100
    remapped_weights = torch.zeros(
        num_classes, 
        feature_dim,
        device=trained_classifier.weight.device,
        dtype=trained_classifier.weight.dtype
    )
    
    remapped_bias = torch.zeros(
        num_classes,
        device=trained_classifier.bias.device,
        dtype=trained_classifier.bias.dtype
    )
    
    # Place each trained class at its correct CIFAR-100 position
    for local_idx, cifar_class in new_to_original.items():
        remapped_weights[cifar_class] = trained_classifier.weight[local_idx]
        remapped_bias[cifar_class] = trained_classifier.bias[local_idx]
    
    return remapped_weights, remapped_bias
```

#### Step 2: Load All Models and Remap

```python
# After all models are trained
models = {
    'outdoors': load_model('outdoors_model.pt'),
    'nature': load_model('nature_model.pt'),
    'people': load_model('people_model.pt'),
    'objects_vehicles': load_model('objects_vehicles_model.pt')
}

# Get the mappings saved during training
mappings = {
    'outdoors': {0: 12, 1: 17, 2: 23, 3: 33, 4: 37, 5: 49},
    'nature': {0: 0, 1: 47, 2: 51, 3: 52, 4: 53, 5: 54, 6: 56, 7: 57, 8: 59},
    'people': {0: 2, 1: 11, 2: 35},
    'objects_vehicles': {0: 5, 1: 8, 2: 9, 3: 10, 4: 13, 5: 16, ...}  # 25 classes
}

# Remap all classifiers
global_weights = torch.zeros(100, feature_dim)
global_bias = torch.zeros(100)

for model_name in ['outdoors', 'nature', 'people', 'objects_vehicles']:
    model = models[model_name]
    mapping = mappings[model_name]
    
    w, b = remap_classifier_to_cifar100(model, mapping, num_classes=100)
    
    # Since classes don't overlap, just add (no need to average)
    global_weights += w
    global_bias += b
    
    print(f"✓ Remapped {model_name}")
    print(f"  Classes: {list(mapping.values())}")
    print(f"  Positions filled in global classifier: {len(mapping)}")
```

#### Step 3: Create Global Model

```python
# Create the final global 100-class model
global_model = ResNet20(num_classes=100)

# Replace its classifier with the aggregated weights
global_model.linear.weight = nn.Parameter(global_weights)
global_model.linear.bias = nn.Parameter(global_bias)

# Save it
torch.save(global_model.state_dict(), 'global_model_100class.pt')

print("✓ Created global model with 100-class classifier")
print(f"  Classifier shape: {global_model.linear.weight.shape}")
print(f"  Trained classes: 43 out of 100")
print(f"  Untrained classes (zero weights): 57 out of 100")
```

---

## Understanding the Remapping

### Example: Model 1 (Outdoors)

**Original trained model:**
```
Classifier: Linear(512, 6)

Weights shape: [6, 512]
  Row 0: trained on CIFAR class 12 (e.g., "mushroom")
  Row 1: trained on CIFAR class 17 (e.g., "castle")
  Row 2: trained on CIFAR class 23 (e.g., "cloud")
  Row 3: trained on CIFAR class 33 (e.g., "forest")
  Row 4: trained on CIFAR class 37 (e.g., "lighthouse")
  Row 5: trained on CIFAR class 49 (e.g., "sea")

new_to_original mapping: {0→12, 1→17, 2→23, 3→33, 4→37, 5→49}
```

**After remapping to CIFAR-100 space:**
```
Remapped weights shape: [100, 512]
  Row 0:   [zeros] (CIFAR class 0, not trained by this model)
  Row 1:   [zeros]
  ...
  Row 12:  [copied from Row 0]  ← Row about "mushroom"
  Row 13:  [zeros]
  Row 14:  [zeros]
  Row 15:  [zeros]
  Row 16:  [zeros]
  Row 17:  [copied from Row 1]  ← Row about "castle"
  ...
  Row 23:  [copied from Row 2]  ← Row about "cloud"
  ...
  Row 49:  [copied from Row 5]  ← Row about "sea"
  ...
  Row 99:  [zeros] (CIFAR class 99, not trained by any model)
```

**Result:**
- 43 rows filled with trained weights
- 57 rows remain zeros
- Global model now "understands" positions 0-99

---

## Testing the Aggregated Model

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load standard CIFAR-100 test set (original labels 0-99)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

test_data = datasets.CIFAR100(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Evaluate
global_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = global_model(images)  # labels are 0-99, model outputs 0-99
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Accuracy on CIFAR-100 test set: {accuracy:.2f}%")

# Expected behavior:
# - 43 trained classes: should perform well
# - 57 untrained classes: predictions will be random (near 50% error each)
# - Overall: depends on test set distribution across trained/untrained classes
```

---

## Key Points for Non-Overlapping Classes

### ✅ What's Simplified

1. **No ambiguity:** Each trained weight belongs at exactly one position
2. **No averaging needed:** Since classes don't overlap, no conflicts to resolve
3. **Simple aggregation:** Just sum the remapped weights (or copy them)
4. **Clear coverage:** Easy to track which classes are trained

### ⚠️ Expected Behavior

1. **Trained classes (43):** Model will make predictions based on learned features
2. **Untrained classes (57):** Model will output near-zero logits (random predictions)
3. **Test set accuracy:** 
   - If test set is balanced: ~43/100 = 43% if random guessing
   - Actual accuracy depends on learned quality

### ❌ What to Avoid

1. **Don't retrain:** Once models are trained, the aggregation step is independent
2. **Don't map during training:** Keep training as-is with mapped labels
3. **Don't average trained classes:** Each class trains on only ONE model
4. **Don't use original labels during training:** Still use mapped labels (0-N)

---

## Troubleshooting

### Problem: "Predictions are still wrong after aggregation"

**Check:**
1. ✓ Did you remap **weights**, not just predictions?
2. ✓ Is `new_to_original` mapping correct?
   - Run: `print(new_to_original)` to verify
3. ✓ Are you using **original CIFAR-100 test set** (labels 0-99)?
   - NOT the mapped labels
4. ✓ Is the remapped global model **actually being used**?
   - Verify: `print(global_model.linear.weight.shape)` should be `[100, 512]`

### Problem: "Only predicts a few classes"

**Likely cause:** Not remapping the classifier before aggregation
- Check: Does your code call `remap_classifier_to_cifar100()`?
- If not, model still predicts in local label spaces

### Problem: "Model performs worse than individual models"

**Expected behavior:** 
- Individual models: Trained on subset, optimized for those classes
- Global model: Trained on different subsets, zero-initialized for untrained classes
- The global model combines knowledge from all specialists, but each class learned by only one model

**Solution:** This is expected. The global model is a **specialist ensemble**, not a unified model.

---

## Summary

For **non-overlapping specialized models**, aggregation is simple:

1. **Remap** each model's classifier from local space to CIFAR-100 space using `new_to_original`
2. **Stack** the remapped classifiers (no complex averaging needed)
3. **Create** the global 100-class model
4. **Test** on original CIFAR-100 (labels 0-99)

The key insight: Since each CIFAR class belongs to exactly one model, there's no conflict or averaging needed. Just place each weight at its correct position.
