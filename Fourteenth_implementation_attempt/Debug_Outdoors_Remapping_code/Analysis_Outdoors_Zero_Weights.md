# Root Cause Analysis: Outdoors Model Zero Weights

## Your Finding (Critical)

```
w for outdoors: ALL ZEROS [100, 64]
b for outdoors: HAS VALUES at positions 23, 33, 37
w for nature, people, objects: HAS VALUES
b for nature, people, objects: HAS VALUES
```

This tells us: **The Outdoors classifier weights are being lost during remapping, but bias is preserved.**

---

## Most Likely Cause: Device or Dtype Mismatch

When you do:
```python
remapped_weights[cifar_class] = classifier.weight[local_idx].to(device)
```

The `.to(device)` call could fail silently if:

### Issue 1: Dtype Mismatch
```python
# Example:
remapped_weights = torch.zeros(100, 64, dtype=torch.float32)  # float32
classifier.weight = torch.tensor(..., dtype=torch.float64)     # float64

# When you do:
remapped_weights[cifar_class] = classifier.weight[local_idx].to(device)

# PyTorch might:
# 1. Convert float64 → float32
# 2. Due to loss of precision, all values → 0.0
# 3. OR the assignment silently fails
```

### Issue 2: Data Getting Lost in Transfer

The classifier weights are on one device (CPU or GPU), and the remapped_weights tensor is on another. The `.to(device)` operation might not preserve the values correctly.

### Issue 3: Outdoors Classifier Weights are Actually Zero

Though you said you printed the outdoors model and it had weights, let's verify this isn't the case.

---

## Immediate Diagnostic: Run This

```python
import torch
from pathlib import Path
from ResNet20_DynamicModel_v13 import ResNet20_DynamicModel

model_dir = Path('./final_specialised_models')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load outdoors model
outdoors_model = ResNet20_DynamicModel(num_classes=6)
state_dict = torch.load(model_dir / 'final_outdoors_model.pt', map_location=device)
outdoors_model.load_state_dict(state_dict)

classifier = outdoors_model.linear

# CHECK 1: Are weights actually loaded?
print("CHECK 1: Outdoors classifier weights")
print(f"  Shape: {classifier.weight.shape}")
print(f"  Dtype: {classifier.weight.dtype}")
print(f"  Device: {classifier.weight.device}")
print(f"  Min: {classifier.weight.min()}")
print(f"  Max: {classifier.weight.max()}")
print(f"  Norm: {classifier.weight.norm()}")

if classifier.weight.norm() == 0.0:
    print("  ❌ WEIGHTS ARE ALL ZEROS!")
else:
    print("  ✓ Weights are non-zero")

# CHECK 2: What happens with .to(device)?
print("\nCHECK 2: Testing .to(device) operation")
original_weight = classifier.weight[0].clone()
print(f"  Original: dtype={original_weight.dtype}, device={original_weight.device}")
print(f"  Original norm: {original_weight.norm()}")

after_to = original_weight.to(device)
print(f"  After .to(device): dtype={after_to.dtype}, device={after_to.device}")
print(f"  After .to(device) norm: {after_to.norm()}")

if original_weight.norm() != after_to.norm():
    print("  ❌ WEIGHTS CHANGED DURING .to(device)!")
else:
    print("  ✓ Weights preserved after .to(device)")

# CHECK 3: Test assignment to zero tensor
print("\nCHECK 3: Testing assignment to zero tensor")
test_tensor = torch.zeros(100, 64, dtype=torch.float32, device=device)
print(f"  Test tensor dtype: {test_tensor.dtype}")
print(f"  Classifier weight dtype: {classifier.weight.dtype}")

test_tensor[12] = classifier.weight[0].to(device)
print(f"  After assignment, test_tensor[12] norm: {test_tensor[12].norm()}")

if test_tensor[12].norm() == 0.0:
    print("  ❌ ASSIGNMENT FAILED - values are zero!")
    print("  This is likely a dtype mismatch issue")
else:
    print("  ✓ Assignment successful")
```

---

## Probable Fix #1: Preserve Dtype

In the `remap_classifier_to_cifar100()` function, ensure dtype matches:

**Current code (WRONG):**
```python
def remap_classifier_to_cifar100(model, new_to_original, num_classes=100, device='cpu'):
    classifier = model.linear
    feature_dim = classifier.weight.shape[1]
    
    # WRONG: Creates float32 by default
    remapped_weights = torch.zeros(num_classes, feature_dim, device=device)
    remapped_bias = torch.zeros(num_classes, device=device)
    
    for local_idx, cifar_class in new_to_original.items():
        remapped_weights[cifar_class] = classifier.weight[local_idx].to(device)
        remapped_bias[cifar_class] = classifier.bias[local_idx].to(device)
```

**Fixed code:**
```python
def remap_classifier_to_cifar100(model, new_to_original, num_classes=100, device='cpu'):
    classifier = model.linear
    feature_dim = classifier.weight.shape[1]
    
    # CORRECT: Match the original dtype
    remapped_weights = torch.zeros(
        num_classes, 
        feature_dim, 
        device=device,
        dtype=classifier.weight.dtype  # ← MATCH ORIGINAL DTYPE
    )
    remapped_bias = torch.zeros(
        num_classes, 
        device=device,
        dtype=classifier.bias.dtype  # ← MATCH ORIGINAL DTYPE
    )
    
    for local_idx, cifar_class in new_to_original.items():
        # Ensure both tensors are on same device/dtype
        w = classifier.weight[local_idx].to(device=device, dtype=remapped_weights.dtype)
        b = classifier.bias[local_idx].to(device=device, dtype=remapped_bias.dtype)
        
        remapped_weights[cifar_class] = w
        remapped_bias[cifar_class] = b
    
    return remapped_weights, remapped_bias
```

---

## Probable Fix #2: Explicit Casting

Alternative approach - be very explicit:

```python
def remap_classifier_to_cifar100(model, new_to_original, num_classes=100, device='cpu'):
    classifier = model.linear
    feature_dim = classifier.weight.shape[1]
    
    # Work in the source dtype first
    source_dtype = classifier.weight.dtype
    source_device = classifier.weight.device
    
    remapped_weights = torch.zeros(
        num_classes, 
        feature_dim, 
        dtype=source_dtype,
        device=source_device  # Keep on source device
    )
    remapped_bias = torch.zeros(
        num_classes,
        dtype=source_dtype,
        device=source_device
    )
    
    for local_idx, cifar_class in new_to_original.items():
        # Keep everything on source device until final transfer
        remapped_weights[cifar_class] = classifier.weight[local_idx]
        remapped_bias[cifar_class] = classifier.bias[local_idx]
    
    # Transfer to target device once
    remapped_weights = remapped_weights.to(device=device)
    remapped_bias = remapped_bias.to(device=device)
    
    return remapped_weights, remapped_bias
```

---

## Probable Fix #3: Check for NaN/Inf

Sometimes .to(device) can produce NaN if there's a numerical issue:

```python
def remap_classifier_to_cifar100(model, new_to_original, num_classes=100, device='cpu'):
    classifier = model.linear
    feature_dim = classifier.weight.shape[1]
    
    remapped_weights = torch.zeros(
        num_classes, 
        feature_dim, 
        device=device,
        dtype=classifier.weight.dtype
    )
    remapped_bias = torch.zeros(
        num_classes, 
        device=device,
        dtype=classifier.bias.dtype
    )
    
    for local_idx, cifar_class in new_to_original.items():
        w = classifier.weight[local_idx].to(device)
        b = classifier.bias[local_idx].to(device)
        
        # Check for NaN/Inf
        if torch.isnan(w).any() or torch.isinf(w).any():
            print(f"  ⚠️  NaN/Inf in weights for class {cifar_class}!")
            print(f"     Using original without conversion")
            w = classifier.weight[local_idx].clone()
        
        remapped_weights[cifar_class] = w
        remapped_bias[cifar_class] = b
    
    return remapped_weights, remapped_bias
```

---

## Why Bias Works But Weights Don't

The bias has 100 values (scalars), while weights have 100×64 values (vectors).

```
Bias assignment:
  remapped_bias[23] = scalar_value  ← Single float, usually robust
  
Weight assignment:
  remapped_weights[23] = vector[64]  ← 64 floats, more likely to have dtype/precision issues
```

This is why bias is preserved but weights aren't!

---

## What to Do Now

1. **Run `Debug_Outdoors_Remapping.py`** immediately
   - This will show exactly where the problem is

2. **Check output for:**
   - Are outdoors classifier weights actually non-zero? (If not, model is corrupted)
   - Do weights change dtype when moving to device? (If yes, that's the bug)
   - Do weights become zero during assignment? (If yes, that's the bug)

3. **Apply the appropriate fix:**
   - **Fix #1** if dtype mismatch detected
   - **Fix #2** if device transfer issue
   - **Fix #3** if NaN/Inf is the problem

4. **Re-run aggregation with fixed code**

---

## Quick Hypothesis

Given that:
- Bias works (scalars, simpler)
- Weights don't (vectors, more complex)
- Only outdoors fails (maybe first model in loop has special conditions)

**I suspect: Dtype mismatch between remapped_weights initialization and classifier.weight dtype**

Most likely fix: **Change this line:**
```python
remapped_weights = torch.zeros(num_classes, feature_dim, device=device)
```

**To this:**
```python
remapped_weights = torch.zeros(num_classes, feature_dim, device=device, dtype=classifier.weight.dtype)
```

Try this simple fix first!
