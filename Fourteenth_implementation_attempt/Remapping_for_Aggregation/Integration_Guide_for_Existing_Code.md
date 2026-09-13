# Integration Guide: Adding Model Aggregation to Your Training Code

This guide shows exactly where to add label remapping logic to your existing training code with minimal changes.

---

## Summary of Changes Needed

You need to add **ONE NEW STEP** after all models are trained:
- Remap each specialized model's classifier from subset space to CIFAR-100 space
- Aggregate the remapped classifiers

**During training:** Keep everything as is (lines 774 in `train_experience()` use mapped labels ✅)

**During evaluation:** Keep everything as is (lines 1734-1735 map back for metrics ✅)

**After training:** Add the aggregation step (NEW)

---

## Step-by-Step Integration

### STEP 1: Save Specialized Models With Metadata

**Location:** After line 2372 (when you save the final specialized model)

**Current code (lines 2371-2375):**
```python
if i == 3:
    torch.save(lwf_trainer.model.state_dict(), f"./final_specialised_models/final_people_model.pt")
    torch.save(lwf_trainer.model.linear.state_dict(), f"./final_specialised_models/final_people_model_CLASSIFIER.pt")
    print("lwf_trainer.model.linear", lwf_trainer.model.linear)
```

**Add this after saving:**
```python
if i == 3:
    torch.save(lwf_trainer.model.state_dict(), f"./final_specialised_models/final_people_model.pt")
    torch.save(lwf_trainer.model.linear.state_dict(), f"./final_specialised_models/final_people_model_CLASSIFIER.pt")
    
    # ===== NEW: Save metadata for aggregation =====
    import pickle
    from pathlib import Path
    
    metadata_dir = Path("./final_specialised_models")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the mappings used in training
    metadata = {
        'class_to_idx_new': class_to_idx_new,      # YOUR EXISTING VARIABLE
        'new_to_original': new_to_original,        # YOUR EXISTING VARIABLE
        'num_subset_classes': len(class_to_idx_new),
        'feature_dim': lwf_trainer.model.linear.weight.shape[1],
        'subset_name': 'people'  # Change this for different subsets
    }
    
    with open(metadata_dir / "people_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print("✓ Saved metadata for aggregation")
    # ===== END NEW =====
```

---

### STEP 2: Load Models for Aggregation

**Location:** After all specialized models have been trained (after line 2410)

**Add this new section:**
```python
# ===== NEW: Aggregation Phase =====
if PERFORM_AGGREGATION:  # Set this flag once all models are trained
    print("\n" + "="*80)
    print("STARTING MODEL AGGREGATION")
    print("="*80 + "\n")
    
    import pickle
    from pathlib import Path
    from typing import Dict
    
    # Configuration
    MODEL_DIR = Path("./final_specialised_models")
    SUBSET_NAMES = ['people', 'objects', 'scenes', 'animals']  # Adjust to your subsets
    CIFAR100_CLASSES = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Function to load a specialized model with metadata
    def load_specialized_model_metadata(subset_name: str):
        """Load model and its label mappings."""
        model_path = MODEL_DIR / f"final_{subset_name}_model.pt"
        metadata_path = MODEL_DIR / f"{subset_name}_metadata.pkl"
        
        # Load model
        model = ResNet20(num_classes=None)  # Will be replaced
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return model, metadata
    
    # Load all models
    print("Loading specialized models...")
    models_and_metadata = []
    for subset_name in SUBSET_NAMES:
        try:
            model, metadata = load_specialized_model_metadata(subset_name)
            models_and_metadata.append((model, metadata))
            print(f"  ✓ Loaded {subset_name}: {metadata['num_subset_classes']} classes")
        except FileNotFoundError:
            print(f"  ⚠ Skipped {subset_name}: files not found")
    
    print(f"\nLoaded {len(models_and_metadata)} models for aggregation\n")
    
    # ===== END NEW =====
```

---

### STEP 3: Implement Classifier Remapping Function

**Location:** Add this as a new method or function right after the aggregation loading section

**Add this code:**
```python
# ===== NEW: Remapping function =====
def remap_classifier_to_cifar100(model, new_to_original: Dict[int, int], 
                                  num_classes: int = 100, device: str = 'cpu'):
    """
    Remap a specialized model's classifier from subset space to CIFAR-100 space.
    
    Args:
        model: Trained model with classifier sized for subset
        new_to_original: {mapped_idx → original_cifar_label}
        num_classes: 100 for CIFAR-100
        device: Device for tensors
    
    Returns:
        remapped_weights [100, feature_dim], remapped_bias [100]
    """
    old_classifier = model.linear
    feature_dim = old_classifier.weight.shape[1]
    
    # Initialize zeros for all 100 classes
    remapped_weights = torch.zeros(
        num_classes, feature_dim,
        device=device,
        dtype=old_classifier.weight.dtype
    )
    remapped_bias = torch.zeros(
        num_classes,
        device=device,
        dtype=old_classifier.bias.dtype
    )
    
    # Copy weights from mapped positions to original positions
    for mapped_idx, original_label in new_to_original.items():
        remapped_weights[original_label] = old_classifier.weight[mapped_idx]
        remapped_bias[original_label] = old_classifier.bias[mapped_idx]
    
    return remapped_weights, remapped_bias

# ===== END NEW =====
```

---

### STEP 4: Perform Aggregation

**Location:** Immediately after the remapping function (before testing)

**Add this code:**
```python
# ===== NEW: Aggregate remapped classifiers =====
print("\n" + "="*60)
print("AGGREGATING CLASSIFIERS")
print("="*60 + "\n")

# Initialize aggregation tensors
aggregated_weights = torch.zeros(
    CIFAR100_CLASSES, 
    models_and_metadata[0][1]['feature_dim'],
    device=DEVICE
)
aggregated_bias = torch.zeros(
    CIFAR100_CLASSES,
    device=DEVICE
)

class_coverage = torch.zeros(CIFAR100_CLASSES, device=DEVICE)

# Aggregate each model
for subset_name, (model, metadata) in zip(SUBSET_NAMES, models_and_metadata):
    print(f"Aggregating {subset_name}...")
    print(f"  - Classes in subset: {metadata['num_subset_classes']}")
    
    # Remap this model's classifier
    remapped_w, remapped_b = remap_classifier_to_cifar100(
        model,
        metadata['new_to_original'],
        CIFAR100_CLASSES,
        DEVICE
    )
    
    # Add to aggregation (simple averaging)
    weight_factor = 1.0 / len(models_and_metadata)
    aggregated_weights += weight_factor * remapped_w
    aggregated_bias += weight_factor * remapped_b
    
    # Track coverage
    for mapped_idx in range(metadata['num_subset_classes']):
        original_label = metadata['new_to_original'][mapped_idx]
        class_coverage[original_label] += 1
    
    print(f"  ✓ Aggregated\n")

# Print coverage report
print("="*60)
print("CLASS COVERAGE REPORT")
print("="*60)
untrained_classes = (class_coverage == 0).sum().item()
trained_classes = CIFAR100_CLASSES - untrained_classes
print(f"Classes trained by ≥1 model: {trained_classes}/{CIFAR100_CLASSES}")
print(f"Classes NOT trained: {untrained_classes}/{CIFAR100_CLASSES}")
if untrained_classes > 0:
    print(f"⚠️  {untrained_classes} classes have zero-initialized weights")
print("="*60 + "\n")

# ===== END NEW =====
```

---

### STEP 5: Create and Save the Global Model

**Location:** After aggregation

**Add this code:**
```python
# ===== NEW: Create and save global model =====
print("\n" + "="*60)
print("CREATING GLOBAL MODEL")
print("="*60 + "\n")

# Create model with full 100-class classifier
global_model = ResNet20(num_classes=CIFAR100_CLASSES)
global_model = global_model.to(DEVICE)

# Replace classifier with aggregated weights
global_model.linear.weight = nn.Parameter(aggregated_weights)
global_model.linear.bias = nn.Parameter(aggregated_bias)

print("✓ Created global model")
print(f"  - Classifier output classes: {global_model.linear.weight.shape[0]}")
print(f"  - Feature dimension: {global_model.linear.weight.shape[1]}")

# Save the global model
GLOBAL_MODEL_PATH = "./final_specialised_models/global_model_aggregated.pt"
torch.save(global_model.state_dict(), GLOBAL_MODEL_PATH)
print(f"✓ Saved global model to {GLOBAL_MODEL_PATH}\n")

# ===== END NEW =====
```

---

### STEP 6: Validate the Global Model

**Location:** After creating the global model

**Add this code:**
```python
# ===== NEW: Validate global model on CIFAR-100 =====
print("\n" + "="*60)
print("VALIDATING GLOBAL MODEL")
print("="*60 + "\n")

# Load original CIFAR-100 test data with original labels (NOT mapped)
# This is crucial - use the original test dataset without any label mapping
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761))
])

cifar100_testset = datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

test_dataloader = DataLoader(
    cifar100_testset,
    batch_size=128,
    shuffle=False,
    num_workers=4
)

# Validation function
def validate_model(model, dataloader, device='cpu'):
    """Validate model on test data with original CIFAR-100 labels."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            predictions = torch.argmax(outputs, dim=1)
            
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
    
    accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    avg_loss = (total_loss / total_samples) if total_samples > 0 else 0.0
    
    return accuracy, avg_loss

accuracy, loss = validate_model(global_model, test_dataloader, DEVICE)

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Test Loss: {loss:.6f}")
print(f"Correct: {int(accuracy * len(cifar100_testset) / 100)}/{len(cifar100_testset)}")
print("="*60 + "\n")

# ===== END NEW =====
```

---

## Critical Checklist

Before aggregation, verify:

✅ **Specialized models trained:** All models have been trained with mapped labels

✅ **Metadata saved:** For each model, `class_to_idx_new` and `new_to_original` are saved

✅ **Same feature dimension:** All models have same feature_dim before classifier

❌ **Don't skip:** The remapping step - this is critical!

✅ **Full CIFAR-100 test:** Use original test data with labels 0-99, not mapped labels

✅ **Save global model:** Before and after, save the aggregated model

---

## Troubleshooting

### Issue: Predictions still wrong after aggregation

**Check:**
1. Did you map labels back in the validation loop? ✓
2. Did each model's `new_to_original` mapping save correctly? ✓
3. Are you testing with original CIFAR-100 labels (0-99)? ✓

### Issue: Some classes have zero weights

**This is expected if:**
- No specialized model was trained on that class
- Solution: Ensure your data splits cover all 100 classes across models

### Issue: Aggregated model performs worse than individual models

**This could mean:**
1. Different models trained on non-overlapping class sets → Normal
2. Label mapping errors during aggregation → Check `new_to_original` dictionaries
3. Feature extractors too different → May need feature alignment

---

## Key Differences: Before vs After Integration

| Phase | Before | After |
|-------|--------|-------|
| **Training** | Use mapped labels (0-N) | ✅ Same (no change) |
| **Loss calc** | Uses mapped labels | ✅ Same (no change) |
| **Eval metrics** | Maps back to 0-99 | ✅ Same (no change) |
| **Aggregation** | ❌ Not done | ✅ **NEW:** Remap + aggregate |
| **Global model** | ❌ Not created | ✅ **NEW:** 100-class classifier |
| **Test validation** | N/A | ✅ **NEW:** On original CIFAR-100 |

---

## Summary

You only need to add code **AFTER** training is complete. The training loop and evaluation loops stay the same. The aggregation code:

1. Loads trained models and their label mappings
2. Remaps each classifier from subset space (0-N) to CIFAR-100 space (0-99)
3. Averages the remapped classifiers
4. Creates a new 100-class model
5. Validates on original CIFAR-100 test data

This way, all label mapping is handled correctly, and your aggregated model will work with standard CIFAR-100 evaluation.
