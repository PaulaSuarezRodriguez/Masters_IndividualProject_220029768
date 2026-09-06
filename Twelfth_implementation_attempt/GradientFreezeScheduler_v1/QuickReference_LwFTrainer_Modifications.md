# Quick Reference: Modifying Your LwFTrainer with Gradient Freeze Strategies

## TL;DR Changes Needed

### 1. **Imports** (add to top of file)
```python
from GradientFreezeScheduler import (
    GradientFreezeScheduler,
    apply_gradient_freezing,
    FreezeConfig
)
```

### 2. **Constructor** (`__init__` method, around line 455)
Add these parameters:
```python
def __init__(
    self,
    model: nn.Module,
    # ... existing parameters ...
    freeze_strategy: str = 'binary',      # NEW: Strategy name
    freeze_config: Optional[Dict] = None, # NEW: Strategy config
):
    # ... existing initialization ...
    
    # NEW: Initialize gradient freeze scheduler
    self.freeze_strategy = freeze_strategy
    self.freeze_config = freeze_config or {}
    self.freeze_scheduler = None
```

### 3. **Training Loop** (`train_experience` method, around line 653)
**At the START of the method** (after `for _ in range(self.train_epochs):`):
```python
# NEW: Initialize scheduler once per experience
if self.freeze_scheduler is None:
    self.freeze_scheduler = GradientFreezeScheduler(
        freeze_strategy=self.freeze_strategy,
        total_epochs=self.train_epochs,
        ft_shared_after=self.ft_shared_after,
        config=self.freeze_config,
        verbose=False
    )
```

**REPLACE** your current gradient freezing code (lines 828-872) with:
```python
# ===== REPLACE LINES 828-872 WITH THIS =====
# Apply gradient freezing strategy (after loss.backward())
freeze_config = self.freeze_scheduler.get_freeze_config(
    current_epoch=_,
    loss=loss.item()
)

apply_gradient_freezing(
    model=self.model,
    freeze_config=freeze_config,
    old_classes=self.old_classes,
    device=self.device,
    verbose=False  # Set to True for debugging
)
```

---

## Original Code vs. New Code

### BEFORE (Your Current Code - Lines 828-872)
```python
if (self.teacher is not None) and (_ <= self.ft_shared_after):

    # Freeze shared layers
    for module in self.model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module is not self.model.linear:
                if module.weight.grad is not None:
                    module.weight.grad.zero_()
                if module.bias is not None:
                    if module.bias.grad is not None:
                        module.bias.grad.zero_()

    # Freeze old classifier rows
    if len(self.old_classes) > 0:
        old_num_classes = len(self.old_classes)
        if self.model.linear.weight.grad is not None:
            self.model.linear.weight.grad[:old_num_classes].zero_()
        if self.model.linear.bias.grad is not None:
            self.model.linear.bias.grad[:old_num_classes].zero_()

self.optimizer.step()
```

### AFTER (With Flexible Strategy Framework)
```python
# Apply gradient freezing strategy (after loss.backward())
freeze_config = self.freeze_scheduler.get_freeze_config(
    current_epoch=_,
    loss=loss.item()
)

apply_gradient_freezing(
    model=self.model,
    freeze_config=freeze_config,
    old_classes=self.old_classes,
    device=self.device,
    verbose=False
)

self.optimizer.step()
```

---

## How to Use Different Strategies

### Strategy 1: Binary (Default - matches your current behavior)
```python
trainer = LwFTrainer(
    model=model,
    freeze_strategy='binary',
    freeze_config={'freeze_before': 8},
    # ... other params ...
)
```

### Strategy 2: Gradual Unfreezing (Recommended starting point)
```python
trainer = LwFTrainer(
    model=model,
    freeze_strategy='gradual_unfreezing',
    freeze_config={
        'start_epoch': 0,
        'end_epoch': 8,
        'unfreeze_shared_first': True  # Shared layers unfreeze first
    },
    # ... other params ...
)
```

### Strategy 3: Two-Phase (Tight then Loose)
```python
trainer = LwFTrainer(
    model=model,
    freeze_strategy='two_phase',
    freeze_config={
        'tight_freeze_epochs': 3,      # First 3 epochs: maximum stability
        'gradual_unfreeze_epochs': 5   # Last 5 epochs: increasing plasticity
    },
    # ... other params ...
)
```

### Strategy 4: Exponential Decay
```python
trainer = LwFTrainer(
    model=model,
    freeze_strategy='exponential_decay',
    freeze_config={
        'decay_rate': 0.5,           # Freeze intensity halves
        'half_life_epochs': 3        # ... every 3 epochs
    },
    # ... other params ...
)
```

### Strategy 5: Layer-Wise Curriculum
```python
trainer = LwFTrainer(
    model=model,
    freeze_strategy='layer_wise_curriculum',
    freeze_config={
        'num_stages': 3,                    # Unfreeze in 3 stages
        'freeze_order': 'deep_to_shallow'   # Deepest layers unfreeze last
    },
    # ... other params ...
)
```

### Strategy 6: Classifier-Only Freezing
```python
# Only freeze classifier rows, let shared layers train immediately
trainer = LwFTrainer(
    model=model,
    freeze_strategy='classifier_only',
    freeze_config={
        'freeze_classifier': True,
        'freeze_shared': False  # Don't freeze shared layers at all
    },
    # ... other params ...
)
```

---

## Experimental Pipeline

### Step 1: Set Up Baseline
```python
# Test your current binary strategy to establish baseline
lwf_binary = LwFTrainer(
    model=ResNet20_DynamicModel(num_classes=0),
    freeze_strategy='binary',
    freeze_config={'freeze_before': 8},
    train_epochs=10,
    # ... other params ...
)

# Log results: BWT, FWT, AvgAcc, Per-experience accuracy
baseline_results = run_training_pipeline(lwf_binary)
print(f"Baseline BWT: {baseline_results['BWT']:.4f}")
print(f"Baseline AvgAcc: {baseline_results['AvgAcc']:.4f}")
```

### Step 2: Test Promising Strategies
```python
strategies_to_test = [
    ('gradual_unfreezing', {'start_epoch': 0, 'end_epoch': 8, 'unfreeze_shared_first': True}),
    ('two_phase', {'tight_freeze_epochs': 3, 'gradual_unfreeze_epochs': 5}),
    ('exponential_decay', {'decay_rate': 0.6, 'half_life_epochs': 2.5}),
]

results_summary = {}
for strategy_name, config in strategies_to_test:
    trainer = LwFTrainer(
        model=ResNet20_DynamicModel(num_classes=0),
        freeze_strategy=strategy_name,
        freeze_config=config,
        train_epochs=10,
        # ... other params ...
    )
    results = run_training_pipeline(trainer)
    results_summary[strategy_name] = results
    print(f"{strategy_name:25s} | BWT: {results['BWT']:.4f} | AvgAcc: {results['AvgAcc']:.4f}")
```

### Step 3: Hyperparameter Fine-Tuning
Once you identify the best strategy, fine-tune its hyperparameters:
```python
# Example: Fine-tune gradual_unfreezing's unfreeze schedule
for unfreeze_shared_first in [True, False]:
    for end_epoch in [6, 8, 10]:
        config = {
            'start_epoch': 0,
            'end_epoch': end_epoch,
            'unfreeze_shared_first': unfreeze_shared_first
        }
        trainer = LwFTrainer(
            model=ResNet20_DynamicModel(num_classes=0),
            freeze_strategy='gradual_unfreezing',
            freeze_config=config,
            # ... other params ...
        )
        results = run_training_pipeline(trainer)
        print(f"end_epoch={end_epoch}, shared_first={unfreeze_shared_first} | "
              f"AvgAcc: {results['AvgAcc']:.4f}")
```

---

## Metrics to Track (Important!)

### 1. **Backward Transfer (BWT)** - Stability indicator
```python
def compute_bwt(accuracies_per_task):
    """
    BWT = mean(accuracy drop on old tasks after learning new ones)
    Higher is better (less forgetting)
    
    accuracies_per_task: Dict[exp_id][exp_id] = accuracy
    """
    bwt_values = []
    for exp_new in range(1, len(accuracies_per_task)):
        for exp_old in range(exp_new):
            # Accuracy on old task after seeing new task - accuracy on old task after learning it
            bwt = accuracies_per_task[exp_new][exp_old] - accuracies_per_task[exp_old][exp_old]
            bwt_values.append(bwt)
    return sum(bwt_values) / len(bwt_values)
```

### 2. **Forward Transfer (FWT)** - Plasticity indicator
```python
def compute_fwt(accuracies_per_task):
    """
    FWT = mean(accuracy gain on new tasks from seeing old ones)
    """
    fwt_values = []
    for exp_new in range(1, len(accuracies_per_task)):
        # This would need zero-shot or oracle accuracies
        # For now, just new task accuracy
        fwt = accuracies_per_task[exp_new][exp_new]
        fwt_values.append(fwt)
    return sum(fwt_values) / len(fwt_values)
```

### 3. **Average Accuracy** - Overall performance
```python
def compute_average_accuracy(accuracies_per_task):
    """
    Average accuracy after all tasks are learned
    """
    final_exp = max(accuracies_per_task.keys())
    all_accuracies = accuracies_per_task[final_exp].values()
    return sum(all_accuracies) / len(all_accuracies)
```

### 4. **Per-Experience Accuracy**
```python
# After each experience, log accuracy on ALL previous + current
self.experience_accuracies[exp_id] = {
    exp_seen_id: accuracy_on_exp_seen_id
    for exp_seen_id in range(exp_id + 1)
}
```

---

## Troubleshooting

### Problem: Gradients are still zeroed everywhere
**Solution**: Check that your `apply_gradient_freezing()` is being called AFTER `loss.backward()` but BEFORE `optimizer.step()`.

### Problem: Strategy isn't making a difference
**Solution**: 
- Print the freeze config to verify it's actually changing: `print(freeze_config)`
- Set `verbose=True` in `apply_gradient_freezing()` to see what's being frozen
- Verify `self.old_classes` is being populated correctly

### Problem: Training is too slow
**Solution**: 
- Some strategies require more bookkeeping (layer_wise_curriculum, adaptive_boosting)
- Start with 'gradual_unfreezing' or 'two_phase' which are lightweight
- Profile which part is slow using `torch.profiler`

### Problem: Memory usage increased
**Solution**: 
- The framework stores freeze configs for logging/debugging
- Disable logging: remove the freeze log tracking code
- Or clean logs periodically: `self.freeze_logs.clear()`

---

## Expected Performance Changes

### By Strategy Type

| Strategy | Expected BWT Impact | Expected FWT Impact | Risk | Recommendation |
|----------|---------------------|---------------------|------|-----------------|
| **Binary** (Baseline) | ±0% | ±0% | Low | Good control baseline |
| **Gradual Unfreezing** | +2-5% | -1-2% | Low | Safe first try |
| **Two-Phase** | +3-7% | -0-1% | Medium | Good stability-plasticity balance |
| **Exponential Decay** | +1-4% | -1-3% | Low | Smooth alternative to binary |
| **Layer-Wise Curriculum** | +3-8% | +0-2% | High | Best if it works, risky if misconfigured |
| **Classifier-Only** | -2-0% | +2-5% | High | Good for plasticity, bad for stability |

**Note**: These are rough expectations based on theory. Your actual results depend on:
- Task similarity (similar tasks → favor stability)
- Task dissimilarity (different tasks → favor plasticity)
- Number of classes per task
- Dataset difficulty

---

## Next Steps

1. **Implement the three file changes** above
2. **Run your baseline** with `freeze_strategy='binary'` to verify nothing breaks
3. **Test gradual_unfreezing** - most likely to help
4. **Track metrics** for at least 3 experiences
5. **Compare against baseline** using BWT, FWT, AvgAcc
6. **Fine-tune** the best strategy's hyperparameters
7. **Report results** with graphs showing BWT/FWT curves over experiences

Good luck! Let me know if you hit any issues with the integration.
