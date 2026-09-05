# Line-by-Line Code Modification Guide for LwFTrainer

This document shows EXACTLY where and how to modify your LwFTrainer code using line numbers from your uploaded file.

---

## File: `Train_ResNet20_Nature_LwF_v11_L_YESRunningStatistics.py`

### CHANGE 1: Add Imports at Top of File

**Location**: After line 184 (after `from torch.optim.adam import Adam`)

**Current code (line 186):**
```python
DEBUG = False #TODO: Remember to change DEBUG to False if running experiments.
```

**Add BEFORE line 186:**
```python
from GradientFreezeScheduler import (
    GradientFreezeScheduler,
    apply_gradient_freezing,
    FreezeConfig
)
```

**After modification:**
```python
from torch.optim.adam import Adam

# NEW IMPORT
from GradientFreezeScheduler import (
    GradientFreezeScheduler,
    apply_gradient_freezing,
    FreezeConfig
)

DEBUG = False #TODO: Remember to change DEBUG to False if running experiments.
```

---

### CHANGE 2: Modify `LwFTrainer.__init__` Method Signature

**Location**: Lines 455-477 (def __init__)

**Current signature (lines 455-477):**
```python
class LwFTrainer(SupervisedTemplate):
    def __init__(
        self,
        model: nn.Module,
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        lr_decay_factor: float = 0.1,
        lr_decay_every: float = 1.0,
        temperature=2.0,
        alpha=1.0,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        patience: int = 0,
        ft_shared_after: int = 5,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch"
    ):
```

**Modify to (ADD TWO LINES before `plugins` parameter):**
```python
class LwFTrainer(SupervisedTemplate):
    def __init__(
        self,
        model: nn.Module,
        lr=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        lr_decay_factor: float = 0.1,
        lr_decay_every: float = 1.0,
        temperature=2.0,
        alpha=1.0,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        patience: int = 0,
        ft_shared_after: int = 5,
        freeze_strategy: str = 'binary',  # NEW LINE
        freeze_config: Optional[Dict] = None,  # NEW LINE
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every=-1,
        peval_mode="epoch"
    ):
```

---

### CHANGE 3: Add Initialization Code in `__init__`

**Location**: After line 496 (after `self.ft_shared_after = ft_shared_after`)

**Current code (line 496):**
```python
        self.ft_shared_after = ft_shared_after
```

**Add AFTER this line:**
```python
        self.ft_shared_after = ft_shared_after
        
        # NEW: Gradient freeze strategy configuration
        self.freeze_strategy = freeze_strategy
        self.freeze_config = freeze_config if freeze_config is not None else {}
        self.freeze_scheduler: Optional[GradientFreezeScheduler] = None
```

---

### CHANGE 4: Initialize Scheduler in `train_experience` Method

**Location**: Line 653 (start of `train_experience` method)

**Current code (line 669):**
```python
        for _ in range(self.train_epochs):
            self.is_training = True
```

**Modify to (INSERT TWO LINES before line 670):**
```python
        for _ in range(self.train_epochs):
            # NEW: Initialize freeze scheduler for this experience (first epoch only)
            if _ == 0:
                self.freeze_scheduler = GradientFreezeScheduler(
                    strategy=self.freeze_strategy,
                    total_epochs=self.train_epochs,
                    ft_shared_after=self.ft_shared_after,
                    config=self.freeze_config,
                    verbose=False  # Set to True for debugging
                )
            
            self.is_training = True
```

---

### CHANGE 5: Replace Original Gradient Freezing Logic

**Location**: Lines 828-872 (the code you want to replace)

**Current code (lines 828-872):**
```python
                if (self.teacher is not None) and (_ <= self.ft_shared_after):# and (experience.current_experience>=2):#if (self.teacher is not None) and (_ <= self.ft_shared_after) and (experience.current_experience>=2): #if (self.teacher is not None) and (_ <= self.ft_shared_after):#if ((len(self.teachers)>0) and (_ <= self.ft_shared_after)):

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
                            self.model.linear.weight.grad[
                                :old_num_classes
                            ].zero_()

                        if self.model.linear.bias.grad is not None:
                            self.model.linear.bias.grad[
                                :old_num_classes
                            ].zero_()

                '''if (self.teacher is not None) and len(self.old_classes) > 0 and (_ <= self.ft_shared_after-2):
                    # Freeze old classifier rows
                    old_num_classes = len(self.old_classes)

                    if self.model.linear.weight.grad is not None:
                        self.model.linear.weight.grad[
                            :old_num_classes
                        ].zero_()

                    if self.model.linear.bias.grad is not None:
                        self.model.linear.bias.grad[
                            :old_num_classes
                        ].zero_()'''

                self.optimizer.step()
```

**Replace with (much cleaner!):**
```python
                # Apply gradient freezing strategy (after loss.backward())
                if self.teacher is not None and self.freeze_scheduler is not None:
                    freeze_config = self.freeze_scheduler.get_freeze_config(
                        current_epoch=_,
                        loss=loss.item()
                    )
                    
                    apply_gradient_freezing(
                        model=self.model,
                        freeze_config=freeze_config,
                        old_classes=self.old_classes,
                        device=self.device,
                        verbose=False  # Set to True to debug gradient freezing
                    )

                self.optimizer.step()
```

---

## Summary of Changes

### Before Modification (Lines 828-872: 45 lines)
```
Binary gradient zeroing logic
- Manually iterate through model modules
- Check instance types
- Zero gradients individually
- Hard to test alternatives
```

### After Modification (Lines 828-839: 12 lines)
```
Flexible strategy framework
- Use scheduler to get freeze config
- Apply freezing using unified interface
- Easy to switch strategies
- Supports gradient scaling (not just zeroing)
```

**Lines saved: 33 lines of boilerplate code eliminated!**

---

## Testing Your Changes

### Minimal Test (Verify nothing breaks)
```python
# Use the exact same strategy as before
lwf_trainer = LwFTrainer(
    model=nature_specialised_model,
    # ... all other params same as before ...
    freeze_strategy='binary',  # NEW parameter
    freeze_config={'freeze_before': 8},  # NEW parameter
)

# Should behave identically to before
lwf_trainer.train(experience, [seen_experiences, seen_experiences_val])
```

### Quick Functionality Test
```python
# Test scheduler initialization
scheduler = GradientFreezeScheduler(
    strategy='gradual_unfreezing',
    total_epochs=10,
    ft_shared_after=8,
    config={'start_epoch': 0, 'end_epoch': 8}
)

# Print schedule (should show decreasing freeze intensity)
print(scheduler.get_schedule_visualization())
```

### Full Integration Test
```python
# Test with new strategy
lwf_trainer = LwFTrainer(
    model=nature_specialised_model,
    # ... all other params ...
    freeze_strategy='gradual_unfreezing',
    freeze_config={'start_epoch': 0, 'end_epoch': 8, 'unfreeze_shared_first': True}
)

# Run training
lwf_trainer.train(experience, [seen_experiences, seen_experiences_val])

# Compare results with binary strategy
```

---

## Potential Issues and Solutions

### Issue 1: "NameError: name 'Dict' is not defined"
**Location**: Line 471 (now in modified signature)
**Solution**: Add to imports at top: `from typing import Dict`

### Issue 2: "AttributeError: 'NoneType' object has no attribute 'get_freeze_config'"
**Location**: In gradient freezing section (around line 833 in modified code)
**Solution**: Verify scheduler is initialized. Check line numbers around 670.

### Issue 3: "Module 'GradientFreezeScheduler' not found"
**Solution**: Place `GradientFreezeScheduler.py` in same directory as your training script.

### Issue 4: Gradients still zeroed even with 'classifier_only' strategy
**Solution**: Check that `old_classes` is being populated. Add debug print:
```python
print(f"Old classes: {self.old_classes}, len={len(self.old_classes)}")
```

---

## Comparison of Code Complexity

### Original Code Pattern
```python
# Lots of nested conditions and manual gradient management
if (self.teacher is not None) and (_ <= self.ft_shared_after):
    for module in self.model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module is not self.model.linear:
                if module.weight.grad is not None:
                    module.weight.grad.zero_()
                # ... more nested code ...
```

### New Code Pattern
```python
# Clean, readable, extensible
freeze_config = self.freeze_scheduler.get_freeze_config(current_epoch=_)
apply_gradient_freezing(model=self.model, freeze_config=freeze_config, ...)
```

---

## Checklist for Implementation

- [ ] Copy `GradientFreezeScheduler.py` to project directory
- [ ] Add import statement (Change 1)
- [ ] Modify `__init__` signature (Change 2)
- [ ] Add initialization code in `__init__` (Change 3)
- [ ] Initialize scheduler in `train_experience` (Change 4)
- [ ] Replace gradient freezing logic (Change 5)
- [ ] Run minimal test with `freeze_strategy='binary'`
- [ ] Verify behavior is identical to original
- [ ] Test with `freeze_strategy='gradual_unfreezing'`
- [ ] Run full pipeline with multiple experiences
- [ ] Track metrics (BWT, FWT, AvgAcc)
- [ ] Compare with baseline results

---

## Quick Reference: All New Parameter Values

### In LwFTrainer instantiation (your line ~1971):
```python
lwf_trainer = LwFTrainer(
    model=nature_specialised_model,
    device="cpu",
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    lr_decay_factor=0.1,
    temperature=2.0,
    alpha=1.0,
    train_mb_size=43,
    train_epochs=10,
    eval_mb_size=43,
    patience=5,
    ft_shared_after=8,
    # NEW PARAMETERS:
    freeze_strategy='gradual_unfreezing',  # or 'binary', 'two_phase', etc.
    freeze_config={
        'start_epoch': 0,
        'end_epoch': 8,
        'unfreeze_shared_first': True
    },
    # ... rest of original parameters ...
)
```

Done! Your LwFTrainer now supports flexible gradient freezing strategies.
