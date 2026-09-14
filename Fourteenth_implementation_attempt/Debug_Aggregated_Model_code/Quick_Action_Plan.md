# Quick Action Plan: Fix "Always Predicts Class 23" Issue

## IMMEDIATE ACTION (5 minutes)

### Step 1: Run the debug script
```bash
python Debug_Aggregated_Model.py
```

### Step 2: Look for these specific warnings:
```
❌ Wrong number of trained classes - remapping may have failed
❌ Batch norm running mean is zero - aggregation averaging failed?
❌ Features have no variance - feature extractor not working
❌ Logits have no variance - critical feature/model problem
❌ Model only predicts ONE class - severe aggregation error
```

**Save the output** - it will tell us exactly what's broken.

---

## DIAGNOSIS BY SYMPTOM

### Symptom A: "Expected 43 trained classes, found 6"

**Problem:** Only Outdoors model loaded

**Cause:** Aggregation loop didn't complete or models not found

**Quick Check:**
```bash
ls ./final_specialised_models/final_*.pt
# You should see 4 files:
# final_outdoors_model.pt
# final_nature_model.pt
# final_people_model.pt
# final_objects_vehicles_model.pt
```

**Quick Fix:**
1. Verify all 4 model files exist
2. Check filenames match what aggregation code expects
3. Re-run aggregation code

---

### Symptom B: "BN running mean is zero"

**Problem:** Batch norm statistics weren't averaged correctly

**Cause:** 
- running_mean/running_var not included in feature extraction
- Dtype mismatch during averaging
- Not cloned properly (references instead of copies)

**Quick Fix:**
Check the aggregation code's `extract_feature_layers()`:

```python
# CORRECT - includes BN stats:
feature_layers['bn1.running_mean'] = state_dict['bn1.running_mean'].clone()
feature_layers['bn1.running_var'] = state_dict['bn1.running_var'].clone()
feature_layers['bn1.num_batches_tracked'] = state_dict['bn1.num_batches_tracked'].clone()

# WRONG - missing BN stats:
# (only copies weight and bias)
```

If missing, update extraction function to include:
- `bn1.weight`
- `bn1.bias`
- `bn1.running_mean` ← MUST HAVE
- `bn1.running_var` ← MUST HAVE
- `bn1.num_batches_tracked` ← MUST HAVE

Then **re-run aggregation**.

---

### Symptom C: "Features have very low variance"

**Problem:** Feature extractor output is same for all inputs

**Cause:** Batch norm running stats are broken OR features never changed from init

**Quick Check in Python:**
```python
import torch
from ResNet20_DynamicModel_v13 import ResNet20_DynamicModel

model = ResNet20_DynamicModel(num_classes=100)
model.load_state_dict(torch.load('global_model.pt'))
model.eval()

# Check batch norm stats
print("BN mean:", model.bn1.running_mean[:5])
print("BN var:", model.bn1.running_var[:5])

# Should NOT be all zeros!
```

**Quick Fix:**
1. Verify batch norm stats in global model
2. If all zeros: Re-run aggregation with corrected code
3. Make sure `.eval()` mode is set when testing

---

### Symptom D: "Logits have very low variance"

**Problem:** Model output is constant for all inputs (critical!)

**Cause:** 
- Feature extractor broken (Symptom C)
- OR classifier weights all zeros/constant
- OR model not in eval() mode properly

**Quick Fix:**
```python
model.eval()  # MUST DO THIS
torch.no_grad()  # Use this in context manager

# Make sure you're not accidentally in train mode!
# Check with:
print(f"Model training mode: {model.training}")  # Should be False
```

---

### Symptom E: "Model only predicts class X"

**Problem:** Only that class has non-zero classifier weights

**Cause:** Aggregation error - classifier remapping failed

**Quick Check:**
```python
weights = model.linear.weight.data
print("Non-zero classes:")
for i in range(100):
    if weights[i].norm() > 0.001:
        print(i, end=' ')
```

**Quick Fix:**
If you see only 6 classes (12, 17, 23, 33, 37, 49):
- Only Outdoors model loaded
- Check aggregation error logs
- Re-run aggregation

If you see scattered classes but most zeros:
- Something wrong with remapping
- Check mapping files exist
- Verify remapping code logic

---

## MOST LIKELY PROBLEMS (Ranked by Probability)

### 1. 🔴 MOST LIKELY: Batch norm running stats not averaged

**Symptoms:**
- Features are constant
- Logits are constant
- Debug script shows: "BN running mean is zero"

**Fix in 2 minutes:**
1. Find `extract_feature_layers()` function
2. Add these lines:
```python
feature_layers['bn1.running_mean'] = state_dict['bn1.running_mean'].clone()
feature_layers['bn1.running_var'] = state_dict['bn1.running_var'].clone()
feature_layers['bn1.num_batches_tracked'] = state_dict['bn1.num_batches_tracked'].clone()
```
3. Re-run aggregation
4. Test

---

### 2. 🟡 LIKELY: Only one model actually loaded

**Symptoms:**
- Debug script shows: "Expected 43 trained classes, found 6"
- Only classes {12, 17, 23, 33, 37, 49} have weights

**Fix in 1 minute:**
1. Check that all 4 model files exist:
   - `final_outdoors_model.pt`
   - `final_nature_model.pt`
   - `final_people_model.pt`
   - `final_objects_vehicles_model.pt`

2. Check filenames in aggregation code match exactly

3. Re-run aggregation

---

### 3. 🟡 LIKELY: Model not in eval() mode

**Symptoms:**
- Debug script might show confusion between train/eval
- Features/logits vary in eval but not in train

**Fix in 30 seconds:**
```python
model.eval()  # Add this line!
with torch.no_grad():
    predictions = model(images)
```

---

## VERIFICATION CHECKLIST

After making a fix, verify with this quick test:

```python
import torch
from ResNet20_DynamicModel_v13 import ResNet20_DynamicModel

model = ResNet20_DynamicModel(num_classes=100)
model.load_state_dict(torch.load('global_model.pt'))
model.eval()  # MUST DO THIS

# Test 1: Do features vary?
input1 = torch.randn(1, 3, 32, 32)
input2 = torch.randn(1, 3, 32, 32)

# Extract features
with torch.no_grad():
    # Forward through feature extractor
    feat1 = model.layer3(model.layer2(model.layer1(
        torch.nn.functional.relu(model.bn1(model.conv1(input1))))))
    feat2 = model.layer3(model.layer2(model.layer1(
        torch.nn.functional.relu(model.bn1(model.conv1(input2))))))

feature_diff = (feat1 - feat2).abs().mean().item()
print(f"Feature difference: {feature_diff}")  # Should be > 0.01
if feature_diff < 0.001:
    print("❌ Features not varying - BN stats broken!")
else:
    print("✓ Features varying correctly")

# Test 2: Do logits vary?
with torch.no_grad():
    logits1 = model(input1)
    logits2 = model(input2)

logit_diff = (logits1 - logits2).abs().mean().item()
print(f"Logit difference: {logit_diff}")  # Should be > 0.1
if logit_diff < 0.01:
    print("❌ Logits not varying - model broken!")
else:
    print("✓ Logits varying correctly")

# Test 3: Do we get different predictions?
pred1 = torch.argmax(logits1).item()
pred2 = torch.argmax(logits2).item()
print(f"Prediction 1: {pred1}, Prediction 2: {pred2}")
if pred1 == pred2:
    print("⚠️  Same prediction (might be okay randomly)")
else:
    print("✓ Different predictions")
```

---

## If Still Broken After Fixes

**Run the full debug script again:**
```bash
python Debug_Aggregated_Model.py > debug_output.txt
```

**Share what it reports** - especially:
1. Number of trained classes (should be 43)
2. Batch norm statistics (should NOT be zero)
3. Feature variance (should be > 0.01)
4. Logit variance (should be > 0.1)
5. Sample predictions (should vary)

---

## One-Minute Summary

**Your problem:** Constant logits = feature extractor or classifier broken

**Most likely cause:** Batch norm stats weren't averaged (or only one model loaded)

**Quick fix:**
1. Run: `python Debug_Aggregated_Model.py`
2. Read warnings
3. Fix identified issue
4. Re-run aggregation
5. Test again

**Expected time:** 5-15 minutes to identify and fix
