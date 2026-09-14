# Debugging Guide: Model Predicts Same Class for Everything

## Your Symptoms

1. **Model predicts class 23 for almost everything**
2. **Raw logits barely change regardless of input data**

## What This Tells Us

**Constant logits = feature extractor output is same for all inputs OR classifier has wrong structure**

This is a critical sign of aggregation failure, not just low accuracy.

---

## Possible Root Causes

### Cause 1: Feature Extractor Producing Constant Features ⚠️ MOST LIKELY

**Symptom:** Features same for every input → logits same for every input

**Why this happens:**
```
Example: Feature output is [0.1, 0.1, 0.1, ..., 0.1] (constant) for all inputs
  ↓
Classifier: [100, 64] × [64] = same logits for all inputs
  ↓
Prediction: Always argmax of same logits = always class 23
```

**Possible reasons:**
1. **Batch norm running statistics wrong** (running_mean/running_var all zeros)
   - Averaging didn't preserve batch norm stats correctly
   - Running stats have wrong dtype or weren't averaged

2. **Feature averaging produced zero tensor**
   - Averaging code had a bug
   - Device mismatch during averaging (CPU vs GPU)
   - Features never updated from initialization

3. **Averaged features collapsed to constant**
   - Floating point precision issue
   - Wrong averaging dimension
   - Features stacked incorrectly

**How to detect:**
- Run debugging script, check "Feature variance across samples"
- If variance < 0.001: Features are constant!
- Check batch norm running_mean and running_var

**How to fix:**
- Verify averaging code is correct
- Check that all 4 models actually loaded
- Ensure running_mean/running_var aren't all zeros
- Make sure batch norm layers transferred correctly

---

### Cause 2: Classifier Weights Improperly Remapped ⚠️ ALSO LIKELY

**Symptom:** Only some classes have non-zero weights

**Why this happens:**
```
Expected: 43 classes with non-zero weights, 57 with zeros
Actual: Only 1-10 classes with non-zero weights

Example of wrong remapping:
  for model_name in ['outdoors', 'nature', 'people', 'objects_vehicles']:
      model = models[model_name]
      mapping = mappings[model_name]
      
      w, b = remap_classifier(model, mapping)
      global_classifier_weights += w
      # If this loop only runs once OR only loads first model:
      # Only ~6 classes get weights!
```

**Possible reasons:**
1. **Only first model's weights were loaded**
   - Loop crashed or didn't complete
   - Subsequent models not found
   - Exception caught silently

2. **Mapping loaded incorrectly**
   - Wrong mapping for some models
   - Mapping is None or empty

3. **Remapping placed weights at wrong positions**
   - Indexing error in remapping code
   - Classes mapped to wrong positions

**How to detect:**
- Run debugging script, check "Classes with non-zero weights"
- Should see 43 classes (distributed across all models)
- If only 6: Only Outdoors model loaded
- If only 15: Only Outdoors + Nature loaded
- etc.

**How to fix:**
- Check aggregation print statements
- Verify all 4 model files exist
- Add error handling to catch exceptions
- Print which models actually got loaded

---

### Cause 3: Only One Model Actually Loaded ⚠️ POSSIBLE

**Symptom:** Only classes from one model have non-zero weights

**Why:**
```
Model 1 (Outdoors): 6 classes {12, 17, 23, 33, 37, 49}
  ↓ Only these 6 classes get trained weights
  ↓ Other 94 classes are all zero
  ↓ Prediction is argmax(6 non-zeros + 94 zeros)
  ↓ Always one of the 6 outdoors classes
  ↓ Often class 23 if it has highest weight
```

**How to detect:**
- Check which classes have non-zero weights
- Should be diverse: {0, 2, 5, 8, 9, 10, 11, 12, 13, 16, 17, 20, 22, 23, 25, 28, 33, 35, 37, 39, 40, 41, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94}
- If only {12, 17, 23, 33, 37, 49}: Only Outdoors loaded!

---

### Cause 4: Batch Norm Running Statistics Wrong

**Symptom:** Batch norm has running_mean=0 and running_var=0 (or wrong values)

**Why this causes problems:**
```
Batch norm formula:
  output = γ * (input - running_mean) / sqrt(running_var + eps) + β

If running_mean=0 and running_var=0:
  output = γ * input / sqrt(eps) + β
  
If eps is small (~1e-5), division by sqrt(eps) is huge!
  → output explodes or collapses
  → all features become same garbage value
```

**How to detect:**
- Run debugging script, check "BN1 running mean/variance"
- Should NOT be all zeros
- Should NOT be all ones (unless training was very quick)

**How to fix:**
- Make sure BN stats were included in averaging
- Check dtype and device during averaging
- Ensure running_mean/running_var were cloned, not referenced

---

### Cause 5: eval() Mode Not Set

**Symptom:** Model works differently in train vs eval mode

**Why:**
```
Train mode:
  Batch norm uses batch statistics
  Features vary per batch

Eval mode:
  Batch norm uses running_mean/running_var
  Features should still vary per sample
  
If running stats are wrong → features collapse in eval mode
```

**How to fix:**
```python
model.eval()  # CRITICAL - must be set!
with torch.no_grad():
    predictions = model(images)
```

---

## Systematic Debugging Steps

### Step 1: Run the Debug Script

```bash
python Debug_Aggregated_Model.py
```

This will tell you:
- How many classes have non-zero weights
- Whether batch norm stats are reasonable
- Whether features vary across inputs
- Whether logits vary across inputs

### Step 2: Interpret the Output

Look for these warning signs:

```
⚠️  WARNING: Expected 43 trained classes, found 6
  → Only Outdoors model loaded! Check aggregation

⚠️  WARNING: Features have very low variance!
  → Feature extractor not working. Check averaging.

⚠️  CRITICAL: Logits have very low variance!
  → Model output is broken. Major aggregation error.

⚠️  CRITICAL: Model only predicts ONE class!
  → Classifier structure is wrong.
```

### Step 3: Check Aggregation Report

```python
import pickle

with open('./final_specialised_models/aggregation_report.pkl', 'rb') as f:
    report = pickle.load(f)

print(f"Trained classes: {report['trained_classes']}")
print(f"Should be: 43")

if report['trained_classes'] != 43:
    print("ERROR: Wrong number of trained classes!")
```

### Step 4: Verify Model Files Exist

```bash
# Check all 4 model files exist
ls -lh ./final_specialised_models/final_*.model.pt

# Expected output:
# final_outdoors_model.pt
# final_nature_model.pt
# final_people_model.pt
# final_objects_vehicles_model.pt

# Check metadata files
ls -lh ./final_specialised_models/*_metadata.pkl

# Check global model
ls -lh ./final_specialised_models/global_model*.pt
```

### Step 5: Verify Aggregation Script Ran Successfully

Check aggregation output for messages like:
```
✓ Outdoors: remapped 6 classes
✓ Nature: remapped 9 classes
✓ People: remapped 3 classes
✓ Objects_vehicles: remapped 25 classes
```

If you don't see all 4, that's the problem!

---

## Common Fixes

### If only one model loaded:

**Check:** Does the aggregation code try to load all 4 models?

```python
model_names = ['outdoors', 'nature', 'people', 'objects_vehicles']

for model_name in model_names:  # ← Must iterate ALL 4
    model = load_model(f'final_{model_name}_model.pt')
    mappings[model_name] = load_mapping(f'{model_name}_metadata.pkl')
```

**Fix:** Make sure all 4 files exist and filenames are correct

---

### If batch norm stats are wrong:

**Check:** Was batch norm included in averaging?

```python
for key in state_dict.keys():
    if 'bn' in key or key.startswith('layer'):
        features[key] = state_dict[key].clone()
```

**Fix:** Ensure batch norm weights, biases, AND running stats are saved

```python
# Correct:
features['bn1.weight'] = ...
features['bn1.bias'] = ...
features['bn1.running_mean'] = ...  # ← Must include!
features['bn1.running_var'] = ...   # ← Must include!
features['bn1.num_batches_tracked'] = ...
```

---

### If features are constant:

**Check:** Are you in eval() mode?

```python
model.eval()  # ← CRITICAL!
```

**Fix:** Set model to eval mode before inference

---

## Expected vs Actual

| Check | Expected | If Wrong |
|-------|----------|----------|
| Trained classes | 43 | Only 6/9/3/25 → One model |
| BN running mean | Not all zeros | All zeros → Averaging failed |
| BN running var | ~1.0 or reasonable | All zeros/ones → Averaging failed |
| Feature variance | > 0.01 | < 0.001 → Feature collapse |
| Logit variance | > 0.1 | < 0.001 → Constant predictions |
| Unique predictions | 43+ | 1 → Broken classifier |

---

## Next Steps

1. **Run the debug script** immediately
2. **Read the warnings** it produces
3. **Fix the identified issue** (likely one of: batch norm stats, missing models, eval mode)
4. **Re-run aggregation** if needed
5. **Re-test** with debug script

The most likely culprit given your symptoms is **either**:
- ✗ Batch norm running statistics weren't averaged correctly
- ✗ Only one model actually loaded during aggregation

Run the debugging script and share what it reports!
