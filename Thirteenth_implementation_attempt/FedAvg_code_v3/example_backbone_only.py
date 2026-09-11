"""
Practical Examples: Backbone-Only FedAvg Aggregation

Scenario: Models trained on non-overlapping class sets
- Model 1: Classes 0-9
- Model 2: Classes 10-19
- Model 3: Classes 20-29

Aggregation: Backbone only (element-wise)
Skip: Classifier (since trained on different classes)
"""

import torch
import torch.nn as nn
from fedavg_aggregator_backbone_only import (
    FedAvgAggregatorBackboneOnly,
    fedavg_element_wise_aggregation
)


# ============================================================================
# Model Definition
# ============================================================================

class ContinualLearningModel(nn.Module):
    """Model with backbone and expanding classifier."""
    
    def __init__(self, input_size=28*28, hidden_size=256, num_classes=10):
        super().__init__()
        
        # Shared backbone (same size across all models)
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        
        # Classifier (different size per model based on task classes)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ============================================================================
# SCENARIO 1: Using the Aggregator Class
# ============================================================================

def scenario_1_aggregator_class():
    """
    Use FedAvgAggregatorBackboneOnly class.
    
    Pros: Clean API, built-in reporting
    Cons: Slightly more abstraction
    """
    print("\n" + "="*80)
    print("SCENARIO 1: Using Aggregator Class")
    print("="*80)
    
    print("""
    Setup:
    - Model 1: Classes 0-9   (1000 training samples)
    - Model 2: Classes 10-19 (1200 training samples)
    - Model 3: Classes 20-29 (900 training samples)
    
    Goal: Aggregate only backbone (feature extractor)
    Skip: Classifier heads (different classes, can't aggregate)
    """)
    
    # Create example models
    print("\nCreating models...")
    model_configs = [
        ('models/model_classes_0to9.pth', 10, 1000),
        ('models/model_classes_10to19.pth', 10, 1200),
        ('models/model_classes_20to29.pth', 10, 900),
    ]
    
    model_paths = []
    dataset_sizes = []
    
    for path, num_classes, n_samples in model_configs:
        model = ContinualLearningModel(num_classes=num_classes)
        # In practice: model = train_with_lwf(model, task_data)
        torch.save(model.state_dict(), path)
        model_paths.append(path)
        dataset_sizes.append(n_samples)
        print(f"  ✓ {path} ({num_classes} classes, {n_samples} samples)")
    
    # Initialize aggregator
    print("\nInitializing aggregator...")
    aggregator = FedAvgAggregatorBackboneOnly(
        weighting_strategy='data_driven',
        exclude_layers=['classifier']  # Only exclude classifier
    )
    
    # Load models
    print("Loading model weights...")
    aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
    
    # Print report
    aggregator.print_report()
    
    # Aggregate
    print("\nPerforming aggregation...")
    aggregated_weights = aggregator.aggregate()
    
    # Save
    print("\nSaving aggregated weights...")
    aggregator.save_aggregated_weights('aggregated_backbone.pth')
    
    # Load into model
    print("\nLoading aggregated backbone into target model...")
    target_model = ContinualLearningModel(num_classes=10)  # For new classes
    target_model = aggregator.aggregate_into_model(target_model)
    print("  ✓ Aggregated backbone loaded")
    print("  ✓ Classifier from target model preserved")
    
    # Verify
    print("\nVerifying aggregation...")
    agg_state = torch.load('aggregated_backbone.pth')
    print(f"  Aggregated parameters:")
    for key in agg_state.keys():
        print(f"    - {key}: {agg_state[key].shape}")
    
    # Cleanup
    import os
    for path, _, _ in model_configs:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists('aggregated_backbone.pth'):
        os.remove('aggregated_backbone.pth')
    
    return aggregator, target_model


# ============================================================================
# SCENARIO 2: Direct Element-Wise Function (Your Code Style)
# ============================================================================

def scenario_2_direct_element_wise():
    """
    Direct element-wise aggregation function.
    
    Pros: Direct control, matches your code style
    Cons: More manual code
    """
    print("\n" + "="*80)
    print("SCENARIO 2: Direct Element-Wise Aggregation")
    print("="*80)
    
    print("""
    Approach similar to your code snippet:
    1. Load weights into list
    2. Group by layer name
    3. Aggregate element-wise
    4. Skip excluded layers
    """)
    
    # Create models
    print("\nCreating models...")
    model_configs = [
        ('model1.pth', 10, 1000),
        ('model2.pth', 10, 1200),
        ('model3.pth', 10, 900),
    ]
    
    model_paths = []
    dataset_sizes = []
    
    for path, num_classes, n_samples in model_configs:
        model = ContinualLearningModel(num_classes=num_classes)
        torch.save(model.state_dict(), path)
        model_paths.append(path)
        dataset_sizes.append(n_samples)
        print(f"  ✓ {path}")
    
    # Load weights (YOUR CODE APPROACH)
    print("\nLoading weights manually...")
    weights_list = [torch.load(path) for path in model_paths]
    print(f"  ✓ Loaded {len(weights_list)} models")
    
    # Direct aggregation
    print("\nPerforming element-wise aggregation...")
    aggregated = fedavg_element_wise_aggregation(
        weights_list=weights_list,
        dataset_sizes=dataset_sizes,
        exclude_layers=['classifier'],
        weighting_strategy='data_driven'
    )
    
    # Save
    print("\nSaving aggregated backbone...")
    torch.save(aggregated, 'aggregated_backbone.pth')
    print("  ✓ Saved to aggregated_backbone.pth")
    
    # Verify what was aggregated
    print("\nAggregated layers:")
    for key in aggregated.keys():
        print(f"  ✓ {key}: {aggregated[key].shape}")
    
    # Cleanup
    import os
    for path in model_paths:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists('aggregated_backbone.pth'):
        os.remove('aggregated_backbone.pth')


# ============================================================================
# SCENARIO 3: Complete Workflow (Training → Aggregation → Fine-tuning)
# ============================================================================

def scenario_3_complete_workflow():
    """
    Complete workflow from continual learning to aggregation.
    """
    print("\n" + "="*80)
    print("SCENARIO 3: Complete Workflow")
    print("="*80)
    
    code = """
import torch
import torch.nn as nn
import torch.optim as optim
from fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly

# ============================================================================
# PHASE 1: CONTINUAL LEARNING TRAINING
# ============================================================================

class ContinualLearningModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

# Define tasks with non-overlapping classes
tasks = [
    {'name': 'task1', 'classes': range(0, 10)},    # Classes 0-9
    {'name': 'task2', 'classes': range(10, 20)},   # Classes 10-19
    {'name': 'task3', 'classes': range(20, 30)},   # Classes 20-29
]

# Train on each task (your LwF training logic)
for task_id, task_info in enumerate(tasks):
    print(f"\\nTraining {task_info['name']}...")
    
    # Initialize model for this task
    model = ContinualLearningModel(num_classes=len(task_info['classes']))
    
    # Your LwF training logic
    # model = lwf_train(model, task_data, old_model)
    
    # Save model
    path = f"models/{task_info['name']}_model.pth"
    torch.save(model.state_dict(), path)
    print(f"  Saved to {path}")

# ============================================================================
# PHASE 2: BACKBONE AGGREGATION
# ============================================================================

model_paths = [
    'models/task1_model.pth',
    'models/task2_model.pth',
    'models/task3_model.pth',
]
dataset_sizes = [1000, 1200, 900]  # Samples per task

# Create aggregator
aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',
    exclude_layers=['classifier']  # Skip classifier
)

# Load and aggregate
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
aggregator.print_report()
aggregator.save_aggregated_weights('models/aggregated_backbone.pth')

# ============================================================================
# PHASE 3: USE AGGREGATED BACKBONE
# ============================================================================

# Option A: Transfer learning on new task
print("\\nTransfer learning setup...")
aggregated_backbone = torch.load('models/aggregated_backbone.pth')

new_model = ContinualLearningModel(num_classes=10)  # For new task
new_state = new_model.state_dict()

# Load aggregated backbone (exclude classifier)
for key in aggregated_backbone.keys():
    if 'classifier' not in key:
        new_state[key] = aggregated_backbone[key]

new_model.load_state_dict(new_state)

# Fine-tune on new task
print("Fine-tuning on new task...")
optimizer = optim.Adam(new_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_data, batch_labels in new_task_loader:
        outputs = new_model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ============================================================================
# PHASE 4: EVALUATION
# ============================================================================

new_model.eval()
with torch.no_grad():
    accuracy = 0.0
    for batch_data, batch_labels in test_loader:
        outputs = new_model(batch_data)
        accuracy += (outputs.argmax(1) == batch_labels).float().mean().item()

print(f"Fine-tuned model accuracy: {accuracy:.2%}")

# Compare with individual models
print("\\nComparison with individual models:")
for i, path in enumerate(model_paths):
    model = ContinualLearningModel(num_classes=10)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    with torch.no_grad():
        acc = 0.0
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            acc += (outputs.argmax(1) == batch_labels).float().mean().item()
    
    print(f"  Model {i+1} accuracy: {acc:.2%}")
"""
    
    print(code)


# ============================================================================
# SCENARIO 4: Custom Exclusion Patterns
# ============================================================================

def scenario_4_custom_exclusion():
    """
    Example with custom layer exclusion patterns.
    """
    print("\n" + "="*80)
    print("SCENARIO 4: Custom Exclusion Patterns")
    print("="*80)
    
    code = """
from fedavg_aggregator_backbone_only import FedAvgAggregatorBackboneOnly

# Exclude multiple layer types
aggregator = FedAvgAggregatorBackboneOnly(
    weighting_strategy='data_driven',
    exclude_layers=[
        'classifier',      # Main classifier
        'head',            # Alternative head name
        'fc_final',        # Final layer
        'output'           # Output layer
    ]
)

aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
aggregator.aggregate()

# This will skip ANY layer containing these patterns:
# Skipped examples:
#   - classifier.weight
#   - classifier.bias
#   - head.0.weight
#   - fc_final.bias
#   - output.weight
#
# Aggregated examples:
#   - backbone.0.weight
#   - backbone.1.weight
#   - feature_extractor.layer1.weight
#   - encoder.weight
"""
    
    print(code)


# ============================================================================
# Element-Wise Aggregation Explained
# ============================================================================

def explain_element_wise_aggregation():
    """Explain how element-wise aggregation works."""
    print("\n" + "="*80)
    print("EXPLANATION: Element-Wise Aggregation")
    print("="*80)
    
    explanation = """
CONCEPT:
────────

Element-wise aggregation means: for EACH individual element in EACH tensor,
compute the weighted average of that element across all models.

EXAMPLE:
────────

Say we have one parameter (e.g., backbone.0.weight) from 3 models:

Model 1:
  [[0.1, 0.2],
   [0.3, 0.4]]

Model 2:
  [[0.5, 0.6],
   [0.7, 0.8]]

Model 3:
  [[0.9, 1.0],
   [1.1, 1.2]]

Aggregation weights (data-driven): [0.303, 0.455, 0.242]

ELEMENT-WISE AGGREGATION:
──────────────────────────

For each element [i,j]:
  result[i,j] = λ₁*model₁[i,j] + λ₂*model₂[i,j] + λ₃*model₃[i,j]

Position [0,0]:
  result[0,0] = 0.303*0.1 + 0.455*0.5 + 0.242*0.9
              = 0.0303 + 0.2275 + 0.2178
              = 0.4756

Position [0,1]:
  result[0,1] = 0.303*0.2 + 0.455*0.6 + 0.242*1.0
              = 0.0606 + 0.2730 + 0.2420
              = 0.5756

Position [1,0]:
  result[1,0] = 0.303*0.3 + 0.455*0.7 + 0.242*1.1
              = 0.0909 + 0.3185 + 0.2662
              = 0.6756

Position [1,1]:
  result[1,1] = 0.303*0.4 + 0.455*0.8 + 0.242*1.2
              = 0.1212 + 0.3640 + 0.2904
              = 0.7756

RESULT:
  [[0.4756, 0.5756],
   [0.6756, 0.7756]]

VECTORIZED (what the code does):
──────────────────────────────────

Stack tensors:
  stacked = torch.stack([model1, model2, model3])  # Shape: [3, 2, 2]

Reshape weights for broadcasting:
  agg_w = torch.tensor([0.303, 0.455, 0.242]).view(3, 1, 1)

Element-wise multiplication and sum:
  result = (stacked * agg_w).sum(dim=0)
  
This computes ALL elements at once:
  result = stacked[0] * 0.303 + stacked[1] * 0.455 + stacked[2] * 0.242
         = model1 * 0.303 + model2 * 0.455 + model3 * 0.242

SAME RESULT:
  [[0.4756, 0.5756],
   [0.6756, 0.7756]]

KEY INSIGHT:
────────────

For ANY tensor shape, the formula is:
  aggregated[i,j,k,...] = Σ λₙ * model_n[i,j,k,...]

Where:
  - aggregated: result tensor (same shape as inputs)
  - λₙ: normalized weight for model n (sum of all λ = 1.0)
  - model_n[i,j,k,...]: specific element in model n
  - Σ: sum over all models
"""
    
    print(explanation)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FedAvg Backbone-Only Aggregation: Practical Examples")
    print("Non-Overlapping Classes Scenario")
    print("="*80)
    
    # Run scenarios
    scenario_1_aggregator_class()
    scenario_2_direct_element_wise()
    scenario_3_complete_workflow()
    scenario_4_custom_exclusion()
    explain_element_wise_aggregation()
    
    print("\n" + "="*80)
    print("Summary: Which Approach to Use?")
    print("="*80)
    print("""
    SCENARIO 1 (Aggregator Class):
    ✓ Cleaner API
    ✓ Built-in error checking
    ✓ Recommended for production
    ✓ Better reporting
    
    SCENARIO 2 (Direct Function):
    ✓ More direct control
    ✓ Matches your code style
    ✓ Good for simple cases
    ✓ Fewer abstractions
    
    RECOMMENDATION:
    Use Scenario 1 for most cases. Clear, tested, production-ready.
    Use Scenario 2 if you prefer maximum control.
    
    KEY POINTS:
    ✓ Element-wise: Each element averaged independently
    ✓ Backbone only: Skip classifier (different classes)
    ✓ Data-driven: Weight by training data size
    ✓ Result: Aggregated feature extractor for transfer learning
    """)
