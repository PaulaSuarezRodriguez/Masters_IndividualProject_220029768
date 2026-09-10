"""
Practical Examples: FedAvg Aggregation with Expanding Classifiers

For continual learning scenarios where models have incrementally growing output layers.
"""

import torch
import torch.nn as nn
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier


# ============================================================================
# Example Model: Growing Classifier
# ============================================================================

class ContinualLearningModel(nn.Module):
    """Model for continual learning with expanding classifier."""
    
    def __init__(self, input_size=28*28, hidden_size=256, num_classes=10):
        super().__init__()
        # Shared backbone
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Expanding classifier (can grow in output dimension)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


# ============================================================================
# Scenario 1: Explicit Expanding Layer Specification
# ============================================================================

def scenario_1_explicit_expanding_layers():
    """
    Explicit approach: Manually specify which layers have expanding dimensions.
    
    Recommended when you know exactly which layers expand (e.g., classifier).
    """
    print("\n" + "="*80)
    print("SCENARIO 1: Explicit Expanding Layer Specification")
    print("="*80)
    print("""
    Use case: You trained models with expanding classifiers.
    - Task 1: 10 classes
    - Task 2: 10 + 5 = 15 classes
    - Task 3: 10 + 5 + 7 = 22 classes
    
    Only the classifier weights/bias expand.
    Backbone (feature_extractor) stays the same.
    
    We want to:
    1. Aggregate only the shared 10 classes
    2. Preserve task-specific new classes
    """)
    
    # Create and save example models with different number of classes
    print("\nCreating models with expanding classifiers...")
    
    model_configs = [
        ('model_task1_lwf.pth', 10, 1000),   # Task 1: 10 classes, 1000 samples
        ('model_task2_lwf.pth', 15, 1200),   # Task 2: 15 classes, 1200 samples
        ('model_task3_lwf.pth', 22, 900),    # Task 3: 22 classes, 900 samples
    ]
    
    model_paths = []
    dataset_sizes = []
    
    for path, num_classes, dataset_size in model_configs:
        model = ContinualLearningModel(num_classes=num_classes)
        torch.save(model.state_dict(), path)
        model_paths.append(path)
        dataset_sizes.append(dataset_size)
        print(f"  ✓ {path} ({num_classes} classes, {dataset_size} samples)")
    
    print("\nInitializing aggregator with explicit expanding layers...")
    aggregator = FedAvgAggregatorExpandingClassifier(
        weighting_strategy='data_driven',
        expanding_layers=['classifier.weight', 'classifier.bias']  # ← Specify which layers expand
    )
    
    print("Loading model weights...")
    aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
    
    print("\nAggregation details:")
    aggregator.print_aggregation_details()
    
    print("\nPerforming aggregation (shared portion only)...")
    aggregated_weights = aggregator.aggregate(expanding_layers=['classifier.weight', 'classifier.bias'])
    
    print(f"  ✓ Aggregated weights keys: {list(aggregated_weights.keys())}")
    print(f"  ✓ Aggregated classifier.weight shape: {aggregated_weights['classifier.weight'].shape}")
    print(f"    → This is the shared portion (10 classes)")
    
    print("\nSaving aggregated model...")
    aggregator.save_aggregated_weights('aggregated_explicit.pth')
    
    # Load into target model
    print("\nLoading aggregated weights into target model...")
    target_model = ContinualLearningModel(num_classes=22)  # Max classes from all tasks
    target_model = aggregator.aggregate_into_model(target_model, expanding_layers=['classifier.weight', 'classifier.bias'])
    
    print("  ✓ Aggregated weights loaded")
    print(f"  ✓ Target model classifier shape: {target_model.classifier.weight.shape}")
    print("    → Shared 10 classes: aggregated from all models")
    print("    → Classes 11-22: from original task-specific models")
    
    # Cleanup
    for path, _, _ in model_configs:
        import os
        os.remove(path)
    
    return aggregator, target_model


# ============================================================================
# Scenario 2: Auto-Detect Expanding Layers
# ============================================================================

def scenario_2_auto_detect_expanding_layers():
    """
    Auto-detect approach: Let the aggregator automatically find expanding layers.
    
    More flexible when you're not sure which layers expand.
    """
    print("\n" + "="*80)
    print("SCENARIO 2: Auto-Detect Expanding Layers")
    print("="*80)
    print("""
    Use case: Automatically detect which layers have shape mismatches.
    
    The aggregator will:
    1. Load all models
    2. Compare their shapes
    3. Find layers with varying dimensions
    4. Handle them appropriately
    """)
    
    # Create models
    print("\nCreating models with different classifiers...")
    
    model_configs = [
        ('model_task1.pth', 10, 1000),
        ('model_task2.pth', 15, 1200),
        ('model_task3.pth', 22, 900),
    ]
    
    model_paths = []
    dataset_sizes = []
    
    for path, num_classes, dataset_size in model_configs:
        model = ContinualLearningModel(num_classes=num_classes)
        torch.save(model.state_dict(), path)
        model_paths.append(path)
        dataset_sizes.append(dataset_size)
        print(f"  ✓ {path} ({num_classes} classes)")
    
    print("\nInitializing aggregator WITHOUT specifying expanding layers...")
    aggregator = FedAvgAggregatorExpandingClassifier(weighting_strategy='data_driven')
    
    print("Loading model weights...")
    aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
    
    print("\nAuto-detecting expanding layers...")
    expanding = aggregator.detect_expanding_layers()
    print(f"  ✓ Found expanding layers: {expanding}")
    
    print("\nAggregation details:")
    aggregator.print_aggregation_details()
    
    print("\nPerforming aggregation (auto-detected)...")
    aggregator.save_aggregated_weights('aggregated_autodetect.pth')
    
    print("  ✓ Aggregation complete!")
    
    # Cleanup
    for path, _, _ in model_configs:
        import os
        os.remove(path)
    
    return aggregator


# ============================================================================
# Scenario 3: Your Custom Code Integration
# ============================================================================

def scenario_3_custom_element_wise():
    """
    Directly using element-wise aggregation like your code snippet.
    
    This shows how to aggregate at the individual tensor element level.
    """
    print("\n" + "="*80)
    print("SCENARIO 3: Custom Element-Wise Aggregation")
    print("="*80)
    print("""
    Direct element-wise aggregation approach.
    
    Similar to your code snippet:
    - Load state dicts
    - Group weights by layer name
    - Aggregate element-by-element
    - Handle shape mismatches manually
    """)
    
    # Simulate your setup
    print("\nLoading weight objects...")
    
    weights_task1 = torch.load('model_task1.pth', map_location='cpu') if False else ContinualLearningModel(10).state_dict()
    weights_task2 = torch.load('model_task2.pth', map_location='cpu') if False else ContinualLearningModel(15).state_dict()
    weights_task3 = torch.load('model_task3.pth', map_location='cpu') if False else ContinualLearningModel(22).state_dict()
    
    # Your code snippet adapted:
    list_load = [weights_task1, weights_task2, weights_task3]
    
    # Group weights by layer name
    weights_by_layer = {}
    for key in list_load[0]:
        if "weight" in key:
            weights_by_layer[key] = []
            
            for state_dict in list_load:
                if key in state_dict:
                    weights_by_layer[key].append(state_dict[key])
    
    print(f"\nLayers found: {list(weights_by_layer.keys())}")
    
    # Now aggregate element-wise with smart handling for expanding dimensions
    print("\nElement-wise aggregation with shape handling...")
    
    aggregated_weights = {}
    agg_weights = np.array([0.303, 0.455, 0.242])  # Data-driven weights
    
    for layer_name, tensors in weights_by_layer.items():
        print(f"\n  Processing: {layer_name}")
        
        # Check if this is an expanding layer
        shapes = [t.shape for t in tensors]
        is_expanding = len(set(shapes)) > 1
        
        if is_expanding:
            # Expanding layer: aggregate only common portion
            print(f"    Shapes: {shapes} (expanding)")
            
            # Find common shape (minimum)
            common_shape = tuple(min(s[i] for s in shapes) for i in range(len(shapes[0])))
            print(f"    Common shape: {common_shape}")
            
            # Extract and aggregate shared portion
            shared_tensors = [t[tuple(slice(0, size) for size in common_shape)] for t in tensors]
            
            # Element-wise weighted sum
            stacked = torch.stack(shared_tensors)
            agg_weights_t = torch.tensor(agg_weights).view(-1, *([1] * (stacked.dim() - 1)))
            aggregated = (stacked * agg_weights_t).sum(dim=0)
            
            aggregated_weights[layer_name] = aggregated
            print(f"    Aggregated shape: {aggregated.shape} (shared only)")
        
        else:
            # Fixed layer: full aggregation
            print(f"    Shape: {shapes[0]} (fixed)")
            
            # Element-wise weighted sum
            stacked = torch.stack(tensors)
            agg_weights_t = torch.tensor(agg_weights).view(-1, *([1] * (stacked.dim() - 1)))
            aggregated = (stacked * agg_weights_t).sum(dim=0)
            
            aggregated_weights[layer_name] = aggregated
            print(f"    Aggregated shape: {aggregated.shape}")
    
    print(f"\n✓ Element-wise aggregation complete!")
    print(f"  Total layers aggregated: {len(aggregated_weights)}")


# ============================================================================
# Scenario 4: Complete Workflow
# ============================================================================

def scenario_4_complete_workflow():
    """
    Complete workflow: Train, aggregate, evaluate.
    """
    print("\n" + "="*80)
    print("SCENARIO 4: Complete Workflow")
    print("="*80)
    
    code = """
import torch
import torch.nn as nn
from fedavg_aggregator_expanded_classifier import FedAvgAggregatorExpandingClassifier

# ============================================================================
# Step 1: Training Phase (Your LwF Training)
# ============================================================================

class ContinualLearningModel(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        return self.classifier(self.feature_extractor(x))

# Train models for each task
num_classes_per_task = [10, 5, 7]  # New classes per task
cumulative_classes = 0
models_info = []

for task_id, new_classes in enumerate(num_classes_per_task):
    cumulative_classes += new_classes
    
    # Initialize model with current total classes
    model = ContinualLearningModel(num_classes=cumulative_classes)
    
    # Train with LwF (your training code)
    # model = lwf_train(model, task_data, old_model)  # Your training
    
    # Save model
    path = f'models/task_{task_id+1}_lwf.pth'
    torch.save(model.state_dict(), path)
    models_info.append({
        'path': path,
        'num_classes': cumulative_classes,
        'dataset_size': len(task_data)
    })

# ============================================================================
# Step 2: Aggregation Phase
# ============================================================================

# Create aggregator
aggregator = FedAvgAggregatorExpandingClassifier(
    weighting_strategy='data_driven',
    expanding_layers=['classifier.weight', 'classifier.bias']
)

# Load all model weights
model_paths = [info['path'] for info in models_info]
dataset_sizes = [info['dataset_size'] for info in models_info]

aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# View aggregation details
aggregator.print_aggregation_details()

# Aggregate (only shared 10 classes)
aggregator.save_aggregated_weights('models/aggregated_fedavg.pth')

# ============================================================================
# Step 3: Load Aggregated Model
# ============================================================================

# Create target model with max classes
max_classes = cumulative_classes
aggregated_model = ContinualLearningModel(num_classes=max_classes)

# Load aggregated weights
aggregated_model = aggregator.aggregate_into_model(aggregated_model)

# ============================================================================
# Step 4: Evaluation
# ============================================================================

aggregated_model.eval()
with torch.no_grad():
    # Test on combined test data
    predictions = aggregated_model(test_data)
    
    # Evaluate
    accuracy = evaluate(predictions, test_labels)
    print(f"Aggregated model accuracy: {accuracy:.2%}")

# Compare with individual models
for task_id, info in enumerate(models_info):
    model = ContinualLearningModel(num_classes=info['num_classes'])
    model.load_state_dict(torch.load(info['path']))
    model.eval()
    
    with torch.no_grad():
        acc = evaluate(model(test_data), test_labels)
    print(f"Task {task_id+1} model accuracy: {acc:.2%}")
"""
    
    print(code)


# ============================================================================
# Comparison: Element-wise vs Full Layer Aggregation
# ============================================================================

def show_comparison():
    """Show the difference between aggregation approaches."""
    print("\n" + "="*80)
    print("COMPARISON: Element-wise vs Full Layer Aggregation")
    print("="*80)
    
    comparison = """
FULL LAYER AGGREGATION (Original FedAvg):
──────────────────────────────────────────
Problem with expanding classifiers:
    Model 1 classifier.weight: [10, 256]
    Model 2 classifier.weight: [15, 256]  ← Different size!
    Model 3 classifier.weight: [22, 256]
    
    Cannot stack these directly (different shapes)
    ERROR: RuntimeError: stack expects equal size

Solution: ELEMENT-WISE AGGREGATION WITH SHAPE HANDLING:
─────────────────────────────────────────────────────────
✓ Extract shared portion from each model
    Model 1: weights[:10, :] = [10, 256]
    Model 2: weights[:10, :] = [10, 256]  ← Extract first 10 classes
    Model 3: weights[:10, :] = [10, 256]
    
✓ Aggregate only the shared portion
    aggregated = λ₁*m1[:10] + λ₂*m2[:10] + λ₃*m3[:10]
    Result shape: [10, 256]

✓ Handle task-specific weights separately
    Model 1: classes 11-15 from Model 2
    Model 2: classes 16-22 from Model 3
    (Or aggregate task-specific weights separately)

KEY BENEFITS:
✓ Works with expanding classifiers
✓ Aggregates shared knowledge (base classes)
✓ Preserves task-specific knowledge
✓ Element-wise control: process at tensor element level
"""
    
    print(comparison)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("\n" + "="*80)
    print("FedAvg Aggregation Examples: Expanding Classifiers")
    print("="*80)
    
    # Run scenarios
    scenario_1_explicit_expanding_layers()
    scenario_2_auto_detect_expanding_layers()
    scenario_3_custom_element_wise()
    scenario_4_complete_workflow()
    show_comparison()
    
    print("\n" + "="*80)
    print("Summary: Which Approach to Use?")
    print("="*80)
    print("""
    SCENARIO 1 (Explicit):
    ✓ Use when you know exactly which layers expand
    ✓ Clearer code and control
    ✓ Best for production
    
    SCENARIO 2 (Auto-detect):
    ✓ Use when unsure which layers expand
    ✓ More flexible
    ✓ Good for exploration
    
    SCENARIO 3 (Custom Element-wise):
    ✓ Use for custom control
    ✓ Like your original code
    ✓ Most flexible but more manual
    
    RECOMMENDED: Use Scenario 1 or 2 with the aggregator class
    - Cleaner code
    - Better error handling
    - Proven implementation
    """)
