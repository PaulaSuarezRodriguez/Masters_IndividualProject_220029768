"""
Practical Example: FedAvg Aggregation for LwF-Trained Models

This example demonstrates how to:
1. Save model weights from LwF training
2. Aggregate them using FedAvg
3. Load and evaluate the aggregated model
"""

import torch
import torch.nn as nn
from fedavg_aggregator import FedAvgAggregator
from pathlib import Path


# ============================================================================
# Example Model Architecture (replace with your actual model)
# ============================================================================

class SimpleFeatureExtractor(nn.Module):
    """Example model - replace with your actual architecture."""
    
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
        features = self.feature_extractor(x)
        return self.classifier(features)


# ============================================================================
# Scenario 1: Uniform Aggregation (Equal weights)
# ============================================================================

def scenario_uniform_aggregation():
    """
    Scenario: Multiple models trained on different tasks with LwF.
    All models should contribute equally to the final aggregated model.
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Uniform FedAvg Aggregation")
    print("="*70)
    print("""
    Use case: You have trained 3 models using Learning without Forgetting
    on different domains/tasks. Each model is equally important.
    
    We aggregate using: w_final = (1/3)*w_1 + (1/3)*w_2 + (1/3)*w_3
    """)
    
    # Initialize aggregator with uniform weighting
    aggregator = FedAvgAggregator(weighting_strategy='uniform')
    
    # Create and save example models (simulating your trained models)
    model_paths = []
    for i in range(1, 4):
        model = SimpleFeatureExtractor()
        # In practice: model.load_state_dict(torch.load(f'task{i}_checkpoint.pth'))
        # Here we just save the initialized model as example
        path = f'model_task{i}_lwf.pth'
        torch.save(model.state_dict(), path)
        model_paths.append(path)
    
    # Load models into aggregator
    print("\nLoading model weights...")
    aggregator.load_model_weights_from_files(model_paths)
    
    # Get aggregation report
    report = aggregator.get_aggregation_report()
    print(f"✓ Loaded {report['num_models']} models")
    print(f"  Aggregation strategy: {report['weighting_strategy']}")
    print(f"  Each model weight: {report['model_weights'][0]:.4f}")
    
    # Aggregate and load into model
    print("\nPerforming FedAvg aggregation...")
    aggregated_model = SimpleFeatureExtractor()
    aggregated_model = aggregator.aggregate_into_model(aggregated_model)
    print("✓ Weights aggregated successfully")
    
    # Save aggregated model
    save_path = 'aggregated_model_uniform.pth'
    aggregator.save_aggregated_weights(save_path)
    print(f"✓ Aggregated model saved to {save_path}")
    
    # Test inference
    print("\nTesting inference on aggregated model...")
    aggregated_model.eval()
    test_input = torch.randn(5, 28*28)
    with torch.no_grad():
        output = aggregated_model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print("✓ Inference successful")
    
    # Cleanup
    for path in model_paths:
        Path(path).unlink(missing_ok=True)
    
    return aggregated_model


# ============================================================================
# Scenario 2: Data-Driven Aggregation (Size-weighted)
# ============================================================================

def scenario_data_driven_aggregation():
    """
    Scenario: Multiple models trained on datasets of different sizes.
    Models trained on larger datasets should contribute more to the aggregation.
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Data-Driven FedAvg Aggregation (Size-Weighted)")
    print("="*70)
    print("""
    Use case: You have trained 3 models using LwF on datasets of different sizes:
    - Model 1: 1000 training samples
    - Model 2: 1500 training samples (more data, higher quality)
    - Model 3: 800 training samples
    
    Weights are proportional to dataset size (n_i / n_total).
    More data = higher contribution to final model.
    """)
    
    # Initialize aggregator with data-driven weighting
    aggregator = FedAvgAggregator(weighting_strategy='data_driven')
    
    # Model training information
    model_configs = [
        ('model_dataset1_lwf.pth', 1000),
        ('model_dataset2_lwf.pth', 1500),
        ('model_dataset3_lwf.pth', 800),
    ]
    
    # Create and save example models
    print("\nCreating and saving trained models...")
    for path, size in model_configs:
        model = SimpleFeatureExtractor()
        torch.save(model.state_dict(), path)
        print(f"  ✓ {path} (trained on {size} samples)")
    
    # Load into aggregator
    model_paths = [config[0] for config in model_configs]
    dataset_sizes = [config[1] for config in model_configs]
    
    aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
    
    # Get aggregation report
    report = aggregator.get_aggregation_report()
    print(f"\nAggregation Details:")
    print(f"  Number of models: {report['num_models']}")
    print(f"  Total samples: {report['total_samples']}")
    
    for i, (size, weight) in enumerate(zip(report['per_model_samples'], 
                                            report['model_weights'])):
        print(f"  Model {i+1}: {size:4d} samples → weight = {weight:.4f}")
    
    # Aggregate
    print("\nPerforming FedAvg aggregation...")
    aggregated_model = SimpleFeatureExtractor()
    aggregated_model = aggregator.aggregate_into_model(aggregated_model)
    print("✓ Weights aggregated successfully (size-weighted)")
    
    # Save aggregated model
    save_path = 'aggregated_model_datadriven.pth'
    aggregator.save_aggregated_weights(save_path)
    print(f"✓ Aggregated model saved to {save_path}")
    
    # Cleanup
    for path, _ in model_configs:
        Path(path).unlink(missing_ok=True)
    
    return aggregated_model


# ============================================================================
# Scenario 3: Custom Ensemble (Different domains)
# ============================================================================

def scenario_domain_ensemble():
    """
    Scenario: Models trained with LwF on different domains/applications.
    Create a universal model by aggregating domain-specific models.
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Domain-Specific Model Ensemble")
    print("="*70)
    print("""
    Use case: You have trained separate models for different applications:
    - Model A: Trained on Medical Imaging domain
    - Model B: Trained on Natural Scene domain
    - Model C: Trained on Document Analysis domain
    
    Goal: Create a universal feature extractor that works across domains.
    """)
    
    domains = [
        ("Medical_Imaging", 1100),
        ("Natural_Scenes", 1400),
        ("Documents", 950),
    ]
    
    aggregator = FedAvgAggregator(weighting_strategy='data_driven')
    
    print("\nDomain-specific models:")
    model_paths = []
    for domain, samples in domains:
        model = SimpleFeatureExtractor()
        path = f'model_{domain}_lwf.pth'
        torch.save(model.state_dict(), path)
        model_paths.append(path)
        print(f"  ✓ {domain:20s} (n={samples:4d} samples)")
    
    # Aggregate
    dataset_sizes = [samples for _, samples in domains]
    aggregator.load_model_weights_from_files(model_paths, dataset_sizes)
    
    report = aggregator.get_aggregation_report()
    print(f"\nAggregation Report:")
    print(f"  Total models: {report['num_models']}")
    print(f"  Total training data: {report['total_samples']} samples")
    print(f"  Contribution weights: {[f'{w:.2%}' for w in report['model_weights']]}")
    
    # Create universal model
    universal_model = SimpleFeatureExtractor()
    universal_model = aggregator.aggregate_into_model(universal_model)
    
    save_path = 'universal_model_across_domains.pth'
    aggregator.save_aggregated_weights(save_path)
    print(f"\n✓ Universal model saved to {save_path}")
    print("  This model combines features from all 3 domains!")
    
    # Cleanup
    for path in model_paths:
        Path(path).unlink(missing_ok=True)
    
    return universal_model


# ============================================================================
# Helper Function: Load Aggregated Model Later
# ============================================================================

def load_aggregated_model(model_class, checkpoint_path):
    """
    Load an aggregated model from checkpoint.
    
    Args:
        model_class: Your model class (e.g., SimpleFeatureExtractor)
        checkpoint_path (str): Path to saved aggregated weights
    
    Returns:
        Model with aggregated weights loaded
    """
    model = model_class()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    return model


# ============================================================================
# Integration with Your LwF Training Pipeline
# ============================================================================

def integration_example():
    """
    Example: How to integrate FedAvg into your existing LwF training workflow.
    """
    print("\n" + "="*70)
    print("INTEGRATION: How to use FedAvg in your LwF pipeline")
    print("="*70)
    
    code = """
# In your LwF training script:

# ... your LwF training loop ...
for task_id, task_data in enumerate(task_list):
    model = initialize_model()
    model = lwf_train(model, task_data)  # Your LwF training function
    
    # Save the model weights after each task
    torch.save(model.state_dict(), f'checkpoints/task_{task_id}_lwf.pth')
    
    print(f"✓ Task {task_id} trained and saved")

# After all tasks are trained, aggregate using FedAvg:

from fedavg_aggregator import FedAvgAggregator

# Collect all trained model paths
model_paths = [f'checkpoints/task_{i}_lwf.pth' for i in range(num_tasks)]
dataset_sizes = [len(task_data) for task_data in task_list]  # Optional

# Initialize aggregator
aggregator = FedAvgAggregator(weighting_strategy='data_driven')
aggregator.load_model_weights_from_files(model_paths, dataset_sizes)

# Get aggregation report
report = aggregator.get_aggregation_report()
print(f"Aggregating {report['num_models']} models...")
print(f"Model contributions: {report['model_weights']}")

# Create and load aggregated model
aggregated_model = YourModel()
aggregated_model = aggregator.aggregate_into_model(aggregated_model)

# Save final aggregated model
aggregator.save_aggregated_weights('final_aggregated_model.pth')

# Use aggregated model for testing
aggregated_model.eval()
test_accuracy = evaluate(aggregated_model, test_data)
print(f"Aggregated model test accuracy: {test_accuracy:.2%}")
"""
    
    print(code)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FedAvg Aggregation Examples for LwF-Trained Models")
    print("="*70)
    
    # Run scenarios
    model1 = scenario_uniform_aggregation()
    model2 = scenario_data_driven_aggregation()
    model3 = scenario_domain_ensemble()
    
    # Show integration
    integration_example()
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
    Key Takeaways:
    
    1. Uniform Aggregation:
       - Use when models are equally important
       - Each model contributes 1/n to final weights
       - Simplest approach
    
    2. Data-Driven Aggregation:
       - Use when models trained on different dataset sizes
       - Larger datasets → higher contribution
       - Prevents bias toward small-dataset models
    
    3. Benefits of FedAvg for LwF:
       - Combines knowledge from multiple tasks without forgetting
       - No server/client infrastructure needed (local aggregation)
       - Produces a single universal model
       - Easy to implement and understand
    
    4. Workflow:
       1. Train multiple models with LwF (save weights)
       2. Create FedAvgAggregator
       3. Load model weights
       4. Aggregate weights
       5. Load into model and use for inference
    """)
