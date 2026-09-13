"""
Complete example: Aggregating 4 non-overlapping specialized models into 1 global model

Models:
  1. Outdoors model:       classes {12, 17, 23, 33, 37, 49}
  2. Nature model:         classes {0, 47, 51, 52, 53, 54, 56, 57, 59}
  3. People model:         classes {2, 11, 35}
  4. Objects/Vehicles:     classes {5, 8, 9, 10, 13, 16, 20, 22, 25, 28, 39, 40, 41, 48, 58, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94}

Total: 43 trained classes + 57 untrained classes = 100 CIFAR-100 classes
"""

import torch
import torch.nn as nn
from pathlib import Path
import pickle
from typing import Dict, Tuple


# ==============================================================================
# STEP 1: Define the remapping function
# ==============================================================================

def remap_classifier_to_cifar100(
    model: nn.Module, 
    new_to_original: Dict[int, int],
    num_classes: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remap a specialized model's classifier from local space to CIFAR-100 space.
    
    Args:
        model: Trained model with classifier [num_subset_classes, feature_dim]
        new_to_original: Dict {local_idx → cifar_class}
                        Example: {0→12, 1→17, 2→23, ...}
        num_classes: Always 100 for CIFAR-100
    
    Returns:
        remapped_weights: [100, feature_dim], trained weights at correct positions
        remapped_bias: [100], trained bias at correct positions
    """
    classifier = model.linear
    feature_dim = classifier.weight.shape[1]
    
    # Initialize with zeros (untrained classes)
    remapped_weights = torch.zeros(num_classes, feature_dim)
    remapped_bias = torch.zeros(num_classes)
    
    # Place each trained class at its CIFAR-100 position
    for local_idx, cifar_class in new_to_original.items():
        remapped_weights[cifar_class] = classifier.weight[local_idx]
        remapped_bias[cifar_class] = classifier.bias[local_idx]
    
    return remapped_weights, remapped_bias


# ==============================================================================
# STEP 2: Load models and mappings
# ==============================================================================

def load_specialized_models_and_mappings():
    """
    Load all 4 trained specialized models and their label mappings.
    
    This should load from wherever you saved them during training.
    """
    
    # Define where models are saved
    MODEL_DIR = Path('./final_specialised_models')
    
    # Load outdoors model
    outdoors_model = torch.load(MODEL_DIR / 'final_outdoors_model.pt')
    outdoors_mapping = {0: 12, 1: 17, 2: 23, 3: 33, 4: 37, 5: 49}
    
    # Load nature model
    nature_model = torch.load(MODEL_DIR / 'final_nature_model.pt')
    nature_mapping = {0: 0, 1: 47, 2: 51, 3: 52, 4: 53, 5: 54, 6: 56, 7: 57, 8: 59}
    
    # Load people model
    people_model = torch.load(MODEL_DIR / 'final_people_model.pt')
    people_mapping = {0: 2, 1: 11, 2: 35}
    
    # Load objects/vehicles model
    objects_vehicles_model = torch.load(MODEL_DIR / 'final_objects_vehicles_model.pt')
    objects_vehicles_mapping = {
        0: 5, 1: 8, 2: 9, 3: 10, 4: 13, 5: 16, 6: 20, 7: 22, 8: 25, 9: 28,
        10: 39, 11: 40, 12: 41, 13: 48, 14: 58, 15: 61, 16: 69, 17: 81,
        18: 84, 19: 85, 20: 86, 21: 87, 22: 89, 23: 90, 24: 94
    }
    
    return {
        'outdoors': (outdoors_model, outdoors_mapping),
        'nature': (nature_model, nature_mapping),
        'people': (people_model, people_mapping),
        'objects_vehicles': (objects_vehicles_model, objects_vehicles_mapping)
    }


# ==============================================================================
# STEP 3: Aggregate classifiers
# ==============================================================================

def aggregate_specialized_models():
    """
    Main aggregation function.
    
    1. Load all 4 models and their mappings
    2. Remap each classifier to CIFAR-100 space
    3. Combine into single 100-class classifier
    4. Create global model
    """
    
    print("\n" + "="*80)
    print("AGGREGATING 4 SPECIALIZED MODELS")
    print("="*80 + "\n")
    
    # Load models
    print("Loading specialized models...")
    models_dict = load_specialized_models_and_mappings()
    
    # Get feature dimension from first model
    first_model, _ = list(models_dict.values())[0]
    feature_dim = first_model.linear.weight.shape[1]
    
    print(f"✓ Loaded 4 models")
    print(f"  Feature dimension: {feature_dim}\n")
    
    # Initialize global classifier
    global_weights = torch.zeros(100, feature_dim)
    global_bias = torch.zeros(100)
    
    trained_classes = set()
    
    # Remap and aggregate
    print("Remapping classifiers to CIFAR-100 space...")
    for model_name, (model, mapping) in models_dict.items():
        print(f"\n  {model_name}:")
        print(f"    - Subset size: {len(mapping)} classes")
        print(f"    - CIFAR classes: {sorted(mapping.values())}")
        
        # Remap this model's classifier
        w, b = remap_classifier_to_cifar100(model, mapping, num_classes=100)
        
        # Add to global classifier (no averaging needed - classes don't overlap)
        global_weights += w
        global_bias += b
        
        # Track trained classes
        trained_classes.update(mapping.values())
        
        print(f"    ✓ Remapped")
    
    # Summary
    untrained_classes = set(range(100)) - trained_classes
    print(f"\n{'='*60}")
    print("AGGREGATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total trained classes: {len(trained_classes)}/100")
    print(f"  - Outdoors: 6 classes")
    print(f"  - Nature: 9 classes")
    print(f"  - People: 3 classes")
    print(f"  - Objects/Vehicles: 25 classes")
    print(f"\nUntrained classes: {len(untrained_classes)}/100")
    print(f"  (Will have zero-initialized weights)")
    
    if len(untrained_classes) > 0 and len(untrained_classes) <= 20:
        print(f"  Classes: {sorted(untrained_classes)}")
    
    print(f"{'='*60}\n")
    
    return global_weights, global_bias


# ==============================================================================
# STEP 4: Create and save global model
# ==============================================================================

def create_global_model(global_weights: torch.Tensor, global_bias: torch.Tensor):
    """
    Create the final 100-class global model.
    
    Args:
        global_weights: Aggregated weights [100, feature_dim]
        global_bias: Aggregated bias [100]
    
    Returns:
        Global model with 100-class classifier
    """
    
    print("Creating global model...")
    
    # Import ResNet20 (or whatever your architecture is)
    from torchvision.models import resnet18  # Or your custom ResNet20
    
    # Create base model
    global_model = resnet18(num_classes=100)
    
    # Replace classifier with aggregated weights
    global_model.linear.weight = nn.Parameter(global_weights)
    global_model.linear.bias = nn.Parameter(global_bias)
    
    print(f"✓ Created global model")
    print(f"  - Classifier shape: {global_model.linear.weight.shape}")
    print(f"  - Output: 100 classes\n")
    
    # Save
    OUTPUT_PATH = Path('./final_specialised_models/global_model_aggregated.pt')
    torch.save(global_model.state_dict(), OUTPUT_PATH)
    print(f"✓ Saved to {OUTPUT_PATH}\n")
    
    return global_model


# ==============================================================================
# STEP 5: Test on CIFAR-100
# ==============================================================================

def test_global_model(global_model: nn.Module):
    """
    Test the aggregated global model on CIFAR-100 test set.
    
    Uses ORIGINAL CIFAR-100 labels (0-99), not mapped labels.
    """
    
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    print("="*80)
    print("TESTING GLOBAL MODEL")
    print("="*80 + "\n")
    
    # Load CIFAR-100 test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])
    
    test_data = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    # Evaluate
    global_model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    print(f"Evaluating on {len(test_data)} test samples...\n")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Forward pass
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Accumulate metrics
            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            # Progress
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {total_samples}/{len(test_data)} samples...")
    
    # Compute final metrics
    avg_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss: {avg_loss:.6f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Correct: {total_correct}/{total_samples}")
    print(f"\nExpected accuracy: 43/100 = 43% if random guessing on trained classes")
    print(f"Actual accuracy: {accuracy:.2f}% (depends on model quality)")
    print(f"{'='*60}\n")
    
    return accuracy, avg_loss


# ==============================================================================
# MAIN: Run complete aggregation pipeline
# ==============================================================================

if __name__ == '__main__':
    
    # Step 1: Load models and aggregate
    global_weights, global_bias = aggregate_specialized_models()
    
    # Step 2: Create global model
    global_model = create_global_model(global_weights, global_bias)
    
    # Step 3: Test
    accuracy, loss = test_global_model(global_model)
    
    print("\n" + "="*80)
    print("AGGREGATION COMPLETE")
    print("="*80)
    print(f"\nYour global model is ready!")
    print(f"  - Handles all CIFAR-100 classes (0-99)")
    print(f"  - 43 classes trained by specialists")
    print(f"  - 57 classes with zero-initialized weights")
    print(f"\nNext steps:")
    print(f"  1. Use global_model for inference on CIFAR-100 images")
    print(f"  2. For untrained classes, consider:")
    print(f"     - Training additional models for remaining classes")
    print(f"     - Using transfer learning to initialize untrained classes")
    print(f"     - Using ensemble methods with specialized models")
    print("="*80 + "\n")
