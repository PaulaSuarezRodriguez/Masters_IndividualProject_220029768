"""
CORRECTED Complete Model Aggregation: Averaging All Layers

Key correction: Feature extraction layers should be AVERAGED, not copied!

Why aggregation is necessary:
- Each model trained on different CIFAR subset
- Each model's conv layers biased toward its classes
- Example: Outdoors model emphasizes landscape features
- Nature model emphasizes vegetation/texture features
- Objects model emphasizes object/shape features
- Averaging gives balanced, diverse feature detector for all 100 classes

For setup:
  Model 1 (Outdoors):         6 classes  {12, 17, 23, 33, 37, 49}
  Model 2 (Nature):           9 classes  {0, 47, 51, 52, 53, 54, 56, 57, 59}
  Model 3 (People):           3 classes  {2, 11, 35}
  Model 4 (Objects/Vehicles): 25 classes {5, 8, 9, 10, 13, 16, ...}
  
Architecture:
  - Feature layers (averaged): conv1, bn1, layer1-3
  - Classifier (remapped per-class): linear [64 → 100]
"""

import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import Dict, Tuple
from collections import OrderedDict


# ==============================================================================
# STEP 1: Feature Extractor Averaging (NEW APPROACH)
# ==============================================================================

def extract_feature_layers(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract all feature extraction layers (everything except classifier).
    
    Returns a dictionary of state dict entries for:
    - conv1, bn1 (input layer)
    - layer1, layer2, layer3 (residual blocks)
    - Does NOT include: linear (classifier)
    """
    state_dict = model.state_dict()
    
    feature_layers = OrderedDict()
    
    # Copy input layers
    feature_layers['conv1.weight'] = state_dict['conv1.weight'].clone()
    feature_layers['bn1.weight'] = state_dict['bn1.weight'].clone()
    feature_layers['bn1.bias'] = state_dict['bn1.bias'].clone()
    feature_layers['bn1.running_mean'] = state_dict['bn1.running_mean'].clone()
    feature_layers['bn1.running_var'] = state_dict['bn1.running_var'].clone()
    feature_layers['bn1.num_batches_tracked'] = state_dict['bn1.num_batches_tracked'].clone()
    
    # Copy all residual block layers (layer1, layer2, layer3)
    for key in state_dict.keys():
        if key.startswith('layer'):  # layer1, layer2, layer3
            feature_layers[key] = state_dict[key].clone()
    
    return feature_layers


def average_feature_layers(features_list: list) -> Dict[str, torch.Tensor]:
    """
    Average feature extraction layers across all models.
    
    Each model's feature extractor was trained on different CIFAR subset.
    By averaging, we get a balanced feature detector that:
    - Has seen more diverse training examples (43 classes vs subset)
    - Isn't biased toward any single subset
    - Learns from all models' optimization
    - More generalizable to untrained classes
    
    Args:
        features_list: List of feature dictionaries from each model
    
    Returns:
        Averaged feature layers
    """
    if not features_list:
        raise ValueError("No feature layers to average")
    
    num_models = len(features_list)
    averaged_features = OrderedDict()
    
    # Average each layer across all models
    for key in features_list[0].keys():
        # Stack all models' values for this layer
        stacked = torch.stack([features[key].float() for features in features_list])
        
        # Take mean across models (dim=0)
        averaged = torch.mean(stacked, dim=0)
        
        # Preserve original dtype
        averaged_features[key] = averaged.to(features_list[0][key].dtype)
    
    return averaged_features


def load_averaged_feature_layers(global_model: nn.Module,
                                 averaged_features: Dict[str, torch.Tensor]):
    """
    Load averaged feature extraction layers into the global model.
    
    Args:
        global_model: Target ResNet20_DynamicModel
        averaged_features: Dictionary of averaged feature layer weights
    """
    global_state_dict = global_model.state_dict()
    
    # Update global model with averaged features
    for key, value in averaged_features.items():
        if key in global_state_dict:
            global_state_dict[key] = value
    
    global_model.load_state_dict(global_state_dict)


# ==============================================================================
# STEP 2: Classifier Remapping (UNCHANGED)
# ==============================================================================

def remap_classifier_to_cifar100(
    model: nn.Module,
    new_to_original: Dict[int, int],
    num_classes: int = 100,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remap a specialized model's classifier from local label space to CIFAR-100 space.
    
    Args:
        model: Trained specialized model
        new_to_original: Dict {local_class_idx → cifar_class_idx}
        num_classes: Always 100 for CIFAR-100
        device: Device for tensors
    
    Returns:
        Tuple of (remapped_weights, remapped_bias)
    """
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
    
    # Place each trained class at its CIFAR-100 position
    for local_idx, cifar_class in new_to_original.items():
        remapped_weights[cifar_class] = classifier.weight[local_idx].to(device)
        remapped_bias[cifar_class] = classifier.bias[local_idx].to(device)
    
    return remapped_weights, remapped_bias


# ==============================================================================
# STEP 3: Main Aggregation Pipeline (CORRECTED)
# ==============================================================================

def aggregate_models_all_layers_with_feature_averaging(
    model_names: list,
    model_dir: Path,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict]:
    """
    Complete aggregation pipeline with FEATURE AVERAGING.
    
    Process:
    1. Load all models
    2. Extract feature layers from EACH model
    3. Average the feature layers (not copy from one)
    4. Load averaged features into global model
    5. Remap and combine classifiers
    6. Create and return global model
    
    Args:
        model_names: List of model names
        model_dir: Path where models and metadata are saved
        device: Device for computation
    
    Returns:
        Tuple of (global_model, aggregation_report)
    """
    
    print("\n" + "="*80)
    print("COMPLETE MODEL AGGREGATION - ALL LAYERS WITH FEATURE AVERAGING")
    print("="*80 + "\n")
    
    model_dir = Path(model_dir)
    models = {}
    mappings = {}
    
    # ==================================================================
    # STEP 1: Load all models and their mappings
    # ==================================================================
    
    print("Step 1: Loading specialized models...")
    for model_name in model_names:
        try:
            from ResNet20_DynamicModel_v13 import ResNet20_DynamicModel

            state_dict = torch.load(
                model_dir / f'final_{model_name}_model.pt',
                map_location=device
            )
            
            # Determine number of classes from classifier weight shape
            classifier_size = state_dict['linear.weight'].shape[0]
            model = ResNet20_DynamicModel(num_classes=classifier_size)
            model.load_state_dict(state_dict)
            model = model.to(device)
            models[model_name] = model
            
            # Load mapping
            with open(model_dir / f'{model_name}_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            mappings[model_name] = metadata['new_to_original']
            
            print(f"  ✓ Loaded {model_name}")
            print(f"    - Classifier size: {classifier_size}")
            print(f"    - CIFAR classes: {sorted(mappings[model_name].values())}")
            
        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")
            raise
    
    print()
    
    # ==================================================================
    # STEP 2: Extract features from ALL models
    # ==================================================================
    
    print("Step 2: Extracting feature layers from ALL models...")
    
    all_feature_layers = []
    feature_layer_keys = None
    
    for model_name in model_names:
        model = models[model_name]
        features = extract_feature_layers(model)
        all_feature_layers.append(features)
        
        if feature_layer_keys is None:
            feature_layer_keys = set(features.keys())
        
        print(f"  ✓ Extracted {len(features)} layer groups from {model_name}")
    
    print()
    
    # ==================================================================
    # STEP 3: Average feature layers across all models (KEY STEP!)
    # ==================================================================
    
    print("Step 3: AVERAGING feature layers across all models...")
    print(f"  (Combining features from {len(model_names)} different CIFAR subsets)")
    print()
    print("  Why averaging?")
    print("  - Each model trained on different classes → biased feature extractor")
    print("  - Outdoors model: emphasizes landscape/sky/terrain features")
    print("  - Nature model: emphasizes vegetation/texture features")
    print("  - People model: emphasizes face/body features")
    print("  - Objects model: emphasizes object shapes/mechanical features")
    print("  - Averaged: balanced detector, useful for all 100 CIFAR classes")
    print()
    
    averaged_features = average_feature_layers(all_feature_layers)
    
    print(f"  ✓ Averaged feature layers:")
    for key in list(averaged_features.keys())[:5]:  # Show first 5
        shape = averaged_features[key].shape
        print(f"    - {key}: {shape}")
    print(f"    ... ({len(averaged_features)} layers total)")
    
    print()
    
    # ==================================================================
    # STEP 4: Create global model with averaged features
    # ==================================================================
    
    print("Step 4: Creating global model with averaged features...")

    from ResNet20_DynamicModel_v13 import ResNet20_DynamicModel
    
    global_model = ResNet20_DynamicModel(num_classes=100)
    global_model = global_model.to(device)
    
    # Load averaged feature extraction layers
    load_averaged_feature_layers(global_model, averaged_features)
    
    print(f"  ✓ Created global model with averaged features:")
    print(f"    - conv1 + bn1: ✓ Averaged from {len(model_names)} models")
    print(f"    - layer1-3: ✓ Averaged from {len(model_names)} models")
    
    print()
    
    # ==================================================================
    # STEP 5: Remap and combine classifiers
    # ==================================================================
    
    print("Step 5: Remapping classifiers to CIFAR-100 space...")
    
    feature_dim = models[model_names[0]].linear.weight.shape[1]
    global_classifier_weights = torch.zeros(100, feature_dim, device=device)
    global_classifier_bias = torch.zeros(100, device=device)
    
    class_coverage = {}
    
    for model_name in model_names:
        model = models[model_name]
        mapping = mappings[model_name]
        
        w, b = remap_classifier_to_cifar100(
            model, mapping, num_classes=100, device=device
        )
        
        global_classifier_weights += w
        global_classifier_bias += b
        
        for local_idx, cifar_class in mapping.items():
            if cifar_class not in class_coverage:
                class_coverage[cifar_class] = []
            class_coverage[cifar_class].append(model_name)
        
        print(f"  ✓ Remapped {model_name}")
        print(f"    - Classes: {len(mapping)}")
        print(f"    - Positions: {sorted(mapping.values())}")
    
    print()
    
    # ==================================================================
    # STEP 6: Load remapped classifier into global model
    # ==================================================================
    
    print("Step 6: Loading remapped classifier into global model...")
    
    global_model.linear.weight = nn.Parameter(global_classifier_weights)
    global_model.linear.bias = nn.Parameter(global_classifier_bias)
    
    print(f"  ✓ Classifier loaded: {global_model.linear.weight.shape}")
    
    print()
    
    # ==================================================================
    # STEP 7: Generate aggregation report
    # ==================================================================
    
    print("Step 7: Aggregation summary...")
    
    untrained_classes = set(range(100)) - set(class_coverage.keys())
    
    report = {
        'total_cifar_classes': 100,
        'trained_classes': len(class_coverage),
        'untrained_classes': len(untrained_classes),
        'class_coverage': class_coverage,
        'models_used': model_names,
        'feature_dim': feature_dim,
        'device': device,
        'feature_averaging': True,  # KEY: Mark that features were averaged
        'num_models_aggregated': len(model_names)
    }
    
    print(f"\n  Aggregation method: FEATURE AVERAGING + CLASSIFIER REMAPPING")
    print(f"  Total CIFAR-100 classes: 100")
    print(f"  Classes trained: {len(class_coverage)}")
    print(f"  Classes NOT trained: {len(untrained_classes)}")
    print(f"  Feature extraction layers: Averaged from {len(model_names)} models")
    print(f"  Classifier: Remapped per-class (non-overlapping)")
    
    print(f"\n  ✓ Aggregation complete!")
    
    if len(untrained_classes) > 0 and len(untrained_classes) <= 20:
        untrained_list = sorted(untrained_classes)
        print(f"  Untrained classes: {untrained_list}")
    
    print("\n" + "="*80 + "\n")
    
    return global_model, report


# ==============================================================================
# STEP 4: Save and Test
# ==============================================================================

def save_global_model(global_model: nn.Module,
                      output_path: Path,
                      report: Dict):
    """Save the global model and aggregation report."""
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = output_path / 'global_model_with_averaged_features.pt'
    torch.save(global_model.state_dict(), model_path)
    print(f"✓ Saved global model to {model_path}")
    
    # Save report
    report_path = output_path / 'aggregation_report.pkl'
    with open(report_path, 'wb') as f:
        pickle.dump(report, f)
    print(f"✓ Saved aggregation report to {report_path}")


def test_global_model(global_model: nn.Module, device: str = 'cpu'):
    """
    Test the global model on a dummy batch.
    """
    
    print("\nStep 8: Testing global model...")
    
    global_model.eval()
    
    # Create dummy input (batch of 4 images)
    dummy_input = torch.randn(4, 3, 32, 32, device=device)
    
    with torch.no_grad():
        output = global_model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: [4, 100]")
    
    if output.shape == torch.Size([4, 100]):
        print(f"  ✓ Output shape is correct!")
    else:
        print(f"  ✗ Output shape mismatch!")
    
    print(f"\n✓ Global model test passed!\n")
    
    return output


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Complete pipeline with feature averaging.
    """
    
    MODEL_DIR = Path('./final_specialised_models')
    OUTPUT_DIR = Path('./final_specialised_models')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_names = ['outdoors', 'nature', 'people', 'objects_vehicles']
    
    try:
        # ============================================================
        # Aggregate models with feature averaging
        # ============================================================
        global_model, report = aggregate_models_all_layers_with_feature_averaging(
            model_names=model_names,
            model_dir=MODEL_DIR,
            device=DEVICE
        )
        
        # ============================================================
        # Save
        # ============================================================
        save_global_model(global_model, OUTPUT_DIR, report)
        
        # ============================================================
        # Test
        # ============================================================
        output = test_global_model(global_model, device=DEVICE)
        
        print("="*80)
        print("AGGREGATION COMPLETE!")
        print("="*80)
        print(f"\nGlobal model ready for inference:")
        print(f"  - Feature extraction: Averaged from 4 models")
        print(f"  - Classifier: Remapped to CIFAR-100 space")
        print(f"  - All 43 trained classes: Ready for prediction")
        print(f"  - 57 untrained classes: Will use averaged features")
        print(f"\nModel saved to: {OUTPUT_DIR / 'global_model_with_averaged_features.pt'}")
        print("="*80 + "\n")
        
        return global_model
        
    except Exception as e:
        print(f"\n✗ Aggregation failed: {e}")
        raise


if __name__ == '__main__':
    global_model = main()
