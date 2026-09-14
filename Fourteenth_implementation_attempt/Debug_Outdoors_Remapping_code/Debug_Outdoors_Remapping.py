"""
Focused debugging for the Outdoors model remapping issue.

Symptoms:
- Outdoors model: w is all zeros, b has some values
- Other models: both w and b have values
- This is very specific - suggests issue with Outdoors classifier weights

Possible causes:
1. Outdoors classifier.weight dtype mismatch
2. Outdoors classifier.weight on wrong device
3. Outdoors classifier.weight corrupted or not loaded properly
4. Bug in remapping function specific to first model
"""

import torch
import pickle
from pathlib import Path


def debug_outdoors_remapping():
    """
    Debug why outdoors model produces all-zero weights during remapping.
    """
    
    print("\n" + "="*80)
    print("DEBUGGING OUTDOORS MODEL REMAPPING")
    print("="*80 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dir = Path('./final_specialised_models')
    
    # ========================================================================
    # STEP 1: Load outdoors model directly
    # ========================================================================
    
    print("STEP 1: Loading Outdoors model directly...")
    
    from ResNet20_DynamicModel_v13 import ResNet20_DynamicModel
    
    outdoors_model_path = model_dir / 'final_outdoors_model.pt'
    
    if not outdoors_model_path.exists():
        print(f"  ✗ Model not found: {outdoors_model_path}")
        return
    
    # Create model with correct size
    outdoors_model = ResNet20_DynamicModel(num_classes=6)
    state_dict = torch.load(outdoors_model_path, map_location=device)
    outdoors_model.load_state_dict(state_dict)
    outdoors_model = outdoors_model.to(device)
    
    print(f"  ✓ Outdoors model loaded")
    print(f"    Device: {device}")
    print(f"    Model training mode: {outdoors_model.training}\n")
    
    # ========================================================================
    # STEP 2: Examine classifier
    # ========================================================================
    
    print("STEP 2: Examining classifier layer...")
    
    classifier = outdoors_model.linear
    
    print(f"  Classifier type: {type(classifier)}")
    print(f"  Classifier weight shape: {classifier.weight.shape}")
    print(f"  Classifier weight dtype: {classifier.weight.dtype}")
    print(f"  Classifier weight device: {classifier.weight.device}")
    print(f"  Classifier bias shape: {classifier.bias.shape}")
    print(f"  Classifier bias dtype: {classifier.bias.dtype}")
    print(f"  Classifier bias device: {classifier.bias.device}\n")
    
    # Check if weights are actually loaded
    print("  Classifier weight values (first 3 rows):")
    for i in range(min(3, classifier.weight.shape[0])):
        row = classifier.weight[i]
        print(f"    Row {i}: min={row.min():.6f}, max={row.max():.6f}, "
              f"mean={row.mean():.6f}, norm={row.norm():.6f}")
    
    # Check for all-zero weights
    if torch.allclose(classifier.weight, torch.zeros_like(classifier.weight)):
        print(f"\n  ❌ CRITICAL: Classifier weights are ALL ZEROS!")
        print(f"     This means the outdoors model wasn't trained or weights corrupted!")
        return
    else:
        print(f"\n  ✓ Classifier weights are NOT all zeros (good)\n")
    
    # ========================================================================
    # STEP 3: Load mapping
    # ========================================================================
    
    print("STEP 3: Loading mapping...")
    
    mapping_path = model_dir / 'outdoors_metadata.pkl'
    
    if not mapping_path.exists():
        print(f"  ✗ Mapping not found: {mapping_path}")
        return
    
    with open(mapping_path, 'rb') as f:
        metadata = pickle.load(f)
    
    new_to_original = metadata['new_to_original']
    
    print(f"  Mapping: {new_to_original}")
    print(f"  Expected for outdoors: {{0: 12, 1: 17, 2: 23, 3: 33, 4: 37, 5: 49}}\n")
    
    # ========================================================================
    # STEP 4: Test remapping function step-by-step
    # ========================================================================
    
    print("STEP 4: Testing remapping step-by-step...")
    
    feature_dim = classifier.weight.shape[1]
    
    # Create remapped tensors
    remapped_weights = torch.zeros(100, feature_dim, device=device, dtype=classifier.weight.dtype)
    remapped_bias = torch.zeros(100, device=device, dtype=classifier.bias.dtype)
    
    print(f"  Created remapped tensors:")
    print(f"    remapped_weights: shape={remapped_weights.shape}, dtype={remapped_weights.dtype}, device={remapped_weights.device}")
    print(f"    remapped_bias: shape={remapped_bias.shape}, dtype={remapped_bias.dtype}, device={remapped_bias.device}\n")
    
    # Manually do the remapping for debugging
    print("  Performing remapping:")
    
    for local_idx, cifar_class in new_to_original.items():
        # Get the weight and bias from classifier
        weight_row = classifier.weight[local_idx]
        bias_val = classifier.bias[local_idx]
        
        print(f"\n    Local index {local_idx} → CIFAR class {cifar_class}:")
        print(f"      Original weight[{local_idx}]: min={weight_row.min():.6f}, max={weight_row.max():.6f}, norm={weight_row.norm():.6f}")
        print(f"      Original bias[{local_idx}]: {bias_val:.6f}")
        
        # Convert to target device
        weight_row_device = weight_row.to(device)
        bias_val_device = bias_val.to(device)
        
        print(f"      After .to(device):")
        print(f"        weight: min={weight_row_device.min():.6f}, max={weight_row_device.max():.6f}, norm={weight_row_device.norm():.6f}")
        print(f"        bias: {bias_val_device:.6f}")
        
        # Try the assignment
        print(f"      Attempting assignment to remapped[{cifar_class}]...")
        remapped_weights[cifar_class] = weight_row_device
        remapped_bias[cifar_class] = bias_val_device
        
        # Verify the assignment worked
        print(f"      After assignment:")
        print(f"        remapped_weights[{cifar_class}]: min={remapped_weights[cifar_class].min():.6f}, max={remapped_weights[cifar_class].max():.6f}, norm={remapped_weights[cifar_class].norm():.6f}")
        print(f"        remapped_bias[{cifar_class}]: {remapped_bias[cifar_class]:.6f}")
        
        # Check if assignment actually worked
        if torch.allclose(remapped_weights[cifar_class], weight_row_device):
            print(f"      ✓ Assignment successful")
        else:
            print(f"      ❌ Assignment FAILED - values didn't stick!")
    
    print()
    
    # ========================================================================
    # STEP 5: Check final remapped tensors
    # ========================================================================
    
    print("STEP 5: Checking final remapped tensors...")
    
    print(f"  remapped_weights statistics:")
    print(f"    Shape: {remapped_weights.shape}")
    print(f"    Min: {remapped_weights.min():.6f}")
    print(f"    Max: {remapped_weights.max():.6f}")
    print(f"    Mean: {remapped_weights.mean():.6f}")
    
    nonzero_count = (remapped_weights.norm(dim=1) > 0).sum().item()
    print(f"    Rows with non-zero weights: {nonzero_count}/100")
    
    if nonzero_count == 0:
        print(f"\n  ❌ CRITICAL: All remapped weights are zero!")
        print(f"     Expected 6 rows with values, got 0")
    else:
        print(f"\n  ✓ Remapped weights have values")
    
    print(f"\n  remapped_bias statistics:")
    print(f"    Shape: {remapped_bias.shape}")
    print(f"    Min: {remapped_bias.min():.6f}")
    print(f"    Max: {remapped_bias.max():.6f}")
    print(f"    Mean: {remapped_bias.mean():.6f}")
    
    nonzero_bias = (remapped_bias != 0).sum().item()
    print(f"    Non-zero bias values: {nonzero_bias}/100")
    
    # ========================================================================
    # STEP 6: Compare with other models (if available)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 6: Comparing with Nature model...")
    print("="*80 + "\n")
    
    nature_model_path = model_dir / 'final_nature_model.pt'
    
    if nature_model_path.exists():
        nature_model = ResNet20_DynamicModel(num_classes=9)
        nature_state = torch.load(nature_model_path, map_location=device)
        nature_model.load_state_dict(nature_state)
        nature_model = nature_model.to(device)
        
        nature_classifier = nature_model.linear
        
        print(f"  Nature classifier weight shape: {nature_classifier.weight.shape}")
        print(f"  Nature classifier weight dtype: {nature_classifier.weight.dtype}")
        print(f"  Nature classifier weight device: {nature_classifier.weight.device}")
        
        print(f"\n  Nature classifier weight values (first 3 rows):")
        for i in range(min(3, nature_classifier.weight.shape[0])):
            row = nature_classifier.weight[i]
            print(f"    Row {i}: min={row.min():.6f}, max={row.max():.6f}, norm={row.norm():.6f}")
        
        print(f"\n  Comparison:")
        print(f"    Outdoors weight norm: {classifier.weight.norm():.6f}")
        print(f"    Nature weight norm: {nature_classifier.weight.norm():.6f}")
        
        if classifier.weight.norm() == 0.0:
            print(f"\n  ❌ Outdoors weights are all zeros - definitely corrupted!")
        elif abs(classifier.weight.norm() - nature_classifier.weight.norm()) < 0.01:
            print(f"\n  ✓ Weight magnitudes similar between models")
        else:
            print(f"\n  ⚠️  Weight magnitudes differ significantly")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    debug_outdoors_remapping()
