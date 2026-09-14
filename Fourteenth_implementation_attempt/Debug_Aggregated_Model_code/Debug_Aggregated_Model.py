"""
Debugging script to identify why aggregated model predicts class 23 for everything.

This script diagnoses:
1. Feature extractor output (are features constant?)
2. Classifier weight distribution (which classes have non-zero weights?)
3. Logits distribution (are they constant?)
4. Model loading (did all models load correctly?)
5. Batch norm statistics (are they reasonable?)
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ResNet20_DynamicModel_v13 import ResNet20_DynamicModel
import pickle


def debug_aggregated_model(model_path='./final_specialised_models/global_model_with_averaged_features.pt'):
    """
    Comprehensive debugging of the aggregated model.
    """
    
    print("\n" + "="*80)
    print("DEBUGGING AGGREGATED MODEL")
    print("="*80 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ========================================================================
    # STEP 1: Load model
    # ========================================================================
    
    print("STEP 1: Loading model...")
    
    try:
        global_model = ResNet20_DynamicModel(num_classes=100)
        state_dict = torch.load(model_path, map_location=device)
        global_model.load_state_dict(state_dict)
        global_model = global_model.to(device)
        global_model.eval()  # IMPORTANT: Set to eval mode!
        print("  ✓ Model loaded successfully\n")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}\n")
        return
    
    # ========================================================================
    # STEP 2: Check classifier weight distribution
    # ========================================================================
    
    print("STEP 2: Examining classifier weights...")
    
    classifier_weights = global_model.linear.weight.data.cpu()  # [100, 64]
    classifier_bias = global_model.linear.bias.data.cpu()  # [100]
    
    # Compute magnitude of each class's weight vector
    weight_magnitudes = torch.norm(classifier_weights, dim=1)  # [100]
    
    print(f"  Classifier weight shape: {classifier_weights.shape}")
    print(f"  Classifier bias shape: {classifier_bias.shape}")
    print()
    
    # Find classes with non-zero weights
    nonzero_threshold = 0.001
    nonzero_classes = (weight_magnitudes > nonzero_threshold).nonzero(as_tuple=True)[0].tolist()
    
    print(f"  Classes with non-zero weights (threshold={nonzero_threshold}):")
    print(f"    Total: {len(nonzero_classes)}/100")
    print(f"    Classes: {sorted(nonzero_classes)}\n")
    
    if len(nonzero_classes) != 43:
        print(f"  ⚠️  WARNING: Expected 43 trained classes, found {len(nonzero_classes)}")
        print(f"      This suggests remapping error!\n")
    
    # Weight statistics
    print(f"  Weight magnitude statistics:")
    print(f"    Min: {weight_magnitudes.min():.6f}")
    print(f"    Max: {weight_magnitudes.max():.6f}")
    print(f"    Mean: {weight_magnitudes.mean():.6f}")
    print(f"    Std: {weight_magnitudes.std():.6f}\n")
    
    # Bias statistics
    print(f"  Bias statistics:")
    print(f"    Min: {classifier_bias.min():.6f}")
    print(f"    Max: {classifier_bias.max():.6f}")
    print(f"    Mean: {classifier_bias.mean():.6f}")
    print(f"    Std: {classifier_bias.std():.6f}\n")
    
    # Top 10 classes by weight magnitude
    top_10_indices = torch.topk(weight_magnitudes, 10)[1].tolist()
    print(f"  Top 10 classes by weight magnitude:")
    for idx in top_10_indices:
        print(f"    Class {idx}: {weight_magnitudes[idx]:.6f}")
    print()
    
    # Check class 23 specifically
    print(f"  Class 23 (the predicted class) analysis:")
    print(f"    Weight magnitude: {weight_magnitudes[23]:.6f}")
    print(f"    Bias value: {classifier_bias[23]:.6f}")
    print(f"    Is class 23 in Outdoors subset? Yes (classes 12, 17, 23, 33, 37, 49)")
    print()
    
    # ========================================================================
    # STEP 3: Check batch norm statistics
    # ========================================================================
    
    print("STEP 3: Examining batch norm statistics...")
    
    bn_running_mean = global_model.bn1.running_mean.data.cpu()
    bn_running_var = global_model.bn1.running_var.data.cpu()
    
    print(f"  BN1 running mean:")
    print(f"    Min: {bn_running_mean.min():.6f}, Max: {bn_running_mean.max():.6f}")
    print(f"    All zeros? {torch.allclose(bn_running_mean, torch.zeros_like(bn_running_mean))}")
    print()
    
    print(f"  BN1 running variance:")
    print(f"    Min: {bn_running_var.min():.6f}, Max: {bn_running_var.max():.6f}")
    print(f"    All ones? {torch.allclose(bn_running_var, torch.ones_like(bn_running_var))}")
    print(f"    All zeros? {torch.allclose(bn_running_var, torch.zeros_like(bn_running_var))}")
    print()
    
    if torch.allclose(bn_running_mean, torch.zeros_like(bn_running_mean)):
        print(f"  ⚠️  WARNING: BN running mean is all zeros!")
        print(f"      This could cause feature collapse!\n")
    
    if torch.allclose(bn_running_var, torch.zeros_like(bn_running_var)):
        print(f"  ⚠️  WARNING: BN running variance is all zeros!")
        print(f"      This will cause division by zero!\n")
    
    # ========================================================================
    # STEP 4: Test feature extraction
    # ========================================================================
    
    print("STEP 4: Testing feature extraction...")
    
    # Create dummy input
    dummy_input = torch.randn(4, 3, 32, 32, device=device)
    
    # Forward through feature extractor (everything except classifier)
    with torch.no_grad():
        # Forward through conv1 and bn1
        features = torch.nn.functional.relu(global_model.bn1(global_model.conv1(dummy_input)))
        
        # Forward through residual layers
        features = global_model.layer1(features)
        features = global_model.layer2(features)
        features = global_model.layer3(features)
        
        # Average pooling
        features = torch.nn.functional.avg_pool2d(features, features.size()[3])
        features = features.view(features.size(0), -1)  # Flatten to [batch, 64]
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Feature output shape: {features.shape}")
    print()
    
    print(f"  Feature statistics (should vary per sample):")
    for i in range(features.shape[0]):
        sample_features = features[i]
        print(f"    Sample {i}: min={sample_features.min():.6f}, max={sample_features.max():.6f}, "
              f"mean={sample_features.mean():.6f}, std={sample_features.std():.6f}")
    print()
    
    # Check if features are constant across samples
    feature_variance = features.var(dim=0).mean().item()
    print(f"  Feature variance across samples: {feature_variance:.6f}")
    if feature_variance < 0.001:
        print(f"  ⚠️  WARNING: Features have very low variance!")
        print(f"      This indicates feature extractor may not be working!\n")
    else:
        print(f"  ✓ Features vary across samples (good)\n")
    
    # ========================================================================
    # STEP 5: Test full forward pass
    # ========================================================================
    
    print("STEP 5: Testing full forward pass...")
    
    with torch.no_grad():
        logits = global_model(dummy_input)
    
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected: [4, 100]\n")
    
    print(f"  Logits statistics (should vary per sample):")
    for i in range(logits.shape[0]):
        sample_logits = logits[i]
        top_5_values, top_5_indices = torch.topk(sample_logits, 5)
        print(f"    Sample {i}:")
        print(f"      Top 5 predictions: {top_5_indices.tolist()}")
        print(f"      Top 5 values: {top_5_values.tolist()}")
        print(f"      Logit range: [{sample_logits.min():.4f}, {sample_logits.max():.4f}]")
    print()
    
    # Check if logits are constant across samples
    logits_variance = logits.var(dim=0).mean().item()
    print(f"  Logit variance across samples: {logits_variance:.6f}")
    if logits_variance < 0.001:
        print(f"  ⚠️  CRITICAL: Logits have very low variance!")
        print(f"      Model produces same output for all inputs!\n")
    else:
        print(f"  ✓ Logits vary across samples (good)\n")
    
    # ========================================================================
    # STEP 6: Test on real CIFAR-100 data
    # ========================================================================
    
    print("STEP 6: Testing on real CIFAR-100 data...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761))
    ])
    
    test_data = datasets.CIFAR100(root='./data', train=False, transform=transform)
    
    # Sample 20 random images
    sample_indices = [0, 10, 20, 30, 40, 50, 100, 200, 500, 1000,
                     2000, 3000, 5000, 7000, 8000, 9000, 9500, 9700, 9800, 9900]
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for idx in sample_indices:
            image, label = test_data[idx]
            image = image.unsqueeze(0).to(device)
            logits = global_model(image)
            pred = torch.argmax(logits, dim=1).item()
            
            predictions.append(pred)
            true_labels.append(label)
    
    print(f"  Sample of predictions on real CIFAR-100 data:")
    print(f"    True label → Predicted class")
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        match = "✓" if true == pred else "✗"
        print(f"    {true:3d} → {pred:3d} {match}")
    
    unique_preds = set(predictions)
    print(f"\n  Unique predictions in sample: {len(unique_preds)}")
    print(f"  Predictions: {sorted(unique_preds)}")
    
    if len(unique_preds) == 1:
        print(f"  ⚠️  CRITICAL: Model only predicts ONE class!")
        print(f"      All predictions are class {unique_preds.pop()}\n")
    
    print()
    
    # ========================================================================
    # STEP 7: Diagnosis summary
    # ========================================================================
    
    print("="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80 + "\n")
    
    issues = []
    
    if len(nonzero_classes) != 43:
        issues.append("❌ Wrong number of trained classes - remapping may have failed")
    
    if torch.allclose(bn_running_mean, torch.zeros_like(bn_running_mean)):
        issues.append("❌ Batch norm running mean is zero - aggregation averaging failed?")
    
    if feature_variance < 0.001:
        issues.append("❌ Features have no variance - feature extractor not working")
    
    if logits_variance < 0.001:
        issues.append("❌ Logits have no variance - critical feature/model problem")
    
    if len(unique_preds) == 1:
        issues.append(f"❌ Model only predicts class {unique_preds.pop()} - severe aggregation error")
    
    if issues:
        print("IDENTIFIED ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        print()
    else:
        print("✓ No critical issues detected in basic diagnostics")
        print("  (But accuracy may still be low due to other factors)\n")
    
    print("="*80 + "\n")
    
    return {
        'nonzero_classes': len(nonzero_classes),
        'feature_variance': feature_variance,
        'logits_variance': logits_variance,
        'unique_predictions': len(unique_preds),
        'issues': issues
    }


# ============================================================================
# HELPER: Check aggregation report
# ============================================================================

def check_aggregation_report(report_path='./final_specialised_models/aggregation_report.pkl'):
    """
    Check the aggregation report from the aggregation process.
    """
    
    print("\n" + "="*80)
    print("CHECKING AGGREGATION REPORT")
    print("="*80 + "\n")
    
    try:
        with open(report_path, 'rb') as f:
            report = pickle.load(f)
        
        print(f"Aggregation method: {report.get('feature_averaging', 'Unknown')}")
        print(f"Number of models: {report.get('num_models_aggregated', 'Unknown')}")
        print(f"Trained classes: {report.get('trained_classes', 'Unknown')}")
        print(f"Untrained classes: {report.get('untrained_classes', 'Unknown')}")
        print()
        
        class_assignment = report.get('class_coverage', {})
        trained_classes = sorted(class_assignment.keys())
        print(f"Trained classes: {trained_classes}")
        print(f"Expected (43): 0, 2, 5, 8, 9, 10, 11, 12, 13, 16, 17, 20, 22, 23, 25, 28, 33, 35, 37, 39, 40, 41, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94")
        print()
        
        if len(trained_classes) != 43:
            print(f"⚠️  WARNING: Expected 43 trained classes, found {len(trained_classes)}")
        
        return report
        
    except FileNotFoundError:
        print(f"  ✗ Report file not found: {report_path}")
        return None


# ============================================================================
# MAIN: Run all diagnostics
# ============================================================================

if __name__ == '__main__':
    # Check aggregation report first
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE DEBUGGING")
    print("="*80)
    
    report = check_aggregation_report()
    
    # Run detailed diagnostics
    diagnostics = debug_aggregated_model()
    
    print("\nDiagnostics complete! See issues listed above.")
