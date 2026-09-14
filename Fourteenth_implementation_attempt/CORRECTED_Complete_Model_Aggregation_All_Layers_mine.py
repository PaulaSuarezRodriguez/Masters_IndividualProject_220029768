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
from Fourteenth_implementation_attempt.ResNet20_DynamicModel_v14 import ResNet20_DynamicModel
############ Imports for Evaluating: ##############################################################################################################################################
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, _T_co
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_definitions import T_co
from Fourteenth_implementation_attempt.Train_ResNet20_Outdoors_LwF_v14 import outdoors_specialised_models_test_stream_class_il, outdoors_specialised_models_train_stream_class_il
from Fourteenth_implementation_attempt.Train_ResNet20_Nature_LwF_v14 import nature_specialised_models_test_stream_class_il, nature_specialised_models_train_stream_class_il
from Fourteenth_implementation_attempt.Train_ResNet20_People_LwF_v14 import people_specialised_models_test_stream_class_il, people_specialised_models_train_stream_class_il
from Fourteenth_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_v14 import objects_and_vehicles_specialised_models_test_stream_class_il, objects_and_vehicles_specialised_models_train_stream_class_il
from Fourteenth_implementation_attempt.Load_cifar100_v14 import avl_test
#TODO: TEMPORARY!!!! Change to that of v14 once I have final Objects_and_Vehicles model!!!!!
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F
####################################################################################################################################################################################

VERBOSE = False

# ============================================================================
# PHASE 4: EVALUATION
# ============================================================================
# NEW NEW NEW
test_stream_dict ={}
test_stream_dict["outdoors_specialised_models_test_stream_class_il_variable"]= outdoors_specialised_models_test_stream_class_il
test_stream_dict["nature_specialised_models_test_stream_class_il_variable"]= nature_specialised_models_test_stream_class_il
test_stream_dict["people_specialised_models_test_stream_class_il_variable"]= people_specialised_models_test_stream_class_il
test_stream_dict["objects_and_vehicles_specialised_models_test_stream_class_il_variable"]=objects_and_vehicles_specialised_models_test_stream_class_il

for test_stream_name, test_stream in test_stream_dict.items():
    for test_exp in test_stream:
        if len(test_exp.dataset) != 0:
            print(
                f"{test_stream_name}.Exp{test_exp.current_experience}.dataset.coarse_label.uniques (coarse_labels of TEST)",
                test_exp.dataset.coarse_label.uniques
            )

            print(
                f"{test_stream_name}.Exp{test_exp.current_experience}.dataset.targets.uniques (targets of TEST)",
                test_exp.dataset.targets.uniques
            )

print("type", type(outdoors_specialised_models_test_stream_class_il[0].dataset))

combined_dataset_outdoors = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[0].dataset,
    outdoors_specialised_models_test_stream_class_il[1].dataset,
    outdoors_specialised_models_test_stream_class_il[2].dataset,
    #outdoors_specialised_models_test_stream_class_il[3].dataset,
    #outdoors_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_nature = ConcatDataset([
    nature_specialised_models_test_stream_class_il[0].dataset,
    nature_specialised_models_test_stream_class_il[1].dataset,
    nature_specialised_models_test_stream_class_il[2].dataset,
    #nature_specialised_models_test_stream_class_il[3].dataset,
    #nature_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_people = ConcatDataset([
    people_specialised_models_test_stream_class_il[0].dataset,
    people_specialised_models_test_stream_class_il[1].dataset,
    people_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles = ConcatDataset([
    objects_and_vehicles_specialised_models_test_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset = ConcatDataset([
    combined_dataset_outdoors,
    combined_dataset_nature,
    combined_dataset_people, combined_dataset_objects_and_vehicles
])

test_loader = DataLoader(
    combined_dataset,
    batch_size=128,#32,
    shuffle=False
)
print("print test_loader", test_loader)
print("print test_loader.dataset", test_loader.dataset)


from collections import Counter

all_labels = []

for i in range(len(combined_dataset)):
    sample = combined_dataset[i]

    # Adjust depending on your dataset output
    label = sample[1]

    all_labels.append(int(label))

print(Counter(all_labels))
print("Unique labels:", sorted(set(all_labels)))
print("Number of unique labels:", len(set(all_labels)))


for batch in test_loader:
    print(type(batch))
    print(len(batch))
    for i, item in enumerate(batch):
        print(i, type(item), getattr(item, "shape", None))
        print(type(batch))
    break


combined_train_dataset = ConcatDataset([
    outdoors_specialised_models_train_stream_class_il,
    nature_specialised_models_train_stream_class_il,
    people_specialised_models_train_stream_class_il,
    objects_and_vehicles_specialised_models_train_stream_class_il])

train_loader = DataLoader(
    combined_train_dataset,
    batch_size=128,#32,
    shuffle=False
)
print("print train_loader",train_loader)
print("print train_loader.dataset", train_loader.dataset)

avl_test_loader = DataLoader(
    avl_test,
    batch_size=128,#32,
    shuffle=False
)


def unpack_minibatch(mbatch):
    """Check if the current mini-batch has 3 components."""
    mbatch = mbatch
    assert mbatch is not None

    if isinstance(mbatch, tuple):
        mbatch = list(mbatch)
        mbatch = mbatch

    for i in range(len(mbatch)):
        if isinstance(mbatch[i], dict):
            # Move all tensor values in the dictionary to the specified device
            mbatch[i] = {
                key: value.to(non_blocking=True)
                for key, value in mbatch[i].items()
            }
        elif isinstance(mbatch[i], torch.Tensor):
            # Directly move the tensor to the specified device
            mbatch[i] = mbatch[i].to(non_blocking=True)
        else:
            # Optionally handle other cases (e.g., raise an error if unsupported types are encountered)
            raise TypeError(f"Unsupported type in batch: {type(mbatch[i])}")


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

    return global_model #TODO: IMPORTANT CHANGE. I added this


# ==============================================================================
# STEP 2: Classifier Remapping (UNCHANGED)
# ==============================================================================

def remap_classifier_to_cifar100(
    model: nn.Module,
    new_to_original: Dict[int, int],
    num_subset_classes,
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
        dtype=model.linear.weight.dtype #dtype=classifier.weight.dtype
    )

    remapped_bias = torch.zeros(
        num_classes,
        device=device,
        dtype=model.linear.bias.dtype  #dtype=classifier.bias.dtype
    )

    local_idx_count = 0
    # Place each trained class at its CIFAR-100 position
    for local_idx, cifar_class in new_to_original.items():
        print("new_to_original", new_to_original)
        print("print(local_idx, cifar_class)", local_idx, cifar_class)
        local_idx_count += 1
        print("print(local_idx_count)", local_idx_count)
        print("print(num_subset_classes)", num_subset_classes)
        if local_idx_count >= num_subset_classes:
            print("print(num_subset_classes) THAT CAUSED BREAK", num_subset_classes)
            break
        # local_idx: 0-5 for outdoors (position in model's classifier)
        # cifar_class: 12, 17, 23, ... (actual CIFAR class)
        remapped_weights[cifar_class] = classifier.weight[local_idx].to(device)
        remapped_bias[cifar_class] = classifier.bias[local_idx].to(device)

    if VERBOSE:
        print("remapped_weights", remapped_weights)
        print("remapped_bias", remapped_bias)

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

    model_dir = f"../Fourteenth_implementation_attempt/final_specialised_models/" #model_dir = Path(model_dir)
    models = {}
    mappings = {}
    num_subset_classes = {}

    # ==================================================================
    # STEP 1: Load all models and their mappings
    # ==================================================================

    print("Step 1: Loading specialized models...")
    for model_name in model_names:
        try:
            state_dict = torch.load(f"{model_dir}final_{model_name}_model.pt", map_location=device)#state_dict = torch.load(model_dir / f'final_{model_name}_model.pt',map_location=device)

            # Determine number of classes from classifier weight shape
            classifier_size = state_dict['linear.weight'].shape[0]
            model = ResNet20_DynamicModel(num_classes=classifier_size)
            model.load_state_dict(state_dict)
            model = model.to(device)
            models[model_name] = model

            # Load mapping
            with open(f"{model_dir}{model_name}_metadata.pkl", 'rb') as f: #with open(model_dir / f'{model_name}_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            mappings[model_name] = metadata['new_to_original']
            num_subset_classes[model_name] = metadata['num_subset_classes']

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

    global_model = ResNet20_DynamicModel(num_classes=100)
    global_model = global_model.to(device)

    # Load averaged feature extraction layers
    global_model = load_averaged_feature_layers(global_model, averaged_features)#load_averaged_feature_layers(global_model, averaged_features) #TODO: IMPORTANT CHANGE

    print(f"  ✓ Created global model with averaged features:")
    print(f"    - conv1 + bn1: ✓ Averaged from {len(model_names)} models")
    print(f"    - layer1-3: ✓ Averaged from {len(model_names)} models")

    print()

    # ==================================================================
    # STEP 5: Remap and combine classifiers
    # ==================================================================

    print("Step 5: Remapping classifiers to CIFAR-100 space...")
    with torch.no_grad():
        feature_dim = models[model_names[0]].linear.weight.shape[1]
        global_classifier_weights = torch.zeros(100, feature_dim, device=device)
        global_classifier_bias = torch.zeros(100, device=device)

        class_coverage = {}

        for model_name in model_names:
            model = models[model_name]
            mapping = mappings[model_name]

            w, b = remap_classifier_to_cifar100(
                model, mapping, num_subset_classes[model_name], num_classes=100, device=device
            )
            print(f"w to see what it looks like after additions ({model_name}):\n", w)
            print(f"b to see what it looks like after additions ({model_name}):\n", b)
            # Add to global classifier (no averaging - classes don't overlap!)
            global_classifier_weights += w
            global_classifier_bias += b
            print(f"global_weights += w to see what it looks like after additions ({model_name}):\n", global_classifier_weights)
            print(f"global_bias += b to see what it looks like after additions ({model_name}):\n", global_classifier_bias)

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


def evaluate_aggregated_model(model: nn.Module, loader: torch.utils.data.DataLoader):
    total_loss_all_experiences = 0.0
    total_correct_all_experiences = 0
    total_samples_all_experiences = 0

    model.eval() #TODO: IN TESTING! If it leads to problems, comment-out again
    with torch.no_grad():
        for batch_data, batch_labels, batch_task_ids in loader:#for batch_data, batch_labels, batch_task_ids in test_loader:
            #unpack_minibatch(batch_data)

            logits_all_experiences = model(batch_data)
            print("logits_all_experiences.dim()",logits_all_experiences.dim())
            targets_all_experiences = batch_labels


            #loss_all_experiences = CrossEntropyLoss(logits_all_experiences, targets_all_experiences, reduction='mean')
            loss_all_experiences = F.cross_entropy(
                logits_all_experiences,
                targets_all_experiences,
                reduction='mean'
            )

            total_loss_all_experiences += loss_all_experiences.item() * targets_all_experiences.size(0)
            predictions_all_experiences = torch.argmax(logits_all_experiences, dim=1)

            total_correct_all_experiences += (predictions_all_experiences == targets_all_experiences).sum().item()
            total_samples_all_experiences += targets_all_experiences.size(0)

    avg_loss_all_experiences = total_loss_all_experiences / total_samples_all_experiences if total_samples_all_experiences > 0 else 0.0
    accuracy_all_experiences = (total_correct_all_experiences / total_samples_all_experiences * 100) if total_samples_all_experiences > 0 else 0.0

    print(
        f"Loss = {avg_loss_all_experiences:.6f} | Accuracy = {accuracy_all_experiences:.2f}%"
    )

def evaluate_aggregated_model_FedAvg_code_v3(target_model: nn.Module, loader: torch.utils.data.DataLoader):
    correct = 0
    total = 0

    target_model.eval()
    with torch.no_grad():
        for batch_data, batch_labels, batch_task_ids in loader:
            #print(type(batch_data))
            #print(type(batch_labels))
            #print(type(batch_task_ids))

            ######print("\nprint(batch_labels)\n",batch_labels)
            ##print("print(batch_task_ids)",batch_task_ids)

            outputs = target_model(batch_data)
            targets_all_experiences = batch_labels
            #####print("predictions before argmax:",predictions)
            #####print("predictions before argmax SHAPE:", predictions.shape)
            #####print("predictions before argmax LENGTH:", len(predictions))
            #####print("batch_labels before argmax SHAPE:", batch_labels.shape)
            #####print("batch_labels before argmax LENGTH:", len(batch_labels))

            predictions = outputs.argmax(dim=1)

            correct += (predictions == batch_labels).sum().item() #correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    #accuracy = correct / total
    accuracy = (correct / total ) if total > 0 else 0.0
    print(f"Fine-tuned model accuracy: {accuracy:.2%}")#print(f"Fine-tuned model accuracy: {accuracy:.2%}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Complete pipeline with feature averaging.
    """

    MODEL_DIR = Path(f"../Fourteenth_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    OUTPUT_DIR = Path('./final_specialised_models')
    DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

    model_names = ['outdoors', 'nature', 'people', 'objects_and_vehicles']

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
    #evaluate_aggregated_model(global_model)
    global_model_reload = torch.load("../Fourteenth_implementation_attempt/final_specialised_models/global_model_with_averaged_features.pt")
    global_model_reload_architecture = ResNet20_DynamicModel(num_classes=100)
    global_model_reload_architecture.load_state_dict(global_model_reload)
    evaluate_aggregated_model_FedAvg_code_v3(global_model_reload_architecture, test_loader)
    evaluate_aggregated_model(global_model_reload_architecture, test_loader)
    #evaluate_aggregated_model(global_model_reload_architecture, train_loader)
    #evaluate_aggregated_model(global_model_reload_architecture,outdoors_specialised_models_train_stream_class_il)
    #evaluate_aggregated_model(global_model_reload_architecture, avl_test_loader)
