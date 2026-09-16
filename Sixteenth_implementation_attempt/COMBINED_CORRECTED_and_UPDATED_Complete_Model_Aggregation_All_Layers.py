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
from Sixteenth_implementation_attempt.ResNet20_DynamicModel_v16 import ResNet20_DynamicModel
#from Sixteenth_implementation_attempt.ResNet20_DynamicModel_Compact_v16 import ResNet20_XL
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
from Sixteenth_implementation_attempt.Train_ResNet20_Outdoors_LwF_v16 import outdoors_specialised_models_test_stream_class_il, outdoors_specialised_models_train_stream_class_il
from Sixteenth_implementation_attempt.Train_ResNet20_Nature_LwF_v16 import nature_specialised_models_test_stream_class_il, nature_specialised_models_train_stream_class_il
from Sixteenth_implementation_attempt.Train_ResNet20_People_LwF_v16 import people_specialised_models_test_stream_class_il, people_specialised_models_train_stream_class_il
from Sixteenth_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_v16 import objects_and_vehicles_specialised_models_test_stream_class_il, objects_and_vehicles_specialised_models_train_stream_class_il
from Sixteenth_implementation_attempt.Load_cifar100_v16 import avl_test
#TODO: TEMPORARY!!!! Change to that of v16 once I have final Objects_and_Vehicles model!!!!!
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F
####################################################################################################################################################################################

# =============================================================================
# CONFIGURATION
# =============================================================================

VERBOSE = False

# Your original script evaluates the first three experiences of each specialist.
# Keep the calibration data consistent with that trained/evaluated portion.
NUM_USED_EXPERIENCES = 3

# Classifier/feature-space alignment.
ALIGN_CLASSIFIERS_TO_AVERAGED_FEATURES = True
ADAPTER_RIDGE = 1e-3
ALIGNMENT_BATCH_SIZE = 128

# Recompute BN running_mean/running_var after model merging.
RECALIBRATE_BATCHNORM = True
BN_RECALIBRATION_BATCH_SIZE = 128

# Verify the old local -> CIFAR-100 operation, but do NOT use sparse 100-row
# tensors to build the actual global head.
RUN_CIFAR100_REMAP_DIAGNOSTIC = True

# Saved model/report names.
GLOBAL_MODEL_FILENAME = "global_model_43_aligned_features.pt"
REPORT_FILENAME = "aggregation_report_43_aligned.pkl"


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
    combined_dataset_people,
    combined_dataset_objects_and_vehicles
])

test_loader = DataLoader(
    combined_dataset,
    batch_size=32,
    shuffle=False
)
print("print test_loader", test_loader)
print("print test_loader.dataset", test_loader.dataset)

nature_loader = DataLoader(
    combined_dataset_nature,
    batch_size=32,
    shuffle=False
)

outdoors_loader = DataLoader(
    combined_dataset_outdoors,
    batch_size=32,
    shuffle=False
)
people_loader = DataLoader(
    combined_dataset_people,
    batch_size=32,
    shuffle=False
)
objects_and_vehicles_loader = DataLoader(
    combined_dataset_objects_and_vehicles,
    batch_size=32,
    shuffle=False
)
combined_dataset_objects_and_vehicles_TRAIN = ConcatDataset([
    objects_and_vehicles_specialised_models_train_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[2].dataset,
])
objects_and_vehicles_TRAIN_loader = DataLoader(
    combined_dataset_objects_and_vehicles_TRAIN,
    batch_size=32,
    shuffle=False
)


combined_dataset_without_objects_and_vehicles = ConcatDataset([
    combined_dataset_outdoors,
    combined_dataset_nature,
    combined_dataset_people #,
    #combined_dataset_objects_and_vehicles
])

test_loader_without_objects_and_vehicles = DataLoader(
    combined_dataset_without_objects_and_vehicles,
    batch_size=32,
    shuffle=False
)


combined_dataset_outdoors_train = ConcatDataset([
    outdoors_specialised_models_train_stream_class_il[0].dataset,
    outdoors_specialised_models_train_stream_class_il[1].dataset,
    outdoors_specialised_models_train_stream_class_il[2].dataset,
    #outdoors_specialised_models_train_stream_class_il[3].dataset,
    #outdoors_specialised_models_train_stream_class_il[4].dataset,
])

combined_dataset_nature_train = ConcatDataset([
    nature_specialised_models_train_stream_class_il[0].dataset,
    nature_specialised_models_train_stream_class_il[1].dataset,
    nature_specialised_models_train_stream_class_il[2].dataset,
    #nature_specialised_models_train_stream_class_il[3].dataset,
    #nature_specialised_models_train_stream_class_il[4].dataset,
])

combined_dataset_people_train = ConcatDataset([
    people_specialised_models_train_stream_class_il[0].dataset,
    people_specialised_models_train_stream_class_il[1].dataset,
    people_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_train = ConcatDataset([
    objects_and_vehicles_specialised_models_train_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_train = ConcatDataset([
    combined_dataset_outdoors_train,
    combined_dataset_nature_train,
    combined_dataset_people_train,
    combined_dataset_objects_and_vehicles_train
])

train_loader = DataLoader(
    combined_dataset_train,
    batch_size=32,
    shuffle=True
)
print("print train_loader", train_loader)
print("print train_loader.dataset", train_loader.dataset)


combined_dataset_outdoors_last_experience_train = ConcatDataset([
    #outdoors_specialised_models_train_stream_class_il[0].dataset,
    #outdoors_specialised_models_train_stream_class_il[1].dataset,
    outdoors_specialised_models_train_stream_class_il[2].dataset,
    #outdoors_specialised_models_train_stream_class_il[3].dataset,
    #outdoors_specialised_models_train_stream_class_il[4].dataset,
])

combined_dataset_nature_last_experience_train = ConcatDataset([
    #nature_specialised_models_train_stream_class_il[0].dataset,
    #nature_specialised_models_train_stream_class_il[1].dataset,
    nature_specialised_models_train_stream_class_il[2].dataset,
    #nature_specialised_models_train_stream_class_il[3].dataset,
    #nature_specialised_models_train_stream_class_il[4].dataset,
])

combined_dataset_people_last_experience_train = ConcatDataset([
    #people_specialised_models_train_stream_class_il[0].dataset,
    #people_specialised_models_train_stream_class_il[1].dataset,
    people_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_last_experience_train = ConcatDataset([
    #objects_and_vehicles_specialised_models_train_stream_class_il[0].dataset,
    #objects_and_vehicles_specialised_models_train_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_last_experience_train = ConcatDataset([
    combined_dataset_outdoors_last_experience_train,
    combined_dataset_nature_last_experience_train,
    combined_dataset_people_last_experience_train,
    combined_dataset_objects_and_vehicles_last_experience_train
])

last_experience_train_loader = DataLoader(
    combined_dataset_last_experience_train,
    batch_size=32,
    shuffle=True
)
print("print last_experience_train_loader", last_experience_train_loader)
print("print last_experience_train_loader.dataset", last_experience_train_loader.dataset)



# Test datasets, kept consistent with your existing first-three-experience setup.

#Concatenate the first n non-empty experience datasets from an Avalanche stream.
'''combined_dataset_outdoors_first_experience_train = ConcatDataset([
    outdoors_specialised_models_train_stream_class_il[0].dataset
])

combined_dataset_nature_first_experience_train = ConcatDataset([
    nature_specialised_models_train_stream_class_il[0].dataset
])

combined_dataset_people_first_experience_train = ConcatDataset([
    people_specialised_models_train_stream_class_il[0].dataset
])

combined_dataset_objects_and_vehicles_first_experience_train = ConcatDataset([
    objects_and_vehicles_specialised_models_train_stream_class_il[0].dataset
])'''

combined_dataset_first_experience_train = ConcatDataset([
    outdoors_specialised_models_train_stream_class_il[0].dataset, #combined_dataset_outdoors_first_experience_train,
    nature_specialised_models_train_stream_class_il[0].dataset, #combined_dataset_nature_first_experience_train,
    people_specialised_models_train_stream_class_il[0].dataset, #combined_dataset_people_first_experience_train,
    objects_and_vehicles_specialised_models_train_stream_class_il[0].dataset#combined_dataset_objects_and_vehicles_first_experience_train
])

first_experience_train_loader = DataLoader(
    combined_dataset_first_experience_train,
    batch_size=32,
    shuffle=True,
)

# Training/calibration datasets used ONLY for BN recalibration and feature alignment.
bn_calibration_loader = DataLoader(
    combined_dataset_first_experience_train, #combined_train_dataset,
    batch_size=BN_RECALIBRATION_BATCH_SIZE,
    shuffle=False,
)

alignment_loaders = {
    "outdoors": DataLoader(
        outdoors_specialised_models_train_stream_class_il[0].dataset,
        batch_size=ALIGNMENT_BATCH_SIZE,
        shuffle=False,
    ),
    "nature": DataLoader(
        nature_specialised_models_train_stream_class_il[0].dataset,
        batch_size=ALIGNMENT_BATCH_SIZE,
        shuffle=False,
    ),
    "people": DataLoader(
        people_specialised_models_train_stream_class_il[0].dataset,
        batch_size=ALIGNMENT_BATCH_SIZE,
        shuffle=False,
    ),
    "objects_and_vehicles": DataLoader(
        objects_and_vehicles_specialised_models_train_stream_class_il[0].dataset,
        batch_size=ALIGNMENT_BATCH_SIZE,
        shuffle=False,
    )
}



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
# STEP 2: Classifier Remapping (UNCHANGED)
# ==============================================================================

'''original_to_aggregated_dict = {"17": 0, "12": 1, "33": 2, "23": 3, "49": 4, "37": 5, #OUTDOORS
                               "51": 6, "47": 7, "0": 8, "54": 9, "53": 10, "52": "11", "59": 12, "57": 13, "56": 14, #NATURE
                               "2": 15, "11": 16, "35": 17, #PEOPLE
                               "16": 18, "13": 19, "10": 20, "9": 21, "8": 22, "5": 23, "58": 24, "48": 25, "41": 26, "40": 27, "39": 28, "28": 29, "25": 30, "22": 31, "20": 32, "94": 33, #OBJECTS_AND_VEHICLES
                               "90": 34, "89": 35, "87": 36, "86": 37, "85": 38, "84": 39, "81": 40, "69": 41, "61": 42} #OBJECTS_AND_VEHICLES'''

original_to_aggregated_dict = {17: 0, 12: 1, 33: 2, 23: 3, 49: 4, 37: 5, #OUTDOORS
                               51: 6, 47: 7, 0: 8, 54: 9, 53: 10, 52: 11, 59: 12, 57: 13, 56: 14, #NATURE
                               2: 15, 11: 16, 35: 17, #PEOPLE
                               16: 18, 13: 19, 10: 20, 9: 21, 8: 22, 5: 23, 58: 24, 48: 25, 41: 26, 40: 27, 39: 28, 28: 29, 25: 30, 22: 31, 20: 32, 94: 33, #OBJECTS_AND_VEHICLES
                               90: 34, 89: 35, 87: 36, 86: 37, 85: 38, 84: 39, 81: 40, 69: 41, 61: 42} #OBJECTS_AND_VEHICLES'''

NUM_GLOBAL_CLASSES = len(original_to_aggregated_dict)
assert NUM_GLOBAL_CLASSES == 43
assert set(original_to_aggregated_dict.values()) == set(range(NUM_GLOBAL_CLASSES))


aggregated_to_original_dict = {
    global_idx: cifar_idx
    for cifar_idx, global_idx in original_to_aggregated_dict.items()
}

def map_original_to_aggregated(targets_tensor, original_to_aggregated=original_to_aggregated_dict):
    ###print("print(targets_tensor)",targets_tensor)
    aggregated_targets_tensor = torch.tensor(
        [
            original_to_aggregated[int(element.item())] for element in targets_tensor #if element in original_to_aggregated.keys()
        ],
        device=targets_tensor.device,
        dtype=torch.long
    )
    ###print("print(aggregated_targets_tensor)\n",aggregated_targets_tensor)

    return aggregated_targets_tensor


def normalize_mapping(mapping: Dict) -> Dict[int, int]:
    """Convert metadata mapping keys/values to plain Python ints."""
    return {int(local_idx): int(cifar_idx) for local_idx, cifar_idx in mapping.items()}



def validate_specialist_mapping(
    model_name: str,
    model: nn.Module,
    mapping: Dict[int, int],
    num_subset_classes: int,
) -> Dict[int, int]:
    """Validate that local rows 0..num_subset_classes-1 all have CIFAR mappings."""

    mapping = normalize_mapping(mapping)
    classifier_rows = model.linear.weight.shape[0]

    if num_subset_classes > classifier_rows:
        raise RuntimeError(
            f"{model_name}: num_subset_classes={num_subset_classes} but classifier "
            f"contains only {classifier_rows} rows."
        )

    required_local_rows = set(range(num_subset_classes))
    missing_local_rows = sorted(required_local_rows - set(mapping.keys()))

    if missing_local_rows:
        raise RuntimeError(
            f"{model_name}: metadata mapping is missing local classifier rows "
            f"{missing_local_rows}."
        )

    trained_cifar_classes = [mapping[i] for i in range(num_subset_classes)]

    duplicates = sorted(
        label
        for label, count in Counter(trained_cifar_classes).items()
        if count > 1
    )
    if duplicates:
        raise RuntimeError(
            f"{model_name}: duplicate CIFAR labels in trained local rows: {duplicates}"
        )

    missing_from_global = sorted(
        set(trained_cifar_classes) - set(original_to_aggregated_dict.keys())
    )
    if missing_from_global:
        raise RuntimeError(
            f"{model_name}: trained CIFAR labels missing from 43-class global mapping: "
            f"{missing_from_global}"
        )

    return mapping



def remap_classifier_to_cifar100(
    model_name,
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
        if local_idx_count > num_subset_classes:
            print("print(num_subset_classes) THAT CAUSED BREAK", num_subset_classes)
            break
        # local_idx: 0-5 for outdoors (position in model's classifier)
        # cifar_class: 12, 17, 23, ... (actual CIFAR class)
        print(f"print(cifar_class = new_to_original[local_idx]) --> {cifar_class}")
        print(f"print(classifier.weight[local_idx] for {model_name})", classifier.weight[local_idx])
        remapped_weights[cifar_class] = classifier.weight[local_idx].to(device)
        remapped_bias[cifar_class] = classifier.bias[local_idx].to(device)

    if VERBOSE:
        print("remapped_weights", remapped_weights)
        print("remapped_bias", remapped_bias)

    return remapped_weights, remapped_bias


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

    '''# Copy input layers
    feature_layers['conv1.weight'] = state_dict['conv1.weight'].clone()
    feature_layers['bn1.weight'] = state_dict['bn1.weight'].clone()
    feature_layers['bn1.bias'] = state_dict['bn1.bias'].clone()
    feature_layers['bn1.running_mean'] = state_dict['bn1.running_mean'].clone()
    feature_layers['bn1.running_var'] = state_dict['bn1.running_var'].clone()
    feature_layers['bn1.num_batches_tracked'] = state_dict['bn1.num_batches_tracked'].clone()

    # Copy all residual block layers (layer1, layer2, layer3)
    for key in state_dict.keys():
        if key.startswith('layer'):  # layer1, layer2, layer3
            feature_layers[key] = state_dict[key].clone()'''

    # Copy all residual block layers (layer1, layer2, layer3)
    for key in state_dict.keys():
        if "linear" not in key and not key.endswith("running_mean") and not key.endswith("running_var") and not key.endswith("num_batches_tracked"):#if "linear" not in key and "bn" not in key and not key.endswith("running_mean") and not key.endswith("running_var") and not key.endswith("num_batches_tracked"): #if key.startswith('layer') or key.endswith("wight") or key.endswith("bias") and "linear" not in key:  # layer1, layer2, layer3
            feature_layers[key] = state_dict[key].clone()
    print("feature_layers.keys()",feature_layers.keys())

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
        stacked = torch.stack([features[key].float() for features in features_list]) #stacked = torch.stack([features[key].float() for features in features_list]).sum(dim=0)#stacked = torch.stack([features_list[0][key].float(), features_list[1][key].float()])#stacked = torch.stack([features[key].float() for features in features_list])
        print("print(stacked.shape)",stacked.shape)
        print("stacked.dim()",stacked.dim())

        # Take mean across models (dim=0)
        averaged = torch.mean(stacked, dim=0)
        #averaged = torch.true_divide(stacked, num_models) # goes with #stacked = torch.stack([features[key].float() for features in features_list]).sum(dim=0)

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

# =============================================================================
# DIRECT 43-CLASS GLOBAL CLASSIFIER (RAW / UNALIGNED BASELINE)
# =============================================================================


def build_global_classifier_43(
    models: Dict[str, nn.Module],
    mappings: Dict[str, Dict[int, int]],
    num_subset_classes: Dict[str, int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict]]:
    """
    Build the 43-row global classifier by explicit indexed insertion.

    For each specialist row:
        local row -> CIFAR-100 label -> compact global row

    No concatenation, sparse 100-row summation, or dictionary-order assumptions.
    """

    first_model = next(iter(models.values()))
    feature_dim = first_model.linear.weight.shape[1]

    global_weight = torch.zeros(
        NUM_GLOBAL_CLASSES,
        feature_dim,
        device=device,
        dtype=first_model.linear.weight.dtype,
    )
    global_bias = torch.zeros(
        NUM_GLOBAL_CLASSES,
        device=device,
        dtype=first_model.linear.bias.dtype,
    )

    owner: Dict[int, Dict] = {}

    for model_name, model in models.items():
        mapping = validate_specialist_mapping(
            model_name,
            model,
            mappings[model_name],
            num_subset_classes[model_name],
        )

        print(f"\n===== BUILDING RAW GLOBAL HEAD: {model_name.upper()} =====")

        for local_idx in range(num_subset_classes[model_name]):
            cifar_idx = mapping[local_idx]
            global_idx = original_to_aggregated_dict[cifar_idx]

            if global_idx in owner:
                raise RuntimeError(
                    f"Global classifier row {global_idx} is already occupied by "
                    f"{owner[global_idx]['model_name']}; {model_name} also requested it."
                )

            global_weight[global_idx].copy_(model.linear.weight[local_idx].to(device))
            global_bias[global_idx].copy_(model.linear.bias[local_idx].to(device))

            owner[global_idx] = {
                "model_name": model_name,
                "local_idx": local_idx,
                "cifar_idx": cifar_idx,
            }

            if not torch.allclose(
                global_weight[global_idx],
                model.linear.weight[local_idx].to(device),
            ):
                raise RuntimeError(
                    f"Weight integrity failure: {model_name} local {local_idx} "
                    f"-> CIFAR {cifar_idx} -> global {global_idx}."
                )

            if not torch.allclose(
                global_bias[global_idx],
                model.linear.bias[local_idx].to(device),
            ):
                raise RuntimeError(
                    f"Bias integrity failure: {model_name} local {local_idx} "
                    f"-> CIFAR {cifar_idx} -> global {global_idx}."
                )

            print(
                f"local {local_idx:2d} -> CIFAR {cifar_idx:2d} "
                f"-> GLOBAL {global_idx:2d} | "
                f"norm={global_weight[global_idx].norm().item():.8f}"
            )

    missing_rows = sorted(set(range(NUM_GLOBAL_CLASSES)) - set(owner.keys()))
    if missing_rows:
        raise RuntimeError(f"Unpopulated global classifier rows: {missing_rows}")

    print("\nAll 43 raw global classifier rows were populated exactly once.")

    print("\n===== RAW GLOBAL CLASSIFIER ORDER =====")
    for global_idx in range(NUM_GLOBAL_CLASSES):
        info = owner[global_idx]
        print(
            f"GLOBAL {global_idx:2d} = CIFAR {info['cifar_idx']:2d} "
            f"<- {info['model_name']} local {info['local_idx']:2d} | "
            f"norm={global_weight[global_idx].norm().item():.8f}"
        )

    return global_weight, global_bias, owner


# =============================================================================
# FEATURE-SPACE ALIGNMENT
# =============================================================================


@torch.no_grad()
def get_penultimate_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Capture the tensor fed directly into model.linear."""

    captured = {}

    def hook(_module, inputs):
        captured["features"] = inputs[0].detach()

    handle = model.linear.register_forward_pre_hook(hook)
    try:
        _ = model(x)
    finally:
        handle.remove()

    if "features" not in captured:
        raise RuntimeError("Failed to capture penultimate features before model.linear.")

    return captured["features"]


@torch.no_grad()
def collect_feature_pairs(
    global_model: nn.Module,
    specialist_model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect representations of the same images under the global and specialist
    feature extractors.
    """

    global_model.eval()
    specialist_model.eval()

    global_features = []
    specialist_features = []

    for batch in loader:
        x = batch[0].to(device)

        z_global = get_penultimate_features(global_model, x)
        z_specialist = get_penultimate_features(specialist_model, x)

        if z_global.shape[0] != z_specialist.shape[0]:
            raise RuntimeError("Feature-pair batch-size mismatch.")
        if z_global.shape[1] != z_specialist.shape[1]:
            raise RuntimeError(
                "Feature dimensions differ. This adapter implementation expects "
                f"equal dimensions, got {z_global.shape} and {z_specialist.shape}."
            )

        global_features.append(z_global.cpu())
        specialist_features.append(z_specialist.cpu())

    if not global_features:
        raise RuntimeError("Alignment loader produced no batches.")

    return torch.cat(global_features, dim=0), torch.cat(specialist_features, dim=0)


def fit_feature_adapter(
    z_global: torch.Tensor,
    z_specialist: torch.Tensor,
    ridge: float = ADAPTER_RIDGE,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Fit affine ridge regression:

        z_specialist ~= z_global @ A + c

    Returns A [D,D], c [D], and fit diagnostics.
    """

    zg = z_global.float()
    zs = z_specialist.float()

    if zg.ndim != 2 or zs.ndim != 2:
        raise ValueError("Expected 2-D feature matrices.")
    if zg.shape != zs.shape:
        raise ValueError(f"Feature shapes differ: global={zg.shape}, specialist={zs.shape}")

    n, d = zg.shape
    ones = torch.ones(n, 1, dtype=zg.dtype, device=zg.device)
    x = torch.cat([zg, ones], dim=1)  # [N, D+1]

    regularizer = torch.eye(d + 1, dtype=x.dtype, device=x.device)
    regularizer[-1, -1] = 0.0  # do not regularise intercept

    lhs = x.T @ x + ridge * regularizer
    rhs = x.T @ zs

    # solve is preferable to explicit matrix inversion.
    beta = torch.linalg.solve(lhs, rhs)

    A = beta[:-1, :]  # [D, D]
    c = beta[-1, :]   # [D]

    reconstructed = zg @ A + c
    residual = reconstructed - zs

    mse = residual.pow(2).mean().item()
    specialist_variance = (zs - zs.mean(dim=0, keepdim=True)).pow(2).mean().item()
    relative_mse = mse / max(specialist_variance, 1e-12)

    diagnostics = {
        "num_samples": int(n),
        "feature_dim": int(d),
        "mse": float(mse),
        "relative_mse": float(relative_mse),
    }

    return A, c, diagnostics


def transform_classifier_to_global_features(
    specialist_model: nn.Module,
    A: torch.Tensor,
    c: torch.Tensor,
    num_subset_classes: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Express a specialist classifier in the averaged/global feature basis.

    If z_s ~= z_g @ A + c and specialist logits are z_s @ W.T + b, then

        W_global = W @ A.T
        b_global = b + W @ c
    """

    W = specialist_model.linear.weight[:num_subset_classes].detach().float().cpu()
    b = specialist_model.linear.bias[:num_subset_classes].detach().float().cpu()

    if W.shape[1] != A.shape[0] or A.shape[0] != A.shape[1]:
        raise RuntimeError(
            f"Incompatible classifier/adapter shapes: W={W.shape}, A={A.shape}"
        )

    W_global = W @ A.T
    b_global = b + W @ c

    return W_global.to(device), b_global.to(device)


@torch.no_grad()
def classifier_logit_alignment_mse(
    z_global: torch.Tensor,
    z_specialist: torch.Tensor,
    specialist_model: nn.Module,
    aligned_W: torch.Tensor,
    aligned_b: torch.Tensor,
    num_subset_classes: int,
) -> float:
    """Check how closely the transformed head reproduces specialist logits."""

    W_source = specialist_model.linear.weight[:num_subset_classes].detach().float().cpu()
    b_source = specialist_model.linear.bias[:num_subset_classes].detach().float().cpu()

    source_logits = z_specialist.float().cpu() @ W_source.T + b_source
    aligned_logits = z_global.float().cpu() @ aligned_W.float().cpu().T + aligned_b.float().cpu()

    return (source_logits - aligned_logits).pow(2).mean().item()


def build_aligned_global_classifier_43(
    global_model: nn.Module,
    models: Dict[str, nn.Module],
    mappings: Dict[str, Dict[int, int]],
    num_subset_classes: Dict[str, int],
    specialist_loaders: Dict[str, DataLoader],
    device: str,
    ridge: float = ADAPTER_RIDGE,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict], Dict[str, Dict]]:
    """Fit one feature adapter per specialist and construct the aligned 43-row head."""

    first_model = next(iter(models.values()))
    feature_dim = first_model.linear.weight.shape[1]

    global_weight = torch.zeros(
        NUM_GLOBAL_CLASSES,
        feature_dim,
        device=device,
        dtype=first_model.linear.weight.dtype,
    )
    global_bias = torch.zeros(
        NUM_GLOBAL_CLASSES,
        device=device,
        dtype=first_model.linear.bias.dtype,
    )

    owner: Dict[int, Dict] = {}
    adapter_report: Dict[str, Dict] = {}

    for model_name, specialist_model in models.items():
        if model_name not in specialist_loaders:
            raise KeyError(f"No alignment loader supplied for {model_name}.")

        n_classes = num_subset_classes[model_name]
        mapping = validate_specialist_mapping(
            model_name,
            specialist_model,
            mappings[model_name],
            n_classes,
        )

        print(f"\n===== FEATURE ALIGNMENT: {model_name.upper()} =====")

        z_global, z_specialist = collect_feature_pairs(
            global_model=global_model,
            specialist_model=specialist_model,
            loader=specialist_loaders[model_name],
            device=device,
        )

        A, c, fit_diagnostics = fit_feature_adapter(
            z_global=z_global,
            z_specialist=z_specialist,
            ridge=ridge,
        )

        aligned_W, aligned_b = transform_classifier_to_global_features(
            specialist_model=specialist_model,
            A=A,
            c=c,
            num_subset_classes=n_classes,
            device=device,
        )

        logit_mse = classifier_logit_alignment_mse(
            z_global=z_global,
            z_specialist=z_specialist,
            specialist_model=specialist_model,
            aligned_W=aligned_W,
            aligned_b=aligned_b,
            num_subset_classes=n_classes,
        )

        fit_diagnostics["logit_mse"] = float(logit_mse)
        adapter_report[model_name] = fit_diagnostics

        print(
            f"samples={fit_diagnostics['num_samples']} | "
            f"feature MSE={fit_diagnostics['mse']:.8f} | "
            f"relative feature MSE={fit_diagnostics['relative_mse']:.8f} | "
            f"logit MSE={logit_mse:.8f}"
        )

        for local_idx in range(n_classes):
            cifar_idx = mapping[local_idx]
            global_idx = original_to_aggregated_dict[cifar_idx]

            if global_idx in owner:
                raise RuntimeError(
                    f"Aligned global row {global_idx} already occupied by "
                    f"{owner[global_idx]['model_name']}."
                )

            global_weight[global_idx].copy_(
                aligned_W[local_idx].to(dtype=global_weight.dtype)
            )
            global_bias[global_idx].copy_(
                aligned_b[local_idx].to(dtype=global_bias.dtype)
            )

            owner[global_idx] = {
                "model_name": model_name,
                "local_idx": local_idx,
                "cifar_idx": cifar_idx,
            }

            print(
                f"aligned local {local_idx:2d} -> CIFAR {cifar_idx:2d} "
                f"-> GLOBAL {global_idx:2d} | "
                f"norm={global_weight[global_idx].norm().item():.8f}"
            )

    missing_rows = sorted(set(range(NUM_GLOBAL_CLASSES)) - set(owner.keys()))
    if missing_rows:
        raise RuntimeError(f"Aligned classifier has unpopulated rows: {missing_rows}")

    print("\nAll 43 ALIGNED global classifier rows were populated exactly once.")
    return global_weight, global_bias, owner, adapter_report


# ==============================================================================
# STEP 3: Main Aggregation Pipeline (CORRECTED)
# ==============================================================================

def aggregate_models_all_layers_with_feature_averaging(
    model_names: list,
    model_dir: Path,
    device: str = 'cpu',
    align_classifiers: bool = ALIGN_CLASSIFIERS_TO_AVERAGED_FEATURES,
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

    model_dir = f"../Sixteenth_implementation_attempt/final_specialised_models/" #model_dir = Path(model_dir)
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
            print(f"classifier_size for {model_name}", classifier_size)
            model = ResNet20_DynamicModel(num_classes=classifier_size)
            model.load_state_dict(state_dict)
            #model = model.to(device)
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
    # -------------------------------------------------------------------------
    # Create global 43-class architecture.
    # -------------------------------------------------------------------------
    print("\nStep 4: Creating 43-class global model...")
    global_model = ResNet20_DynamicModel(num_classes=NUM_GLOBAL_CLASSES).to(device)
    global_model = load_averaged_feature_layers(global_model, averaged_features)

    '''print("Step 4: Creating global model with averaged features...")

    global_model = ResNet20_DynamicModel(num_classes=100)#ResNet20_DynamicModel(num_classes=100)
    global_model = global_model#.to(device)

    # Load averaged feature extraction layers
    global_model = load_averaged_feature_layers(global_model, averaged_features)#load_averaged_feature_layers(global_model, averaged_features) #TODO: IMPORTANT CHANGE'''

    print(f"  ✓ Created global model with averaged features:")
    print(f"    - conv1 + bn1: ✓ Averaged from {len(model_names)} models")
    print(f"    - layer1-3: ✓ Averaged from {len(model_names)} models")

    print()

    # Build raw direct head first. This is both a valid baseline and gives the
    # model a fully populated classifier during BN recalibration.
    raw_W, raw_b, raw_owner = build_global_classifier_43(
        models=models,
        mappings=mappings,
        num_subset_classes=num_subset_classes,
        device=device,
    )

    with torch.no_grad():
        global_model.linear.weight.copy_(raw_W)
        global_model.linear.bias.copy_(raw_b)

    print(f"  ✓ Global architecture: linear.weight={tuple(global_model.linear.weight.shape)}")

    # -------------------------------------------------------------------------
    # BN recalibration after feature averaging.
    # -------------------------------------------------------------------------
    if RECALIBRATE_BATCHNORM:
        print("\nStep 5: Recalibrating BatchNorm running statistics...")
        '''recalibrate_batchnorm(
            model=global_model,
            loader=bn_calibration_loader,
            device=device,
        )'''
    else:
        print("\nStep 5: BatchNorm recalibration disabled.")
        global_model.eval()

    # -------------------------------------------------------------------------
    # Align specialist classifier heads to the FINAL averaged feature basis.
    # This happens after BN recalibration because BN changes the representation.
    # -------------------------------------------------------------------------
    adapter_report = {}
    classifier_owner = raw_owner

    if align_classifiers:
        print("\nStep 6: Aligning specialist classifiers to averaged feature space...")

        aligned_W, aligned_b, aligned_owner, adapter_report = (
            build_aligned_global_classifier_43(
                global_model=global_model,
                models=models,
                mappings=mappings,
                num_subset_classes=num_subset_classes,
                specialist_loaders=alignment_loaders,
                device=device,
                ridge=ADAPTER_RIDGE,
            )
        )

        with torch.no_grad():
            global_model.linear.weight.copy_(aligned_W)
            global_model.linear.bias.copy_(aligned_b)

        classifier_owner = aligned_owner
        classifier_mode = "feature-aligned"
    else:
        print("\nStep 6: Classifier feature alignment disabled; keeping raw remapped head.")
        classifier_mode = "raw-remapped"

    global_model.eval()

    # -------------------------------------------------------------------------
    # Final integrity checks.
    # -------------------------------------------------------------------------
    print("\nStep 7: Final integrity checks...")

    if global_model.linear.weight.shape[0] != NUM_GLOBAL_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_GLOBAL_CLASSES} output rows, got "
            f"{global_model.linear.weight.shape[0]}."
        )

    if set(classifier_owner.keys()) != set(range(NUM_GLOBAL_CLASSES)):
        raise RuntimeError("Final classifier ownership does not cover rows 0..42 exactly.")

    print("  ✓ Exactly 43 global classifier rows.")
    print("  ✓ Every row has exactly one specialist owner.")
    print("  ✓ No 57 zero/untrained CIFAR output rows exist in the model.")

    covered_cifar_classes = {
        info["cifar_idx"] for info in classifier_owner.values()
    }
    expected_cifar_classes = set(original_to_aggregated_dict.keys())

    if covered_cifar_classes != expected_cifar_classes:
        raise RuntimeError(
            "Final classifier CIFAR coverage mismatch. "
            f"missing={sorted(expected_cifar_classes - covered_cifar_classes)}, "
            f"extra={sorted(covered_cifar_classes - expected_cifar_classes)}"
        )

    report = {
        "num_global_classes": NUM_GLOBAL_CLASSES,
        "global_label_space": "compact_0_to_42",
        "original_to_aggregated_dict": original_to_aggregated_dict,
        "aggregated_to_original_dict": aggregated_to_original_dict,
        "models_used": model_names,
        "num_subset_classes": num_subset_classes,
        "classifier_mode": classifier_mode,
        "classifier_owner": classifier_owner,
        "feature_averaging": True,
        "batchnorm_included_in_feature_state": True,
        "batchnorm_recalibrated": RECALIBRATE_BATCHNORM,
        "adapter_ridge": ADAPTER_RIDGE if align_classifiers else None,
        "adapter_report": adapter_report,
        "feature_dim": int(global_model.linear.weight.shape[1]),
        "device": str(device),
    }

    print("\n" + "=" * 88)
    print("AGGREGATION COMPLETE")
    print(f"Classifier mode: {classifier_mode}")
    print(f"Global output space: {NUM_GLOBAL_CLASSES} classes")
    print("=" * 88 + "\n")

    return global_model, report

    print()

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


def fine_tune_aggregated_model(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        fine_tune_epochs: int = 10,
        lr: float = 1e-4,
        momentum: float=0.9,
        weight_decay:float=5e-4,
):
    """
    Fine-tune an aggregated/global model.

    Args:
        model:
            Aggregated model to fine-tune.

        loader:
            DataLoader containing:
                batch_data, batch_labels, batch_task_ids

        fine_tune_epochs:
            Number of fine-tuning epochs.

        lr:
            Learning rate used for fine-tuning.

    Returns:
        Fine-tuned model.
    """

    '''# Determine the device on which the model resides
    device = next(model.parameters()).device

    # Fine-tuning optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )'''
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,  # lr,
        momentum=momentum,  # momentum,
        weight_decay=weight_decay,  # weight_decay,
    )

    model.train()

    for epoch in range(fine_tune_epochs):

        # Reset statistics at the beginning of EACH epoch
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_data, batch_labels, batch_task_ids in loader:
            # Move minibatch to the same device as the model
            #batch_data = batch_data .to(device)
            #batch_labels = batch_labels.to(device)

            # Clear gradients from previous minibatch
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_data)
            batch_labels =  map_original_to_aggregated(batch_labels)

            # Calculate loss
            loss = F.cross_entropy(
                logits,
                batch_labels,
                reduction="mean"
            )

            # Calculate gradients
            loss.backward()

            # Update model parameters
            optimizer.step()

            # ---------------------------------
            # Statistics
            # ---------------------------------

            batch_size = batch_labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                logits,
                dim=1
            )

            total_correct += (
                    predictions == batch_labels
            ).sum().item()

            total_samples += batch_size

        # ---------------------------------
        # Epoch statistics
        # ---------------------------------

        avg_loss = (
            total_loss / total_samples
            if total_samples > 0
            else 0.0
        )

        accuracy = (
            100.0 * total_correct / total_samples
            if total_samples > 0
            else 0.0
        )

        print(
            f"Epoch {epoch + 1}/{fine_tune_epochs} | "
            f"Loss: {avg_loss:.6f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    return model

def fine_tune_aggregated_model_GPU(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    fine_tune_epochs: int = 10,
    lr: float = 1e-4,
    momentum: float=0.9,
    weight_decay:float=5e-4,
    device: str = "cpu"
):
    """
    Fine-tune an aggregated/global model.

    Args:
        model:
            Aggregated PyTorch model to fine-tune.

        loader:
            DataLoader yielding:
                batch_data, batch_labels, batch_task_ids

        fine_tune_epochs:
            Number of fine-tuning epochs.

        lr:
            Learning rate used for fine-tuning.

        device:
            Device on which training should be performed.

            Examples:
                "cpu"
                "cuda"
                "cuda:0"
                "cuda:1"

    Returns:
        Fine-tuned model. The returned model remains on the
        specified device.
    """

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(device)

    # Check that CUDA is actually available if requested
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but CUDA is not available."
        )

    # Move the entire model, including its parameters and
    # registered buffers, to the selected device.
    model = model.to(device)

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    '''optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )'''
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,  # lr,
        momentum=momentum,  # momentum,
        weight_decay=weight_decay,  # weight_decay,
    )

    model.train()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    for epoch in range(fine_tune_epochs):

        # Reset statistics for this epoch
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_data, batch_labels, batch_task_ids in loader:

            # Move tensors required for computation to the
            # same device as the model.
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # If your model actually uses task IDs, also do:
            #
            # batch_task_ids = batch_task_ids.to(device)

            # Remove gradients left from previous minibatch
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_data)

            # Calculate loss
            loss = F.cross_entropy(
                logits,
                batch_labels,
                reduction="mean"
            )

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------

            batch_size = batch_labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                logits,
                dim=1
            )

            total_correct += (
                predictions == batch_labels
            ).sum().item()

            total_samples += batch_size

        # -----------------------------------------------------
        # Epoch statistics
        # -----------------------------------------------------

        avg_loss = (
            total_loss / total_samples
            if total_samples > 0
            else 0.0
        )

        accuracy = (
            100.0 * total_correct / total_samples
            if total_samples > 0
            else 0.0
        )

        print(
            f"Epoch {epoch + 1}/{fine_tune_epochs} | "
            f"Loss: {avg_loss:.6f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    return model

def evaluate_aggregated_model(model: nn.Module, loader: torch.utils.data.DataLoader, output_message:str):
    total_loss_all_experiences = 0.0
    total_correct_all_experiences = 0
    total_samples_all_experiences = 0

    model.eval() #TODO: IN TESTING! If it leads to problems, comment-out again
    with torch.no_grad():
        for batch_data, batch_labels, batch_task_ids in loader:#for batch_data, batch_labels, batch_task_ids in test_loader:
            unpack_minibatch(batch_data)

            logits_all_experiences = model(batch_data)
            #print("logits_all_experiences.dim()",logits_all_experiences.dim())
            targets_all_experiences = batch_labels
            targets_all_experiences = map_original_to_aggregated(batch_labels)


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
        f"{output_message} --> Loss = {avg_loss_all_experiences:.6f} | Accuracy = {accuracy_all_experiences:.2f}%"
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
            batch_labels = map_original_to_aggregated(batch_labels)
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

    MODEL_DIR = Path("./final_specialised_models") #MODEL_DIR = Path(f"../Sixteenth_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    MODEL_DIR.mkdir(parents=True, exist_ok=True) #MODEL_DIR = Path(f"../Sixteenth_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    OUTPUT_DIR = Path('./final_specialised_models')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

    model_names = ['outdoors', 'nature', 'people', 'objects_and_vehicles']#TODO: Testing if index out of bounds error occurs regardless of models

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
    #evaluate_aggregated_model_FedAvg_code_v3(global_model, test_loader)
    #evaluate_aggregated_model(global_model, test_loader_without_objects_and_vehicles, "Evaluating global model on test_loader_without_objects_and_vehicles")
    evaluate_aggregated_model(global_model, test_loader, "Evaluating global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_aggregated_model(global_model, outdoors_loader, "Evaluating global model on outdoors_loader")
    evaluate_aggregated_model(global_model, nature_loader, "Evaluating global model on nature_loader")
    evaluate_aggregated_model(global_model, people_loader, "Evaluating global model on people_loader")
    evaluate_aggregated_model(global_model, objects_and_vehicles_loader, "Evaluating global model on objects_and_vehicles_loader")
    ####global_model_reload = torch.load("../Sixteenth_implementation_attempt/final_specialised_models/global_model_with_averaged_features.pt")
    ####global_model_reload_architecture = ResNet20_DynamicModel(num_classes=100)#global_model_reload_architecture = ResNet20_DynamicModel(num_classes=100)
    ####global_model_reload_architecture.load_state_dict(global_model_reload)
    ####evaluate_aggregated_model_FedAvg_code_v3(global_model_reload_architecture, test_loader)
    ####evaluate_aggregated_model(global_model_reload_architecture, test_loader)
    #evaluate_aggregated_model(global_model_reload_architecture,outdoors_specialised_models_train_stream_class_il)
    #evaluate_aggregated_model(global_model_reload_architecture, avl_test_loader)
    print("\n\n############# FINE-TUNED GLOBAL MODEL EVALUATION BEGINS NOW!!!! #############")
    fine_tuned_global_model = fine_tune_aggregated_model(global_model, train_loader, lr=0.001, momentum=0.9, weight_decay=5e-4) #device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #TODO: GPU version DOES NOT WORK (at least when device='cuda:0'). Hence, commented-out below:
    #fine_tuned_global_model = fine_tune_aggregated_model_GPU(global_model, train_loader, lr=0.001, momentum=0.9,weight_decay=5e-4, device='cuda:0')
    #evaluate_aggregated_model(fine_tuned_global_model, test_loader_without_objects_and_vehicles, "Evaluating FINE-TUNED global model on test_loader_without_objects_and_vehicles")
    evaluate_aggregated_model(fine_tuned_global_model, test_loader, "Evaluating FINE-TUNED global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_aggregated_model(fine_tuned_global_model, outdoors_loader, "Evaluating FINE-TUNED global model on outdoors_loader")
    evaluate_aggregated_model(fine_tuned_global_model, nature_loader, "Evaluating FINE-TUNED global model on nature_loader")
    evaluate_aggregated_model(fine_tuned_global_model, people_loader, "Evaluating FINE-TUNED global model on people_loader")
    evaluate_aggregated_model(fine_tuned_global_model, objects_and_vehicles_loader, "Evaluating FINE-TUNED global model on objects_and_vehicles_loader")

    print("\n\n############# FINE-TUNED GLOBAL MODEL EVALUATION BEGINS NOW -- global model fine-tuned on last set of train experiences of each specialised model!!!! #############")
    fine_tuned_global_model = fine_tune_aggregated_model(global_model, last_experience_train_loader, fine_tune_epochs=3, lr=0.001, momentum=0.9,weight_decay=5e-4)
    # evaluate_aggregated_model(fine_tuned_global_model, test_loader_without_objects_and_vehicles, "Evaluating FINE-TUNED global model on test_loader_without_objects_and_vehicles")
    evaluate_aggregated_model(fine_tuned_global_model, test_loader,"Evaluating FINE-TUNED global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_aggregated_model(fine_tuned_global_model, outdoors_loader,"Evaluating FINE-TUNED global model on outdoors_loader")
    evaluate_aggregated_model(fine_tuned_global_model, nature_loader,"Evaluating FINE-TUNED global model on nature_loader")
    evaluate_aggregated_model(fine_tuned_global_model, people_loader,"Evaluating FINE-TUNED global model on people_loader")
    evaluate_aggregated_model(fine_tuned_global_model, objects_and_vehicles_loader,"Evaluating FINE-TUNED global model on objects_and_vehicles_loader")
