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
from Report_implementation_attempt.ResNet20_DynamicModel_vReport import ResNet20_DynamicModel
#from Report_implementation_attempt.ResNet20_DynamicModel_Compact_vReport import ResNet20_XL
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
from Report_implementation_attempt.Train_ResNet20_Outdoors_LwF_vReport import outdoors_specialised_models_test_stream_class_il, outdoors_specialised_models_train_stream_class_il
from Report_implementation_attempt.Train_ResNet20_Nature_LwF_vReport import nature_specialised_models_test_stream_class_il, nature_specialised_models_train_stream_class_il
from Report_implementation_attempt.Train_ResNet20_People_LwF_vReport import people_specialised_models_test_stream_class_il, people_specialised_models_train_stream_class_il
from Report_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_vReport import objects_and_vehicles_specialised_models_test_stream_class_il, objects_and_vehicles_specialised_models_train_stream_class_il
from Report_implementation_attempt.Load_cifar100_vReport import avl_test
#TODO: TEMPORARY!!!! Change to that of vReport once I have final Objects_and_Vehicles model!!!!!
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


combined_dataset_outdoors_last_experience_test = ConcatDataset([
    #outdoors_specialised_models_test_stream_class_il[0].dataset,
    #outdoors_specialised_models_test_stream_class_il[1].dataset,
    outdoors_specialised_models_test_stream_class_il[2].dataset,
    #outdoors_specialised_models_test_stream_class_il[3].dataset,
    #outdoors_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_nature_last_experience_test = ConcatDataset([
    #nature_specialised_models_test_stream_class_il[0].dataset,
    #nature_specialised_models_test_stream_class_il[1].dataset,
    nature_specialised_models_test_stream_class_il[2].dataset,
    #nature_specialised_models_test_stream_class_il[3].dataset,
    #nature_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_people_last_experience_test = ConcatDataset([
    #people_specialised_models_test_stream_class_il[0].dataset,
    #people_specialised_models_test_stream_class_il[1].dataset,
    people_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_last_experience_test = ConcatDataset([
    #objects_and_vehicles_specialised_models_test_stream_class_il[0].dataset,
    #objects_and_vehicles_specialised_models_test_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_last_experience_test = ConcatDataset([
    combined_dataset_outdoors_last_experience_test,
    combined_dataset_nature_last_experience_test,
    combined_dataset_people_last_experience_test,
    combined_dataset_objects_and_vehicles_last_experience_test
])

last_experience_test_loader = DataLoader(
    combined_dataset_last_experience_test,
    batch_size=32,
    shuffle=False
)
last_experience_test_outdoors = DataLoader(
    combined_dataset_outdoors_last_experience_test,
    batch_size=32,
    shuffle=False
)
last_experience_test_nature = DataLoader(
    combined_dataset_nature_last_experience_test,
    batch_size=32,
    shuffle=False
)
last_experience_test_people = DataLoader(
    combined_dataset_people_last_experience_test,
    batch_size=32,
    shuffle=False
)
last_experience_test_objects_and_vehicles = DataLoader(
    combined_dataset_objects_and_vehicles_last_experience_test,
    batch_size=32,
    shuffle=False
)
print("print last_experience_test_loader", last_experience_test_loader)
print("print last_experience_test_loader.dataset", last_experience_test_loader.dataset)

combined_dataset_outdoors_middle_experience_test = ConcatDataset([
    #outdoors_specialised_models_test_stream_class_il[0].dataset,
    outdoors_specialised_models_test_stream_class_il[1].dataset,
    #outdoors_specialised_models_test_stream_class_il[2].dataset,
    #outdoors_specialised_models_test_stream_class_il[3].dataset,
    #outdoors_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_nature_middle_experience_test = ConcatDataset([
    #nature_specialised_models_test_stream_class_il[0].dataset,
    nature_specialised_models_test_stream_class_il[1].dataset,
    #nature_specialised_models_test_stream_class_il[2].dataset,
    #nature_specialised_models_test_stream_class_il[3].dataset,
    #nature_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_people_middle_experience_test = ConcatDataset([
    #people_specialised_models_test_stream_class_il[0].dataset,
    people_specialised_models_test_stream_class_il[1].dataset,
    #people_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_middle_experience_test = ConcatDataset([
    #objects_and_vehicles_specialised_models_test_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_test_stream_class_il[1].dataset,
    #objects_and_vehicles_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_middle_experience_test = ConcatDataset([
    combined_dataset_outdoors_middle_experience_test,
    combined_dataset_nature_middle_experience_test,
    combined_dataset_people_middle_experience_test,
    combined_dataset_objects_and_vehicles_middle_experience_test
])

middle_experience_test_loader = DataLoader(
    combined_dataset_middle_experience_test,
    batch_size=32,
    shuffle=False
)
middle_experience_test_outdoors = DataLoader(
    combined_dataset_outdoors_middle_experience_test,
    batch_size=32,
    shuffle=False
)
middle_experience_test_nature = DataLoader(
    combined_dataset_nature_middle_experience_test,
    batch_size=32,
    shuffle=False
)
middle_experience_test_people = DataLoader(
    combined_dataset_people_middle_experience_test,
    batch_size=32,
    shuffle=False
)
middle_experience_test_objects_and_vehicles = DataLoader(
    combined_dataset_objects_and_vehicles_middle_experience_test,
    batch_size=32,
    shuffle=False
)

combined_dataset_outdoors_first_experience_test = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[0].dataset,
    #outdoors_specialised_models_test_stream_class_il[1].dataset,
    #outdoors_specialised_models_test_stream_class_il[2].dataset,
    #outdoors_specialised_models_test_stream_class_il[3].dataset,
    #outdoors_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_nature_first_experience_test = ConcatDataset([
    nature_specialised_models_test_stream_class_il[0].dataset,
    #nature_specialised_models_test_stream_class_il[1].dataset,
    #nature_specialised_models_test_stream_class_il[2].dataset,
    #nature_specialised_models_test_stream_class_il[3].dataset,
    #nature_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_people_first_experience_test = ConcatDataset([
    people_specialised_models_test_stream_class_il[0].dataset,
    #people_specialised_models_test_stream_class_il[1].dataset,
    #people_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_first_experience_test = ConcatDataset([
    objects_and_vehicles_specialised_models_test_stream_class_il[0].dataset,
    #objects_and_vehicles_specialised_models_test_stream_class_il[1].dataset,
    #objects_and_vehicles_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_first_experience_test = ConcatDataset([
    combined_dataset_outdoors_first_experience_test,
    combined_dataset_nature_first_experience_test,
    combined_dataset_people_first_experience_test,
    combined_dataset_objects_and_vehicles_first_experience_test
])

first_experience_test_loader = DataLoader(
    combined_dataset_first_experience_test,
    batch_size=32,
    shuffle=False
)
first_experience_test_outdoors = DataLoader(
    combined_dataset_outdoors_first_experience_test,
    batch_size=32,
    shuffle=False
)
first_experience_test_nature = DataLoader(
    combined_dataset_nature_first_experience_test,
    batch_size=32,
    shuffle=False
)
first_experience_test_people = DataLoader(
    combined_dataset_people_first_experience_test,
    batch_size=32,
    shuffle=False
)
first_experience_test_objects_and_vehicles = DataLoader(
    combined_dataset_objects_and_vehicles_first_experience_test,
    batch_size=32,
    shuffle=False
)
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
# STEP 1: Feature Extractor Averaging (NEW APPROACH)
# ==============================================================================

def extract_feature_layers(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract all feature extraction layers (everything except classifier) and add it to  feature_layers dict.

    """
    state_dict = model.state_dict()

    feature_layers = OrderedDict()

    EXCLUDE_STATISTICS = True

    if EXCLUDE_STATISTICS == True:
        # Copy all residual block layers (layer1, layer2, layer3)
        for key in state_dict.keys():
            if "linear" not in key and not key.endswith("running_mean") and not key.endswith("running_var") and not key.endswith("num_batches_tracked"):#if "linear" not in key and "bn" not in key and not key.endswith("running_mean") and not key.endswith("running_var") and not key.endswith("num_batches_tracked"): #if key.startswith('layer') or key.endswith("wight") or key.endswith("bias") and "linear" not in key:  # layer1, layer2, layer3
                feature_layers[key] = state_dict[key].clone()
        print("feature_layers.keys()",feature_layers.keys())

    if EXCLUDE_STATISTICS == False:
        # Copy all residual block layers (layer1, layer2, layer3)
        for key in state_dict.keys():
            if "linear" not in key:#if "linear" not in key and "bn" not in key and not key.endswith("running_mean") and not key.endswith("running_var") and not key.endswith("num_batches_tracked"): #if key.startswith('layer') or key.endswith("wight") or key.endswith("bias") and "linear" not in key:  # layer1, layer2, layer3
                feature_layers[key] = state_dict[key].clone()
        print("feature_layers.keys()",feature_layers.keys())

    return feature_layers


def average_feature_layers(features_list: list) -> Dict[str, torch.Tensor]:
    """
    Average feature extraction layers across all models by averaging corresponding parameters (element-wise) of corresponding layers.
    It takes as input the feature_layers dictionary produced by extract_feature_layers() and returns dict containing aggregated layers for feature extractor.

    """
    DATA_DRIVEN = True
    if DATA_DRIVEN == True:
        if not features_list:
            raise ValueError("No feature layers to average")

        num_models = len(features_list)

        training_samples_per_model = [2400, 3600, 1200, 10000]

        if len(training_samples_per_model) != num_models:
            raise ValueError(
                f"Number of training-sample counts "
                f"({len(training_samples_per_model)}) does not match "
                f"number of models ({num_models})"
            )

        averaged_features = OrderedDict()

        # Average each layer across all models
        for key in features_list[0].keys():
            # Stack all models' values for this layer.
            # Dimension 0 is now the MODEL dimension.
            stacked = torch.stack(
                [features[key].float() for features in features_list],
                dim=0
            )

            print("key:", key)
            print("stacked.shape:", stacked.shape)
            print("stacked.dim():", stacked.dim())

            # Create sample-count weights on same device/dtype as parameters
            weights = torch.tensor(
                training_samples_per_model,
                dtype=stacked.dtype,
                device=stacked.device
            )

            # Normalize according to training-set size
            weights = weights / weights.sum()

            print("weights:", weights)

            # Reshape weights so that they apply ONLY across dim=0, which is where the aggregated feature extractor is,
            # given dim=0 in torch.stack
            weight_shape = [num_models] + [1] * (stacked.dim() - 1)

            weights = weights.view(*weight_shape)

            # Weighted average across MODEL dimension (dim=0)
            averaged = torch.sum(
                stacked * weights,
                dim=0
            )

            # Preserve original dtype
            averaged_features[key] = averaged.to(
                features_list[0][key].dtype
            )

    if DATA_DRIVEN == False:
        if not features_list:
            raise ValueError("No feature layers to average")

        #num_models = len(features_list)
        averaged_features = OrderedDict()

        # Average each layer across all models
        for key in features_list[0].keys():
            # Stack all models' values for this layer
            stacked = torch.stack([features[key].float() for features in features_list])
            print("print(stacked.shape)",stacked.shape)
            print("stacked.dim()",stacked.dim())

            # Take mean across models (dim=0)
            averaged = torch.mean(stacked, dim=0)

            # Preserve original dtype
            averaged_features[key] = averaged.to(features_list[0][key].dtype)

    return averaged_features


def load_averaged_feature_layers(global_model: nn.Module,
                                 averaged_features: Dict[str, torch.Tensor]):
    """
    Load dict containing aggregated layers for feature extractor into the ResNet20_Dynamic architecture.
    """
    global_state_dict = global_model.state_dict()

    # Update global model with averaged features
    for key, value in averaged_features.items():
        if key in global_state_dict:
            global_state_dict[key] = value

    global_model.load_state_dict(global_state_dict)

    return global_model #TODO: IMPORTANT CHANGE. I added this


# Classifier Remapping

original_to_aggregated_dict = {17: 0, 12: 1, 33: 2, 23: 3, 49: 4, 37: 5, #OUTDOORS
                               51: 6, 47: 7, 0: 8, 54: 9, 53: 10, 52: 11, 59: 12, 57: 13, 56: 14, #NATURE
                               2: 15, 11: 16, 35: 17, #PEOPLE
                               16: 18, 13: 19, 10: 20, 9: 21, 8: 22, 5: 23, 58: 24, 48: 25, 41: 26, 40: 27, 39: 28, 28: 29, 25: 30, 22: 31, 20: 32, 94: 33, #OBJECTS_AND_VEHICLES
                               90: 34, 89: 35, 87: 36, 86: 37, 85: 38, 84: 39, 81: 40, 69: 41, 61: 42} #OBJECTS_AND_VEHICLES'''

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

    remapped_weights_aggregated = torch.zeros(
        43,#num_subset_classes,
        feature_dim,
        device=device,
        dtype=model.linear.weight.dtype #dtype=classifier.weight.dtype
    )

    remapped_bias_aggregated = torch.zeros(
        43, #num_subset_classes,
        device=device,
        dtype=model.linear.bias.dtype  #dtype=classifier.bias.dtype
    )

    #local_idx_count = 0
    # Place each trained class at its CIFAR-100 position
    for cifar_class, aggregated in original_to_aggregated_dict.items():
        print("original_to_aggregated_dict", original_to_aggregated_dict)
        print("print(cifar_class, aggregated)", cifar_class, aggregated)
        #local_idx_count += 1
        #print("print(local_idx_count)", local_idx_count)
        #print("print(num_subset_classes)", num_subset_classes)
        #if local_idx_count > num_subset_classes:
        #    print("print(num_subset_classes) THAT CAUSED BREAK", num_subset_classes)
        #    break
        # cifar_class: 12, 17, 23, ... (actual CIFAR class)
        # aggregated: 0-5 for outdoors (position in AGGREGATED classifier)
        print(f"print(aggregated = original_to_aggregated_dict[cifar_class]) --> {aggregated} (for CIFAR-100 class {cifar_class})")
        print(f"print(remapped_weights[cifar_class] for {model_name})", remapped_weights[cifar_class])
        remapped_weights_aggregated[aggregated] = remapped_weights[cifar_class].to(device)
        remapped_bias_aggregated[aggregated] = remapped_bias[cifar_class].to(device)

        print("print(remapped_weights_aggregated.shape, len(remapped_weights_aggregated))",remapped_weights_aggregated.shape, len(remapped_weights_aggregated))

    return remapped_weights_aggregated, remapped_bias_aggregated, #ORIGINAL remapped_weights, remapped_bias


# Main process -- calls functions to aggregate feature extractor layers, load all models, extract feature layers from each model, aggregate feature extractor layers, load averaged
# features into global model, remap and combine classifiers into global classifier for global model

def aggregate_models_all_layers_with_feature_averaging(
    model_names: list,
    model_dir: Path,
    device: str = 'cpu'
) -> nn.Module:
    """
    Main process -- calls functions to aggregate feature extractor layers, load all models, extract feature layers from each model, aggregate feature extractor layers, load averaged
    features into global model, remap and combine classifiers into global classifier for global model
    """

    print("\n" + "="*80)
    print("MAIN PROCESS -- aggregate feature extractor layers, load all models, extract feature layers from each model, aggregate feature extractor layers, load averaged features into"
          " global model, remap and combine classifiers into global classifier for global model")
    print("="*80 + "\n")

    model_dir = f"../Report_implementation_attempt/final_specialised_models/" #model_dir = Path(model_dir)
    models = {}
    mappings = {}
    num_subset_classes = {}


    # Load all models and their mappings

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

            print(f"  Loaded {model_name}")
            print(f"    - Classifier size: {classifier_size}")
            print(f"    - CIFAR classes: {sorted(mappings[model_name].values())}")

        except Exception as e:
            print(f"  Failed to load {model_name}: {e}")
            raise

    print()

    # Extract features from ALL models

    print("Step 2: Extracting feature layers from ALL models...")

    all_feature_layers = []
    feature_layer_keys = None

    for model_name in model_names:
        model = models[model_name]
        features = extract_feature_layers(model)
        all_feature_layers.append(features)

        if feature_layer_keys is None:
            feature_layer_keys = set(features.keys())

    averaged_features = average_feature_layers(all_feature_layers)

    print()


    # Create global model and load averaged features onto its layers

    print("Step 4: Creating global model with averaged features...")

    global_model = ResNet20_DynamicModel(num_classes=43)#ResNet20_DynamicModel(num_classes=100)
    global_model = global_model#.to(device)

    # Load averaged feature extraction layers
    global_model = load_averaged_feature_layers(global_model, averaged_features)#load_averaged_feature_layers(global_model, averaged_features) #TODO: IMPORTANT CHANGE
    #print("global_model.linear.shape", global_model.linear.shape)
    for name, param in global_model.named_parameters():
        print(f"Parameter Name: {name}, Parameter Shape: {param.shape}")

    print()

    # ==================================================================
    # STEP 5: Remap and combine classifiers
    # ==================================================================

    print("Step 5: Remapping classifiers to CIFAR-100 labels and from CIFARR-100 to aggregated labels such that classifier is of appropriate size...")
    with torch.no_grad():
        feature_dim = models[model_names[0]].linear.weight.shape[1]

        global_classifier_weights = torch.zeros(
            43, feature_dim,
            device=device,
            dtype=global_model.linear.weight.dtype  # ORIGINALLY: #global_classifier_weights = torch.zeros(100, feature_dim, device=device)
        )
        global_classifier_bias = torch.zeros(
            43,
            device=device,
            dtype=global_model.linear.bias.dtype  # ORIGINALLY: #global_classifier_bias = torch.zeros(100, device=device)
        )
        #global_classifier_weights = global_model.linear.weight
        print("global_classifier_weights INSTEAD OF torch.zero\n", global_classifier_weights)
        print("global_classifier_weights.shape INSTEAD OF torch.zero \n", global_classifier_weights.shape)
        #global_classifier_bias = global_model.linear.bias
        print("global_classifier_bias INSTEAD OF torch.zero\n", global_classifier_bias)
        print("global_classifier_bias.shape INSTEAD OF torch.zero \n", global_classifier_bias.shape)

        all_weights = {}
        all_bias = {}

        for model_name in model_names:
            model = models[model_name]
            mapping = mappings[model_name]

            w, b = remap_classifier_to_cifar100(
                model_name, model, mapping, num_subset_classes[model_name], num_classes=100, device=device
            )
            print(f"w to see what it looks like after additions ({model_name}):\n", w)
            print(f"b to see what it looks like after additions ({model_name}):\n", b)

            '''row_norms = w.norm(p=2, dim=1)

            nonzero_rows = torch.where(row_norms > 1e-12)[0]

            print(f"\n{model_name} non-zero classifier rows:")
            print(nonzero_rows.tolist())

            for row in nonzero_rows.tolist():
                print(
                    f"CIFAR row {row}: "
                    f"norm={row_norms[row].item():.8f}, "
                    f"first values={w[row, :8]}"
                )

            print("\nSOURCE classifier:")
            for local_idx in range(model.linear.weight.shape[0]):
                print(
                    local_idx,
                    model.linear.weight[local_idx].norm().item()
                )

            print("\nREMAPPED classifier:")
            for cifar_idx in [12, 17, 23, 33, 37, 49]:
                print(
                    cifar_idx,
                    w[cifar_idx].norm().item()
                )'''

            all_weights[model_name] = w
            all_bias[model_name] =b

            print("all_weights",all_weights)

            # TODO: Back to original code for global_classifier_weights += w and global_classifier_bias += b. This and the version I developed seem to do the same!!!
            # TODO: This is the version I developed
            global_classifier_weights = torch.stack([m.float() for m in all_weights.values()]).sum(dim=0)
            global_classifier_bias = torch.stack([b.float() for b in all_bias.values()]).sum(dim=0)

            print(f"global_weights += w to see what it looks like after additions ({model_name}):\n",global_classifier_weights)
            print(f"global_bias += b to see what it looks like after additions ({model_name}):\n", global_classifier_bias)

            print("print(global_classifier_weights.shape)",global_classifier_weights.shape)
            print("global_classifier_bias.shape",global_classifier_bias.shape)

            #global_classifier_weights = global_classifier_weights
            #global_classifier_bias = global_classifier_bias


    print()

    # ==================================================================
    # STEP 6: Load remapped classifier into global model
    # ==================================================================

    print("Step 6: Loading remapped classifier into global model...")

    global_model.linear.weight = nn.Parameter(global_classifier_weights)
    global_model.linear.bias = nn.Parameter(global_classifier_bias)

    test={}
    for layer_name, layer in global_model.state_dict().items():
        print(f"{layer_name}.shape:",layer.shape)
        test[layer_name]=layer.shape

    ###print("global_model.state_dict() AFTER nn.Parameter",global_model.state_dict())
    ###print("global_model.linear.state_dict() AFTER nn.Parameter", global_model.linear.state_dict())

    for layer_name, layer in global_model.state_dict().items():
        #print(f"{layer_name}.shape AFTER nn.Parameter:",layer.shape)
        if test[layer_name] == layer.shape:
            print(f"{test[layer_name]} EQUAL SHAPE {layer.shape}")
        if test[layer_name] != layer.shape:
            print(f"{test[layer_name]} NOT equal SHAPE {layer.shape}")

    print(f"  ✓ Classifier loaded: {global_model.linear.weight.shape}")

    print()

    print("\n" + "="*80 + "\n")

    return global_model


# ==============================================================================
# STEP 4: Save and Test
# ==============================================================================

def save_global_model(global_model: nn.Module,
                      output_path: Path):
    """Save the global model and aggregation report."""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = output_path / 'summed_features_global_model.pt'
    torch.save(global_model.state_dict(), model_path)
    print(f"✓ Saved global model to {model_path}")



def fine_tune_classifier(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        fine_tune_epochs: int = 10,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
):
    """
    Fine-tune ONLY the classifier of an aggregated/global model.

    The feature extractor is frozen and remains unchanged.

    Args:
        model:
            Aggregated model to fine-tune.

        loader:
            DataLoader containing:
                batch_data, batch_labels, batch_task_ids

        fine_tune_epochs:
            Number of fine-tuning epochs.

        lr:
            Learning rate used for classifier fine-tuning.

        momentum:
            SGD momentum.

        weight_decay:
            SGD weight decay.

    Returns:
        Model with fine-tuned classifier and frozen feature extractor.
    """
    loss_counter=0
    prev_loss=100
    prev_accuracy= 0

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = next(model.parameters()).device

    # ---------------------------------------------------------
    # FREEZE FEATURE EXTRACTOR
    # ---------------------------------------------------------
    for param_name, param in model.named_parameters(): #for param in model.feature_extractor.parameters():
        if "linear" not in param_name:
            param.requires_grad = False
        #print("print(param) of if linear not in param_name:", param_name)

    # ---------------------------------------------------------
    # Ensure classifier remains trainable
    # ---------------------------------------------------------
    for param in model.linear.parameters():
        param.requires_grad = True

    # ---------------------------------------------------------
    # Optimizer: ONLY classifier parameters
    # ---------------------------------------------------------
    optimizer = torch.optim.SGD(
        model.linear.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # ---------------------------------------------------------
    # Training modes
    # ---------------------------------------------------------

    # Put whole model into training mode first
    #model.train()

    # But keep frozen feature extractor in eval mode.
    #
    # This is important for BatchNorm / Dropout:
    # BatchNorm running statistics should not change inside
    # the frozen feature extractor.
    #model.feature_extractor.eval()

    # Classifier remains in training mode
    model.linear.train()

    # ---------------------------------------------------------
    # Fine-tuning
    # ---------------------------------------------------------
    for epoch in range(fine_tune_epochs):

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_data, batch_labels, batch_task_ids in loader:

            # Move minibatch to model device
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            # Clear classifier gradients
            optimizer.zero_grad()

            # -------------------------------------------------
            # Forward pass
            # -------------------------------------------------
            logits = model(batch_data)
            batch_labels = map_original_to_aggregated(batch_labels)

            # -------------------------------------------------
            # Loss
            # -------------------------------------------------
            loss = F.cross_entropy(
                logits,
                batch_labels,
                reduction="mean"
            )

            # -------------------------------------------------
            # Backpropagation
            # -------------------------------------------------
            loss.backward()

            # ONLY classifier parameters are updated because
            # the optimizer contains only classifier parameters.
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
            f"Accuracy: {accuracy:.2f}% | "
            f"lr: {lr}"
        )
        if epoch < 12:
            lr = lr * 0.1
        if epoch > 12:
            lr = lr * 0.1 * 0.1 * 0.1
        if epoch == 19:
            # ---------------------------------------------------------
            print("FEATURE EXTRACTOR UNFROZEN")
            # ---------------------------------------------------------
            for param_name, param in model.named_parameters():  # for param in model.feature_extractor.parameters():
                if "linear" not in param_name:
                    param.requires_grad = True
                # print("print(param) of if linear not in param_name:", param_name)

            # ---------------------------------------------------------
            # Ensure classifier remains trainable
            # ---------------------------------------------------------
            for param in model.linear.parameters():
                param.requires_grad = True

            # ---------------------------------------------------------
            # Optimizer: ONLY classifier parameters
            # ---------------------------------------------------------
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        if avg_loss >= prev_loss : #and accuracy > prev_accuracy:
            loss_counter += 1
            if loss_counter == 3:
                lr = lr * 0.1
                prev_loss = avg_loss
                prev_accuracy = accuracy
            if loss_counter == 6 and accuracy > prev_accuracy:
                lr = lr * 0.1
                prev_loss = avg_loss
                prev_accuracy = accuracy
            if loss_counter == 6 and accuracy <= prev_accuracy:
                prev_loss = avg_loss
                prev_accuracy = accuracy
                break

    return model

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

    loss_counter=0
    prev_loss=100
    prev_accuracy= 0
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
            batch_labels = map_original_to_aggregated(batch_labels)

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
            f"lr: {lr:.6f}"
        )
        if avg_loss >= prev_loss : #and accuracy > prev_accuracy:
            loss_counter += 1
            if loss_counter == 3:
                lr = lr * 0.1
                prev_loss = avg_loss
                prev_accuracy = accuracy
            if loss_counter == 6 and accuracy > prev_accuracy:
                lr = lr * 0.1
                prev_loss = avg_loss
                prev_accuracy = accuracy
            if loss_counter == 6 and accuracy <= prev_accuracy:
                prev_loss = avg_loss
                prev_accuracy = accuracy
                break


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
            #unpack_minibatch(batch_data)

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
        f"{output_message} --> Loss = {avg_loss_all_experiences:.6f} | Accuracy = {accuracy_all_experiences:.2f}% [correct:{total_correct_all_experiences}, total:{total_samples_all_experiences}. {total_correct_all_experiences} of {total_samples_all_experiences} are correct]"
    )

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Complete pipeline with feature averaging.
    """

    MODEL_DIR = Path("./final_specialised_models") #MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    MODEL_DIR.mkdir(parents=True, exist_ok=True) #MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    OUTPUT_DIR = Path('./final_specialised_models')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

    model_names = ['outdoors', 'nature', 'people', 'objects_and_vehicles']#TODO: Testing if index out of bounds error occurs regardless of models

    try:
        # ============================================================
        # Aggregate models with feature averaging
        # ============================================================
        global_model = aggregate_models_all_layers_with_feature_averaging(
            model_names=model_names,
            model_dir=MODEL_DIR,
            device=DEVICE
        )

        # ============================================================
        # Save
        # ============================================================
        save_global_model(global_model, OUTPUT_DIR)


        print("="*80)
        print(f"\nModel saved to: {OUTPUT_DIR / 'summed_features_global_model.pt'}")
        print("="*80 + "\n")

        return global_model

    except Exception as e:
        print(f"\n✗ Aggregation failed: {e}")
        raise


if __name__ == '__main__':

    global_model = main()
    evaluate_aggregated_model(global_model, test_loader, "Evaluating global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_aggregated_model(global_model, outdoors_loader, "Evaluating global model on outdoors_loader")
    evaluate_aggregated_model(global_model, nature_loader, "Evaluating global model on nature_loader")
    evaluate_aggregated_model(global_model, people_loader, "Evaluating global model on people_loader")
    evaluate_aggregated_model(global_model, objects_and_vehicles_loader, "Evaluating global model on objects_and_vehicles_loader")
    print("\n\n############# FINE-TUNED CLASSIFIER (not feature extractor) EVALUATION BEGINS NOW!!!! #############")
    fine_tuned_classifier = fine_tune_classifier(global_model, last_experience_train_loader, fine_tune_epochs=30, lr=0.001, momentum=0.9,weight_decay=5e-4) #device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #TODO: GPU version DOES NOT WORK (at least when device='cuda:0'). Hence, commented-out below:
    evaluate_aggregated_model(fine_tuned_classifier, test_loader, "Evaluating FINE-TUNED CLASSIFIER (not feature extractor)  on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_aggregated_model(fine_tuned_classifier, outdoors_loader, "Evaluating FINE-TUNED CLASSIFIER (not feature extractor)  on outdoors_loader")
    evaluate_aggregated_model(fine_tuned_classifier, nature_loader, "Evaluating FINE-TUNED CLASSIFIER (not feature extractor)  on nature_loader")
    evaluate_aggregated_model(fine_tuned_classifier, people_loader, "Evaluating FINE-TUNED CLASSIFIER (not feature extractor)  on people_loader")
    evaluate_aggregated_model(fine_tuned_classifier, objects_and_vehicles_loader, "Evaluating FINE-TUNED CLASSIFIER (not feature extractor)  on objects_and_vehicles_loader")

    '''
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
    '''

    global_model = main()
    # evaluate_aggregated_model_FedAvg_code_v3(global_model, test_loader)
    # evaluate_aggregated_model(global_model, test_loader_without_objects_and_vehicles, "Evaluating global model on test_loader_without_objects_and_vehicles")
    evaluate_aggregated_model(global_model, test_loader,
                              "Evaluating global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_aggregated_model(global_model, outdoors_loader, "Evaluating global model on outdoors_loader")
    evaluate_aggregated_model(global_model, nature_loader, "Evaluating global model on nature_loader")
    evaluate_aggregated_model(global_model, people_loader, "Evaluating global model on people_loader")
    evaluate_aggregated_model(global_model, objects_and_vehicles_loader,
                              "Evaluating global model on objects_and_vehicles_loader")
    print(
        "\n\n############# Evaluating global model on last_experiences -- is it the same as Evaluating global model on last_experience_test_loader? If so, then Spatial-CF may be fixed by my strategy, BUT CLEARLY, TEMPORAL-CF IS NOT!!! #############")
    # ORIGINALLY print read: print("\n\n############# Evaluating global model on last_experiences -- is it the same as Evaluating global model on last_experience_test_loader? If so, then Spatial-CF may be fixed by my strategy, BUT CLEARLY, TEMPORAL-CF IS NOT!!! #############")
    evaluate_aggregated_model(global_model, last_experience_test_loader,
                              "Evaluating global model on last_experience_test_loader")
    evaluate_aggregated_model(global_model, last_experience_test_outdoors,
                              "Evaluating global model on  last_experience_test_outdoors")
    evaluate_aggregated_model(global_model, last_experience_test_nature,
                              "Evaluating global model on last_experience_test_nature")
    evaluate_aggregated_model(global_model, last_experience_test_people,
                              "Evaluating global model on last_experience_test_people")
    evaluate_aggregated_model(global_model, last_experience_test_objects_and_vehicles,
                              "Evaluating global model on last_experience_test_objects_and_vehicles")

    print(
        "\n\n############# Evaluating global model on middle_experiences -- is it the same as Evaluating global model on middle_experience_test_loader? If so, then Spatial-CF may be fixed by my strategy, BUT CLEARLY, TEMPORAL-CF IS NOT!!! #############")
    evaluate_aggregated_model(global_model, middle_experience_test_loader,
                              "Evaluating global model on middle_experience_test_loader")
    evaluate_aggregated_model(global_model, middle_experience_test_outdoors,
                              "Evaluating global model on  middle_experience_test_outdoors")
    evaluate_aggregated_model(global_model, middle_experience_test_nature,
                              "Evaluating global model on middle_experience_test_nature")
    evaluate_aggregated_model(global_model, middle_experience_test_people,
                              "Evaluating global model on middle_experience_test_people")
    evaluate_aggregated_model(global_model, middle_experience_test_objects_and_vehicles,
                              "Evaluating global model on middle_experience_test_objects_and_vehicles")

    print(
        "\n\n############# Evaluating global model on first_experiences -- is it the same as Evaluating global model on first_experience_test_loader? If so, then Spatial-CF may be fixed by my strategy, BUT CLEARLY, TEMPORAL-CF IS NOT!!! #############")
    evaluate_aggregated_model(global_model, first_experience_test_loader,
                              "Evaluating global model on first_experience_test_loader")
    evaluate_aggregated_model(global_model, first_experience_test_outdoors,
                              "Evaluating global model on  first_experience_test_outdoors")
    evaluate_aggregated_model(global_model, first_experience_test_nature,
                              "Evaluating global model on first_experience_test_nature")
    evaluate_aggregated_model(global_model, first_experience_test_people,
                              "Evaluating global model on first_experience_test_people")
    evaluate_aggregated_model(global_model, first_experience_test_objects_and_vehicles,
                              "Evaluating global model on first_experience_test_objects_and_vehicles")



    print("\n\n############# FINE-TUNED GLOBAL MODEL EVALUATION BEGINS NOW -- global model fine-tuned on last set of train experiences of each specialised model!!!! #############")
    fine_tuned_global_model = fine_tune_aggregated_model(global_model, last_experience_train_loader, fine_tune_epochs=30, lr=0.001, momentum=0.9,weight_decay=5e-4)#fine_tune_epochs=3, lr=0.001, momentum=0.9,weight_decay=5e-4)
    # evaluate_aggregated_model(fine_tuned_global_model, test_loader_without_objects_and_vehicles, "Evaluating FINE-TUNED global model on test_loader_without_objects_and_vehicles")
    evaluate_aggregated_model(fine_tuned_global_model, test_loader,"Evaluating FINE-TUNED global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_aggregated_model(fine_tuned_global_model, outdoors_loader,"Evaluating FINE-TUNED global model on outdoors_loader")
    evaluate_aggregated_model(fine_tuned_global_model, nature_loader,"Evaluating FINE-TUNED global model on nature_loader")
    evaluate_aggregated_model(fine_tuned_global_model, people_loader,"Evaluating FINE-TUNED global model on people_loader")
    evaluate_aggregated_model(fine_tuned_global_model, objects_and_vehicles_loader,"Evaluating FINE-TUNED global model on objects_and_vehicles_loader")

    print(
        "\n\n############# Evaluating FINE-TUNED global model on last_experiences -- is it the same as Evaluating FINE-TUNED global model on last_experience_test_loader? If so, then Spatial-CF may be fixed by my strategy, BUT CLEARLY, TEMPORAL-CF IS NOT!!! #############")
    #ORIGINALLY print read: print("\n\n############# Evaluating FINE-TUNED global model on last_experiences -- is it the same as Evaluating FINE-TUNED global model on last_experience_test_loader? If so, then Spatial-CF may be fixed by my strategy, BUT CLEARLY, TEMPORAL-CF IS NOT!!! #############")
    evaluate_aggregated_model(fine_tuned_global_model, last_experience_test_loader, "Evaluating FINE-TUNED global model on last_experience_test_loader")
    evaluate_aggregated_model(fine_tuned_global_model, last_experience_test_outdoors, "Evaluating FINE-TUNED global model on  last_experience_test_outdoors")
    evaluate_aggregated_model(fine_tuned_global_model, last_experience_test_nature, "Evaluating FINE-TUNED global model on last_experience_test_nature")
    evaluate_aggregated_model(fine_tuned_global_model, last_experience_test_people, "Evaluating FINE-TUNED global model on last_experience_test_people")
    evaluate_aggregated_model(fine_tuned_global_model, last_experience_test_objects_and_vehicles, "Evaluating FINE-TUNED global model on last_experience_test_objects_and_vehicles")
    
    print("\n\n############# Evaluating FINE-TUNED global model on middle_experiences -- is it the same as Evaluating FINE-TUNED global model on middle_experience_test_loader? If so, then Spatial-CF may be fixed by my strategy, BUT CLEARLY, TEMPORAL-CF IS NOT!!! #############")
    evaluate_aggregated_model(fine_tuned_global_model, middle_experience_test_loader, "Evaluating FINE-TUNED global model on middle_experience_test_loader")
    evaluate_aggregated_model(fine_tuned_global_model, middle_experience_test_outdoors, "Evaluating FINE-TUNED global model on  middle_experience_test_outdoors")
    evaluate_aggregated_model(fine_tuned_global_model, middle_experience_test_nature, "Evaluating FINE-TUNED global model on middle_experience_test_nature")
    evaluate_aggregated_model(fine_tuned_global_model, middle_experience_test_people, "Evaluating FINE-TUNED global model on middle_experience_test_people")
    evaluate_aggregated_model(fine_tuned_global_model, middle_experience_test_objects_and_vehicles, "Evaluating FINE-TUNED global model on middle_experience_test_objects_and_vehicles")
    
    print("\n\n############# Evaluating FINE-TUNED global model on first_experiences -- is it the same as Evaluating FINE-TUNED global model on first_experience_test_loader? If so, then Spatial-CF may be fixed by my strategy, BUT CLEARLY, TEMPORAL-CF IS NOT!!! #############")
    evaluate_aggregated_model(fine_tuned_global_model, first_experience_test_loader, "Evaluating FINE-TUNED global model on first_experience_test_loader")
    evaluate_aggregated_model(fine_tuned_global_model, first_experience_test_outdoors, "Evaluating FINE-TUNED global model on  first_experience_test_outdoors")
    evaluate_aggregated_model(fine_tuned_global_model, first_experience_test_nature, "Evaluating FINE-TUNED global model on first_experience_test_nature")
    evaluate_aggregated_model(fine_tuned_global_model, first_experience_test_people, "Evaluating FINE-TUNED global model on first_experience_test_people")
    evaluate_aggregated_model(fine_tuned_global_model, first_experience_test_objects_and_vehicles, "Evaluating FINE-TUNED global model on first_experience_test_objects_and_vehicles")
