
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
import numpy as np
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
from Sixteenth_implementation_attempt.Train_ResNet20_Outdoors_LwF_v16 import outdoors_specialised_models_test_stream_class_il, outdoors_specialised_models_train_stream_class_il, outdoors_specialised_models_val_stream_class_il
from Sixteenth_implementation_attempt.Train_ResNet20_Nature_LwF_v16 import nature_specialised_models_test_stream_class_il, nature_specialised_models_train_stream_class_il, nature_specialised_models_val_stream_class_il
from Sixteenth_implementation_attempt.Train_ResNet20_People_LwF_v16 import people_specialised_models_test_stream_class_il, people_specialised_models_train_stream_class_il, people_specialised_models_val_stream_class_il
from Sixteenth_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_v16 import objects_and_vehicles_specialised_models_test_stream_class_il, objects_and_vehicles_specialised_models_train_stream_class_il, objects_and_vehicles_specialised_models_val_stream_class_il
from Sixteenth_implementation_attempt.Load_cifar100_v16 import avl_test
#TODO: TEMPORARY!!!! Change to that of v16 once I have final Objects_and_Vehicles model!!!!!
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F
####################################################################################################################################################################################
MLP_1 =True #Original used in all previous experiments. Simplest MLP
MLP_2 =False # slightly more powerful MLP, as it contains 1 hidden layer, but it brought no improvements over MPL_1
MLP_3 =True

VERBOSE = False
EXPERT_NAMES = [
    "outdoors",
    "nature",
    "people",
    "objects_and_vehicles",
]
'''MULTILAYER_NAMES = (
    "layer1",
    "layer2",
    "layer3",
    "penultimate",
)'''
MULTILAYER_NAMES = (
    "layer1",
    "layer2",
    "penultimate",
)

NUM_MAHALANOBIS_LAYERS = len(
    MULTILAYER_NAMES
)

NUM_ROUTER_FEATURES = (
    len(EXPERT_NAMES)
    *
    NUM_MAHALANOBIS_LAYERS
)
# ============================================================================
# MAHALANOBIS CONFIGURATION
# ============================================================================

MAHALANOBIS_L2_NORMALIZE = True

# Options:
#   "full"
#   "diagonal"
#   "shrinkage"
MAHALANOBIS_COVARIANCE_MODE = "shrinkage"

# 0 = original full covariance
# 1 = diagonal covariance
MAHALANOBIS_SHRINKAGE = 0.10

# Relative ridge regularization.
MAHALANOBIS_COV_REG = 1e-3

# Expert routing:
# False = hard-select one expert
# True  = probabilistically combine all experts
SOFT_EXPERT_ROUTING = True

ROUTER_TEMPERATURE = 1.0
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

# TRAIN DATASET all specialised train subsets together
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


outdoors_train_loader = DataLoader(
    combined_dataset_outdoors_train,
    batch_size=64,
    shuffle=False,
)

nature_train_loader = DataLoader(
    combined_dataset_nature_train,
    batch_size=64,
    shuffle=False,
)

people_train_loader = DataLoader(
    combined_dataset_people_train,
    batch_size=64,
    shuffle=False,
)

objects_and_vehicles_train_loader = DataLoader(
    combined_dataset_objects_and_vehicles_train,
    batch_size=64,
    shuffle=False,
)

specialist_train_loaders = {
    "outdoors": outdoors_train_loader,
    "nature": nature_train_loader,
    "people": people_train_loader,
    "objects_and_vehicles": objects_and_vehicles_train_loader,
}

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



# VAL DATASET all specialised val subsets together
combined_dataset_outdoors_val = ConcatDataset([
    outdoors_specialised_models_val_stream_class_il[0].dataset,
    outdoors_specialised_models_val_stream_class_il[1].dataset,
    outdoors_specialised_models_val_stream_class_il[2].dataset,
    #outdoors_specialised_models_val_stream_class_il[3].dataset,
    #outdoors_specialised_models_val_stream_class_il[4].dataset,
])

combined_dataset_nature_val = ConcatDataset([
    nature_specialised_models_val_stream_class_il[0].dataset,
    nature_specialised_models_val_stream_class_il[1].dataset,
    nature_specialised_models_val_stream_class_il[2].dataset,
    #nature_specialised_models_val_stream_class_il[3].dataset,
    #nature_specialised_models_val_stream_class_il[4].dataset,
])

combined_dataset_people_val = ConcatDataset([
    people_specialised_models_val_stream_class_il[0].dataset,
    people_specialised_models_val_stream_class_il[1].dataset,
    people_specialised_models_val_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_val = ConcatDataset([
    objects_and_vehicles_specialised_models_val_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_val_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_val_stream_class_il[2].dataset,
])

combined_dataset_val = ConcatDataset([
    combined_dataset_outdoors_val,
    combined_dataset_nature_val,
    combined_dataset_people_val,
    combined_dataset_objects_and_vehicles_val
])

val_loader = DataLoader(
    combined_dataset_val,
    batch_size=32,
    shuffle=True
)
print("print val_loader", val_loader)
print("print val_loader.dataset", val_loader.dataset)


combined_dataset_outdoors_first_experience_test = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[0].dataset,
])

combined_dataset_nature_first_experience_test = ConcatDataset([
    nature_specialised_models_test_stream_class_il[0].dataset,
])

combined_dataset_people_first_experience_test = ConcatDataset([
    people_specialised_models_test_stream_class_il[0].dataset,
])

combined_dataset_objects_and_vehicles_first_experience_test = ConcatDataset([
    objects_and_vehicles_specialised_models_test_stream_class_il[0].dataset,
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

combined_dataset_outdoors_middle_experience_test = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[1].dataset,
])

combined_dataset_nature_middle_experience_test = ConcatDataset([
    nature_specialised_models_test_stream_class_il[1].dataset,
])

combined_dataset_people_middle_experience_test = ConcatDataset([
    people_specialised_models_test_stream_class_il[1].dataset,
])

combined_dataset_objects_and_vehicles_middle_experience_test = ConcatDataset([
    objects_and_vehicles_specialised_models_test_stream_class_il[1].dataset,
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
    shuffle=False)


combined_dataset_outdoors_last_experience_test = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_nature_last_experience_test = ConcatDataset([
    nature_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_people_last_experience_test = ConcatDataset([
    people_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_last_experience_test = ConcatDataset([
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


original_to_aggregated_dict = {12: 0, 17: 1, 23: 2, 33: 3, 37: 4, 49: 5, #OUTDOORS
                               0: 6, 47: 7, 51: 8, 52: 9, 53: 10, 54: 11, 56: 12, 57: 13, 59: 14, #NATURE
                               2: 15, 11: 16, 35: 17, #PEOPLE
                               5: 18, 8: 19, 9: 20, 10: 21, 13: 22, 16: 23, 20: 24, 22: 25, 25: 26, 28: 27, 39: 28, 40: 29, 41: 30, 48: 31, 58: 32, 61: 33, 69: 34,#OBJECTS_AND_VEHICLES
                               81: 35, 84: 36, 85: 37, 86: 38, 87: 39, 89: 40, 90: 41, 94: 42} #OBJECTS_AND_VEHICLES


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


def build_cifar_to_global_lookup(
    original_to_aggregated_dict,
    num_cifar_classes=100,
):

    lookup = torch.full(
        (num_cifar_classes,),
        -1,
        dtype=torch.long,
    )

    for cifar_class, global_idx in (
        original_to_aggregated_dict.items()
    ):
        lookup[cifar_class] = global_idx

    return lookup


def build_local_to_global_indices(
    model_name,
    new_to_original,
    num_subset_classes,
    original_to_aggregated_dict,
):
    """
    Convert:

        local specialist classifier index
                ↓
        original CIFAR-100 label
                ↓
        global 43-class index

    Returns:
        LongTensor of shape [num_subset_classes]

    Example:
        result[0] = global class position for local classifier row 0
    """

    # Normalise metadata keys/values in case pickle contains
    # numpy ints, strings, etc.
    new_to_original = {
        int(local_idx): int(cifar_idx)
        for local_idx, cifar_idx in new_to_original.items()
    }

    local_to_global = []

    print(f"\n===== {model_name.upper()} MAPPING =====")

    for local_idx in range(num_subset_classes):

        if local_idx not in new_to_original:
            raise RuntimeError(
                f"{model_name}: local classifier row "
                f"{local_idx} is missing from new_to_original"
            )

        # STEP 1:
        # local specialist index -> CIFAR-100 class
        cifar_class = new_to_original[local_idx]

        if cifar_class not in original_to_aggregated_dict:
            raise RuntimeError(
                f"{model_name}: CIFAR class {cifar_class} "
                "is not present in the global 43-class mapping"
            )

        # STEP 2:
        # CIFAR-100 class -> global 0..42 index
        global_idx = original_to_aggregated_dict[cifar_class]

        local_to_global.append(global_idx)

        print(
            f"local {local_idx:2d}"
            f" -> CIFAR {cifar_class:2d}"
            f" -> GLOBAL {global_idx:2d}"
        )

        local_to_global_tensor = torch.tensor(
        local_to_global,
        dtype=torch.long)

        print("local_to_global_tensor", local_to_global_tensor)


    return local_to_global_tensor


def build_cifar_to_expert_lookup(
    mappings,
    num_subset_classes,
    expert_names=EXPERT_NAMES,
    num_cifar_classes=100,
):
    """
    CIFAR class -> specialist index

    0 = outdoors
    1 = nature
    2 = people
    3 = objects_and_vehicles
    """

    lookup = torch.full(
        (num_cifar_classes,),
        -1,
        dtype=torch.long,
    )

    for expert_idx, model_name in enumerate(
        expert_names
    ):

        mapping = {
            int(local_idx): int(cifar_class)
            for local_idx, cifar_class
            in mappings[model_name].items()
        }

        n = int(
            num_subset_classes[model_name]
        )

        for local_idx in range(n):

            cifar_class = mapping[local_idx]

            if lookup[cifar_class] != -1:
                raise RuntimeError(
                    f"CIFAR class {cifar_class} "
                    "belongs to more than one expert."
                )

            lookup[cifar_class] = expert_idx

    return lookup


def validate_global_class_coverage(
    local_to_global_indices,
    num_global_classes=43,
):

    all_indices = torch.cat(
        list(local_to_global_indices.values())
    )

    print("\nAll assigned global indices:")
    print(all_indices.tolist())

    if len(all_indices) != num_global_classes:
        raise RuntimeError(
            f"Expected {num_global_classes} classifier "
            f"positions, but found {len(all_indices)}"
        )

    unique_indices = torch.unique(all_indices)

    if len(unique_indices) != num_global_classes:
        raise RuntimeError(
            "Duplicate global class positions detected. "
            "Two specialists appear to own the same "
            "global classifier position."
        )

    expected = torch.arange(num_global_classes)

    if not torch.equal(
        torch.sort(unique_indices).values,
        expected,
    ):
        raise RuntimeError(
            "The specialist mappings do not cover "
            "every global class index from 0 to 42."
        )

    print(
        "✓ All 43 global class positions are "
        "covered exactly once."
    )


def invert_new_to_original(
    new_to_original,
    num_subset_classes,
):
    """
    Existing metadata:
        local -> CIFAR

    Returns:
        CIFAR -> local
    """

    result = {}

    mapping = {
        int(k): int(v)
        for k, v in new_to_original.items()
    }

    for local_idx in range(
        num_subset_classes
    ):
        cifar_class = mapping[local_idx]

        result[cifar_class] = local_idx

    return result


if MLP_1:
    class MahalanobisExpertRouter(
        nn.Module
    ):

        def __init__(
            self,
            num_features,
            num_experts=4,
        ):

            super().__init__()

            self.linear = nn.Linear(
                num_features,
                num_experts,
            )


        def forward(
            self,
            x,
        ):

            return self.linear(
                x
            )

if MLP_2:
    class MahalanobisExpertRouter(nn.Module):

        def __init__(
            self,
            num_features,
            num_experts=4,
            hidden_dim=16,
        ):
            super().__init__()

            self.net = nn.Sequential(

                # ----------------------------------------------------
                # Input:
                # standardized log-Mahalanobis distances
                #
                # Current setup:
                # 4 experts × 3 layers = 12 features
                # ----------------------------------------------------
                nn.Linear(num_features,hidden_dim,),

                #WORSE nn.Softmax(dim=0),
                #SAME AS original; i.e. SAME AS ReLU() --> nn.Softmax(dim=1),
                nn.ReLU(),
                #SAME AS original; i.e. SAME AS ReLU() --> nn.Softmax(dim=-1),

                # ----------------------------------------------------
                # Output:
                # one routing logit per expert
                # ----------------------------------------------------
                nn.Linear(
                    hidden_dim,
                    num_experts,
                ),
            )

        def forward(
            self,
            x,
        ):
            return self.net(x)

if MLP_3:
    class MahalanobisExpertRouter(nn.Module):

        def __init__(
                self,
                num_features,
                num_experts=4,
                hidden_dim1=64,
                hidden_dim2=32,
                hidden_dim3=16,
                dropout=0.15,
        ):
            super().__init__()

            self.net = nn.Sequential(

                # ====================================================
                # Input block
                #
                # Current setup:
                # 4 experts × 3 Mahalanobis layers = 12 inputs
                # ====================================================

                nn.Linear(
                    num_features,
                    hidden_dim1,
                ),

                nn.LayerNorm(
                    hidden_dim1
                ),

                nn.GELU(),

                nn.Dropout(
                    dropout
                ),

                # ====================================================
                # Hidden block 2
                # ====================================================

                nn.Linear(
                    hidden_dim1,
                    hidden_dim2,
                ),

                #nn.LayerNorm(hidden_dim2),

                nn.GELU(),

                nn.Dropout(
                    dropout
                ),

                # ====================================================
                # Hidden block 3
                # ====================================================

                nn.Linear(
                    hidden_dim2,
                    hidden_dim3,
                ),

                #nn.LayerNorm(hidden_dim3),

                nn.GELU(),

                # ====================================================
                # Expert-routing output
                # ====================================================

                nn.Linear(
                    hidden_dim3,
                    num_experts,
                ),
            )

        def forward(
                self,
                x,
        ):
            return self.net(
                x
            )

def make_router_class_weights(
    targets,
    num_experts,
    alpha=0.5,
):
    print("Alpha value:", alpha)

    counts = torch.bincount(
        targets,
        minlength=num_experts,
    ).float()

    weights = (
        1.0
        /
        counts.clamp_min(1.0)
    ).pow(alpha)

    # Average weight = 1
    weights = (
        weights
        /
        weights.mean()
    )
    #####RUNS BUT GAVE WORSE RESULTS counts = 600, 900, 300, 2500
        #dataset_sizes = [2400, 3600, 1200, 10000]
        #weights = [600, 900, 300, 2500] #since total_samples = 4
    #dataset_sizes = torch.Tensor(counts)#np.array(counts, dtype=np.float32)
    #####RUNS BUT GAVE WORSE RESULTS weights = torch.Tensor(counts)
    #therefore total_samples = dataset_sizes.sum() and weights = dataset_sizes / total_samples, NOT NEEDED

    return weights


@torch.no_grad()
def collect_multilayer_mahalanobis_router_dataset(
    models,
    feature_models,
    validation_loader,
    mappings,
    num_subset_classes,
    device="cpu",
):
    """
    Produces:

        distances:
            [N, 16]

        targets:
            [N]

    Feature order:

        outdoors layer1
        outdoors layer2
        outdoors layer3
        outdoors penultimate

        nature layer1
        nature layer2
        nature layer3
        nature penultimate

        people ...

        objects_and_vehicles ...
    """

    device = torch.device(
        device
    )


    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings=mappings,
            num_subset_classes=(
                num_subset_classes
            ),
            expert_names=EXPERT_NAMES,
        ).to(device)
    )


    all_distances = []
    all_targets = []


    for model_name in EXPERT_NAMES:

        models[
            model_name
        ] = (
            models[
                model_name
            ]
            .to(device)
        )

        models[
            model_name
        ].eval()


    for (
        batch_data,
        batch_labels,
        batch_task_ids,
    ) in validation_loader:

        batch_data = batch_data.to(
            device
        )

        batch_labels = batch_labels.to(
            device
        )


        true_expert = (
            cifar_to_expert[
                batch_labels
            ]
        )


        # Each element eventually [B]
        batch_router_features = []


        # ========================================================
        # EXPERT-MAJOR ORDER
        # ========================================================

        for model_name in EXPERT_NAMES:

            n = int(
                num_subset_classes[
                    model_name
                ]
            )


            layer_features, _ = (
                get_multilayer_features_and_logits(
                    model=models[
                        model_name
                    ],

                    x=batch_data,

                    num_classes=n,
                )
            )


            # ====================================================
            # LAYER-MINOR ORDER
            # ====================================================

            for layer_name in MULTILAYER_NAMES:

                features = (
                    prepare_mahalanobis_features(
                        layer_features[
                            layer_name
                        ]
                    )
                )


                layer_model = (
                    feature_models[
                        model_name
                    ][
                        layer_name
                    ]
                )


                class_means = (
                    layer_model[
                        "class_means"
                    ]
                    .to(device)
                    .float()
                )


                precision = (
                    layer_model[
                        "precision"
                    ]
                    .to(device)
                    .float()
                )


                distance, _ = (
                    specialist_mahalanobis_distance(
                        features=features,
                        class_means=(
                            class_means
                        ),
                        precision=(
                            precision
                        ),
                    )
                )


                batch_router_features.append(
                    distance
                )


        # ========================================================
        # [B, 16]
        # ========================================================

        batch_router_features = (
            torch.stack(
                batch_router_features,
                dim=1,
            )
        )


        if (
            batch_router_features.shape[1]
            !=
            NUM_ROUTER_FEATURES
        ):

            raise RuntimeError(
                "Wrong router feature count. "
                f"Expected "
                f"{NUM_ROUTER_FEATURES}, "
                f"got "
                f"{batch_router_features.shape[1]}"
            )


        all_distances.append(
            batch_router_features.cpu()
        )


        all_targets.append(
            true_expert.cpu()
        )


    distances = torch.cat(
        all_distances,
        dim=0,
    )


    targets = torch.cat(
        all_targets,
        dim=0,
    )


    print(
        "\nMULTI-LAYER ROUTER DATASET:"
    )

    print(
        "distances.shape =",
        distances.shape,
    )

    print(
        "targets.shape =",
        targets.shape,
    )


    return (
        distances,
        targets,
    )

def prepare_router_features(
    distances,
    mean=None,
    std=None,
):

    x = torch.log(
        distances.clamp_min(
            1e-8
        )
    )

    if mean is None:

        mean = x.mean(
            dim=0
        )

    if std is None:

        std = x.std(
            dim=0,
            unbiased=False,
        ).clamp_min(
            1e-6
        )

    x = (
        x
        -
        mean.unsqueeze(0)
    ) / std.unsqueeze(0)

    return x, mean, std

def fit_mahalanobis_expert_router(
    distances,
    targets,
    device="cpu",
    epochs=500,
    lr=1e-2,
):

    device = torch.device(
        device
    )

    distances = distances.to(
        device
    )

    targets = targets.to(
        device
    )

    router_features, feature_mean, feature_std = (
        prepare_router_features(
            distances
        )
    )

    num_router_features = (
        router_features.shape[1]
    )

    print(
        "num_router_features =",
        num_router_features,
    )

    router = (
        MahalanobisExpertRouter(
            num_features=(
                num_router_features
            ),

            num_experts=len(
                EXPERT_NAMES
            ),
        )
        .to(device)
    )

    optimizer = torch.optim.Adam(
        router.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    # ============================================================
    # Expert-balanced routing loss VERSION 2
    # ============================================================

    class_weights = make_router_class_weights(
        targets=targets,
        num_experts=len(EXPERT_NAMES),
        alpha=0.5, #alpha_values = [0.0,0.25,0.5,0.75,1.0,]
    )

    print(
        "Router expert counts:",
        torch.bincount(
            targets,
            minlength=len(EXPERT_NAMES),
        ).tolist()
    )

    print(
        "Router class weights:",
        class_weights.tolist()
    )

    for epoch in range(
        epochs
    ):

        optimizer.zero_grad()

        router_logits = router(
            router_features
        )

        loss = F.cross_entropy(
            router_logits,
            targets,
            weight=class_weights,
        )

        loss.backward()

        optimizer.step()

        if (
            epoch == 0
            or (epoch + 1) % 100 == 0
        ):

            '''with torch.no_grad():

                predictions = (
                    router_logits.argmax(
                        dim=1
                    )
                )

                accuracy = (
                    predictions
                    == targets
                ).float().mean()

            print(
                f"Router epoch "
                f"{epoch + 1:4d}/{epochs} | "
                f"loss={loss.item():.6f} | "
                f"routing acc="
                f"{100.0 * accuracy.item():.2f}%"
            )'''
            with torch.no_grad():

                predictions = router_logits.argmax(
                    dim=1
                )

                accuracy = (
                        predictions == targets
                ).float().mean()

                recalls = []

                for expert_idx in range(
                        len(EXPERT_NAMES)
                ):

                    mask = (
                            targets == expert_idx
                    )

                    if mask.any():
                        expert_recall = (
                                predictions[mask]
                                == expert_idx
                        ).float().mean()

                        recalls.append(
                            expert_recall
                        )

                balanced_accuracy = (
                    torch.stack(recalls).mean()
                )
                print(
                    f"Router epoch "
                    f"{epoch + 1:4d}/{epochs} | "
                    f"loss={loss.item():.6f} | "
                    f"routing acc="
                    f"{accuracy.item() * 100:.2f}% | "
                    f"balanced routing acc="
                    f"{balanced_accuracy.item() * 100:.2f}%"
                )

    router.eval()

    for parameter in router.parameters():

        parameter.requires_grad = False

    return (
        router,
        feature_mean.detach(),
        feature_std.detach(),
    )


'''def prepare_mahalanobis_features(
    features: torch.Tensor,
):
    features = features.float()

    if MAHALANOBIS_L2_NORMALIZE:

        features = F.normalize(
            features,
            p=2,
            dim=1,
        )

    return features'''
def prepare_mahalanobis_features(
    features,
):
    """
    Prepare either:

        1. A single feature tensor [B, D]

    or:

        2. A dictionary of multi-layer features:

            {
                "layer1":      Tensor[B, D1],
                "layer2":      Tensor[B, D2],
                "layer3":      Tensor[B, D3],
                "penultimate": Tensor[B, D4],
            }

    L2 normalization is applied independently to each layer.
    """

    # ============================================================
    # MULTI-LAYER CASE
    # ============================================================

    if isinstance(
        features,
        dict,
    ):

        prepared_features = {}

        for (
            layer_name,
            layer_features,
        ) in features.items():

            prepared_features[
                layer_name
            ] = (
                prepare_mahalanobis_features(
                    layer_features
                )
            )

        return prepared_features


    # ============================================================
    # SINGLE-TENSOR CASE
    # ============================================================

    if not isinstance(
        features,
        torch.Tensor,
    ):

        raise TypeError(
            "features must be either "
            "a torch.Tensor or a dict "
            "of torch.Tensor objects. "
            f"Received: {type(features)}"
        )


    features = features.float()


    # ------------------------------------------------------------
    # Mahalanobis fitting expects one vector per sample:
    #
    # [B, D]
    #
    # Your get_multilayer_features_and_logits()
    # should already global-average-pool convolutional outputs.
    # ------------------------------------------------------------

    if features.dim() != 2:

        raise RuntimeError(
            "Mahalanobis features must have shape "
            "[B, D]. "
            f"Received shape: {features.shape}"
        )


    # ============================================================
    # Optional L2 normalization
    # ============================================================

    if MAHALANOBIS_L2_NORMALIZE:

        features = F.normalize(
            features,
            p=2,
            dim=1,
        )


    return features


@torch.no_grad()
def fit_multilayer_specialist_feature_distribution(
    model,
    loader,
    new_to_original,
    num_subset_classes,
    device="cpu",
    covariance_regularization=1e-3,
    covariance_mode="shrinkage",
    shrinkage=0.10,
):
    """
    Fits a separate class-conditional Mahalanobis model
    for EACH selected feature layer.

    Structure returned:

        {
            "layer1": {
                "class_means": ...,
                "precision": ...
            },

            "layer2": {...},

            "layer3": {...},

            "penultimate": {...}
        }
    """

    device = torch.device(
        device
    )

    model = model.to(
        device
    )

    model.eval()


    cifar_to_local = (
        invert_new_to_original(
            new_to_original,
            num_subset_classes,
        )
    )


    # ============================================================
    # layer -> local class -> features
    # ============================================================

    features_by_layer_and_class = {

        layer_name: {

            local_idx: []

            for local_idx in range(
                num_subset_classes
            )

        }

        for layer_name
        in MULTILAYER_NAMES
    }


    # ============================================================
    # Collect TRAIN representations
    # ============================================================

    for (
        batch_data,
        batch_labels,
        batch_task_ids,
    ) in loader:

        batch_data = batch_data.to(
            device
        )


        layer_features, _ = (
            get_multilayer_features_and_logits(
                model=model,
                x=batch_data,
                num_classes=(
                    num_subset_classes
                ),
            )
        )


        # ========================================================
        # Process every representation
        # ========================================================

        for layer_name in MULTILAYER_NAMES:

            features = (
                prepare_mahalanobis_features(
                    layer_features[
                        layer_name
                    ]
                )
            )


            for feature, cifar_label in zip(
                features,
                batch_labels,
            ):

                cifar_class = int(
                    cifar_label.item()
                )

                if (
                    cifar_class
                    not in cifar_to_local
                ):

                    raise RuntimeError(
                        f"CIFAR class "
                        f"{cifar_class} "
                        "does not belong to "
                        "this specialist."
                    )


                local_idx = (
                    cifar_to_local[
                        cifar_class
                    ]
                )


                features_by_layer_and_class[
                    layer_name
                ][
                    local_idx
                ].append(
                    feature.cpu()
                )


    # ============================================================
    # Fit one Mahalanobis model PER LAYER
    # ============================================================

    layer_models = {}


    for layer_name in MULTILAYER_NAMES:

        class_means = []

        centered_features = []


        for local_idx in range(
            num_subset_classes
        ):

            if len(
                features_by_layer_and_class[
                    layer_name
                ][
                    local_idx
                ]
            ) == 0:

                raise RuntimeError(
                    f"No training features for "
                    f"layer={layer_name}, "
                    f"class={local_idx}"
                )


            class_features = torch.stack(
                features_by_layer_and_class[
                    layer_name
                ][
                    local_idx
                ]
            ).float()


            class_mean = (
                class_features.mean(
                    dim=0
                )
            )


            class_means.append(
                class_mean
            )


            centered_features.append(
                class_features
                -
                class_mean.unsqueeze(0)
            )


        class_means = torch.stack(
            class_means,
            dim=0,
        )


        centered = torch.cat(
            centered_features,
            dim=0,
        )


        n = centered.shape[0]


        covariance = (
            centered.T
            @ centered
        ) / max(
            n - 1,
            1,
        )


        feature_dim = (
            covariance.shape[0]
        )


        # ========================================================
        # Covariance estimator
        # ========================================================

        if covariance_mode == "full":

            pass


        elif covariance_mode == "diagonal":

            covariance = torch.diag(
                torch.diag(
                    covariance
                )
            )


        elif covariance_mode == "shrinkage":

            diagonal_covariance = (
                torch.diag(
                    torch.diag(
                        covariance
                    )
                )
            )


            covariance = (
                (1.0 - shrinkage)
                * covariance

                +

                shrinkage
                * diagonal_covariance
            )


        else:

            raise ValueError(
                "covariance_mode must be "
                "'full', 'diagonal', "
                "or 'shrinkage'."
            )


        # ========================================================
        # Scale-aware ridge regularization
        # ========================================================

        covariance_scale = (
            torch.diag(
                covariance
            )
            .mean()
            .clamp_min(
                1e-8
            )
        )


        covariance = (
            covariance

            +

            covariance_regularization
            * covariance_scale
            * torch.eye(
                feature_dim,
                dtype=covariance.dtype,
                device=covariance.device,
            )
        )


        precision = torch.linalg.pinv(
            covariance
        )


        layer_models[
            layer_name
        ] = {

            "class_means":
                class_means,

            "precision":
                precision,

            "covariance":
                covariance,
        }


        print(
            f"  {layer_name:12s} | "
            f"class_means="
            f"{tuple(class_means.shape)} | "
            f"precision="
            f"{tuple(precision.shape)}"
        )


    return layer_models

@torch.no_grad()
def get_multilayer_features_and_logits(
    model,
    x,
    num_classes,
):
    """
    Extracts:

        layer1
        layer2
        layer3
        penultimate

    plus the specialist's native logits.

    Intermediate convolutional representations are
    global-average-pooled to [B, C].

    Returns:

        features = {
            "layer1":      [B, D1],
            "layer2":      [B, D2],
            "layer3":      [B, D3],
            "penultimate": [B, D4],
        }

        logits:
            [B, num_classes]
    """

    captured = {}

    # ============================================================
    # Forward-hook factory for residual stages
    # ============================================================

    def make_output_hook(
        layer_name,
    ):

        def hook(
            module,
            inputs,
            output,
        ):

            z = output.detach()

            # Typical ResNet residual-stage output:
            #
            # [B, C, H, W]
            #
            # Convert this to one feature vector per image.
            if z.dim() == 4:

                z = F.adaptive_avg_pool2d(
                    z,
                    output_size=1,
                )

                z = torch.flatten(
                    z,
                    start_dim=1,
                )

            elif z.dim() > 2:

                z = torch.flatten(
                    z,
                    start_dim=1,
                )

            captured[
                layer_name
            ] = z

        return hook


    # ============================================================
    # Penultimate representation:
    # input entering model.linear
    # ============================================================

    def capture_penultimate(
        module,
        inputs,
    ):

        captured[
            "penultimate"
        ] = inputs[0].detach()


    handles = []

    # ============================================================
    # Register hooks
    # ============================================================

    handles.append(
        model.layer1.register_forward_hook(
            make_output_hook(
                "layer1"
            )
        )
    )

    handles.append(
        model.layer2.register_forward_hook(
            make_output_hook(
                "layer2"
            )
        )
    )

    handles.append(
        model.layer3.register_forward_hook(
            make_output_hook(
                "layer3"
            )
        )
    )

    handles.append(
        model.linear.register_forward_pre_hook(
            capture_penultimate
        )
    )


    # ============================================================
    # ONE specialist forward pass captures every representation
    # ============================================================

    try:

        logits = model(x)

    finally:

        for handle in handles:

            handle.remove()


    logits = logits[
        :,
        :num_classes
    ]


    # ============================================================
    # Safety checks
    # ============================================================

    for layer_name in MULTILAYER_NAMES:

        if layer_name not in captured:

            raise RuntimeError(
                f"Failed to capture "
                f"{layer_name}."
            )


    return (
        captured,
        logits,
    )


def specialist_mahalanobis_distance(
    features,
    class_means,
    precision,
):
    """
    features:
        [B, D]

    class_means:
        [C, D]

    Returns:
        min_distance: [B]
        nearest_class: [B]

    LOWER is better / more in-distribution.
    """

    # [B, C, D]
    diff = (
        features.unsqueeze(1)
        -
        class_means.unsqueeze(0)
    )

    # Mahalanobis squared distance:
    #
    # (x-mu)^T P (x-mu)

    distances = torch.einsum(
        "bcd,de,bce->bc",
        diff,
        precision,
        diff,
    )

    min_distance, nearest_class = (
        distances.min(dim=1)
    )

    return (
        min_distance,
        nearest_class,
    )


@torch.no_grad()
def fit_multilayer_feature_distance_thresholds(
    models,
    feature_models,
    validation_loader,
    mappings,
    num_subset_classes,
    device="cpu",
    quantile=0.95,
):
    """
    Fits one ID threshold for each:

        expert x layer

    Used for diagnostics / accepted_as_id,
    NOT for choosing the expert.
    """

    device = torch.device(
        device
    )


    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings=mappings,
            num_subset_classes=(
                num_subset_classes
            ),
            expert_names=EXPERT_NAMES,
        ).to(device)
    )


    id_distances = {

        expert_name: {

            layer_name: []

            for layer_name
            in MULTILAYER_NAMES
        }

        for expert_name
        in EXPERT_NAMES
    }


    for (
        batch_data,
        batch_labels,
        batch_task_ids,
    ) in validation_loader:

        batch_data = batch_data.to(
            device
        )

        batch_labels = batch_labels.to(
            device
        )


        true_expert = (
            cifar_to_expert[
                batch_labels
            ]
        )


        for expert_idx, model_name in enumerate(
            EXPERT_NAMES
        ):

            n = int(
                num_subset_classes[
                    model_name
                ]
            )


            layer_features, _ = (
                get_multilayer_features_and_logits(
                    model=models[
                        model_name
                    ],

                    x=batch_data,

                    num_classes=n,
                )
            )


            id_mask = (
                true_expert
                ==
                expert_idx
            )


            if not id_mask.any():

                continue


            for layer_name in MULTILAYER_NAMES:

                features = (
                    prepare_mahalanobis_features(
                        layer_features[
                            layer_name
                        ]
                    )
                )


                layer_model = (
                    feature_models[
                        model_name
                    ][
                        layer_name
                    ]
                )


                class_means = (
                    layer_model[
                        "class_means"
                    ]
                    .to(device)
                    .float()
                )


                precision = (
                    layer_model[
                        "precision"
                    ]
                    .to(device)
                    .float()
                )


                distances, _ = (
                    specialist_mahalanobis_distance(
                        features=features,
                        class_means=(
                            class_means
                        ),
                        precision=(
                            precision
                        ),
                    )
                )


                id_distances[
                    model_name
                ][
                    layer_name
                ].append(
                    distances[
                        id_mask
                    ].cpu()
                )


    thresholds = {}


    print(
        "\n===== MULTI-LAYER "
        "MAHALANOBIS THRESHOLDS ====="
    )


    for model_name in EXPERT_NAMES:

        thresholds[
            model_name
        ] = {}


        print(
            f"\n{model_name.upper()}"
        )


        for layer_name in MULTILAYER_NAMES:

            distances = torch.cat(
                id_distances[
                    model_name
                ][
                    layer_name
                ]
            )


            threshold = torch.quantile(
                distances,
                quantile,
            )


            median = torch.median(
                distances
            )


            thresholds[
                model_name
            ][
                layer_name
            ] = threshold


            print(
                f"  {layer_name:12s} | "
                f"median="
                f"{median.item():.4f} | "
                f"95%="
                f"{threshold.item():.4f}"
            )

    return thresholds


def gaussian_log_probability(
    x,
    mean,
    std,
):
    """
    Element-wise Gaussian log probability,
    ignoring the constant -0.5*log(2*pi),
    which cancels in the likelihood ratio.
    """

    variance = std.pow(2)

    return (
        -torch.log(std)
        -
        0.5
        * ((x - mean).pow(2) / variance)
    )

@torch.no_grad()
def select_expert_by_features(
    models,
    feature_models,
    thresholds,
    x,
    num_subset_classes,
    device="cpu",
):
    """
    No task ID.

    Every specialist sees every input.

    Returns the expert whose features are most
    in-distribution relative to its own learned
    feature distribution.
    """

    device = torch.device(device)

    x = x.to(device)

    relative_distances = []

    raw_distances = []

    for model_name in EXPERT_NAMES:

        n = num_subset_classes[
            model_name
        ]

        features, _ = get_multilayer_features_and_logits(
            models[model_name],
            x,
            n,
        )

        features = prepare_mahalanobis_features(
            features
        )

        feature_model = feature_models[
            model_name
        ]

        class_means = feature_model[
            "class_means"
        ].to(device)

        precision = feature_model[
            "precision"
        ].to(device)

        distance, nearest_class = (
            specialist_mahalanobis_distance(
                features.float(),
                class_means,
                precision,
            )
        )

        threshold = thresholds[
            model_name
        ].to(device)

        relative = (
            distance / threshold
        )

        raw_distances.append(
            distance
        )

        relative_distances.append(
            relative
        )

    # [B, 4]
    relative_distances = torch.stack(
        relative_distances,
        dim=1,
    )

    raw_distances = torch.stack(
        raw_distances,
        dim=1,
    )

    # LOWEST distance wins
    selected_expert = (
        relative_distances.argmin(
            dim=1
        )
    )

    return {
        "selected_expert":
            selected_expert,

        "relative_distances":
            relative_distances,

        "raw_distances":
            raw_distances,
    }

@torch.no_grad()
def fit_mahalanobis_id_ood_calibration(
    models,
    feature_models,
    validation_loader,
    mappings,
    num_subset_classes,
    device="cpu",
    expert_names=EXPERT_NAMES,
    layer_names=MULTILAYER_NAMES,
):
    """
    MULTI-LAYER version.

    For every expert AND every selected feature layer, fit
    Gaussian distributions to:

        log(Mahalanobis distance | ID)
        log(Mahalanobis distance | OOD)

    Expected feature_models structure:

        feature_models[expert][layer] = {
            "class_means": ...,
            "precision": ...,
            "covariance": ...
        }

    Returns:

        calibration[expert][layer] = {
            "id_mean": ...,
            "id_std": ...,
            "ood_mean": ...,
            "ood_std": ...
        }

    NOTE:
        These statistics are useful diagnostically.

        If expert selection is now performed by the learned
        joint 16-D Mahalanobis router, these statistics do NOT
        need to be used by GlobalSpecialistModel.forward().
    """

    device = torch.device(device)

    # ============================================================
    # CIFAR class -> true expert
    # ============================================================

    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings=mappings,
            num_subset_classes=num_subset_classes,
            expert_names=expert_names,
        ).to(device)
    )

    # ============================================================
    # Storage:
    #
    # expert -> layer -> list of tensors
    # ============================================================

    id_distances = {
        model_name: {
            layer_name: []
            for layer_name in layer_names
        }
        for model_name in expert_names
    }

    ood_distances = {
        model_name: {
            layer_name: []
            for layer_name in layer_names
        }
        for model_name in expert_names
    }

    # ============================================================
    # Evaluation mode
    # ============================================================

    for model_name in expert_names:

        models[model_name] = (
            models[model_name]
            .to(device)
        )

        models[model_name].eval()

    # ============================================================
    # Collect validation distances
    # ============================================================

    for (
        batch_data,
        batch_labels,
        batch_task_ids,
    ) in validation_loader:

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        # --------------------------------------------------------
        # True owner of each image
        # --------------------------------------------------------

        true_expert = cifar_to_expert[
            batch_labels
        ]

        if torch.any(true_expert < 0):

            bad_labels = torch.unique(
                batch_labels[
                    true_expert < 0
                ]
            )

            raise RuntimeError(
                "Validation set contains classes "
                "not owned by any expert: "
                f"{bad_labels.tolist()}"
            )

        # ========================================================
        # Run every image through every expert
        # ========================================================

        for expert_idx, model_name in enumerate(
            expert_names
        ):

            n = int(
                num_subset_classes[
                    model_name
                ]
            )

            # ----------------------------------------------------
            # ONE forward pass gives all feature layers
            # ----------------------------------------------------

            layer_features, _ = (
                get_multilayer_features_and_logits(
                    model=models[
                        model_name
                    ],
                    x=batch_data,
                    num_classes=n,
                )
            )

            # ----------------------------------------------------
            # Apply the same preprocessing used when fitting
            # class means / covariance.
            #
            # Returns dict:
            #
            # {
            #   "layer1": ...,
            #   "layer2": ...,
            #   ...
            # }
            # ----------------------------------------------------

            layer_features = (
                prepare_mahalanobis_features(
                    layer_features
                )
            )

            # ----------------------------------------------------
            # Masks are shared across all layers of this expert
            # ----------------------------------------------------

            id_mask = (
                true_expert
                ==
                expert_idx
            )

            ood_mask = (
                true_expert
                !=
                expert_idx
            )

            # ====================================================
            # Mahalanobis distance for each feature layer
            # ====================================================

            for layer_name in layer_names:

                features = (
                    layer_features[
                        layer_name
                    ]
                )

                layer_model = (
                    feature_models[
                        model_name
                    ][
                        layer_name
                    ]
                )

                class_means = (
                    layer_model[
                        "class_means"
                    ]
                    .to(device)
                    .float()
                )

                precision = (
                    layer_model[
                        "precision"
                    ]
                    .to(device)
                    .float()
                )

                distances, _ = (
                    specialist_mahalanobis_distance(
                        features=features,
                        class_means=class_means,
                        precision=precision,
                    )
                )

                # -----------------------------------------------
                # Store ID distances
                # -----------------------------------------------

                if id_mask.any():

                    id_distances[
                        model_name
                    ][
                        layer_name
                    ].append(
                        distances[
                            id_mask
                        ]
                        .detach()
                        .cpu()
                    )

                # -----------------------------------------------
                # Store OOD distances
                # -----------------------------------------------

                if ood_mask.any():

                    ood_distances[
                        model_name
                    ][
                        layer_name
                    ].append(
                        distances[
                            ood_mask
                        ]
                        .detach()
                        .cpu()
                    )

    # ============================================================
    # Fit Gaussian distributions in LOG-DISTANCE space
    # ============================================================

    calibration = {}

    print(
        "\n"
        "===== MULTI-LAYER MAHALANOBIS "
        "ID/OOD CALIBRATION ====="
    )

    for model_name in expert_names:

        calibration[
            model_name
        ] = {}

        print(
            f"\n{model_name.upper()}"
        )

        print(
            "-" * 80
        )

        for layer_name in layer_names:

            if len(
                id_distances[
                    model_name
                ][
                    layer_name
                ]
            ) == 0:

                raise RuntimeError(
                    f"No ID distances for "
                    f"{model_name}/{layer_name}"
                )

            if len(
                ood_distances[
                    model_name
                ][
                    layer_name
                ]
            ) == 0:

                raise RuntimeError(
                    f"No OOD distances for "
                    f"{model_name}/{layer_name}"
                )

            # ----------------------------------------------------
            # Concatenate all batches
            # ----------------------------------------------------

            id_d = torch.cat(
                id_distances[
                    model_name
                ][
                    layer_name
                ]
            ).float()

            ood_d = torch.cat(
                ood_distances[
                    model_name
                ][
                    layer_name
                ]
            ).float()

            # ----------------------------------------------------
            # Log transform
            # ----------------------------------------------------

            log_id = torch.log(
                id_d.clamp_min(
                    1e-8
                )
            )

            log_ood = torch.log(
                ood_d.clamp_min(
                    1e-8
                )
            )

            # ----------------------------------------------------
            # Gaussian parameters
            # ----------------------------------------------------

            id_mean = (
                log_id.mean()
            )

            id_std = (
                log_id.std(
                    unbiased=False
                )
                .clamp_min(
                    1e-6
                )
            )

            ood_mean = (
                log_ood.mean()
            )

            ood_std = (
                log_ood.std(
                    unbiased=False
                )
                .clamp_min(
                    1e-6
                )
            )

            calibration[
                model_name
            ][
                layer_name
            ] = {
                "id_mean":
                    id_mean,

                "id_std":
                    id_std,

                "ood_mean":
                    ood_mean,

                "ood_std":
                    ood_std,
            }

            print(
                f"  {layer_name:12s} | "
                f"ID log-d: "
                f"{id_mean.item():.4f} "
                f"± {id_std.item():.4f} | "
                f"OOD log-d: "
                f"{ood_mean.item():.4f} "
                f"± {ood_std.item():.4f}"
            )

    return calibration

class GlobalSpecialistModel(nn.Module):

    def __init__(
        self,
        outdoors_model,
        nature_model,
        people_model,
        objects_and_vehicles_model,
        outdoors_indices,
        nature_indices,
        people_indices,
        objects_and_vehicles_indices,
        outdoors_num_classes,
        nature_num_classes,
        people_num_classes,
        objects_num_classes,
        feature_models,
        feature_thresholds,
        #mahalanobis_calibration:Optional,
        mahalanobis_router,
        router_feature_mean,
        router_feature_std,
        num_classes=43,
    ):
        super().__init__()

        self.outdoors = outdoors_model
        self.nature = nature_model
        self.people = people_model
        self.objects_and_vehicles = (
            objects_and_vehicles_model
        )

        self.outdoors_num_classes = (
            outdoors_num_classes
        )

        self.nature_num_classes = (
            nature_num_classes
        )

        self.people_num_classes = (
            people_num_classes
        )

        self.objects_num_classes = (
            objects_num_classes
        )

        self.num_classes = num_classes


        self.register_buffer(
            "outdoors_indices",
            outdoors_indices.long()
        )

        self.register_buffer(
            "nature_indices",
            nature_indices.long()
        )

        self.register_buffer(
            "people_indices",
            people_indices.long()
        )

        self.register_buffer(
            "objects_and_vehicles_indices",
            objects_and_vehicles_indices.long()
        )

        # ============================================================
        # MULTI-LAYER Mahalanobis feature statistics
        # ============================================================

        for model_name in EXPERT_NAMES:

            for layer_name in MULTILAYER_NAMES:
                layer_model = (
                    feature_models[
                        model_name
                    ][
                        layer_name
                    ]
                )

                # --------------------------------------------------------
                # Class means / prototypes
                # --------------------------------------------------------

                self.register_buffer(
                    f"{model_name}_{layer_name}_class_means",
                    layer_model[
                        "class_means"
                    ].float(),
                )

                # --------------------------------------------------------
                # Precision matrix
                # --------------------------------------------------------

                self.register_buffer(
                    f"{model_name}_{layer_name}_precision",
                    layer_model[
                        "precision"
                    ].float(),
                )

        # ============================================================
        # MULTI-LAYER Mahalanobis thresholds
        #
        # Shape:
        #     [4 experts, 4 layers]
        # ============================================================

        self.register_buffer(
            "expert_layer_thresholds",
            torch.stack(
                [
                    torch.stack(
                        [
                            feature_thresholds[
                                model_name
                            ][
                                layer_name
                            ].float()

                            for layer_name
                            in MULTILAYER_NAMES
                        ],
                        dim=0,
                    )

                    for model_name
                    in EXPERT_NAMES
                ],
                dim=0,
            )
        )

        # ============================================================
        # Learned multi-layer Mahalanobis router
        # ============================================================

        self.mahalanobis_router = (
            mahalanobis_router
        )

        self.register_buffer(
            "router_feature_mean",
            router_feature_mean.float(),
        )

        self.register_buffer(
            "router_feature_std",
            router_feature_std.float(),
        )

        # ============================================================
        # Sanity checks
        # ============================================================

        expected_num_router_features = (
                len(EXPERT_NAMES)
                *
                len(MULTILAYER_NAMES)
        )

        if (
                self.router_feature_mean.numel()
                != expected_num_router_features
        ):
            raise RuntimeError(
                "router_feature_mean has wrong size. "
                f"Expected "
                f"{expected_num_router_features}, "
                f"got "
                f"{self.router_feature_mean.numel()}."
            )

        if (
                self.router_feature_std.numel()
                != expected_num_router_features
        ):
            raise RuntimeError(
                "router_feature_std has wrong size. "
                f"Expected "
                f"{expected_num_router_features}, "
                f"got "
                f"{self.router_feature_std.numel()}."
            )

        expected_threshold_shape = (
            len(EXPERT_NAMES),
            len(MULTILAYER_NAMES),
        )

        if (
                tuple(
                    self.expert_layer_thresholds.shape
                )
                !=
                expected_threshold_shape
        ):
            raise RuntimeError(
                "expert_layer_thresholds has wrong shape. "
                f"Expected "
                f"{expected_threshold_shape}, "
                f"got "
                f"{tuple(self.expert_layer_thresholds.shape)}."
            )

        print(
            "\n===== GLOBAL MULTI-LAYER "
            "MAHALANOBIS CONFIGURATION ====="
        )

        print(
            "expert_layer_thresholds:",
            self.expert_layer_thresholds.shape
        )

        print(
            "router_feature_mean:",
            self.router_feature_mean.shape
        )

        print(
            "router_feature_std:",
            self.router_feature_std.shape
        )

        print(
            "router input features:",
            expected_num_router_features
        )


        global_class_owner = torch.full(
            (num_classes,),
            -1,
            dtype=torch.long,
        )

        expert_global_indices = [
            self.outdoors_indices,
            self.nature_indices,
            self.people_indices,
            self.objects_and_vehicles_indices,
        ]

        for expert_idx, indices in enumerate(
                expert_global_indices
        ):
            global_class_owner[
                indices
            ] = expert_idx

        if torch.any(
                global_class_owner < 0
        ):
            raise RuntimeError(
                "Not all global classes have an expert owner."
            )

        self.register_buffer(
            "global_class_owner",
            global_class_owner,
        )

    def forward(
            self,
            x,
            return_gate_info=False,
    ):

        # ============================================================
        # 1. Extract ALL selected feature layers + native logits
        #
        # Each feature object is a dictionary:
        #
        # {
        #     "layer1":      [B, 16],
        #     "layer2":      [B, 32],
        #     "layer3":      [B, 64],
        #     "penultimate": [B, 64],
        # }
        # ============================================================

        outdoors_features, outdoors_logits = (
            get_multilayer_features_and_logits(
                self.outdoors,
                x,
                self.outdoors_num_classes,
            )
        )

        nature_features, nature_logits = (
            get_multilayer_features_and_logits(
                self.nature,
                x,
                self.nature_num_classes,
            )
        )

        people_features, people_logits = (
            get_multilayer_features_and_logits(
                self.people,
                x,
                self.people_num_classes,
            )
        )

        objects_features, objects_logits = (
            get_multilayer_features_and_logits(
                self.objects_and_vehicles,
                x,
                self.objects_num_classes,
            )
        )

        # ============================================================
        # 2. Apply EXACTLY the same preprocessing used when fitting
        #    the Mahalanobis distributions.
        #
        # prepare_mahalanobis_features() now accepts dictionaries.
        # ============================================================

        outdoors_features = (
            prepare_mahalanobis_features(
                outdoors_features
            )
        )

        nature_features = (
            prepare_mahalanobis_features(
                nature_features
            )
        )

        people_features = (
            prepare_mahalanobis_features(
                people_features
            )
        )

        objects_features = (
            prepare_mahalanobis_features(
                objects_features
            )
        )

        features_by_expert = [
            outdoors_features,
            nature_features,
            people_features,
            objects_features,
        ]

        # ============================================================
        # 3. Calculate Mahalanobis distances
        #
        # Ordering is IMPORTANT.
        #
        # expert-major, layer-minor:
        #
        # 0  outdoors layer1
        # 1  outdoors layer2
        # 2  outdoors layer3
        # 3  outdoors penultimate
        #
        # 4  nature layer1
        # ...
        #
        # 15 objects penultimate
        #
        # This MUST match the ordering used when creating the
        # validation router dataset.
        # ============================================================

        raw_distances_list = []
        nearest_classes_list = []

        '''for expert_idx, model_name in enumerate(
                EXPERT_NAMES
        ):

            expert_features = (
                features_by_expert[
                    expert_idx
                ]
            )

            for layer_name in MULTILAYER_NAMES:
                # ----------------------------------------------------
                # Feature tensor for this expert/layer
                # ----------------------------------------------------

                features = expert_features[
                    layer_name
                ]'''
        for expert_idx, model_name in enumerate(EXPERT_NAMES):

            for layer_name in MULTILAYER_NAMES:
                features = features_by_expert[
                    expert_idx
                ][
                    layer_name
                ]

                # ----------------------------------------------------
                # Retrieve this expert/layer's fitted Mahalanobis
                # statistics.
                #
                # Example:
                #
                # outdoors_layer1_class_means
                # outdoors_layer1_precision
                # ----------------------------------------------------

                class_means = getattr(
                    self,
                    f"{model_name}_{layer_name}_class_means"
                )

                precision = getattr(
                    self,
                    f"{model_name}_{layer_name}_precision"
                )

                # ----------------------------------------------------
                # Nearest-class Mahalanobis distance
                # ----------------------------------------------------

                distance, nearest_class = (
                    specialist_mahalanobis_distance(
                        features=features,
                        class_means=class_means,
                        precision=precision,
                    )
                )

                raw_distances_list.append(
                    distance
                )

                nearest_classes_list.append(
                    nearest_class
                )

        # ============================================================
        # 4. Stack all expert/layer distances
        #
        # raw_distances:
        #     [B, 16]
        #
        # nearest_classes:
        #     [B, 16]
        # ============================================================

        raw_distances = torch.stack(
            raw_distances_list,
            dim=1,
        )

        nearest_classes = torch.stack(
            nearest_classes_list,
            dim=1,
        )

        expected_num_router_features = (
                len(EXPERT_NAMES)
                *
                len(MULTILAYER_NAMES)
        )

        if (
                raw_distances.shape[1]
                != expected_num_router_features
        ):
            raise RuntimeError(
                "Wrong number of Mahalanobis "
                "router features. "
                f"Expected "
                f"{expected_num_router_features}, "
                f"got "
                f"{raw_distances.shape[1]}."
            )

        # ============================================================
        # 5. Relative distances
        #
        # Thresholds have shape:
        #
        #     [4 experts, 4 layers]
        #
        # Flatten them into the SAME ordering as raw_distances.
        #
        # These are ONLY used for the 95%-ID diagnostic.
        # They do NOT choose the expert.
        # ============================================================

        flat_thresholds = (
            self.expert_layer_thresholds
            .reshape(-1)
        )

        if (
                flat_thresholds.numel()
                != raw_distances.shape[1]
        ):
            raise RuntimeError(
                "Threshold count does not match "
                "Mahalanobis feature count. "
                f"thresholds="
                f"{flat_thresholds.numel()}, "
                f"distances="
                f"{raw_distances.shape[1]}"
            )

        relative_distances = (
                raw_distances
                /
                flat_thresholds.unsqueeze(0)
        )

        # ============================================================
        # 6. Prepare the MULTI-LAYER router features
        #
        # This transformation MUST match the transformation used
        # inside fit_mahalanobis_expert_router().
        #
        # raw distances
        #       ↓
        # log()
        #       ↓
        # standardize using validation mean/std
        # ============================================================

        router_features = torch.log(
            raw_distances.clamp_min(
                1e-8
            )
        )

        router_features = (
                                  router_features
                                  -
                                  self.router_feature_mean.unsqueeze(0)
                          ) / (
                              self.router_feature_std.unsqueeze(0)
                          )

        # ============================================================
        # 7. JOINT multi-layer Mahalanobis expert routing
        #
        # [B, 16]
        #     ↓
        # Linear(16, 4)
        #     ↓
        # [B, 4]
        # ============================================================

        router_logits = (
            self.mahalanobis_router(
                router_features
            )
        )

        if SOFT_EXPERT_ROUTING:

            router_log_probs = F.log_softmax(
                router_logits / ROUTER_TEMPERATURE,
                dim=1,
            )

            global_logits = (
                outdoors_logits.new_full(
                    (
                        x.size(0),
                        self.num_classes,
                    ),
                    -1e9,
                )
            )

            specialist_logits = [
                outdoors_logits,
                nature_logits,
                people_logits,
                objects_logits,
            ]

            specialist_indices = [
                self.outdoors_indices,
                self.nature_indices,
                self.people_indices,
                self.objects_and_vehicles_indices,
            ]

            for expert_idx, (
                    local_logits,
                    global_indices,
            ) in enumerate(
                zip(
                    specialist_logits,
                    specialist_indices,
                )
            ):
                local_log_probs = F.log_softmax(
                    local_logits,
                    dim=1,
                )

                global_logits[
                    :,
                    global_indices
                ] = (
                        local_log_probs
                        +
                        router_log_probs[
                            :,
                            expert_idx
                        ].unsqueeze(1)
                )

        else:

            # your existing hard-routing code

            router_probabilities = F.softmax(
                router_logits,
                dim=1,
            )

            # ============================================================
            # 8. Select expert
            #
            # HIGHER router probability wins.
            # ============================================================

            (
                sorted_probabilities,
                sorted_experts,
            ) = torch.sort(
                router_probabilities,
                dim=1,
                descending=True,
            )

            selected_expert = (
                sorted_experts[
                    :,
                    0
                ]
            )

            best_probability = (
                sorted_probabilities[
                    :,
                    0
                ]
            )

            second_best_probability = (
                sorted_probabilities[
                    :,
                    1
                ]
            )

            routing_margin = (
                    best_probability
                    -
                    second_best_probability
            )

            # ============================================================
            # 9. 95%-ID diagnostic
            #
            # The router used all 16 distances.
            #
            # For accepted_as_id, use the PENULTIMATE distance belonging
            # to whichever expert was selected.
            #
            # Example if 4 layers:
            #
            # expert 0 penultimate -> index 3
            # expert 1 penultimate -> index 7
            # expert 2 penultimate -> index 11
            # expert 3 penultimate -> index 15
            # ============================================================

            num_layers = len(
                MULTILAYER_NAMES
            )

            penultimate_layer_idx = (
                MULTILAYER_NAMES.index(
                    "penultimate"
                )
            )

            selected_penultimate_feature_idx = (
                    selected_expert
                    *
                    num_layers
                    +
                    penultimate_layer_idx
            )

            selected_relative_distance = (
                relative_distances.gather(
                    dim=1,

                    index=(
                        selected_penultimate_feature_idx
                        .unsqueeze(1)
                    ),
                )
                .squeeze(1)
            )

            accepted_as_id = (
                    selected_relative_distance
                    <= 1.0
            )

            # ============================================================
            # 10. Construct the global 43-class logit vector
            # ============================================================

            global_logits = (
                outdoors_logits.new_full(
                    (
                        x.size(0),
                        self.num_classes,
                    ),
                    -1e9,
                )
            )

            global_logits[
                :,
                self.outdoors_indices
            ] = outdoors_logits

            global_logits[
                :,
                self.nature_indices
            ] = nature_logits

            global_logits[
                :,
                self.people_indices
            ] = people_logits

            global_logits[
                :,
                self.objects_and_vehicles_indices
            ] = objects_logits

            # ============================================================
            # 11. HARD expert routing
            #
            # Only the classes belonging to the selected expert
            # are allowed to compete.
            # ============================================================

            class_owner = (
                self.global_class_owner
                .unsqueeze(0)
            )

            selected_owner = (
                selected_expert
                .unsqueeze(1)
            )

            wrong_expert_mask = (
                    class_owner
                    !=
                    selected_owner
            )

            global_logits = (
                global_logits.masked_fill(
                    wrong_expert_mask,
                    -1e9,
                )
            )

            # ============================================================
            # 12. Diagnostics
            # ============================================================

            if return_gate_info:
                gate_info = {

                    # --------------------------------------------
                    # Selected expert
                    # --------------------------------------------

                    "selected_expert":
                        selected_expert,

                    # --------------------------------------------
                    # Multi-layer Mahalanobis information
                    #
                    # All three are [B, 16]
                    # --------------------------------------------

                    "raw_distances":
                        raw_distances,

                    "relative_distances":
                        relative_distances,

                    "nearest_classes":
                        nearest_classes,

                    # --------------------------------------------
                    # Router information
                    # --------------------------------------------

                    "router_features":
                        router_features,

                    "router_logits":
                        router_logits,

                    "router_probabilities":
                        router_probabilities,

                    # --------------------------------------------
                    # Routing confidence
                    # --------------------------------------------

                    "best_probability":
                        best_probability,

                    "second_best_probability":
                        second_best_probability,

                    "routing_margin":
                        routing_margin,

                    # --------------------------------------------
                    # Separate 95%-ID diagnostic
                    # --------------------------------------------

                    "selected_relative_distance":
                        selected_relative_distance,

                    "accepted_as_id":
                        accepted_as_id,
                }

                return (
                    global_logits,
                    gate_info,
                )

        return global_logits



# ==============================================================================
# STEP 3: Main Aggregation Pipeline (CORRECTED)
# ==============================================================================

def FORMERLY_aggregate_models_all_layers_with_feature_averaging(
    model_names: list,
    model_dir: Path,
    device: str = 'cpu'
) -> nn.Module: #-> Tuple[nn.Module, Dict]:
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
    local_to_global_indices = {}
    feature_models = {}
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


            local_to_global_indices[model_name] = (
                build_local_to_global_indices(
                    model_name=model_name,
                    new_to_original=mappings[model_name],
                    num_subset_classes=num_subset_classes[
                        model_name
                    ],
                    original_to_aggregated_dict=(
                        original_to_aggregated_dict
                    ),
                )
            )

            print(
                f"\nFitting feature distribution for "
                f"{model_name}"
            )

            '''feature_models[model_name] = (
                fit_specialist_feature_distribution(
                    model=models[model_name],

                    loader=specialist_train_loaders[
                        model_name
                    ],

                    new_to_original=mappings[
                        model_name
                    ],

                    num_subset_classes=(
                        num_subset_classes[
                            model_name
                        ]
                    ),

                    device=device,

                    covariance_regularization=1e-3,
                )
            )'''

        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")
            raise

        print(f"  ✓ Loaded {model_name}")
        print(f"    - Classifier size: {classifier_size}")
        print(f"    - CIFAR classes: {sorted(mappings[model_name].values())}")


    validate_global_class_coverage(
        local_to_global_indices,
        num_global_classes=43,
    )

    # ============================================================
    # STEP 2: Fit Mahalanobis feature distributions
    # ============================================================

    feature_models = {}

    for model_name in model_names:
        print(
            f"\nFitting MULTI-LAYER "
            f"Mahalanobis distributions "
            f"for {model_name}"
        )

        feature_models[
            model_name
        ] = (
            fit_multilayer_specialist_feature_distribution(
                model=models[
                    model_name
                ],

                loader=(
                    specialist_train_loaders[
                        model_name
                    ]
                ),

                new_to_original=(
                    mappings[
                        model_name
                    ]
                ),

                num_subset_classes=(
                    num_subset_classes[
                        model_name
                    ]
                ),

                device=device,

                covariance_regularization=(
                    MAHALANOBIS_COV_REG
                ),

                covariance_mode=(
                    MAHALANOBIS_COVARIANCE_MODE
                ),

                shrinkage=(
                    MAHALANOBIS_SHRINKAGE
                ),
            )
        )



    # ============================================================
    # STEP 2B:
    # Analyze Mahalanobis ID/OOD separation
    # ============================================================

    '''mahalanobis_analysis = (
        analyze_mahalanobis_id_ood_separation(
            models=models,

            feature_models=feature_models,

            validation_loader=val_loader,

            mappings=mappings,

            num_subset_classes=(
                num_subset_classes
            ),

            device=device,

            expert_names=EXPERT_NAMES,
        )
    )'''

    # ============================================================
    # STEP 3: Fit Mahalanobis ID thresholds on VALIDATION data
    # ============================================================

    print(
        "\n===== FITTING MAHALANOBIS "
        "FEATURE-DISTANCE THRESHOLDS ====="
    )

    feature_thresholds = (
        fit_multilayer_feature_distance_thresholds(
            models=models,

            feature_models=feature_models,

            validation_loader=val_loader,

            mappings=mappings,

            num_subset_classes=(
                num_subset_classes
            ),

            device=device,

            quantile=0.95,
        )
    )

    # ============================================================
    # Fit ID-vs-OOD Mahalanobis calibration
    # ============================================================

    #mahalanobis_calibration = (
    fit_mahalanobis_id_ood_calibration(
        models=models,

        feature_models=feature_models,

        validation_loader=val_loader,

        mappings=mappings,

        num_subset_classes=(
            num_subset_classes
        ),

        device=device,

        expert_names=EXPERT_NAMES,
    )
    #)

    router_distances, router_targets = (
        collect_multilayer_mahalanobis_router_dataset(
            models=models,

            feature_models=feature_models,

            validation_loader=val_loader,

            mappings=mappings,

            num_subset_classes=(
                num_subset_classes
            ),

            device=device,
        )
    )

    # ============================================================
    # Sanity check
    # ============================================================

    assert (
            router_distances.shape[1]
            ==
            NUM_ROUTER_FEATURES
    ), (
        f"Expected "
        f"{NUM_ROUTER_FEATURES} "
        f"router features, got "
        f"{router_distances.shape}"
    )

    (
        mahalanobis_router,
        router_feature_mean,
        router_feature_std,
    ) = fit_mahalanobis_expert_router(
        distances=router_distances,

        targets=router_targets,

        device=device,

        epochs=500,

        lr=1e-2,
    )


    global_model = GlobalSpecialistModel(
        outdoors_model=models["outdoors"],

        nature_model=models["nature"],

        people_model=models["people"],

        objects_and_vehicles_model=(
            models["objects_and_vehicles"]
        ),

        outdoors_indices=(
            local_to_global_indices["outdoors"]
        ),

        nature_indices=(
            local_to_global_indices["nature"]
        ),

        people_indices=(
            local_to_global_indices["people"]
        ),

        objects_and_vehicles_indices=(
            local_to_global_indices[
                "objects_and_vehicles"
            ]
        ),

        outdoors_num_classes=(
            num_subset_classes["outdoors"]
        ),

        nature_num_classes=(
            num_subset_classes["nature"]
        ),

        people_num_classes=(
            num_subset_classes["people"]
        ),

        objects_num_classes=(
            num_subset_classes[
                "objects_and_vehicles"
            ]
        ),

        feature_models = feature_models,
        feature_thresholds=(
            feature_thresholds
        ),
        #feature_thresholds = feature_thresholds,
        #mahalanobis_calibration =mahalanobis_calibration,
        mahalanobis_router=mahalanobis_router,
        router_feature_mean=router_feature_mean,
        router_feature_std=router_feature_std,

        num_classes=43,
    )

    #evaluate_mahalanobis_router(global_model, test_loader, mappings, num_subset_classes)

    return global_model


# ==============================================================================
# STEP 4: Save and Test
# ==============================================================================

def save_global_model(global_model: nn.Module,
                      output_path: Path): #,report: Dict):
    """Save the global model and aggregation report."""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = output_path / 'global_model_with_averaged_features.pt'
    torch.save(global_model.state_dict(), model_path)
    print(f"✓ Saved global model to {model_path}")

    '''# Save report
    report_path = output_path / 'aggregation_report.pkl'
    with open(report_path, 'wb') as f:
        pickle.dump(report, f)
    print(f"✓ Saved aggregation report to {report_path}")'''


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

def evaluate_global_specialist_model(
    global_model,
    loader,
    original_to_aggregated_dict:dict,
    output_message:str,
    device="cpu",
):

    global_model.eval()

    cifar_to_global = (
        build_cifar_to_global_lookup(
            original_to_aggregated_dict
        ).to(device)
    )

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():

        for (
            batch_data,
            batch_labels,
            batch_task_ids,
        ) in loader:

            batch_data = batch_data.to(device)

            batch_labels = batch_labels.to(device)

            # -----------------------------------------
            # Original CIFAR label -> global 0..42
            # -----------------------------------------

            global_targets = cifar_to_global[
                batch_labels
            ]

            if torch.any(global_targets < 0):

                invalid_labels = torch.unique(
                    batch_labels[
                        global_targets < 0
                    ]
                )

                raise RuntimeError(
                    "Found classes outside the "
                    "43-class global model: "
                    f"{invalid_labels.tolist()}"
                )

            # -----------------------------------------
            # One task-agnostic global forward pass
            # -----------------------------------------

            '''global_logits = global_model(
                batch_data
            )'''
            if SOFT_EXPERT_ROUTING == False:
                global_logits, gate_info = global_model(
                    batch_data,
                    return_gate_info=True,
                )
            if SOFT_EXPERT_ROUTING == True:
                global_logits = global_model(
                    batch_data
                )

            assert global_logits.shape[1] == 43

            # -----------------------------------------
            # Loss
            # -----------------------------------------

            loss = F.cross_entropy(
                global_logits,
                global_targets,
            )

            # -----------------------------------------
            # Prediction
            # -----------------------------------------

            predictions = global_logits.argmax(
                dim=1
            )

            batch_size = batch_labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            correct += (
                predictions == global_targets
            ).sum().item()

            total += batch_size

    accuracy = correct / total

    avg_loss = total_loss / total

    print(
        f"{output_message} --> Loss = {avg_loss:.6f} | Accuracy = {accuracy * 100:.2f}%"
    )

    #return accuracy

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
            #targets_all_experiences = map_original_to_aggregated(batch_labels)


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
            #batch_labels = map_original_to_aggregated(batch_labels)
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


@torch.no_grad()
def evaluate_mahalanobis_router(
    global_model,
    loader,
    mappings,
    num_subset_classes,
    device="cpu",
):

    global_model.eval()

    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings,
            num_subset_classes,
            EXPERT_NAMES,
        ).to(device)
    )

    confusion = torch.zeros(
        4,
        4,
        dtype=torch.long,
    )

    routing_correct = 0
    classification_correct_when_route_correct = 0
    routed_correct_total = 0
    total = 0

    correct_route_margins = []
    wrong_route_margins = []

    for batch_data, batch_labels, _ in loader:

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        true_expert = cifar_to_expert[
            batch_labels
        ]

        global_logits, gate_info = (
            global_model(
                batch_data,
                return_gate_info=True,
            )
        )

        selected_expert = gate_info[
            "selected_expert"
        ]

        route_correct = (
            selected_expert
            == true_expert
        )

        routing_correct += (
            route_correct.sum().item()
        )

        predictions = global_logits.argmax(
            dim=1
        )

        global_targets = (
            build_cifar_to_global_lookup(
                original_to_aggregated_dict
            )
            .to(device)[batch_labels]
        )

        if route_correct.any():

            classification_correct_when_route_correct += (
                predictions[route_correct]
                ==
                global_targets[route_correct]
            ).sum().item()

            routed_correct_total += (
                route_correct.sum().item()
            )

        margins = gate_info[
            "routing_margin"
        ]

        correct_route_margins.append(
            margins[
                route_correct
            ].cpu()
        )

        wrong_route_margins.append(
            margins[
                ~route_correct
            ].cpu()
        )

        flat = (
            true_expert * 4
            +
            selected_expert
        )

        confusion += torch.bincount(
            flat,
            minlength=16,
        ).reshape(4, 4).cpu()

        total += batch_labels.size(0)

    print(
        "Routing accuracy:",
        100.0 * routing_correct / total
    )

    print(
        "Classification accuracy GIVEN correct route:",
        100.0
        * classification_correct_when_route_correct
        / max(routed_correct_total, 1)
    )

    print("\nRouting confusion:")
    print(confusion)

    row_totals = confusion.sum(
        dim=1
    )

    per_expert_recall = (
            confusion.diag().float()
            /
            row_totals.clamp_min(1).float()
    )

    balanced_routing_accuracy = (
        per_expert_recall.mean()
    )

    print(
        "\nPer-expert routing recall:"
    )

    for expert_idx, model_name in enumerate(
            EXPERT_NAMES
    ):
        print(
            f"{model_name:22s}: "
            f"{per_expert_recall[expert_idx].item() * 100:.2f}%"
        )

    print(
        "Balanced routing accuracy:",
        balanced_routing_accuracy.item()
        * 100.0
    )

    return confusion
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
        global_model = FORMERLY_aggregate_models_all_layers_with_feature_averaging(#global_model, report = FORMERLY_aggregate_models_all_layers_with_feature_averaging(
            model_names=model_names,
            model_dir=MODEL_DIR,
            device=DEVICE
        )

        # ============================================================
        # Save
        # ============================================================
        save_global_model(global_model, OUTPUT_DIR)#, report)

        # ============================================================
        # Test
        # ============================================================
        #output = test_global_model(global_model, device=DEVICE)

        return global_model

    except Exception as e:
        print(f"\n✗ Aggregation failed: {e}")
        raise


if __name__ == '__main__':
    global_model = main()
    # evaluate_aggregated_model_FedAvg_code_v3(global_model, test_loader)
    # evaluate_aggregated_model(global_model, test_loader_without_objects_and_vehicles, "Evaluating global model on test_loader_without_objects_and_vehicles")
    '''evaluate_aggregated_model(global_model, test_loader,
                              "Evaluating global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_aggregated_model(global_model, outdoors_loader, "Evaluating global model on outdoors_loader")
    evaluate_aggregated_model(global_model, nature_loader, "Evaluating global model on nature_loader")
    evaluate_aggregated_model(global_model, people_loader, "Evaluating global model on people_loader")
    evaluate_aggregated_model(global_model, objects_and_vehicles_loader,
                              "Evaluating global model on objects_and_vehicles_loader")'''

    #evaluate_aggregated_model_FedAvg_code_v3(global_model, test_loader)
    #evaluate_global_specialist_model(global_model, test_loader_without_objects_and_vehicles, "Evaluating global model on test_loader_without_objects_and_vehicles")
    evaluate_global_specialist_model(global_model, test_loader, original_to_aggregated_dict,"Evaluating global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_global_specialist_model(global_model, outdoors_loader, original_to_aggregated_dict,"Evaluating global model on outdoors_loader")
    evaluate_global_specialist_model(global_model, nature_loader, original_to_aggregated_dict,"Evaluating global model on nature_loader")
    evaluate_global_specialist_model(global_model, people_loader, original_to_aggregated_dict,"Evaluating global model on people_loader")
    evaluate_global_specialist_model(global_model, objects_and_vehicles_loader, original_to_aggregated_dict,"Evaluating global model on objects_and_vehicles_loader")
    '''print("\n\n############# FINE-TUNED GLOBAL MODEL EVALUATION BEGINS NOW!!!! #############")
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
    evaluate_aggregated_model(fine_tuned_global_model, objects_and_vehicles_loader,"Evaluating FINE-TUNED global model on objects_and_vehicles_loader")'''
    print("\n\n")
    evaluate_global_specialist_model(global_model, first_experience_test_loader, original_to_aggregated_dict,
                              "Evaluating Mahalanobis global model on first_experience_test_loader")
    evaluate_global_specialist_model(global_model, middle_experience_test_loader, original_to_aggregated_dict,
                              "Evaluating Mahalanobis global model on middle_experience_test_loader")
    evaluate_global_specialist_model(global_model, last_experience_test_loader, original_to_aggregated_dict,
                              "Evaluating Mahalanobis global model on last_experience_test_loader")