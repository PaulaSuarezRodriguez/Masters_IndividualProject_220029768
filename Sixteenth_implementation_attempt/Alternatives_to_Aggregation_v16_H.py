
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

VERBOSE = False
EXPERT_NAMES = [
    "outdoors",
    "nature",
    "people",
    "objects_and_vehicles",
]
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

class MahalanobisExpertRouter(nn.Module):

    def __init__(
        self,
        num_experts=4,
    ):
        super().__init__()

        self.linear = nn.Linear(
            num_experts,
            num_experts,
        )

    def forward(
        self,
        normalized_log_distances,
    ):

        return self.linear(
            normalized_log_distances
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

    return weights


@torch.no_grad()
def collect_mahalanobis_router_dataset(
    models,
    feature_models,
    validation_loader,
    mappings,
    num_subset_classes,
    device="cpu",
):

    device = torch.device(device)

    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings=mappings,
            num_subset_classes=num_subset_classes,
            expert_names=EXPERT_NAMES,
        ).to(device)
    )

    all_distances = []
    all_targets = []

    for model_name in EXPERT_NAMES:

        models[model_name].to(device)
        models[model_name].eval()

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

        true_expert = cifar_to_expert[
            batch_labels
        ]

        batch_distances = []

        for model_name in EXPERT_NAMES:

            n = int(
                num_subset_classes[
                    model_name
                ]
            )

            features, _ = (
                get_features_and_logits(
                    models[model_name],
                    batch_data,
                    n,
                )
            )

            features = (
                prepare_mahalanobis_features(
                    features
                )
            )

            class_means = (
                feature_models[
                    model_name
                ]["class_means"]
                .to(device)
                .float()
            )

            precision = (
                feature_models[
                    model_name
                ]["precision"]
                .to(device)
                .float()
            )

            distance, _ = (
                specialist_mahalanobis_distance(
                    features=features,
                    class_means=class_means,
                    precision=precision,
                )
            )

            batch_distances.append(
                distance
            )

        batch_distances = torch.stack(
            batch_distances,
            dim=1,
        )

        all_distances.append(
            batch_distances.cpu()
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

    return distances, targets

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

    router = MahalanobisExpertRouter(
        num_experts=len(
            EXPERT_NAMES
        )
    ).to(device)
    '''num_router_features = (
        router_features.shape[1]
    )

    router = MahalanobisExpertRouter(
        num_features=num_router_features,
        num_experts=len(EXPERT_NAMES),
    ).to(device)'''

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


def prepare_mahalanobis_features(
    features: torch.Tensor,
):
    features = features.float()

    if MAHALANOBIS_L2_NORMALIZE:

        features = F.normalize(
            features,
            p=2,
            dim=1,
        )

    return features


@torch.no_grad()
def fit_specialist_feature_distribution(
    model,
    loader,
    new_to_original,
    num_subset_classes,
    device="cpu",
    covariance_regularization=MAHALANOBIS_COV_REG,
    covariance_mode=MAHALANOBIS_COVARIANCE_MODE,
    shrinkage=MAHALANOBIS_SHRINKAGE,
):

    device = torch.device(device)

    model = model.to(device)
    model.eval()

    cifar_to_local = invert_new_to_original(
        new_to_original,
        num_subset_classes,
    )

    features_by_class = {
        local_idx: []
        for local_idx in range(
            num_subset_classes
        )
    }

    # ============================================================
    # Collect TRAIN features
    # ============================================================

    for (
        batch_data,
        batch_labels,
        batch_task_ids,
    ) in loader:

        batch_data = batch_data.to(device)

        features, _ = get_features_and_logits(
            model,
            batch_data,
            num_subset_classes,
        )

        # IMPORTANT:
        # Same feature transformation used later at inference.
        features = prepare_mahalanobis_features(
            features
        )

        for feature, cifar_label in zip(
            features,
            batch_labels,
        ):

            cifar_class = int(
                cifar_label.item()
            )

            if cifar_class not in cifar_to_local:

                raise RuntimeError(
                    f"CIFAR class {cifar_class} "
                    "does not belong to this specialist."
                )

            local_idx = cifar_to_local[
                cifar_class
            ]

            features_by_class[
                local_idx
            ].append(
                feature.cpu()
            )

    # ============================================================
    # Class prototypes
    # ============================================================

    class_means = []

    centered_features = []

    for local_idx in range(
        num_subset_classes
    ):

        class_features = torch.stack(
            features_by_class[
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
            - class_mean.unsqueeze(0)
        )

    class_means = torch.stack(
        class_means
    )

    # ============================================================
    # Shared within-class covariance
    # ============================================================

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

    feature_dim = covariance.shape[0]

    # ============================================================
    # Covariance estimator
    # ============================================================

    if covariance_mode == "full":

        pass

    elif covariance_mode == "diagonal":

        covariance = torch.diag(
            torch.diag(
                covariance
            )
        )

    elif covariance_mode == "shrinkage":

        diagonal_covariance = torch.diag(
            torch.diag(
                covariance
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
            "'full', 'diagonal', or 'shrinkage'"
        )

    # ============================================================
    # Scale-aware ridge regularization
    #
    # Better than blindly adding absolute 1e-3 after L2
    # normalization.
    # ============================================================

    covariance_scale = (
        torch.diag(
            covariance
        )
        .mean()
        .clamp_min(1e-8)
    )

    covariance = (
        covariance
        +
        covariance_regularization
        * covariance_scale
        * torch.eye(
            feature_dim
        )
    )

    precision = torch.linalg.pinv(
        covariance
    )

    print(
        "class_means:",
        class_means.shape,
        "| precision:",
        precision.shape,
        "| covariance:",
        covariance_mode,
        "| L2:",
        MAHALANOBIS_L2_NORMALIZE,
    )

    return {
        "class_means":
            class_means,

        "precision":
            precision,

        "covariance":
            covariance,
    }

@torch.no_grad()
def get_features_and_logits(
    model,
    x,
    num_classes,
):
    """
    Returns:
        features: [B, D]
        logits:   [B, num_classes]

    features are the input to model.linear.
    """

    captured = {}

    def capture_features(module, inputs):
        captured["features"] = inputs[0].detach()

    handle = model.linear.register_forward_pre_hook(
        capture_features
    )

    try:
        logits = model(x)
    finally:
        handle.remove()

    features = captured["features"]

    logits = logits[:, :num_classes]

    return features, logits


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
def fit_feature_distance_thresholds(
    models,
    feature_models,
    validation_loader,
    mappings,
    num_subset_classes,
    device="cpu",
    quantile=0.95,
):
    """
    Fits an ID-distance threshold for every expert.

    Only validation samples belonging to that expert are
    used to define its normal feature-distance range.
    """

    device = torch.device(device)

    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings,
            num_subset_classes,
            EXPERT_NAMES,
        ).to(device)
    )

    id_distances = {
        name: []
        for name in EXPERT_NAMES
    }

    for batch_data, batch_labels, batch_task_ids in (
        validation_loader
    ):

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        true_expert = cifar_to_expert[
            batch_labels
        ]

        for expert_idx, model_name in enumerate(
            EXPERT_NAMES
        ):

            n = num_subset_classes[
                model_name
            ]

            features, _ = get_features_and_logits(
                models[model_name],
                batch_data,
                n,
            )

            features = prepare_mahalanobis_features(
                features
            )

            feature_model = feature_models[
                model_name
            ]

            means = feature_model[
                "class_means"
            ].to(device)

            precision = feature_model[
                "precision"
            ].to(device)

            distances, _ = (
                specialist_mahalanobis_distance(
                    features.float(),
                    means,
                    precision,
                )
            )

            mask = (
                true_expert
                == expert_idx
            )

            if mask.any():

                id_distances[
                    model_name
                ].append(
                    distances[
                        mask
                    ].cpu()
                )

    thresholds = {}

    medians = {}

    for model_name in EXPERT_NAMES:

        distances = torch.cat(
            id_distances[
                model_name
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
        ] = threshold

        medians[
            model_name
        ] = median

        print(
            f"{model_name:22s} | "
            f"median={median.item():.4f} | "
            f"{int(quantile * 100)}% "
            f"threshold="
            f"{threshold.item():.4f}"
        )

    return thresholds

@torch.no_grad()
def analyze_mahalanobis_id_ood_separation(
    models,
    feature_models,
    validation_loader,
    mappings,
    num_subset_classes,
    device="cpu",
    expert_names=EXPERT_NAMES,
):
    """
    For every specialist, collect:

        ID distances:
            validation samples belonging to that specialist

        OOD distances:
            validation samples belonging to all OTHER specialists

    Then print:
        - ID median
        - OOD median
        - ID 95th percentile
        - OOD 5th percentile
        - separation gap

    Positive separation gap:
        OOD 5% > ID 95%

        => even most unusually distant ID samples are still
           closer than most unusually close OOD samples.

        This is excellent separation.

    Negative separation gap:
        OOD 5% < ID 95%

        => ID and OOD distance distributions overlap.

        The more negative this is, the harder routing becomes.
    """

    device = torch.device(device)

    # ------------------------------------------------------------
    # CIFAR-100 original class -> correct specialist
    #
    # 0 = outdoors
    # 1 = nature
    # 2 = people
    # 3 = objects_and_vehicles
    # ------------------------------------------------------------

    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings=mappings,
            num_subset_classes=num_subset_classes,
            expert_names=expert_names,
        ).to(device)
    )

    # ------------------------------------------------------------
    # Put specialists into evaluation mode
    # ------------------------------------------------------------

    for model_name in expert_names:

        models[model_name] = (
            models[model_name]
            .to(device)
        )

        models[model_name].eval()

    # ------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------

    id_distances = {
        model_name: []
        for model_name in expert_names
    }

    ood_distances = {
        model_name: []
        for model_name in expert_names
    }

    # ============================================================
    # Run the entire VALIDATION SET through every specialist
    # ============================================================

    for (
        batch_data,
        batch_labels,
        batch_task_ids,
    ) in validation_loader:

        batch_data = batch_data.to(device)

        batch_labels = batch_labels.to(device)

        # --------------------------------------------------------
        # Which specialist SHOULD own each sample?
        # --------------------------------------------------------

        true_expert = cifar_to_expert[
            batch_labels
        ]

        if torch.any(true_expert < 0):

            invalid_labels = torch.unique(
                batch_labels[
                    true_expert < 0
                ]
            )

            raise RuntimeError(
                "Validation set contains CIFAR classes "
                "which are not owned by any specialist: "
                f"{invalid_labels.tolist()}"
            )

        # ========================================================
        # Evaluate the SAME validation batch with EVERY specialist
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
            # Extract that specialist's feature representation
            # ----------------------------------------------------

            features, _ = get_features_and_logits(
                models[model_name],
                batch_data,
                n,
            )

            features = prepare_mahalanobis_features(
                features
            )

            # ----------------------------------------------------
            # Retrieve fitted Mahalanobis distribution
            # ----------------------------------------------------

            class_means = (
                feature_models[
                    model_name
                ]["class_means"]
                .to(device)
                .float()
            )

            precision = (
                feature_models[
                    model_name
                ]["precision"]
                .to(device)
                .float()
            )

            # ----------------------------------------------------
            # Distance to nearest class prototype
            # ----------------------------------------------------

            distances, nearest_class = (
                specialist_mahalanobis_distance(
                    features=features.float(),
                    class_means=class_means,
                    precision=precision,
                )
            )

            # ----------------------------------------------------
            # Split this specialist's distances into:
            #
            # ID:
            #   sample actually belongs to this specialist
            #
            # OOD:
            #   sample belongs to one of the OTHER specialists
            # ----------------------------------------------------

            id_mask = (
                true_expert
                == expert_idx
            )

            ood_mask = (
                true_expert
                != expert_idx
            )

            if id_mask.any():

                id_distances[
                    model_name
                ].append(
                    distances[
                        id_mask
                    ].detach().cpu()
                )

            if ood_mask.any():

                ood_distances[
                    model_name
                ].append(
                    distances[
                        ood_mask
                    ].detach().cpu()
                )

    # ============================================================
    # Calculate statistics
    # ============================================================

    results = {}

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MAHALANOBIS ID vs OOD DISTANCE ANALYSIS"
    )

    print(
        "=" * 90
    )

    for model_name in expert_names:

        if len(
            id_distances[model_name]
        ) == 0:

            raise RuntimeError(
                f"No ID validation distances "
                f"collected for {model_name}"
            )

        if len(
            ood_distances[model_name]
        ) == 0:

            raise RuntimeError(
                f"No OOD validation distances "
                f"collected for {model_name}"
            )

        id_d = torch.cat(
            id_distances[
                model_name
            ]
        ).float()

        ood_d = torch.cat(
            ood_distances[
                model_name
            ]
        ).float()

        # --------------------------------------------------------
        # Main statistics
        # --------------------------------------------------------

        id_median = torch.median(
            id_d
        ).item()

        ood_median = torch.median(
            ood_d
        ).item()

        id_95 = torch.quantile(
            id_d,
            0.95,
        ).item()

        ood_05 = torch.quantile(
            ood_d,
            0.05,
        ).item()

        id_mean = id_d.mean().item()

        ood_mean = ood_d.mean().item()

        # --------------------------------------------------------
        # Important:
        #
        # Positive:
        #     there is a clean gap between ID and OOD
        #
        # Negative:
        #     the distributions overlap
        # --------------------------------------------------------

        separation_gap = (
            ood_05
            -
            id_95
        )

        # --------------------------------------------------------
        # Median separation ratio
        #
        # > 1 means OOD median is farther than ID median.
        # --------------------------------------------------------

        median_ratio = (
            ood_median
            /
            max(
                id_median,
                1e-12,
            )
        )

        results[
            model_name
        ] = {
            "id_distances":
                id_d,

            "ood_distances":
                ood_d,

            "id_mean":
                id_mean,

            "ood_mean":
                ood_mean,

            "id_median":
                id_median,

            "ood_median":
                ood_median,

            "id_95":
                id_95,

            "ood_05":
                ood_05,

            "separation_gap":
                separation_gap,

            "median_ratio":
                median_ratio,
        }

        print(
            f"\n{model_name.upper()}"
        )

        print(
            "-" * 60
        )

        print(
            f"Number ID samples:  "
            f"{len(id_d)}"
        )

        print(
            f"Number OOD samples: "
            f"{len(ood_d)}"
        )

        print(
            f"ID mean:            "
            f"{id_mean:.4f}"
        )

        print(
            f"OOD mean:           "
            f"{ood_mean:.4f}"
        )

        print(
            f"ID median:          "
            f"{id_median:.4f}"
        )

        print(
            f"OOD median:         "
            f"{ood_median:.4f}"
        )

        print(
            f"ID 95%:             "
            f"{id_95:.4f}"
        )

        print(
            f"OOD 5%:             "
            f"{ood_05:.4f}"
        )

        print(
            f"Separation gap "
            f"(OOD5 - ID95): "
            f"{separation_gap:.4f}"
        )

        print(
            f"OOD/ID median ratio: "
            f"{median_ratio:.4f}"
        )

        # --------------------------------------------------------
        # Human-readable diagnosis
        # --------------------------------------------------------

        if separation_gap > 0:

            print(
                "✓ GOOD: ID and OOD have a "
                "clean 90%-central separation."
            )

        else:

            print(
                "✗ OVERLAP: ID and OOD "
                "distance distributions overlap."
            )

    print(
        "\n"
        + "=" * 90
    )

    return results

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

        features, _ = get_features_and_logits(
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
):
    """
    For each expert, fit Gaussian distributions to:

        log(Mahalanobis distance | ID)
        log(Mahalanobis distance | OOD)

    Returns calibration statistics used for expert routing.
    """

    device = torch.device(device)

    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings=mappings,
            num_subset_classes=num_subset_classes,
            expert_names=expert_names,
        ).to(device)
    )

    id_distances = {
        name: []
        for name in expert_names
    }

    ood_distances = {
        name: []
        for name in expert_names
    }

    for model_name in expert_names:
        models[model_name].to(device)
        models[model_name].eval()

    # ============================================================
    # Collect ID and OOD distances
    # ============================================================

    for (
        batch_data,
        batch_labels,
        batch_task_ids,
    ) in validation_loader:

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        true_expert = cifar_to_expert[
            batch_labels
        ]

        for expert_idx, model_name in enumerate(
            expert_names
        ):

            n = int(
                num_subset_classes[
                    model_name
                ]
            )

            features, _ = (
                get_features_and_logits(
                    models[model_name],
                    batch_data,
                    n,
                )
            )

            # If you introduced L2 normalization:
            features = prepare_mahalanobis_features(
                features
            )

            class_means = (
                feature_models[
                    model_name
                ]["class_means"]
                .to(device)
                .float()
            )

            precision = (
                feature_models[
                    model_name
                ]["precision"]
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

            id_mask = (
                true_expert == expert_idx
            )

            ood_mask = (
                true_expert != expert_idx
            )

            if id_mask.any():

                id_distances[
                    model_name
                ].append(
                    distances[
                        id_mask
                    ].cpu()
                )

            if ood_mask.any():

                ood_distances[
                    model_name
                ].append(
                    distances[
                        ood_mask
                    ].cpu()
                )

    # ============================================================
    # Fit distributions in LOG-DISTANCE space
    # ============================================================

    calibration = {}

    print(
        "\n===== MAHALANOBIS ID/OOD CALIBRATION ====="
    )

    for model_name in expert_names:

        id_d = torch.cat(
            id_distances[
                model_name
            ]
        ).float()

        ood_d = torch.cat(
            ood_distances[
                model_name
            ]
        ).float()

        # Log transformation helps with the strongly
        # right-skewed Mahalanobis distance distributions.
        log_id = torch.log(
            id_d.clamp_min(1e-8)
        )

        log_ood = torch.log(
            ood_d.clamp_min(1e-8)
        )

        id_mean = log_id.mean()
        id_std = log_id.std(
            unbiased=False
        ).clamp_min(1e-6)

        ood_mean = log_ood.mean()
        ood_std = log_ood.std(
            unbiased=False
        ).clamp_min(1e-6)

        calibration[
            model_name
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
            f"{model_name:22s} | "
            f"ID log-d: "
            f"{id_mean.item():.4f} ± "
            f"{id_std.item():.4f} | "
            f"OOD log-d: "
            f"{ood_mean.item():.4f} ± "
            f"{ood_std.item():.4f}"
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
        mahalanobis_calibration,
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
        # Mahalanobis feature statistics
        # ============================================================

        for model_name in EXPERT_NAMES:
            self.register_buffer(
                f"{model_name}_class_means",
                feature_models[
                    model_name
                ]["class_means"].float()
            )

            self.register_buffer(
                f"{model_name}_precision",
                feature_models[
                    model_name
                ]["precision"].float()
            )

        self.register_buffer(
            "expert_thresholds",
            torch.stack(
                [
                    feature_thresholds[name].float()
                    for name in EXPERT_NAMES
                ]
            )
        )

        # ============================================================
        # Register the calibration statistics
        # ============================================================
        self.register_buffer(
            "id_log_distance_means",
            torch.stack([
                mahalanobis_calibration[
                    name
                ]["id_mean"].float()
                for name in EXPERT_NAMES
            ])
        )

        self.register_buffer(
            "id_log_distance_stds",
            torch.stack([
                mahalanobis_calibration[
                    name
                ]["id_std"].float()
                for name in EXPERT_NAMES
            ])
        )

        self.register_buffer(
            "ood_log_distance_means",
            torch.stack([
                mahalanobis_calibration[
                    name
                ]["ood_mean"].float()
                for name in EXPERT_NAMES
            ])
        )

        self.register_buffer(
            "ood_log_distance_stds",
            torch.stack([
                mahalanobis_calibration[
                    name
                ]["ood_std"].float()
                for name in EXPERT_NAMES
            ])
        )

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
        # 1. Extract BOTH penultimate features and native logits
        # ============================================================

        outdoors_features, outdoors_logits = (
            get_features_and_logits(
                self.outdoors,
                x,
                self.outdoors_num_classes,
            )
        )

        nature_features, nature_logits = (
            get_features_and_logits(
                self.nature,
                x,
                self.nature_num_classes,
            )
        )

        people_features, people_logits = (
            get_features_and_logits(
                self.people,
                x,
                self.people_num_classes,
            )
        )

        objects_features, objects_logits = (
            get_features_and_logits(
                self.objects_and_vehicles,
                x,
                self.objects_num_classes,
            )
        )

        features_by_expert = [
            outdoors_features,
            nature_features,
            people_features,
            objects_features,
        ]

        # ============================================================
        # 2. Calculate Mahalanobis distances for ALL four experts
        # ============================================================

        raw_distances_list = []

        # Still retained for the 95%-ID acceptance diagnostic.
        # This is NO LONGER used to choose the expert.
        relative_distances_list = []

        nearest_classes_list = []

        for expert_idx, model_name in enumerate(
                EXPERT_NAMES
        ):
            features = features_by_expert[
                expert_idx
            ].float()

            # --------------------------------------------------------
            # IMPORTANT:
            # If you fitted the Mahalanobis distributions using
            # L2-normalized features, apply exactly the same
            # transformation here.
            # --------------------------------------------------------

            # Uncomment/use this if you have implemented it:
            #
            features = prepare_mahalanobis_features(features)

            class_means = getattr(
                self,
                f"{model_name}_class_means"
            )

            precision = getattr(
                self,
                f"{model_name}_precision"
            )

            # --------------------------------------------------------
            # Distance to nearest known class belonging to this expert
            # --------------------------------------------------------

            distance, nearest_class = (
                specialist_mahalanobis_distance(
                    features=features,
                    class_means=class_means,
                    precision=precision,
                )
            )

            # --------------------------------------------------------
            # Existing 95%-ID threshold.
            #
            # KEEP THIS ONLY AS AN ID/OOD ACCEPTANCE DIAGNOSTIC.
            # It will NOT determine which expert wins anymore.
            # --------------------------------------------------------

            threshold = self.expert_thresholds[
                expert_idx
            ]

            relative_distance = (
                    distance
                    /
                    threshold
            )

            raw_distances_list.append(
                distance
            )

            relative_distances_list.append(
                relative_distance
            )

            nearest_classes_list.append(
                nearest_class
            )

        # ============================================================
        # IMPORTANT:
        # Stack AFTER all four experts have been processed.
        #
        # Shapes:
        #
        # raw_distances:
        #     [B, 4]
        #
        # relative_distances:
        #     [B, 4]
        #
        # nearest_classes:
        #     [B, 4]
        # ============================================================

        raw_distances = torch.stack(
            raw_distances_list,
            dim=1,
        )

        relative_distances = torch.stack(
            relative_distances_list,
            dim=1,
        )

        nearest_classes = torch.stack(
            nearest_classes_list,
            dim=1,
        )

        # ============================================================
        # 3. ID-vs-OOD CALIBRATED MAHALANOBIS ROUTING
        #
        # This REPLACES:
        #
        #     selected_expert =
        #         relative_distances.argmin(dim=1)
        #
        # We no longer choose whichever expert has the smallest
        # distance / threshold.
        # ============================================================

        # ------------------------------------------------------------
        # Mahalanobis distances are positive and typically right-skewed.
        #
        # Fit/calibration was performed in log-distance space,
        # therefore inference must use the same transformation.
        # ------------------------------------------------------------

        log_distances = torch.log(
            raw_distances.clamp_min(
                1e-8
            )
        )

        # ------------------------------------------------------------
        # Log probability of each observed distance assuming
        # the sample IS in-distribution for that expert.
        #
        # Output:
        #     [B, 4]
        # ------------------------------------------------------------

        id_log_probability = (
            gaussian_log_probability(
                x=log_distances,

                mean=(
                    self.id_log_distance_means
                    .unsqueeze(0)
                ),

                std=(
                    self.id_log_distance_stds
                    .unsqueeze(0)
                ),
            )
        )

        # ------------------------------------------------------------
        # Log probability of each observed distance assuming
        # the sample is OOD for that expert.
        #
        # Output:
        #     [B, 4]
        # ------------------------------------------------------------

        ood_log_probability = (
            gaussian_log_probability(
                x=log_distances,

                mean=(
                    self.ood_log_distance_means
                    .unsqueeze(0)
                ),

                std=(
                    self.ood_log_distance_stds
                    .unsqueeze(0)
                ),
            )
        )

        # ============================================================
        # JOINT Mahalanobis expert routing
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
                          ) / self.router_feature_std.unsqueeze(0)

        router_logits = (
            self.mahalanobis_router(
                router_features
            )
        )

        router_probabilities = F.softmax(
            router_logits,
            dim=1,
        )

        sorted_probabilities, sorted_experts = (
            torch.sort(
                router_probabilities,
                dim=1,
                descending=True,
            )
        )

        selected_expert = sorted_experts[
            :, 0
        ]

        best_probability = (
            sorted_probabilities[
                :, 0
            ]
        )

        second_best_probability = (
            sorted_probabilities[
                :, 1
            ]
        )

        routing_margin = (
                best_probability
                -
                second_best_probability
        )

        # ============================================================
        # 7. Separate 95%-ID acceptance diagnostic
        #
        # We need the relative Mahalanobis distance belonging to
        # whichever expert was selected above.
        # ============================================================

        selected_relative_distance = (
            relative_distances.gather(
                dim=1,

                index=(
                    selected_expert
                    .unsqueeze(1)
                ),
            )
            .squeeze(1)
        )

        # ------------------------------------------------------------
        # <= 1 means:
        #
        # raw Mahalanobis distance
        # <=
        # that expert's 95%-ID threshold
        #
        # IMPORTANT:
        # This does NOT choose the expert.
        # ------------------------------------------------------------

        accepted_as_id = (
                selected_relative_distance
                <= 1.0
        )

        # ============================================================
        # 8. Construct the global 43-class logit vector
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
        # 9. HARD EXPERT ROUTING
        #
        # Only classes belonging to the expert selected by the
        # ID-vs-OOD likelihood ratio may compete.
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
        # 10. Return diagnostics if requested
        # ============================================================

        if return_gate_info:
            gate_info = {

                # --------------------------------------------
                # Selected specialist
                # --------------------------------------------

                "selected_expert":
                    selected_expert,

                # --------------------------------------------
                # Mahalanobis distances
                # --------------------------------------------

                "raw_distances":
                    raw_distances,

                "relative_distances":
                    relative_distances,

                "nearest_classes":
                    nearest_classes,

                # --------------------------------------------
                # ID/OOD calibration information
                # --------------------------------------------

                "id_log_probability":
                    id_log_probability,

                "ood_log_probability":
                    ood_log_probability,

                # --------------------------------------------
                # Routing confidence
                # --------------------------------------------

                "routing_margin":
                    routing_margin,

                # --------------------------------------------
                # Separate OOD acceptance diagnostic
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
            f"\nFitting Mahalanobis feature distribution "
            f"for {model_name}"
        )

        feature_models[model_name] = (
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
        )



    # ============================================================
    # STEP 2B:
    # Analyze Mahalanobis ID/OOD separation
    # ============================================================

    mahalanobis_analysis = (
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
    )

    # ============================================================
    # STEP 3: Fit Mahalanobis ID thresholds on VALIDATION data
    # ============================================================

    print(
        "\n===== FITTING MAHALANOBIS "
        "FEATURE-DISTANCE THRESHOLDS ====="
    )

    feature_thresholds = (
        fit_feature_distance_thresholds(
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

    mahalanobis_calibration = (
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
    )

    router_distances, router_targets = (
        collect_mahalanobis_router_dataset(
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
        mahalanobis_calibration =mahalanobis_calibration,
        mahalanobis_router=mahalanobis_router,
        router_feature_mean=router_feature_mean,
        router_feature_std=router_feature_std,

        num_classes=43,
    )

    evaluate_mahalanobis_router(global_model, test_loader, mappings, num_subset_classes)

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
            global_logits, gate_info = global_model(
                batch_data,
                return_gate_info=True,
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
