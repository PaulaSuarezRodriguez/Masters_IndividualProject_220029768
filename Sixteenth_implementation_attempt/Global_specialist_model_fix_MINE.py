"""
Drop-in replacement pieces for Alternatives_to_Aggregation_v16_B.py

Replaces the broken `scale_specialised_logits_globally` (global min-max
stretch) with:
  1. Per-expert temperature scaling (proper calibration, fit on a held-out
     validation set with NLL loss).
  2. Confidence comparison done on *softmax probabilities*, not rescaled
     logit magnitudes.
  3. Optional confidence threshold with a fallback path for "no expert is
     confident enough" cases.

Integrate by:
  - Adding TemperatureScaler + fit_temperature to your model file.
  - Fitting one temperature per expert on each expert's own validation
    stream (in its own label space) BEFORE building GlobalSpecialistModel.
  - Passing the fitted temperatures into GlobalSpecialistModel.
  - Replacing scale_specialised_logits_globally / forward() with the
    versions below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    shuffle=True
)
print("print first_experience_test_loader", first_experience_test_loader)
print("print first_experience_test_loader.dataset", first_experience_test_loader.dataset)

combined_dataset_outdoors_middle_experience_test = ConcatDataset([
    #outdoors_specialised_models_test_stream_class_il[0].dataset,
    #outdoors_specialised_models_test_stream_class_il[1].dataset,
    outdoors_specialised_models_test_stream_class_il[2].dataset,
    #outdoors_specialised_models_test_stream_class_il[3].dataset,
    #outdoors_specialised_models_test_stream_class_il[4].dataset,
])

combined_dataset_nature_middle_experience_test = ConcatDataset([
    #nature_specialised_models_test_stream_class_il[0].dataset,
    #nature_specialised_models_test_stream_class_il[1].dataset,
    nature_specialised_models_test_stream_class_il[2].dataset,
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
    shuffle=True
)
print("print middle_experience_test_loader", middle_experience_test_loader)
print("print middle_experience_test_loader.dataset", middle_experience_test_loader.dataset)


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
    shuffle=True
)
print("print last_experience_test_loader", last_experience_test_loader)
print("print last_experience_test_loader.dataset", last_experience_test_loader.dataset)


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

outdoors_val_loader =DataLoader(
    combined_dataset_outdoors_val,
    batch_size=32,
    shuffle=True
)

nature_val_loader =DataLoader(
    combined_dataset_nature_val,
    batch_size=32,
    shuffle=True
)

people_val_loader =DataLoader(
    combined_dataset_people_val,
    batch_size=32,
    shuffle=True
)

objects_and_vehicles_val_loader =DataLoader(
    combined_dataset_objects_and_vehicles_val,
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

# ==============================================================================
# 1. TEMPERATURE SCALING (real calibration, per expert)
# ==============================================================================

class TemperatureScaler(nn.Module):
    """
    Learns a single scalar T for one expert such that softmax(logits / T)
    is calibrated (i.e. confidence ~= accuracy) on a held-out validation
    set. This is the standard Guo et al. 2017 approach.
    """

    def __init__(self):
        super().__init__()
        # Parameterise as log_T so T = exp(log_T) stays positive during
        # optimisation.
        self.log_T = nn.Parameter(torch.zeros(1))

    @property
    def T(self):
        return self.log_T.exp()

    def forward(self, logits):
        return logits / self.T


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    """
    Fit one temperature to (logits, labels) from a single expert's
    validation set, in that expert's OWN label space (i.e. labels already
    remapped to 0..num_expert_classes-1, exactly like the loss you already
    compute during training).

    Returns the fitted scalar T as a python float.
    """
    logits = logits.detach()
    labels = labels.detach()

    cifar_to_global = (
        build_cifar_to_global_lookup(
            original_to_aggregated_dict
        )
    )

    remapped_labels = cifar_to_global[labels]

    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.log_T], lr=0.01, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(logits), remapped_labels)#loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        pre_nll = nll(logits, remapped_labels).item()#pre_nll = nll(logits, labels).item()
        post_nll = nll(scaler(logits), remapped_labels).item()#post_nll = nll(scaler(logits), labels).item()
    print(f"Temperature fit: T={scaler.T.item():.4f}  NLL {pre_nll:.4f} -> {post_nll:.4f}")

    return scaler.T.item()


# ==============================================================================
# 2. HOW TO FIT TEMPERATURES (run this once, before building GlobalSpecialistModel)
# ==============================================================================

@torch.no_grad()
def collect_logits_and_labels(model: nn.Module, loader, num_classes: int, device: str = "cpu"):
    """
    Runs `model` over `loader`, slicing logits to the expert's own
    num_classes (matching what happens inside GlobalSpecialistModel.forward),
    and returns concatenated (logits, labels) for temperature fitting.

    `loader` must yield labels already in the expert's own 0..num_classes-1
    space (i.e. the same val stream / label remapping used to train that
    expert).
    """
    model.eval()
    all_logits, all_labels = [], []
    for batch_data, batch_labels, *_ in loader:
        batch_data = batch_data.to(device)
        logits = model(batch_data)[:, :num_classes]
        all_logits.append(logits.cpu())
        all_labels.append(batch_labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


# Example usage (adapt loaders to your existing *_val_stream_class_il objects):
#
#   outdoors_val_logits, outdoors_val_labels = collect_logits_and_labels(
#       outdoors_model, outdoors_val_loader, outdoors_num_classes
#   )
#   outdoors_T = fit_temperature(outdoors_val_logits, outdoors_val_labels)
#
#   ...repeat for nature_T, people_T, objects_T...
#
#   global_model = GlobalSpecialistModel(
#       ..., outdoors_T=outdoors_T, nature_T=nature_T,
#       people_T=people_T, objects_T=objects_T,
#   )


# ==============================================================================
# 3. CORRECTED GlobalSpecialistModel forward pass
# ==============================================================================

class GlobalSpecialistModelFixed(nn.Module):
    """
    Same structure as your GlobalSpecialistModel, but:
      - calibrates each expert with its OWN fitted temperature
      - compares softmax PROBABILITIES across experts (comparable, bounded,
        confidence-preserving) instead of a shared min-max stretch on logits
      - exposes a confidence-threshold fallback instead of blindly trusting
        whichever expert wins argmax
    """

    def __init__(
        self,
        outdoors_model, nature_model, people_model, objects_and_vehicles_model,
        outdoors_indices, nature_indices, people_indices, objects_and_vehicles_indices,
        outdoors_num_classes, nature_num_classes, people_num_classes, objects_num_classes,
        outdoors_T: float = 1.0,
        nature_T: float = 1.0,
        people_T: float = 1.0,
        objects_T: float = 1.0,
        num_classes: int = 43,
        confidence_threshold: float = 0.0,  # 0.0 = disabled; e.g. 0.5 to enable
    ):
        super().__init__()

        self.outdoors = outdoors_model
        self.nature = nature_model
        self.people = people_model
        self.objects_and_vehicles = objects_and_vehicles_model

        self.outdoors_num_classes = outdoors_num_classes
        self.nature_num_classes = nature_num_classes
        self.people_num_classes = people_num_classes
        self.objects_num_classes = objects_num_classes
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

        # Fitted, FIXED temperatures -- one scalar per expert, each fit on
        # that expert's own validation set. These are buffers (not
        # parameters): calibration is done offline, once, not during the
        # forward pass of the combined model.
        self.register_buffer("outdoors_T", torch.tensor(float(outdoors_T)))
        self.register_buffer("nature_T", torch.tensor(float(nature_T)))
        self.register_buffer("people_T", torch.tensor(float(people_T)))
        self.register_buffer("objects_T", torch.tensor(float(objects_T)))

        self.register_buffer("outdoors_indices", outdoors_indices.long())
        self.register_buffer("nature_indices", nature_indices.long())
        self.register_buffer("people_indices", people_indices.long())
        self.register_buffer("objects_and_vehicles_indices", objects_and_vehicles_indices.long())

    def forward(self, x, return_confidence: bool = False):
        outdoors_logits = self.outdoors(x)[:, :self.outdoors_num_classes]
        nature_logits = self.nature(x)[:, :self.nature_num_classes]
        people_logits = self.people(x)[:, :self.people_num_classes]
        objects_logits = self.objects_and_vehicles(x)[:, :self.objects_num_classes]

        # --- Calibrate: divide by each expert's OWN fitted temperature ---
        outdoors_probs = F.softmax(outdoors_logits / self.outdoors_T, dim=1)
        nature_probs = F.softmax(nature_logits / self.nature_T, dim=1)
        people_probs = F.softmax(people_logits / self.people_T, dim=1)
        objects_probs = F.softmax(objects_logits / self.objects_T, dim=1)

        # --- Assemble into one comparable global probability vector ---
        # Each column is now a genuine, calibrated P(class | expert, x),
        # so comparing magnitudes across experts is meaningful in a way
        # raw/min-max-scaled logits never were.
        global_probs = outdoors_probs.new_zeros((x.size(0), self.num_classes))
        global_probs[:, self.outdoors_indices] = outdoors_probs
        global_probs[:, self.nature_indices] = nature_probs
        global_probs[:, self.people_indices] = people_probs
        global_probs[:, self.objects_and_vehicles_indices] = objects_probs

        if not return_confidence:
            return global_probs

        confidence, prediction = global_probs.max(dim=1)

        if self.confidence_threshold > 0.0:
            below_threshold = confidence < self.confidence_threshold
            # Fallback behaviour when no expert is confident enough.
            # Pick whichever policy fits your use case, e.g.:
            #   - leave the argmax prediction as-is but flag it (below_threshold)
            #   - route to a reject / "unknown" class
            #   - fall back to the coarse-label gate (see note below)
            # Here we just expose the mask so the caller decides.
            return global_probs, prediction, confidence, below_threshold

        return global_probs, prediction, confidence


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

    outdoors_val_logits, outdoors_val_labels = collect_logits_and_labels(models["outdoors"], outdoors_val_loader, num_subset_classes["outdoors"])
    nature_val_logits, nature_val_labels = collect_logits_and_labels(models["nature"], nature_val_loader, num_subset_classes["nature"])
    people_val_logits, people_val_labels = collect_logits_and_labels(models["people"], people_val_loader, num_subset_classes["people"])
    objects_and_vehicles_val_logits, objects_and_vehicles_val_labels = collect_logits_and_labels(models["objects_and_vehicles"], objects_and_vehicles_val_loader, num_subset_classes["objects_and_vehicles"])

    outdoors_T = fit_temperature(outdoors_val_logits, outdoors_val_labels)
    nature_T = fit_temperature(nature_val_logits, nature_val_labels)
    people_T = fit_temperature(people_val_logits, people_val_labels)
    objects_T = fit_temperature(objects_and_vehicles_val_logits, objects_and_vehicles_val_labels)

    global_model = GlobalSpecialistModelFixed(
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
        outdoors_T = outdoors_T,
        nature_T = nature_T,
        people_T = people_T,
        objects_T=objects_T,

        confidence_threshold=0.0,

        num_classes=43,
    )

    return global_model


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
        #save_global_model(global_model, OUTPUT_DIR, report)

        return global_model

    except Exception as e:
        print(f"\n✗ Aggregation failed: {e}")
        raise
# ==============================================================================
# 4. Using it at eval time
# ==============================================================================
#
#   global_probs, predictions, confidence = global_model(batch_data, return_confidence=True)
#   # predictions is already the final class prediction: argmax over the
#   # CALIBRATED probability vector, which is what you want in place of
#   # `global_logits.argmax(dim=1)`.
#
# NOTE on `evaluate_global_specialist_model` (line ~1458 in your file) and
# `evaluate_aggregated_model_FedAvg_code_v3`: both currently call
# `.argmax(dim=1)` on the raw/min-max-scaled `global_logits`. Once you swap
# to GlobalSpecialistModelFixed, argmax on `global_probs` (probabilities,
# not logits) is the correct final prediction -- no separate thresholding
# step is required unless you want a reject/fallback option.



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

            global_probs, predictions, confidence = global_model(batch_data, return_confidence=True) #global_logits = global_model(batch_data)

            assert global_probs.shape[1] == 43#assert global_logits.shape[1] == 43

            # -----------------------------------------
            # Loss
            # -----------------------------------------

            loss = F.cross_entropy(
                global_probs,
                global_targets,
            ) # loss = F.cross_entropy(global_logits,global_targets,)

            # -----------------------------------------
            # Prediction
            # -----------------------------------------

            predictions = global_probs.argmax(dim=1)#predictions = global_logits.argmax(dim=1)

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


if __name__ == '__main__':
    global_model = main()
    evaluate_global_specialist_model(global_model, test_loader, original_to_aggregated_dict,
                                     "Evaluating global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_global_specialist_model(global_model, outdoors_loader, original_to_aggregated_dict,
                                     "Evaluating global model on outdoors_loader")
    evaluate_global_specialist_model(global_model, nature_loader, original_to_aggregated_dict,
                                     "Evaluating global model on nature_loader")
    evaluate_global_specialist_model(global_model, people_loader, original_to_aggregated_dict,
                                     "Evaluating global model on people_loader")
    evaluate_global_specialist_model(global_model, objects_and_vehicles_loader, original_to_aggregated_dict,
                                     "Evaluating global model on objects_and_vehicles_loader")
