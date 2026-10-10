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
from Report_implementation_attempt.Train_ResNet20_Outdoors_LwF_vReport import outdoors_specialised_models_test_stream_class_il, outdoors_specialised_models_train_stream_class_il, outdoors_specialised_models_val_stream_class_il
from Report_implementation_attempt.Train_ResNet20_Nature_LwF_vReport import nature_specialised_models_test_stream_class_il, nature_specialised_models_train_stream_class_il, nature_specialised_models_val_stream_class_il
from Report_implementation_attempt.Train_ResNet20_People_LwF_vReport import people_specialised_models_test_stream_class_il, people_specialised_models_train_stream_class_il, people_specialised_models_val_stream_class_il
from Report_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_vReport import objects_and_vehicles_specialised_models_test_stream_class_il, objects_and_vehicles_specialised_models_train_stream_class_il, objects_and_vehicles_specialised_models_val_stream_class_il
from Report_implementation_attempt.Load_cifar100_vReport import avl_test
#TODO: TEMPORARY!!!! Change to that of vReport once I have final Objects_and_Vehicles model!!!!!
from Report_implementation_attempt.Global_model_LwF_BASELINE_vReport import global_model_lwf_baseline_val_stream_class_il
from Report_implementation_attempt.Global_model_NAIVE_BASELINE_vReport import global_model_naive_baseline_val_stream_class_il
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F
import tracemalloc
import torch
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
    shuffle=False#True
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
    shuffle=False#True
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
    shuffle=False#True
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
outdoors_val_loader = DataLoader(
    combined_dataset_outdoors_val,
    batch_size=32,
    shuffle=False
)
nature_val_loader = DataLoader(
    combined_dataset_nature_val,
    batch_size=32,
    shuffle=False
)
people_val_loader = DataLoader(
    combined_dataset_people_val,
    batch_size=32,
    shuffle=False
)
objects_and_vehicles_val_loader = DataLoader(
    combined_dataset_objects_and_vehicles_val,
    batch_size=32,
    shuffle=False
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



def map_targets(class_to_idx_new, labels_tensor):
    targets_tensor = torch.tensor([class_to_idx_new[int(element.item())] for element in labels_tensor],
                                  device=labels_tensor.device,  # fine_label.device,
                                  dtype=torch.long
                                  )
    # print("print(class_to_idx_new)\n", class_to_idx_new)
    ##print("print(targets_tensor)\n",targets_tensor)
    return targets_tensor


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


# ==============================================================================
# STEP 2: Classifier Remapping (UNCHANGED)
# ==============================================================================

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


def load_baselines(
    model_names: list,
    model_dir: str,#Path,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict]: #-> Tuple[nn.Module, Dict]:
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

    #model_dir = f"../Report_implementation_attempt/final_specialised_models_BASELINE/" #model_dir = Path(model_dir)
    models = {}
    mappings = {}
    num_subset_classes = {}

    # ==================================================================
    # STEP 1: Load all models and their mappings
    # ==================================================================

    print("Step 1: Loading BASELINES...")
    local_to_global_indices = {}
    for model_name in model_names:
        try:
            if "BASELINE" not in model_name:
                state_dict = torch.load(f"{model_dir}final_{model_name}_model.pt", map_location=device)
            else:
                state_dict = torch.load(f"{model_dir}{model_name}_model.pt", map_location=device)#state_dict = torch.load(model_dir / f'final_{model_name}_model.pt',map_location=device)

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

        class_to_idx_new = metadata['class_to_idx_new']

        return model, class_to_idx_new


# ==============================================================================
# STEP 4: Test
# ===============================================

def evaluate_baseline_vs_specialist(
    model,
    loader,
    class_to_idx_new:dict,
    output_message:str,
    device="cpu",
):

    model.eval()

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
            targets = map_targets(class_to_idx_new, batch_labels)

            # -----------------------------------------
            # One task-agnostic global forward pass
            # -----------------------------------------

            logits = model(batch_data)

            #assert global_logits.shape[1] == 43

            # -----------------------------------------
            # Loss
            # -----------------------------------------

            loss = F.cross_entropy(
                logits,
                targets,
            )

            # -----------------------------------------
            # Prediction
            # -----------------------------------------

            predictions = logits.argmax(
                dim=1
            )

            batch_size = batch_labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += batch_size

    accuracy = correct / total

    avg_loss = total_loss / total

    print(
        f"{output_message} --> Loss = {avg_loss:.6f} | Accuracy = {accuracy * 100:.2f}%"
    )

    #return accuracy


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main(MODEL_DIR, DEVICE, model_names):
    """
    Complete pipeline with feature averaging.
    """

    #NEW MODEL_DIR = f"../Report_implementation_attempt/final_specialised_models_BASELINE/" #MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    #OLD MODEL_DIR = Path("./final_specialised_models_BASELINE")
    #OLD MODEL_DIR.mkdir(parents=True, exist_ok=True) #MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    #NEW DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

    #OLD model_names = ['people']#model_names = ['outdoors', 'nature', 'people', 'objects_and_vehicles']#TODO: Testing if index out of bounds error occurs regardless of models

    try:
        # ============================================================
        # Aggregate models with feature averaging
        # ============================================================
        model, class_to_idx_new = load_baselines(#used to be FORMERLY_aggregate_models_all_layers_with_feature_averaging(
            model_names=model_names,
            model_dir=MODEL_DIR,
            device=DEVICE
        )

        return model, class_to_idx_new

    except Exception as e:
        print(f"\n✗ Aggregation failed: {e}")
        raise



'''
    train_list = [33, 68, 37, 71, 12, 76, 17, 49, 23, 60, 0, 96, 70, 47, 82, 51, 52, 53, 54, 83, 56, 57, 59, 92, 62, 2,
                  35, 98, 11, 46, 1, 3, 4, 6, 7, 14, 15, 18, 19, 21, 24, 26, 27, 29, 30,
                  31, 32, 34, 36, 38, 42, 43, 44, 45, 50, 55, 63, 64, 65, 66, 67, 72, 73, 74, 75, 77, 78, 79, 80, 88,
                  91, 93, 95, 97, 99,
                  5, 8, 9, 10, 13, 16, 20, 22, 25, 28, 39, 40, 41, 48, 58, 61, 69, 81, 84, 85, 86, 87, 89, 90, 94]

    specialised_models_indices_list=[23, 17, 71, 49, 76, 33, 68, 12, 37, 60, 0, 96, 82, 59, 70, 52, 47, 56, 51, 83, 53, 92, 62, 54, 57,
     11, 98, 35, 46, 2, 19, 29, 1, 31, 97, 80, 74, 64, 42, 65, 21, 24, 78, 45, 73, 14, 6, 36, 55, 72, 43,
     27, 50, 15, 18, 75, 38, 66, 77, 95, 99, 93, 4, 34, 32, 88, 67, 30, 63, 26, 79, 44, 7, 91, 3, 86, 90,
     28, 39, 8, 87, 84, 22, 81, 89, 9, 20, 69, 61, 94, 40, 48, 85, 41,
     13, 25, 10, 5, 58, 16]

    delete_list = {12: 0, 17: 1, 23: 2, 33: 3, 37: 4, 49: 5,  # OUTDOORS
                                   0: 6, 47: 7, 51: 8, 52: 9, 53: 10, 54: 11, 56: 12, 57: 13, 59: 14,  # NATURE
                                   2: 15, 11: 16, 35: 17,  # PEOPLE
                                   5: 18, 8: 19, 9: 20, 10: 21, 13: 22, 16: 23, 20: 24, 22: 25, 25: 26, 28: 27, 39: 28,
                                   40: 29, 41: 30, 48: 31, 58: 32, 61: 33, 69: 34,  # OBJECTS_AND_VEHICLES
                                   81: 35, 84: 36, 85: 37, 86: 38, 87: 39, 89: 40, 90: 41,
                                   94: 42}  # OBJECTS_AND_VEHICLES

    second_list =[]
    for item in train_list:
        if item not in original_to_aggregated_dict.keys():
            train_list.remove(item)
            second_list.append(item)
            print(item)
            print("train_list AFTER REMOVE:", train_list, "\nlen",len(train_list))
    print("second_list:", second_list, "\nlen", len(second_list))'''

if __name__ == '__main__':

    MODEL_DIR_BASELINE = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE = 'cpu'
    model_names_BASELINE = ['NAIVE_BASELINE_outdoors']
    baseline, class_to_idx_new_baseline = main(MODEL_DIR_BASELINE, DEVICE_BASELINE, model_names_BASELINE)

    MODEL_DIR_Standard_Supervised_Learning = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_Standard_Supervised_Learning = 'cpu'
    model_names_Standard_Supervised_Learning = ['STANDARD_SUPERVISED_LEARNING_BASELINE_outdoors']
    standard_Supervised_Learning, class_to_idx_new_Standard_Supervised_Learning = main(MODEL_DIR_Standard_Supervised_Learning, DEVICE_Standard_Supervised_Learning, model_names_Standard_Supervised_Learning)


    MODEL_DIR_final_specialised = f"../Report_implementation_attempt/final_specialised_models/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_final_specialised = 'cpu'
    model_names_final_specialised = ['outdoors']
    final_specialised_model, class_to_idx_new_final_specialised = main(MODEL_DIR_final_specialised,
                                                                       DEVICE_final_specialised,
                                                                       model_names_final_specialised)
    print(
        "############################# outdoors model --- baseline (outdoors model trained in standard supervised learning) vs final_specialised_outdoors_model: #############################")
    evaluate_baseline_vs_specialist(baseline, outdoors_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE outdoors model on outdoors_loader")
    evaluate_baseline_vs_specialist(standard_Supervised_Learning, outdoors_loader, class_to_idx_new_Standard_Supervised_Learning,
                                    "Evaluating standard_Supervised_Learning_BASELINE on outdoors_loader")
    evaluate_baseline_vs_specialist(final_specialised_model, outdoors_loader, class_to_idx_new_final_specialised,
                                    "Evaluating final_specialised_outdoors_model on outdoors_loader")

    ###########################################################################################################################################################################################################################################

    MODEL_DIR_BASELINE = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE = 'cpu'
    model_names_BASELINE = ['NAIVE_BASELINE_nature']
    baseline, class_to_idx_new_baseline = main(MODEL_DIR_BASELINE, DEVICE_BASELINE, model_names_BASELINE)

    MODEL_DIR_Standard_Supervised_Learning = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_Standard_Supervised_Learning = 'cpu'
    model_names_Standard_Supervised_Learning = ['STANDARD_SUPERVISED_LEARNING_BASELINE_nature']
    standard_Supervised_Learning, class_to_idx_new_Standard_Supervised_Learning = main(MODEL_DIR_Standard_Supervised_Learning, DEVICE_Standard_Supervised_Learning, model_names_Standard_Supervised_Learning)

    MODEL_DIR_final_specialised = f"../Report_implementation_attempt/final_specialised_models/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_final_specialised = 'cpu'
    model_names_final_specialised = ['nature']
    final_specialised_model, class_to_idx_new_final_specialised = main(MODEL_DIR_final_specialised,
                                                                       DEVICE_final_specialised,
                                                                       model_names_final_specialised)
    print(
        "############################# nature model --- baseline (nature model trained in standard supervised learning) vs final_specialised_nature_model: #############################")
    evaluate_baseline_vs_specialist(baseline, nature_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE nature model on nature_loader")
    evaluate_baseline_vs_specialist(standard_Supervised_Learning, nature_loader, class_to_idx_new_Standard_Supervised_Learning,
                                    "Evaluating standard_Supervised_Learning_BASELINE on nature_loader")
    evaluate_baseline_vs_specialist(final_specialised_model, nature_loader, class_to_idx_new_final_specialised,
                                    "Evaluating final_specialised_nature_model on nature_loader")

    ###########################################################################################################################################################################################################################################

    MODEL_DIR_BASELINE = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE = 'cpu'
    model_names_BASELINE = ['NAIVE_BASELINE_people']
    baseline, class_to_idx_new_baseline = main(MODEL_DIR_BASELINE, DEVICE_BASELINE, model_names_BASELINE)

    MODEL_DIR_Standard_Supervised_Learning = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_Standard_Supervised_Learning = 'cpu'
    model_names_Standard_Supervised_Learning = ['STANDARD_SUPERVISED_LEARNING_BASELINE_people']
    standard_Supervised_Learning, class_to_idx_new_Standard_Supervised_Learning = main(MODEL_DIR_Standard_Supervised_Learning, DEVICE_Standard_Supervised_Learning, model_names_Standard_Supervised_Learning)

    MODEL_DIR_final_specialised = f"../Report_implementation_attempt/final_specialised_models/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_final_specialised = 'cpu'
    model_names_final_specialised = ['people']
    final_specialised_model, class_to_idx_new_final_specialised = main(MODEL_DIR_final_specialised,
                                                                       DEVICE_final_specialised,
                                                                       model_names_final_specialised)
    print(
        "############################# people model --- baseline (people model trained in standard supervised learning) vs final_specialised_people_model: #############################")
    evaluate_baseline_vs_specialist(baseline, people_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE people model on people_loader")
    evaluate_baseline_vs_specialist(standard_Supervised_Learning, people_loader, class_to_idx_new_Standard_Supervised_Learning,
                                    "Evaluating standard_Supervised_Learning_BASELINE on people_loader")
    evaluate_baseline_vs_specialist(final_specialised_model, people_loader, class_to_idx_new_final_specialised,
                                    "Evaluating final_specialised_people_model on people_loader")
    print("\n\n")

    ###########################################################################################################################################################################################################################################

    '''MODEL_DIR_BASELINE = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE = 'cpu'
    model_names_BASELINE = ['NAIVE_BASELINE_objects_and_vehicles']
    baseline, class_to_idx_new_baseline = main(MODEL_DIR_BASELINE, DEVICE_BASELINE, model_names_BASELINE)

    MODEL_DIR_Standard_Supervised_Learning = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_Standard_Supervised_Learning = 'cpu'
    model_names_Standard_Supervised_Learning = ['STANDARD_SUPERVISED_LEARNING_BASELINE_objects_and_vehicles']
    standard_Supervised_Learning, class_to_idx_new_Standard_Supervised_Learning = main(MODEL_DIR_Standard_Supervised_Learning, DEVICE_Standard_Supervised_Learning, model_names_Standard_Supervised_Learning)

    MODEL_DIR_final_specialised = f"../Report_implementation_attempt/final_specialised_models/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_final_specialised = 'cpu'
    model_names_final_specialised = ['objects_and_vehicles']
    final_specialised_model, class_to_idx_new_final_specialised = main(MODEL_DIR_final_specialised,
                                                                       DEVICE_final_specialised,
                                                                       model_names_final_specialised)
    print(
        "############################# objects_and_vehicles model --- baseline (objects_and_vehicles model trained in standard supervised learning) vs final_specialised_objects_and_vehicles_model: #############################")
    evaluate_baseline_vs_specialist(baseline, objects_and_vehicles_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE objects_and_vehicles model on objects_and_vehicles_loader")
    evaluate_baseline_vs_specialist(standard_Supervised_Learning, objects_and_vehicles_loader, class_to_idx_new_Standard_Supervised_Learning,
                                    "Evaluating standard_Supervised_Learning_BASELINE on objects_and_vehicles_loader")
    evaluate_baseline_vs_specialist(final_specialised_model, objects_and_vehicles_loader, class_to_idx_new_final_specialised,
                                    "Evaluating final_specialised_objects_and_vehicles_model on objects_and_vehicles_loader")
    print("\n\n")'''

    ###########################################################################################################################################################################################################################################
    MODEL_DIR_BASELINE = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE = 'cpu'
    model_names_BASELINE = ['BASELINE_standard_supervised_global']
    baseline, class_to_idx_new_baseline = main(MODEL_DIR_BASELINE, DEVICE_BASELINE, model_names_BASELINE)

    MODEL_DIR_BASELINE_naive = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE_naive = 'cpu'
    model_names_BASELINE_naive = ['NAIVE_BASELINE_global']
    baseline_naive, class_to_idx_new_baseline_naive = main(MODEL_DIR_BASELINE_naive, DEVICE_BASELINE_naive, model_names_BASELINE_naive)

    MODEL_DIR_BASELINE_naive_2 = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE_naive_2 = 'cpu'
    model_names_BASELINE_naive_2 = ['TEST2_NAIVE_BASELINE_global']
    baseline_naive_2, class_to_idx_new_baseline_naive_2 = main(MODEL_DIR_BASELINE_naive_2, DEVICE_BASELINE_naive_2, model_names_BASELINE_naive_2)

    MODEL_DIR_BASELINE_lwf = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE_lwf = 'cpu'
    model_names_BASELINE_lwf = ['lwf_BASELINE_global']
    baseline_lwf, class_to_idx_new_baseline_lwf = main(MODEL_DIR_BASELINE_lwf, DEVICE_BASELINE_lwf, model_names_BASELINE_lwf)

    MODEL_DIR_BASELINE_lwf_CORRECTED = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE_lwf_CORRECTED = 'cpu'
    model_names_BASELINE_lwf_CORRECTED = ['TEST_lwf_BASELINE_global']
    baseline_lwf_CORRECTED, class_to_idx_new_baseline_lwf_CORRECTED = main(MODEL_DIR_BASELINE_lwf_CORRECTED, DEVICE_BASELINE_lwf_CORRECTED, model_names_BASELINE_lwf_CORRECTED)

    MODEL_DIR_BASELINE_lwf_CORRECTED_2 = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE_lwf_CORRECTED_2 = 'cpu'
    model_names_BASELINE_lwf_CORRECTED_2 = ['TEST2_lwf_BASELINE_global']
    baseline_lwf_CORRECTED_2, class_to_idx_new_baseline_lwf_CORRECTED_2 = main(MODEL_DIR_BASELINE_lwf_CORRECTED_2, DEVICE_BASELINE_lwf_CORRECTED_2, model_names_BASELINE_lwf_CORRECTED_2)


    print(
        "############################# BASELINE_standard_supervised_global (global model trained in standard supervised learning) vs NAIVE_BASELINE_global vs lwf_baseline_global: #############################")
    evaluate_baseline_vs_specialist(baseline, test_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on test_loader")
    evaluate_baseline_vs_specialist(baseline, outdoors_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on outdoors_loader")
    evaluate_baseline_vs_specialist(baseline, nature_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on nature_loader")
    evaluate_baseline_vs_specialist(baseline, people_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on people_loader")
    evaluate_baseline_vs_specialist(baseline, objects_and_vehicles_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on objects_and_vehicles_loader")

    print("\n")
    evaluate_baseline_vs_specialist(baseline, last_experience_test_loader,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on last_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline, last_experience_test_outdoors,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on last_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline, last_experience_test_nature,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on last_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline, last_experience_test_people,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on last_experience_test_people")
    evaluate_baseline_vs_specialist(baseline, last_experience_test_objects_and_vehicles,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on last_experience_test_objects_and_vehicles")

    print("\n")
    evaluate_baseline_vs_specialist(baseline, middle_experience_test_loader,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on middle_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline, middle_experience_test_outdoors,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on middle_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline, middle_experience_test_nature,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on middle_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline, middle_experience_test_people,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on middle_experience_test_people")
    evaluate_baseline_vs_specialist(baseline, middle_experience_test_objects_and_vehicles,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on middle_experience_test_objects_and_vehicles")

    print("\n")
    evaluate_baseline_vs_specialist(baseline, first_experience_test_loader,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on first_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline, first_experience_test_outdoors,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on first_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline, first_experience_test_nature,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on first_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline, first_experience_test_people,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on first_experience_test_people")
    evaluate_baseline_vs_specialist(baseline, first_experience_test_objects_and_vehicles,
                                    class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on first_experience_test_objects_and_vehicles")


    print("\n")
    evaluate_baseline_vs_specialist(baseline_naive, test_loader, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on test_loader")
    evaluate_baseline_vs_specialist(baseline_naive, outdoors_loader, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on outdoors_loader")
    evaluate_baseline_vs_specialist(baseline_naive, nature_loader, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on nature_loader")
    evaluate_baseline_vs_specialist(baseline_naive, people_loader, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on people_loader")
    evaluate_baseline_vs_specialist(baseline_naive, objects_and_vehicles_loader, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on objects_and_vehicles_loader")
    print("\n")
    evaluate_baseline_vs_specialist(baseline_naive, last_experience_test_loader, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on last_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_naive, last_experience_test_outdoors, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on last_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_naive, last_experience_test_nature, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on last_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_naive, last_experience_test_people, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on last_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_naive, last_experience_test_objects_and_vehicles, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on last_experience_test_objects_and_vehicles")
    print("\n")
    evaluate_baseline_vs_specialist(baseline_naive, middle_experience_test_loader, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on middle_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_naive, middle_experience_test_outdoors, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on middle_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_naive, middle_experience_test_nature, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on middle_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_naive, middle_experience_test_people, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on middle_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_naive, middle_experience_test_objects_and_vehicles, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on middle_experience_test_objects_and_vehicles")
    print("\n")
    evaluate_baseline_vs_specialist(baseline_naive, first_experience_test_loader, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on first_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_naive, first_experience_test_outdoors, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on first_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_naive, first_experience_test_nature, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on first_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_naive, first_experience_test_people, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on first_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_naive, first_experience_test_objects_and_vehicles, class_to_idx_new_baseline_naive,
                                    "Evaluating NAIVE_BASELINE_global on first_experience_test_objects_and_vehicles")

    print("\n")
    evaluate_baseline_vs_specialist(baseline_naive_2, test_loader, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on test_loader")
    evaluate_baseline_vs_specialist(baseline_naive_2, outdoors_loader, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on outdoors_loader")
    evaluate_baseline_vs_specialist(baseline_naive_2, nature_loader, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on nature_loader")
    evaluate_baseline_vs_specialist(baseline_naive_2, people_loader, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on people_loader")
    evaluate_baseline_vs_specialist(baseline_naive_2, objects_and_vehicles_loader, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on objects_and_vehicles_loader")

    print("\n") #global_model_lwf_baseline_val_stream_class_il
    global_model_naive_baseline_val_stream_last_experience = DataLoader(global_model_naive_baseline_val_stream_class_il[2].dataset, batch_size=32, shuffle=False)
    global_model_naive_baseline_val_stream_middle_experience = DataLoader(global_model_naive_baseline_val_stream_class_il[1].dataset,batch_size=32, shuffle=False)
    global_model_naive_baseline_val_stream_first_experience = DataLoader(global_model_naive_baseline_val_stream_class_il[0].dataset,batch_size=32, shuffle=False)

    evaluate_baseline_vs_specialist(baseline_naive_2, global_model_naive_baseline_val_stream_last_experience, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on last_experience (global_model_naive_baseline_val_stream_class_il[2])")
    print("\n")
    evaluate_baseline_vs_specialist(baseline_naive_2, global_model_naive_baseline_val_stream_middle_experience, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on middle_experience (global_model_naive_baseline_val_stream_class_il[2])")
    print("\n")
    evaluate_baseline_vs_specialist(baseline_naive_2, global_model_naive_baseline_val_stream_first_experience, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on first_experience (global_model_naive_baseline_val_stream_class_il[0])")
    
    '''print("\n")
    evaluate_baseline_vs_specialist(baseline_naive_2, last_experience_test_loader, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on last_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_naive_2, last_experience_test_outdoors, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on last_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_naive_2, last_experience_test_nature, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on last_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_naive_2, last_experience_test_people, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on last_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_naive_2, last_experience_test_objects_and_vehicles, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on last_experience_test_objects_and_vehicles")
    print("\n")
    evaluate_baseline_vs_specialist(baseline_naive_2, middle_experience_test_loader, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on middle_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_naive_2, middle_experience_test_outdoors, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on middle_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_naive_2, middle_experience_test_nature, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on middle_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_naive_2, middle_experience_test_people, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on middle_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_naive_2, middle_experience_test_objects_and_vehicles, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on middle_experience_test_objects_and_vehicles")
    print("\n")
    evaluate_baseline_vs_specialist(baseline_naive_2, first_experience_test_loader, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on first_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_naive_2, first_experience_test_outdoors, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on first_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_naive_2, first_experience_test_nature, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on first_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_naive_2, first_experience_test_people, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on first_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_naive_2, first_experience_test_objects_and_vehicles, class_to_idx_new_baseline_naive_2,
                                    "Evaluating NAIVE_BASELINE_global_2 on first_experience_test_objects_and_vehicles")'''
    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf, test_loader, class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on test_loader")
    evaluate_baseline_vs_specialist(baseline_lwf, outdoors_loader, class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on outdoors_loader")
    evaluate_baseline_vs_specialist(baseline_lwf, nature_loader, class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on nature_loader")
    evaluate_baseline_vs_specialist(baseline_lwf, people_loader, class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on people_loader")
    evaluate_baseline_vs_specialist(baseline_lwf, objects_and_vehicles_loader, class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on objects_and_vehicles_loader")

    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf, last_experience_test_loader,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on last_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_lwf, last_experience_test_outdoors,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on last_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_lwf, last_experience_test_nature,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on last_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_lwf, last_experience_test_people,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on last_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_lwf, last_experience_test_objects_and_vehicles,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on last_experience_test_objects_and_vehicles")

    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf, middle_experience_test_loader,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on middle_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_lwf, middle_experience_test_outdoors,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on middle_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_lwf, middle_experience_test_nature,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on middle_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_lwf, middle_experience_test_people,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on middle_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_lwf, middle_experience_test_objects_and_vehicles,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on middle_experience_test_objects_and_vehicles")

    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf, first_experience_test_loader,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on first_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_lwf, first_experience_test_outdoors,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on first_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_lwf, first_experience_test_nature,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on first_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_lwf, first_experience_test_people,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on first_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_lwf, first_experience_test_objects_and_vehicles,
                                    class_to_idx_new_baseline_lwf,
                                    "Evaluating lwf_baseline_global on first_experience_test_objects_and_vehicles")
    '''
    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED, test_loader, class_to_idx_new_baseline_lwf_CORRECTED,
                                    "Evaluating lwf_baseline_global_CORRECTED on test_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED, outdoors_loader, class_to_idx_new_baseline_lwf_CORRECTED,
                                    "Evaluating lwf_baseline_global_CORRECTED on outdoors_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED, nature_loader, class_to_idx_new_baseline_lwf_CORRECTED,
                                    "Evaluating lwf_baseline_global_CORRECTED on nature_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED, people_loader, class_to_idx_new_baseline_lwf_CORRECTED,
                                    "Evaluating lwf_baseline_global_CORRECTED on people_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED, objects_and_vehicles_loader, class_to_idx_new_baseline_lwf_CORRECTED,
                                    "Evaluating lwf_baseline_global_CORRECTED on objects_and_vehicles_loader")'''

    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, test_loader, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on test_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, outdoors_loader, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on outdoors_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, nature_loader, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on nature_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, people_loader, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on people_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, objects_and_vehicles_loader, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on objects_and_vehicles_loader")
    
    print("\n") #global_model_lwf_baseline_val_stream_class_il
    global_model_lwf_baseline_val_stream_last_experience = DataLoader(global_model_lwf_baseline_val_stream_class_il[2].dataset, batch_size=32, shuffle=False)
    global_model_lwf_baseline_val_stream_middle_experience = DataLoader(global_model_lwf_baseline_val_stream_class_il[1].dataset,batch_size=32, shuffle=False)
    global_model_lwf_baseline_val_stream_first_experience = DataLoader(global_model_lwf_baseline_val_stream_class_il[0].dataset,batch_size=32, shuffle=False)

    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, global_model_lwf_baseline_val_stream_last_experience,
                                    class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on last_experience (global_model_lwf_baseline_val_stream_class_il[2])")

    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, global_model_lwf_baseline_val_stream_middle_experience,
                                    class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on middle_experience (global_model_lwf_baseline_val_stream_class_il[1])")

    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, global_model_lwf_baseline_val_stream_first_experience,
                                    class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on first_experience (global_model_lwf_baseline_val_stream_class_il[0],)")
    '''
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, last_experience_test_loader, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on last_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, last_experience_test_outdoors, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on last_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, last_experience_test_nature, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on last_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, last_experience_test_people, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on last_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, last_experience_test_objects_and_vehicles, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on last_experience_test_objects_and_vehicles")

    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, middle_experience_test_loader, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on middle_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, middle_experience_test_outdoors, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on middle_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, middle_experience_test_nature, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on middle_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, middle_experience_test_people, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on middle_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, middle_experience_test_objects_and_vehicles, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on middle_experience_test_objects_and_vehicles")
    
    print("\n")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, first_experience_test_loader, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on first_experience_test_loader")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, first_experience_test_outdoors, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on first_experience_test_outdoors")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, first_experience_test_nature, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on first_experience_test_nature")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, first_experience_test_people, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on first_experience_test_people")
    evaluate_baseline_vs_specialist(baseline_lwf_CORRECTED_2, first_experience_test_objects_and_vehicles, class_to_idx_new_baseline_lwf_CORRECTED_2,
                                    "Evaluating lwf_baseline_global_CORRECTED_2 on first_experience_test_objects_and_vehicles")'''

    MODEL_DIR_final_specialised = f"../Report_implementation_attempt/final_specialised_models/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_final_specialised = 'cpu'
    model_names_final_outdoors = ['outdoors']
    final_outdoors_model, class_to_idx_new_final_outdoors = main(MODEL_DIR_final_specialised,
                                                                 DEVICE_final_specialised,
                                                                 model_names_final_outdoors)

    model_names_final_nature = ['nature']
    final_nature_model, class_to_idx_new_final_nature = main(MODEL_DIR_final_specialised,
                                                             DEVICE_final_specialised,
                                                             model_names_final_nature)

    model_names_final_people = ['people']
    final_people_model, class_to_idx_new_final_people = main(MODEL_DIR_final_specialised,
                                                             DEVICE_final_specialised,
                                                             model_names_final_people)

    model_names_final_objects_and_vehicles = ['objects_and_vehicles']
    final_objects_and_vehicles_model, class_to_idx_new_final_objects_and_vehicles = main(MODEL_DIR_final_specialised,
                                                                                         DEVICE_final_specialised,
                                                                                         model_names_final_objects_and_vehicles)
    print(
        "\n\n############################# Every specialised model evaluated on its own test set: #############################")
    evaluate_baseline_vs_specialist(final_outdoors_model, outdoors_loader, class_to_idx_new_final_outdoors,
                                    "Evaluating final_specialised_outdoors_model on outdoors_loader")

    evaluate_baseline_vs_specialist(final_nature_model, nature_loader, class_to_idx_new_final_nature,
                                    "Evaluating final_specialised_nature_model on nature_loader")

    evaluate_baseline_vs_specialist(final_people_model, people_loader, class_to_idx_new_final_people,
                                    "Evaluating final_specialised_people_model on people_loader")

    evaluate_baseline_vs_specialist(final_objects_and_vehicles_model, objects_and_vehicles_loader,
                                    class_to_idx_new_final_objects_and_vehicles,
                                    "Evaluating final_specialised_objects_and_vehicles_model on objects_and_vehicles_loader")

    print(
        "\n\n############################# Every specialised model evaluated on its own VAL set: #############################")
    evaluate_baseline_vs_specialist(final_outdoors_model, outdoors_val_loader, class_to_idx_new_final_outdoors,
                                    "Evaluating final_specialised_outdoors_model on outdoors_val_loader")

    evaluate_baseline_vs_specialist(final_nature_model, nature_val_loader, class_to_idx_new_final_nature,
                                    "Evaluating final_specialised_nature_model on nature_val_loader")

    evaluate_baseline_vs_specialist(final_people_model, people_val_loader, class_to_idx_new_final_people,
                                    "Evaluating final_specialised_people_model on people_val_loader")

    evaluate_baseline_vs_specialist(final_objects_and_vehicles_model, objects_and_vehicles_val_loader,
                                    class_to_idx_new_final_objects_and_vehicles,
                                    "Evaluating final_specialised_objects_and_vehicles_model on objects_and_vehicles_val_loader")

    ##############################################################################################################################################################################################################################
    MODEL_DIR_final_specialised = f"../Report_implementation_attempt/final_specialised_models/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_final_specialised = 'cpu'
    model_names_final_outdoors = ['outdoors']
    final_outdoors_model, class_to_idx_new_final_outdoors = main(MODEL_DIR_final_specialised,
                                                                 DEVICE_final_specialised,
                                                                 model_names_final_outdoors)

    model_names_final_nature = ['nature']
    final_nature_model, class_to_idx_new_final_nature = main(MODEL_DIR_final_specialised,
                                                             DEVICE_final_specialised,
                                                             model_names_final_nature)

    model_names_final_people = ['people']
    final_people_model, class_to_idx_new_final_people = main(MODEL_DIR_final_specialised,
                                                             DEVICE_final_specialised,
                                                             model_names_final_people)

    model_names_final_objects_and_vehicles = ['objects_and_vehicles']
    final_objects_and_vehicles_model, class_to_idx_new_final_objects_and_vehicles = main(MODEL_DIR_final_specialised,
                                                                                         DEVICE_final_specialised,
                                                                                         model_names_final_objects_and_vehicles)
    print(
        "\n\n############################# Every specialised model evaluated on its own test set FOR EACH EXPERIENCE SEPARATELY: #############################")
    evaluate_baseline_vs_specialist(final_outdoors_model, outdoors_loader, class_to_idx_new_final_outdoors,
                                    "Evaluating final_specialised_outdoors_model on outdoors_loader")
    evaluate_baseline_vs_specialist(final_outdoors_model, first_experience_test_outdoors, class_to_idx_new_final_outdoors,
                                    "Evaluating final_specialised_outdoors_model on first_experience_test_outdoors")
    evaluate_baseline_vs_specialist(final_outdoors_model, middle_experience_test_outdoors, class_to_idx_new_final_outdoors,
                                    "Evaluating final_specialised_outdoors_model on middle_experience_test_outdoors")
    evaluate_baseline_vs_specialist(final_outdoors_model, last_experience_test_outdoors, class_to_idx_new_final_outdoors,
                                    "Evaluating final_specialised_outdoors_model on last_experience_test_outdoors")
    print("\n\n")
    evaluate_baseline_vs_specialist(final_nature_model, nature_loader, class_to_idx_new_final_nature,
                                    "Evaluating final_specialised_nature_model on nature_loader")
    evaluate_baseline_vs_specialist(final_nature_model, first_experience_test_nature, class_to_idx_new_final_nature,
                                    "Evaluating final_specialised_nature_model on first_experience_test_nature")
    evaluate_baseline_vs_specialist(final_nature_model, middle_experience_test_nature, class_to_idx_new_final_nature,
                                    "Evaluating final_specialised_nature_model on middle_experience_test_nature")
    evaluate_baseline_vs_specialist(final_nature_model, last_experience_test_nature, class_to_idx_new_final_nature,
                                    "Evaluating final_specialised_nature_model on last_experience_test_nature")
    print("\n\n")
    evaluate_baseline_vs_specialist(final_people_model, people_loader, class_to_idx_new_final_people,
                                    "Evaluating final_specialised_people_model on people_loader")
    evaluate_baseline_vs_specialist(final_people_model, first_experience_test_people, class_to_idx_new_final_people,
                                    "Evaluating final_specialised_people_model on first_experience_test_people")
    evaluate_baseline_vs_specialist(final_people_model, middle_experience_test_people, class_to_idx_new_final_people,
                                    "Evaluating final_specialised_people_model on middle_experience_test_people")
    evaluate_baseline_vs_specialist(final_people_model, last_experience_test_people, class_to_idx_new_final_people,
                                    "Evaluating final_specialised_people_model on last_experience_test_people")
    print("\n\n")
    evaluate_baseline_vs_specialist(final_objects_and_vehicles_model, objects_and_vehicles_loader, class_to_idx_new_final_objects_and_vehicles,
                                    "Evaluating final_specialised_objects_and_vehicles_model on objects_and_vehicles_loader")
    evaluate_baseline_vs_specialist(final_objects_and_vehicles_model, first_experience_test_objects_and_vehicles, class_to_idx_new_final_objects_and_vehicles,
                                    "Evaluating final_specialised_objects_and_vehicles_model on first_experience_test_objects_and_vehicles")
    evaluate_baseline_vs_specialist(final_objects_and_vehicles_model, middle_experience_test_objects_and_vehicles, class_to_idx_new_final_objects_and_vehicles,
                                    "Evaluating final_specialised_objects_and_vehicles_model on middle_experience_test_objects_and_vehicles")
    evaluate_baseline_vs_specialist(final_objects_and_vehicles_model, last_experience_test_objects_and_vehicles, class_to_idx_new_final_objects_and_vehicles,
                                    "Evaluating final_specialised_objects_and_vehicles_model on last_experience_test_objects_and_vehicles")
    ##############################################################################################################################################################################################################################

    # COMPARISONS THAT CANNOT BE MADE IN THIS FILE DUE TO THE STRUCTURE OF MoE models (whether baselines or final MoE models). Peform this same comparisons by running evaluating the
    # MoE models in their own files and comparing with the results in this file !!!!!!!!!!!!!!!!
    '''MODEL_DIR_BASELINE = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE = 'cpu'
    model_names_BASELINE = ['MoE_TASK-IL_BASELINE']
    baseline, class_to_idx_new_baseline = main(MODEL_DIR_BASELINE, DEVICE_BASELINE, model_names_BASELINE)

    MODEL_DIR_global_model_moe = f"../Report_implementation_attempt/final_specialised_models/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_global_model_moe = 'cpu'
    model_names_global_model_moe = ['MoE_gating_network_global']
    global_model_moe, class_to_idx_new_global_model_moe = main(MODEL_DIR_global_model_moe, DEVICE_global_model_moe, model_names_global_model_moe)

    print(
        "############################# MoE_TASK-IL_BASELINE vs global model produced via my proposed CL strategy: #############################\n"
        "###### How close did the part of my strategy for combining the specialised models into the final global model get to being as good as it possibly could \n"
        "(i.e. how close is it to giving the same results as MoE_TASK-IL_BASELINE?): ######")
    evaluate_baseline_vs_specialist(baseline, test_loader, class_to_idx_new_baseline,
                                    "Evaluating MoE_TASK-IL_BASELINE on test_loader")
    evaluate_baseline_vs_specialist(global_model_moe, test_loader, class_to_idx_new_global_model_moe,
                                    "Evaluating MoE_gating_network_global_model on test_loader")

    ###########################################################################################################################################################################################################################################
    MODEL_DIR_BASELINE = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE = 'cpu'
    model_names_BASELINE = ['BASELINE_standard_supervised_global']
    baseline, class_to_idx_new_baseline = main(MODEL_DIR_BASELINE, DEVICE_BASELINE, model_names_BASELINE)

    MODEL_DIR_global_model_moe = f"../Report_implementation_attempt/final_specialised_models/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_global_model_moe = 'cpu'
    model_names_global_model_moe = ['MoE_gating_network_global_model']
    global_model_moe, class_to_idx_new_global_model_moe = main(MODEL_DIR_global_model_moe, DEVICE_global_model_moe, model_names_global_model_moe)

    print(
        "############################# BASELINE_standard_supervised_global (global model trained in standard supervised learning) vs global model produced via my proposed CL strategy: #############################")
    evaluate_baseline_vs_specialist(baseline, test_loader, class_to_idx_new_baseline,
                                    "Evaluating BASELINE_standard_supervised_global on test_loader")
    evaluate_baseline_vs_specialist(global_model_moe, test_loader, class_to_idx_new_global_model_moe,
                                    "Evaluating MoE_gating_network_global_model on test_loader")

    ###########################################################################################################################################################################################################################################
    MODEL_DIR_BASELINE = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE = 'cpu'
    model_names_BASELINE = ['NAIVE_BASELINE_global']
    baseline, class_to_idx_new_baseline = main(MODEL_DIR_BASELINE, DEVICE_BASELINE, model_names_BASELINE)

    MODEL_DIR_global_model_moe = f"../Report_implementation_attempt/final_specialised_models/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_global_model_moe = 'cpu'
    model_names_global_model_moe = ['MoE_gating_network_global_model']
    global_model_moe, class_to_idx_new_global_model_moe = main(MODEL_DIR_BASELINE, DEVICE_BASELINE, model_names_BASELINE)

    MODEL_DIR_BASELINE_MoE_TASK_IL = f"../Report_implementation_attempt/final_specialised_models_BASELINE/"  # MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    DEVICE_BASELINE_MoE_TASK_IL = 'cpu'
    model_names_BASELINE_MoE_TASK_IL = ['MoE_TASK-IL_BASELINE']
    baseline_MoE_TASK_IL, class_to_idx_new_baseline_MoE_TASK_IL = main(MODEL_DIR_BASELINE_MoE_TASK_IL, DEVICE_BASELINE_MoE_TASK_IL, model_names_BASELINE_MoE_TASK_IL)

    print(
        "############################# NAIVE_BASELINE_global_model (global model trained in Naive CL) vs global model produced via my proposed CL strategy: #############################\n"
        "ANSWERS THE KEY QUESTION OF MY RESEARCH PROJECT: Does my proposed CL strategy mitigate CF further than when no specialised models are used?")
    evaluate_baseline_vs_specialist(baseline, test_loader, class_to_idx_new_baseline,
                                    "Evaluating NAIVE_BASELINE_global_model on test_loader")
    evaluate_baseline_vs_specialist(global_model_moe, test_loader, class_to_idx_new_global_model_moe,
                                    "Evaluating MoE_gating_network_global_model on test_loader")

    print(
        "############################# NAIVE_BASELINE_global_model (global model trained in Naive CL) vs MoE_TASK-IL_BASELINE: #############################\n"
        "Does my proposed CL strategy mitigate CF further than when no specialised models are used? IF NOT, would it had if I had managed to perfection the global model aggregation stage"
        " of my strategy to the extent necessary for it to give the best results possible (i.e. the results MoE_TASK-IL_BASELINE gives)? Let us determine this by comparing "
        "NAIVE_BASELINE_global_model to MoE_TASK-IL_BASELINE")
    evaluate_baseline_vs_specialist(baseline, test_loader, class_to_idx_new_baseline,
                                    "Evaluating NAIVE_BASELINE_global_model on test_loader")
    evaluate_baseline_vs_specialist(baseline_MoE_TASK_IL, test_loader, class_to_idx_new_baseline_MoE_TASK_IL,
                                    "Evaluating MoE_TASK-IL_BASELINE on test_loader")'''