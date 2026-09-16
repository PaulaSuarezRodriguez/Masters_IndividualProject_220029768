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

'''original_to_aggregated_dict = {
"outdoors" : {17: 0, 12: 1, 33: 2, 23: 3, 49: 4, 37: 5},#, #OUTDOORS}
"nature": {51: 6, 47: 7, 0: 8, 54: 9, 53: 10, 52: 11, 59: 12, 57: 13, 56: 14},#, #NATURE}
"people" : {2: 15, 11: 16, 35: 17}, #, #PEOPLE
"objects_and_vehicles" : {16: 18, 13: 19, 10: 20, 9: 21, 8: 22, 5: 23, 58: 24, 48: 25, 41: 26, 40: 27, 39: 28, 28: 29, 25: 30, 22: 31, 20: 32, 94: 33, #OBJECTS_AND_VEHICLES
                               90: 34, 89: 35, 87: 36, 86: 37, 85: 38, 84: 39, 81: 40, 69: 41, 61: 42} }'''

'''original_to_aggregated_dict = {
"nature": {0: 0, 47: 1, 51: 2, 52: 3, 53: 4, 54: 5, 56: 6, 57: 7, 59: 8}, #, 62: 9}, #70, 82, 83, 92, 96}#{51: 0, 47: 1, 0: 2, 54: 3, 53: 4, 52: 5, 59: 6, 57: 7, 56: 8},#, #NATURE}
"people" : {2: 9, 11: 10, 35: 11}#3: 46, 4: 98},#{2: 9, 11: 10, 35: 11}, #, #PEOPLE
}

original_to_aggregated_dict_ALL = {0: 0, 47: 1, 51: 2, 52: 3, 53: 4, 54: 5, 56: 6, 57: 7, 59: 8, #, 62: 9}, #70, 82, 83, 92, 96}#{51: 0, 47: 1, 0: 2, 54: 3, 53: 4, 52: 5, 59: 6, 57: 7, 56: 8},#, #NATURE}
2: 9, 11: 10, 35: 11}#3: 46, 4: 98},#{2: 9, 11: 10, 35: 11}, #, #PEOPLE'''




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

    return remapped_weights, remapped_bias

'''def remap_classifier_to_aggregated(
    #model_name,
    model: nn.Module,
    w,
    b,
    original_to_aggregated_dict: Dict[int, int],
    #num_subset_classes,
    num_classes: int = 12,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:

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
    #remapped_weights = w
    #remapped_bias = b

    cifar_idx_count = 0
    # Place each trained class at its CIFAR-100 position
    for cifar_idx, aggregated_idx in original_to_aggregated_dict.items():
        print("original_to_aggregated_dict", original_to_aggregated_dict)
        print("print(cifar_idx, aggregated_idx)", cifar_idx, aggregated_idx)
        cifar_idx_count += 1
        print("print(cifar_idx_count)", cifar_idx_count)
        #print("print(num_subset_classes)", num_subset_classes)
        #if cifar_idx_count > num_subset_classes:
         #   print("print(num_subset_classes) THAT CAUSED BREAK", num_subset_classes)
          #  break
        # local_idx: 0-5 for outdoors (position in model's classifier)
        # cifar_class: 12, 17, 23, ... (actual CIFAR class)
        print(f"print(cifar_class = new_to_original[cifar_idx]) --> {cifar_idx}")
        #print(f"print(w[cifar_idx])", w[cifar_idx])
        remapped_weights[aggregated_idx] = w[cifar_idx].to(device)
        remapped_bias[aggregated_idx] = b[cifar_idx].to(device)

    if VERBOSE:
        print("remapped_weights", remapped_weights)
        print("remapped_bias", remapped_bias)

    return remapped_weights, remapped_bias'''


'''def remap_classifier_to_cifar100(model_name, model, new_to_original, num_trained_classes, num_classes=100, device='cpu'):
    """
    Remap classifier from local to CIFAR-100 space.

    Only maps the first num_trained_classes indices, since those are the ones
    that were actually trained. Ignore extra entries in new_to_original that
    came from untrained experiences.
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

    # Only map indices 0 to num_trained_classes-1
    for local_idx in range(num_trained_classes):
        if local_idx in new_to_original:
            cifar_class = new_to_original[local_idx]
            print(f"print(cifar_class = new_to_original[local_idx]) --> {cifar_class}")
            print(f"print(classifier.weight[local_idx] for {model_name})",classifier.weight[local_idx])
            remapped_weights[cifar_class] = classifier.weight[local_idx]#.to(device)
            remapped_bias[cifar_class] = classifier.bias[local_idx]#.to(device)

    return remapped_weights, remapped_bias'''
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

    print("Step 4: Creating global model with averaged features...")

    global_model = ResNet20_DynamicModel(num_classes=100)#ResNet20_DynamicModel(num_classes=100)
    global_model = global_model#.to(device)

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

        global_classifier_weights = torch.zeros(
            100, feature_dim,
            device=device,
            dtype=global_model.linear.weight.dtype  # ORIGINALLY: #global_classifier_weights = torch.zeros(100, feature_dim, device=device)
        )
        global_classifier_bias = torch.zeros(
            100,
            device=device,
            dtype=global_model.linear.bias.dtype  # ORIGINALLY: #global_classifier_bias = torch.zeros(100, device=device)
        )
        #global_classifier_weights = global_model.linear.weight
        print("global_classifier_weights INSTEAD OF torch.zero\n", global_classifier_weights)
        print("global_classifier_weights.shape INSTEAD OF torch.zero \n", global_classifier_weights.shape)
        #global_classifier_bias = global_model.linear.bias
        print("global_classifier_bias INSTEAD OF torch.zero\n", global_classifier_bias)
        print("global_classifier_bias.shape INSTEAD OF torch.zero \n", global_classifier_bias.shape)

        class_coverage = {}
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

            row_norms = w.norm(p=2, dim=1)

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
                )

            '''w, b = remap_classifier_to_aggregated(
                model_name, model, w, b, original_to_aggregated_dict[model_name], num_subset_classes[model_name], num_classes=13, device=device
            )
            print(f"w to see what it looks like after remap_classifier_to_aggregated ({model_name}):\n", w)
            print(f"b to see what it looks like after remap_classifier_to_aggregated ({model_name}):\n", b)'''

            #TODO: Original code for global_classifier_weights += w and global_classifier_bias += b. This and the version I developed seem to do the same!!!
            #Add to global classifier (no averaging - classes don't overlap!)
            #########global_classifier_weights += w
            #########global_classifier_bias += b
            #########print(f"global_weights += w to see what it looks like after additions ({model_name}):\n", global_classifier_weights)
            #########print(f"global_bias += b to see what it looks like after additions ({model_name}):\n", global_classifier_bias)

            all_weights[model_name] = w
            all_bias[model_name] =b

            '''global_classifier_weights = torch.stack([m.float() for m in all_weights.values()])
            global_classifier_bias = torch.stack([b.float() for b in all_bias.values()])'''

            '''global_classifier_weights = torch.cat([m.float() for m in all_weights.values()], dim=0)
            global_classifier_bias = torch.cat([b.float() for b in all_bias.values()], dim=0)'''

            # TODO: Back to original code for global_classifier_weights += w and global_classifier_bias += b. This and the version I developed seem to do the same!!!
            # TODO: This is the version I developed
            global_classifier_weights = torch.stack([m.float() for m in all_weights.values()]).sum(dim=0)
            global_classifier_bias = torch.stack([b.float() for b in all_bias.values()]).sum(dim=0)

            print(f"global_weights += w to see what it looks like after additions ({model_name}):\n",global_classifier_weights)
            print(f"global_bias += b to see what it looks like after additions ({model_name}):\n", global_classifier_bias)


            for local_idx, cifar_class in mapping.items():
                if cifar_class not in class_coverage:
                    class_coverage[cifar_class] = []
                class_coverage[cifar_class].append(model_name)

            print(f"  ✓ Remapped {model_name}")
            print(f"    - Classes: {len(mapping)}")
            print(f"    - Positions: {sorted(mapping.values())}")

            global_classifier_weights = global_classifier_weights
            global_classifier_bias = global_classifier_bias

    '''global_classifier_weights, global_classifier_bias = remap_classifier_to_aggregated(global_model, global_classifier_weights, global_classifier_bias, original_to_aggregated_dict_ALL, num_classes=13, device=device)
    print(f"w to see what it looks like after remap_classifier_to_aggregated:\n", global_classifier_weights)
    print(f"b to see what it looks like after remap_classifier_to_aggregated:\n", global_classifier_bias)'''

    print()

    # ==================================================================
    # STEP 6: Load remapped classifier into global model
    # ==================================================================

    print("Step 6: Loading remapped classifier into global model...")

    global_model.linear.weight = nn.Parameter(global_classifier_weights)
    #transposed_weight = nn.Parameter(global_classifier_weights).transpose(0, -1)#.to(device)
    global_model.linear.bias = nn.Parameter(global_classifier_bias)
    #transposed_bias = nn.Parameter(global_classifier_bias).transpose(0, -1)#.to(device)

    #FAILED global_model.linear.weight = nn.Parameter(global_classifier_weights).data
    #FAILED global_model.linear.bias = nn.Parameter(global_classifier_bias).data

    ###print("global_model.state_dict()",global_model.state_dict())
    ###print("global_model.linear.state_dict()", global_model.linear.state_dict())

    test={}
    for layer_name, layer in global_model.state_dict().items():
        #print(f"{layer_name}.shape:",layer.shape)
        test[layer_name]=layer.shape

    #WORKS but does not seem to have any effect so, back to original
    #global_model.linear.weight.data = nn.Parameter(global_classifier_weights).data
    #global_model.linear.bias.data = nn.Parameter(global_classifier_bias).data

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
