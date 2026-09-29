"""
Complete Model Aggregation via PFNM (Probabilistic Federated Neural Matching)

This is a drop-in replacement for the FedAvg version of this script.  The ONLY
thing that changed is the aggregation algorithm:

    BEFORE:  element-wise mean of the feature layers          (FedAvg)
    NOW:     Beta-Bernoulli neuron matching + Gaussian
             posterior mean of the MATCHED neurons            (PFNM)

    Yurochkin et al., "Bayesian Nonparametric Federated Learning of
    Neural Networks", ICML 2019.

Why this should work better than FedAvg here
--------------------------------------------
FedAvg averages channel i of model 1 with channel i of model 2.  Those two
channels are unrelated: neural networks are permutation invariant in their
hidden units, so independently trained models put "the same" feature in
different positions.  Averaging them destroys both.  That is exactly the
"large parameter divergence" you are seeing.

PFNM first *matches* neurons across models (Hungarian assignment under a
Beta-Bernoulli process prior), then averages only neurons that were matched
to each other.  No training, no gradients, no data pooling - a single
"communication round" in the paper's terms.

Three things the original PFNM code could not do, added in pfnm_resnet.py
------------------------------------------------------------------------
* Convolutions   - a filter is the analogue of a hidden neuron.
* BatchNorm      - BN parameters are carried by the permutation of the
                   channels they normalise, never matched independently.
* Residual adds  - `out = block(x) + shortcut(x)` forces every tensor writing
                   into the same residual stream to share ONE permutation.
                   The paper lists CNNs/ResNets as future work; this is the
                   part that had to be worked out.

Every permutation is verified to be function preserving before it is used
(see `verify_permutation_invariance`), so a wrong grouping fails loudly
instead of silently producing a broken model.

ALSO FIXED (this alone may explain a large part of the bad FedAvg numbers)
--------------------------------------------------------------------------
The previous `extract_feature_layers()` deliberately skipped
`running_mean` / `running_var` / `num_batches_tracked`, and
`load_averaged_feature_layers()` only writes keys it is given.  The global
model therefore kept its *freshly initialised* BatchNorm statistics
(mean 0, var 1).  At eval time (`model.eval()`) BN uses exactly those
statistics, so every activation in the network was being normalised with
garbage constants.  PFNM here merges the BN buffers too, and optionally
re-estimates them with forward passes only (no training) - see
RECALIBRATE_BN below.

For setup:
  Model 1 (Outdoors):         6 classes  {12, 17, 23, 33, 37, 49}
  Model 2 (Nature):           9 classes  {0, 47, 51, 52, 53, 54, 56, 57, 59}
  Model 3 (People):           3 classes  {2, 11, 35}
  Model 4 (Objects/Vehicles): 25 classes {5, 8, 9, 10, 13, 16, ...}
  
Architecture:
  - Feature layers (averaged): conv1, bn1, layer1-3
  - Classifier (remapped per-class): linear [64 → 100]
"""

import copy
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import Dict, Tuple
from collections import OrderedDict
from Report_implementation_attempt.ResNet20_DynamicModel_vReport import ResNet20_DynamicModel

# ---------------------------------------------------------------------------
# PFNM engine (new).  Put pfnm_resnet.py next to this file.
# ---------------------------------------------------------------------------
from pfnm_resnet import pfnm_merge_state_dicts, recalibrate_batchnorm
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
# PFNM CONFIGURATION
# ----------------------------------------------------------------------------
# sigma / sigma0 / gamma are the paper's hyper-parameters (Prop. 2 and 3).
# The authors' experiment.py defaults them all to 1.0 and then grid-searches
# them; PFNM is reported to be fairly sensitive to them, so treat the block
# below as the first thing to tune if results are not what you expect.
#
#   sigma   : likelihood variance of a local neuron around its global atom.
#             Smaller  -> matching trusts the weights more (sharper matching).
#   sigma0  : prior variance of a global atom. Larger -> merged weights are the
#             plain mean of the matched neurons; smaller -> shrunk toward zero.
#   gamma   : IBP mass parameter. Only has an effect when PFNM_ALLOW_GROWTH is
#             True (it controls how eagerly new global neurons are created).
#   it      : leave-one-out iterations inside a group (pfnm.py uses 5).
#   sweeps  : outer coordinate-descent passes over all permutation groups.
#             A ResNet is a DAG, not a chain, so one top-down pass is not
#             enough; 3 is usually plenty (it prints when it converges).
# ============================================================================
PFNM_SIGMA = 1.0
PFNM_SIGMA0 = 1.0
PFNM_GAMMA = 1.0
PFNM_IT = 5
PFNM_SWEEPS = 3

# 'full'     : atoms = incoming weights + BN + outgoing weights.  USE THIS.
# 'outgoing' : the paper's literal top-down formulation (outgoing weights only).
#              It is degenerate in YOUR setting: your specialists have disjoint
#              label sets, so their remapped classifier rows have disjoint
#              support, every pairing gives almost the same cost, and the top
#              layer gets matched essentially at random.
PFNM_ATOM_MODE = 'full'

# PFNM is nonparametric: it can invent new global neurons, giving a global
# network WIDER than the locals.  A fixed ResNet-20 cannot represent that, so
# growth is off.  Turn it on only with a width-configurable architecture.
PFNM_ALLOW_GROWTH = False

# Re-estimate BatchNorm running statistics on the merged model.
# Forward passes only: no gradients, no labels, no parameter updates - so this
# does NOT violate your "no training" constraint.  Merging channels invalidates
# the inherited BN statistics (the "variance collapse" that REPAIR, Jordan et
# al. 2023, fixes for interpolated models); this is usually worth several
# accuracy points and is strongly recommended.
RECALIBRATE_BN = True
RECALIBRATE_BN_BATCHES = 100

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


# ============================================================================
# Calibration loader for BatchNorm re-estimation (NEW, used by PFNM step 5)
# ----------------------------------------------------------------------------
# Uses TRAINING data from all four specialists so that the merged model sees
# the same input distribution it will be evaluated on.  Labels are never used:
# only forward passes, purely to recompute BN running_mean / running_var.
# ============================================================================

def build_calibration_loader(batch_size: int = 128):
    """Concatenate the four specialists' training streams into one loader."""
    train_streams = [
        outdoors_specialised_models_train_stream_class_il,
        nature_specialised_models_train_stream_class_il,
        people_specialised_models_train_stream_class_il,
        objects_and_vehicles_specialised_models_train_stream_class_il,
    ]

    datasets = []
    for stream in train_streams:
        for exp in stream:
            if len(exp.dataset) != 0:
                datasets.append(exp.dataset)

    if not datasets:
        raise RuntimeError("No training data found for BatchNorm calibration")

    return DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True)


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
    [UNUSED by the PFNM pipeline - kept for reference.  Note the bug this had:
     it filtered out running_mean / running_var / num_batches_tracked, so the
     global model kept its randomly-initialised BatchNorm statistics.]

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
    [UNUSED - this is the old FedAvg aggregation, kept only for reference /
     A-B comparison.  The pipeline now calls pfnm_merge_state_dicts() instead.]

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
    2. Remap each specialist's classifier into the shared CIFAR-100 space
    3. PFNM: match neurons across models, then merge the matched neurons
       (this replaces the old FedAvg element-wise averaging)
    4. Load the merged weights into the global model
    5. Re-estimate BatchNorm statistics (forward passes only)
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

    model_dir = f"../final_specialised_models/"  #model_dir = Path(model_dir)
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
    # STEP 2: Remap every specialist classifier into the shared CIFAR-100
    #         label space, BEFORE matching.
    #
    #         Why before?  PFNM needs all J state_dicts to have identical
    #         shapes.  Your specialists have 6 / 9 / 3 / 25 output units, so
    #         without this the final layer cannot participate at all.  After
    #         remapping, every model is a 100-way classifier whose rows are
    #         zero outside its own classes.
    # ==================================================================

    print("Step 2: Remapping classifiers to CIFAR-100 space (done BEFORE matching)...")

    class_coverage = {}
    pfnm_state_dicts = []
    feature_dim = models[model_names[0]].linear.weight.shape[1]

    with torch.no_grad():
        for model_name in model_names:
            model = models[model_name]
            mapping = mappings[model_name]

            w, b = remap_classifier_to_cifar100(
                model_name, model, mapping, num_subset_classes[model_name],
                num_classes=100, device=device
            )

            # Full state_dict (features + BN buffers + remapped classifier).
            # NOTE: unlike the old extract_feature_layers(), the BatchNorm
            # running_mean / running_var / num_batches_tracked buffers are
            # included here.  Dropping them left the global model with
            # freshly-initialised BN statistics, which is fatal at eval time.
            sd = OrderedDict(
                (k, v.detach().clone()) for k, v in model.state_dict().items()
            )
            sd['linear.weight'] = w
            sd['linear.bias'] = b
            pfnm_state_dicts.append(sd)

            for local_idx, cifar_class in mapping.items():
                if cifar_class not in class_coverage:
                    class_coverage[cifar_class] = []
                class_coverage[cifar_class].append(model_name)

            print(f"  \u2713 Remapped {model_name}")
            print(f"    - Classes: {len(mapping)}")
            print(f"    - Positions: {sorted(mapping.values())}")

    print()

    # ==================================================================
    # STEP 3: PFNM AGGREGATION  (this replaces the FedAvg averaging)
    # ==================================================================

    print("Step 3: PFNM aggregation across all models...")
    print(f"  (Matching neurons from {len(model_names)} different CIFAR subsets)")
    print()
    print("  Why matching instead of averaging?")
    print("  - Hidden units are permutation invariant: channel i of the Nature")
    print("    model and channel i of the Objects model are unrelated features.")
    print("  - FedAvg averages them positionally, which destroys both.")
    print("  - PFNM first solves a Hungarian assignment under a Beta-Bernoulli")
    print("    prior (paper Prop. 2/3), then averages only neurons that were")
    print("    actually matched to each other (paper Prop. 1/4).")
    print("  - No gradients, no data pooling: a single communication round.")
    print()

    merged_state_dict, pfnm_info = pfnm_merge_state_dicts(
        pfnm_state_dicts,
        model_factory=lambda: ResNet20_DynamicModel(num_classes=43), #model_factory=lambda: ResNet20_DynamicModel(num_classes=100),
        sigma=PFNM_SIGMA,
        sigma0=PFNM_SIGMA0,
        gamma=PFNM_GAMMA,
        it=PFNM_IT,
        sweeps=PFNM_SWEEPS,
        atom_mode=PFNM_ATOM_MODE,
        allow_growth=PFNM_ALLOW_GROWTH,
        # The classifier rows are disjoint across specialists, so summing is
        # exactly "place each specialist's rows at its own class indices" -
        # identical to what the FedAvg version of this script did.
        sum_keys=('linear.weight', 'linear.bias'),
        include_classifier_in_atoms=False,
        verbose=True,
    )

    print()
    print(f"  \u2713 PFNM merge complete")
    print(f"    - Permutation groups matched: {len(pfnm_info['matched_groups'])}")
    print(f"    - Shortcut style detected: {pfnm_info['shortcut_style']}")
    print(f"    - Posterior shrinkage factor: {pfnm_info['shrinkage']:.4f}")

    print()

    # ==================================================================
    # STEP 4: Create global model from the merged weights
    # ==================================================================

    print("Step 4: Creating global model with PFNM-merged weights...")

    global_model = ResNet20_DynamicModel(num_classes=43)#ResNet20_DynamicModel(num_classes=100)
    global_model.load_state_dict(merged_state_dict)

    print(f"  \u2713 Created global model:")
    print(f"    - conv1 + bn1: \u2713 PFNM-merged from {len(model_names)} models")
    print(f"    - layer1-3:    \u2713 PFNM-merged from {len(model_names)} models")
    print(f"    - BN buffers:  \u2713 merged (were previously left at init!)")
    print(f"    - Classifier:  \u2713 {global_model.linear.weight.shape}")

    print()

    # ==================================================================
    # STEP 5/6: BatchNorm re-estimation (forward passes only, no training)
    # ==================================================================

    if RECALIBRATE_BN:
        print("Step 5: Re-estimating BatchNorm statistics (forward passes only)...")
        try:
            calib_loader = build_calibration_loader()
            recalibrate_batchnorm(
                global_model, calib_loader,
                num_batches=RECALIBRATE_BN_BATCHES, device=device, verbose=True
            )
        except Exception as e:
            print(f"  [!] BN recalibration skipped ({e}).")
            print(f"      Results will be noticeably worse without it.")
        print()
    else:
        print("Step 5: BatchNorm recalibration DISABLED (RECALIBRATE_BN = False)")
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
        'feature_averaging': False,  # KEY: features are MATCHED then merged, not averaged
        'aggregation_method': 'PFNM',
        'num_models_aggregated': len(model_names),
        # --- PFNM specifics (assignments/spec stripped out: not picklable-friendly) ---
        'pfnm': {k: v for k, v in pfnm_info.items() if k not in ('assignments', 'spec')},
        'pfnm_assignments': pfnm_info.get('assignments'),
        'batchnorm_recalibrated': RECALIBRATE_BN,
    }

    print(f"\n  Aggregation method: PFNM NEURON MATCHING + CLASSIFIER REMAPPING")
    print(f"  Total CIFAR-100 classes: 100")
    print(f"  Classes trained: {len(class_coverage)}")
    print(f"  Classes NOT trained: {len(untrained_classes)}")
    print(f"  Feature extraction layers: PFNM-matched + merged from {len(model_names)} models")
    print(f"  Permutation groups matched: {len(pfnm_info['matched_groups'])} "
          f"of {len(pfnm_info['groups'])}")
    print(f"  Permutation invariance verified: {pfnm_info.get('permutation_invariance_ok')}")
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

    MODEL_DIR = Path("../final_specialised_models") #MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    MODEL_DIR.mkdir(parents=True, exist_ok=True) #MODEL_DIR = Path(f"../Report_implementation_attempt/final_specialised_models") #MODEL_DIR = Path('./final_specialised_models')
    OUTPUT_DIR = Path('../final_specialised_models')
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
        print(f"  - Feature extraction: PFNM-matched + merged from 4 models")
        print(f"  - Classifier: Remapped to CIFAR-100 space")
        print(f"  - All 43 trained classes: Ready for prediction")
        print(f"  - 57 untrained classes: Will use the merged features")
        print(f"\nModel saved to: {OUTPUT_DIR / 'global_model_with_averaged_features.pt'}")
        print("="*80 + "\n")

        return global_model

    except Exception as e:
        print(f"\n✗ Aggregation failed: {e}")
        raise


if __name__ == '__main__':
    global_model = main()
    #evaluate_aggregated_model_FedAvg_code_v3(global_model, test_loader)
    evaluate_aggregated_model(global_model, test_loader)
    evaluate_aggregated_model(global_model, outdoors_loader)
    evaluate_aggregated_model(global_model, nature_loader)
    evaluate_aggregated_model(global_model, people_loader)
    evaluate_aggregated_model(global_model, objects_and_vehicles_loader)
    '''global_model_reload = torch.load("../Report_implementation_attempt/final_specialised_models/global_model_with_averaged_features.pt")
    global_model_reload_architecture = ResNet20_DynamicModel(num_classes=100)'''#global_model_reload_architecture = ResNet20_DynamicModel(num_classes=100)
    ####global_model_reload_architecture.load_state_dict(global_model_reload)
    ####evaluate_aggregated_model_FedAvg_code_v3(global_model_reload_architecture, test_loader)
    ####evaluate_aggregated_model(global_model_reload_architecture, test_loader)
    #evaluate_aggregated_model(global_model_reload_architecture,outdoors_specialised_models_train_stream_class_il)
    #evaluate_aggregated_model(global_model_reload_architecture, avl_test_loader)
