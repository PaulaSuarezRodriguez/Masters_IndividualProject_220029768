"""
Complete Model Aggregation via FedMA (Federated Matched Averaging)

Drop-in replacement for the FedAvg version of this script.  Only the
aggregation algorithm changed:

    BEFORE:  element-wise mean of the feature layers            (FedAvg)
    NOW:     BBP-MAP channel matching, front-to-back, then the
             1/J mean of the MATCHED channels                   (FedMA)

    Wang, Yurochkin, Sun, Papailiopoulos, Khazaeni,
    "Federated Learning with Matched Averaging", ICLR 2020.

Why this should beat FedAvg
---------------------------
FedAvg averages channel i of model 1 with channel i of model 2.  Those two
channels are unrelated: CNNs are permutation invariant in their channels
(FedMA eq. 5), so independently trained models put "the same" filter in
different positions.  Averaging them destroys both.  That is the parameter
divergence you are seeing.  FedMA first undoes the permutation, then averages.

IMPORTANT - what is missing, and why it matters
------------------------------------------------
FedMA is matched averaging PLUS layer-wise retraining: match layer 1, freeze
it, have every client RETRAIN all later layers, match layer 2, retrain again,
and so on (Algorithm 1).  You asked for aggregation only, so the retraining
loop is omitted.

The paper is explicit that the retraining is the contribution.  Section 2.3
and Figure 1 show that matching straight through a deep CNN without it "fails
on deep architectures", and that BBP-MAP applied to CNNs this way "only gives
very minor improvements over weight averaging".  Expect a modest gain over
FedAvg here, not the numbers in the paper's tables.

What stays FedMA rather than PFNM
---------------------------------
  a. Atoms are INCOMING weights: rows of Pi_{j,n-1}^T W_{j,n} flattened over
     C_in x w x h (eq. 5).
  b. Matching is front-to-back, layer n only after layer n-1 is fixed
     (eq. 4/5, recursion base = the input image).
  c. Aggregation is the plain 1/J mean of matched channels (Algorithm 1),
     not PFNM's shrinking posterior mean.
  d. The last layer is not matched; it is averaged by class proportions p_jk.
     Your specialists are disjoint, so p_jk is 1 for the owning model and 0
     otherwise, and the rule reduces to stacking their classifiers.
  e. gamma0 = 7 (FedMA Appendix D.2) rather than PFNM's 1.0 - though gamma
     has no effect unless FEDMA_ALLOW_GROWTH is on.

Beyond the paper
----------------
FedMA ignores BatchNorm ("we ignore all batch normalization layers in the VGG
architecture and leave it for future work") and lists residual connections as
future work.  ResNet-20 has both.  The residual-stream grouping and the BN
handling are my construction; every permutation is verified function
preserving before use.

ALSO FIXED (may explain much of the bad FedAvg numbers on its own)
-------------------------------------------------------------------
The old extract_feature_layers() skipped running_mean / running_var /
num_batches_tracked, and load_averaged_feature_layers() only writes the keys
it is handed, so the global model kept its freshly initialised BatchNorm
statistics - and evaluate_aggregated_model() calls model.eval(), which is
exactly when BN uses those buffers.  Every activation was normalised with
meaningless constants.  Those buffers are now merged, and optionally
re-estimated with forward passes only (RECALIBRATE_BN).

For setup:
  Model 1 (Outdoors):         6 classes  {12, 17, 23, 33, 37, 49}
  Model 2 (Nature):           9 classes  {0, 47, 51, 52, 53, 54, 56, 57, 59}
  Model 3 (People):           3 classes  {2, 11, 35}
  Model 4 (Objects/Vehicles): 25 classes {5, 8, 9, 10, 13, 16, ...}
  
Architecture:
  - Feature layers (averaged): conv1, bn1, layer1-3
  - Classifier (concatenated, remapped per-class): linear [64 → 100]
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
# FedMA engine (new).  Put fedma_resnet.py AND pfnm_resnet.py next to this
# file: FedMA's solver is BBP-MAP, which the FedMA paper takes unchanged from
# PFNM, so the solver and the ResNet analysis are shared, not duplicated.
# ---------------------------------------------------------------------------
from fedma_resnet import fedma_merge_state_dicts, recalibrate_batchnorm
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
# FedMA CONFIGURATION
# ----------------------------------------------------------------------------
# sigma / sigma0 / gamma are the BBP-MAP hyper-parameters FedMA inherits from
# PFNM.  FedMA Appendix D.2 reports sigma=1, sigma0=1, gamma0=7 for VGG-9 on
# CIFAR-10.  Tune these first if results disappoint.
#
#   sigma   : likelihood variance of a local channel around its global atom.
#             Smaller -> matching trusts the weights more (sharper matching).
#   sigma0  : prior variance of a global atom.
#   gamma   : IBP mass parameter, controls creation of new global channels.
#             NO EFFECT unless FEDMA_ALLOW_GROWTH is True.
#   it      : leave-one-out iterations inside a group (BBP-MAP uses 5).
# ============================================================================
FEDMA_SIGMA = 1.0
FEDMA_SIGMA0 = 1.0
FEDMA_GAMMA = 7.0
FEDMA_IT = 5

# 1 = FedMA's single front-to-back recursion (eq. 4/5), the faithful setting.
# A ResNet is a DAG rather than a chain, so the first residual stream is also
# written by convs whose input groups are matched after it; passes=2 fixes
# that, at the cost of no longer being a pure recursion. Worth trying both.
FEDMA_PASSES = 1

# FedMA lets the global model grow (max_j L_j <= L <= sum_j L_j; the paper
# reports 1.11x for VGG-9). A fixed ResNet-20 cannot hold a wider model, so
# this is off and the engine raises a clear error if a group would grow.
FEDMA_ALLOW_GROWTH = False

# Re-estimate BatchNorm running statistics on the merged model.
# Forward passes only: no gradients, no labels, no parameter updates, so this
# does NOT violate your "no training" constraint. Merging channels invalidates
# the BN statistics inherited from the parents; usually worth several accuracy
# points. (FedMA sidesteps this by ignoring BN entirely, which a ResNet cannot.)
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
# Calibration loader for BatchNorm re-estimation (NEW, used by FedMA step 5)
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
    [UNUSED by the FedMA pipeline - kept for reference.  Note the bug this had:
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
     A-B comparison.  The pipeline now calls fedma_merge_state_dicts() instead.]

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
    3. FedMA: match channels front-to-back, then average matched channels
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
    # STEP 2: Prepare the specialists for matching.
    #
    #         The classifier is NOT aggregated.  It is remapped into CIFAR-100
    #         label space here (exactly as STEP 5 of the original FedAvg
    #         script did) and carried through the merge only so that its INPUT
    #         columns get permuted along with the feature channels they read.
    #         A classifier whose columns are not permuted would be wired to
    #         the wrong features after matching.
    #
    #         Only the first `n_local` entries of `new_to_original` are real:
    #         the metadata lists more classes (10 / 15 / 5 / 25) than the
    #         classifier has rows (6 / 9 / 3 / 25) because it also lists
    #         experiences that were never trained.  The extras are ignored.
    # ==================================================================

    print("Step 2: Remapping classifiers to CIFAR-100 space (NOT aggregated)...")

    class_coverage = {}
    fedma_state_dicts = []
    feature_dim = models[model_names[0]].linear.weight.shape[1]

    with torch.no_grad():
        for model_name in model_names:
            model = models[model_name]
            mapping = mappings[model_name]
            n_local = model.linear.weight.shape[0]

            w, b = remap_classifier_to_cifar100(
                model_name, model, mapping,
                min(num_subset_classes[model_name], n_local),
                num_classes=100, device=device
            )

            # Full state_dict (features + BN buffers + remapped classifier).
            # NOTE: unlike the old extract_feature_layers(), the BatchNorm
            # running_mean / running_var / num_batches_tracked buffers are
            # included.  Dropping them left the global model with freshly
            # initialised BN statistics, which is fatal at eval time.
            sd = OrderedDict(
                (k, v.detach().clone()) for k, v in model.state_dict().items()
            )
            sd['linear.weight'] = w
            sd['linear.bias'] = b
            fedma_state_dicts.append(sd)

            for local_idx in range(n_local):
                cifar_class = int(mapping[local_idx])
                class_coverage.setdefault(cifar_class, []).append(model_name)

            print(f"  \u2713 Remapped {model_name}")
            print(f"    - Classes: {n_local}")
            print(f"    - Positions: {sorted(int(mapping[i]) for i in range(n_local))}")

    print()

    # ==================================================================
    # STEP 3: FedMA AGGREGATION  (this replaces the FedAvg averaging)
    # ==================================================================

    print("Step 3: FedMA matched averaging across all models...")
    print(f"  (Matching channels from {len(model_names)} different CIFAR subsets)")
    print()
    print("  Why matching instead of averaging?")
    print("  - CNN channels are permutation invariant (FedMA eq. 5): channel i")
    print("    of the Nature model and channel i of the Objects model are")
    print("    unrelated filters.")
    print("  - FedAvg averages them positionally, which destroys both.")
    print("  - FedMA solves a Hungarian assignment with BBP-MAP (eq. 2/3),")
    print("    front-to-back, then takes the 1/J mean of matched channels.")
    print("  - No gradients, no data pooling.")
    print()
    print("  NOTE: FedMA's layer-wise RETRAINING loop (Algorithm 1) is omitted,")
    print("        because you asked for aggregation only. The paper shows that")
    print("        is the half that makes FedMA work on deep CNNs, so expect a")
    print("        modest gain over FedAvg rather than the paper's numbers.")
    print()

    merged_state_dict, fedma_info = fedma_merge_state_dicts(
        fedma_state_dicts,
        model_factory=lambda: ResNet20_DynamicModel(num_classes=100),
        sigma=FEDMA_SIGMA,
        sigma0=FEDMA_SIGMA0,
        gamma=FEDMA_GAMMA,
        it=FEDMA_IT,
        passes=FEDMA_PASSES,
        allow_growth=FEDMA_ALLOW_GROWTH,
        # Algorithm 1, else branch: the last layer is averaged by class
        # proportions p_jk. Disjoint specialists -> p_jk is 1 for the owner and
        # 0 elsewhere, so this reduces to stacking (class_proportions=None).
        # The classifier is excluded from aggregation: permuted, never merged.
        # FedMA's Algorithm 1 last-layer rule (average by class proportions
        # p_jk) reduces to exactly this for non-overlapping specialists anyway,
        # since p_jk is 1 for the owning model and 0 for every other.
        exclude_keys=('linear.weight', 'linear.bias'),
        verbose=True,
    )

    print()
    print(f"  \u2713 FedMA merge complete")
    print(f"    - Groups matched: {len(fedma_info['matched_groups'])}")
    print(f"    - Shortcut style detected: {fedma_info['shortcut_style']}")
    print(f"    - Aggregation: plain 1/J mean of matched channels (features only)")

    print()

    # ==================================================================
    # STEP 3b: Build the global classifier by CONCATENATION, not aggregation
    #
    #          Each specialist's classifier was remapped into CIFAR-100 space
    #          (zero everywhere except its own classes) and permuted by the
    #          merge so its columns follow the matched feature channels.  The
    #          classes are non-overlapping, so summing the remapped matrices
    #          is exactly "place each specialist's rows at its own CIFAR-100
    #          indices" - i.e. a concatenation, which is what STEP 5 of the
    #          original FedAvg script did.  No averaging of any kind touches
    #          the classifier.
    # ==================================================================

    print("Step 3b: Concatenating specialist classifiers (no aggregation)...")

    permuted_classifiers = fedma_info['permuted_excluded']

    global_classifier_weights = torch.zeros(
        100, feature_dim, device=device,
        dtype=fedma_state_dicts[0]['linear.weight'].dtype)
    global_classifier_bias = torch.zeros(
        100, device=device,
        dtype=fedma_state_dicts[0]['linear.bias'].dtype)

    for model_name, pc in zip(model_names, permuted_classifiers):
        global_classifier_weights += pc['linear.weight'].to(
            device=device, dtype=global_classifier_weights.dtype)
        global_classifier_bias += pc['linear.bias'].to(
            device=device, dtype=global_classifier_bias.dtype)
        n_rows = int((pc['linear.weight'].abs().sum(dim=1) > 0).sum())
        print(f"  \u2713 Placed {model_name}: {n_rows} class rows")

    n_filled = int((global_classifier_weights.abs().sum(dim=1) > 0).sum())
    print(f"  \u2713 Global classifier: {tuple(global_classifier_weights.shape)}, "
          f"{n_filled} of 100 class rows filled")

    merged_state_dict['linear.weight'] = global_classifier_weights
    merged_state_dict['linear.bias'] = global_classifier_bias

    print()

    # ==================================================================
    # STEP 4: Create global model from the merged weights
    # ==================================================================

    print("Step 4: Creating global model with FedMA-merged weights...")

    global_model = ResNet20_DynamicModel(num_classes=100)
    global_model.load_state_dict(merged_state_dict)

    print(f"  \u2713 Created global model:")
    print(f"    - conv1 + bn1: \u2713 FedMA-merged from {len(model_names)} models")
    print(f"    - layer1-3:    \u2713 FedMA-merged from {len(model_names)} models")
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

    untrained_classes = set(range(100)) - set(class_coverage.keys())  # CIFAR-100 classes not covered

    report = {
        'total_cifar_classes': 100,
        'num_output_classes': 100,
        'classifier_aggregated': False,
        'trained_classes': len(class_coverage),
        'untrained_classes': len(untrained_classes),
        'class_coverage': class_coverage,
        'models_used': model_names,
        'feature_dim': feature_dim,
        'device': device,
        'feature_averaging': False,  # KEY: features are MATCHED then merged, not averaged
        'aggregation_method': 'FedMA',
        'num_models_aggregated': len(model_names),
        # --- FedMA specifics (assignments/spec stripped out: not picklable-friendly) ---
        'fedma': {k: v for k, v in fedma_info.items() if k not in ('assignments', 'spec')},
        'fedma_assignments': fedma_info.get('assignments'),
        'batchnorm_recalibrated': RECALIBRATE_BN,
    }

    print(f"\n  Aggregation method: FedMA MATCHED AVERAGING + CLASSIFIER STACKING")
    print(f"  Total CIFAR-100 classes: 100")
    print(f"  Classes trained: {len(class_coverage)}")
    print(f"  Classes NOT trained: {len(untrained_classes)}")
    print(f"  Feature extraction layers: FedMA-matched + averaged from {len(model_names)} models")
    print(f"  Permutation groups matched: {len(fedma_info['matched_groups'])} "
          f"of {len(fedma_info['groups'])}")
    print(f"  Permutation invariance verified: {fedma_info.get('permutation_invariance_ok')}")
    print(f"  Classifier: Concatenated per-class in CIFAR-100 space (not aggregated)")

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


def test_global_model(global_model: nn.Module, device: str = 'cpu', num_classes: int = 100):
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
    print(f"  Expected shape: [4, {num_classes}]")

    if output.shape == torch.Size([4, num_classes]):
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
            targets_all_experiences = batch_labels  # already CIFAR-100 labels; classifier is in CIFAR-100 space


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
            #batch_labels = map_original_to_aggregated(batch_labels)  # not needed: CIFAR-100 space
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
        print(f"  - Feature extraction: FedMA-matched + averaged from 4 models")
        print(f"  - Classifier: Concatenated, remapped to CIFAR-100 space (NOT aggregated)")
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
    global_model_reload = torch.load("../final_specialised_models/global_model_with_averaged_features.pt")
    global_model_reload_architecture = ResNet20_DynamicModel(num_classes=100)
    ####global_model_reload_architecture.load_state_dict(global_model_reload)
    ####evaluate_aggregated_model_FedAvg_code_v3(global_model_reload_architecture, test_loader)
    ####evaluate_aggregated_model(global_model_reload_architecture, test_loader)
    #evaluate_aggregated_model(global_model_reload_architecture,outdoors_specialised_models_train_stream_class_il)
    #evaluate_aggregated_model(global_model_reload_architecture, avl_test_loader)
