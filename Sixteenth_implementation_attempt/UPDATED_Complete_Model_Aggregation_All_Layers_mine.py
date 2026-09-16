"""
Updated specialised-model aggregation for the LwF project.

Main changes relative to CORRECTED_Complete_Model_Aggregation_All_Layers_mine.py:

1. The global model predicts ONLY the 43 classes learned by the specialists.
   The 57 untrained CIFAR-100 rows are no longer present as zero-logit competitors.

2. Classifier construction no longer relies on concatenation, dictionary insertion
   order, sparse 100-row tensors, stacking, or summation. Every classifier row is
   placed explicitly using:

       specialist local row -> original CIFAR-100 label -> global 0..42 row

3. CIFAR-100 remapping is retained as a DIAGNOSTIC only. It verifies that a source
   classifier row is copied exactly to the corresponding CIFAR-100 row.

4. BatchNorm is no longer silently excluded from feature aggregation. All feature
   extractor state (everything except linear.*) is included. Floating-point state
   is averaged; integer counters such as num_batches_tracked are handled separately.

5. BatchNorm running statistics can be recalibrated on the union of the specialist
   training data after feature averaging.

6. OPTIONAL but enabled by default: each specialist classifier is transformed into
   the averaged/global feature basis. For specialist i we fit an affine adapter

       z_specialist ~= z_global @ A_i + c_i

   using paired penultimate features from the same calibration images. The original
   specialist head is then transformed as

       W_i_global = W_i @ A_i.T
       b_i_global = b_i + W_i @ c_i

   before its rows are inserted into the 43-class global classifier.

This keeps label-space remapping and feature-space alignment as two explicit,
separately verifiable operations.
"""

from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from Fourteenth_implementation_attempt.ResNet20_DynamicModel_v14 import (
    ResNet20_DynamicModel,
)
from Fourteenth_implementation_attempt.Train_ResNet20_Outdoors_LwF_v14 import (
    outdoors_specialised_models_test_stream_class_il,
    outdoors_specialised_models_train_stream_class_il,
)
from Fourteenth_implementation_attempt.Train_ResNet20_Nature_LwF_v14 import (
    nature_specialised_models_test_stream_class_il,
    nature_specialised_models_train_stream_class_il,
)
from Fourteenth_implementation_attempt.Train_ResNet20_People_LwF_v14 import (
    people_specialised_models_test_stream_class_il,
    people_specialised_models_train_stream_class_il,
)
from Fourteenth_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_v14 import (
    objects_and_vehicles_specialised_models_test_stream_class_il,
    objects_and_vehicles_specialised_models_train_stream_class_il,
)


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


# =============================================================================
# GLOBAL LABEL SPACE
# =============================================================================

# Original CIFAR-100 label -> compact global label (0..42).
# This preserves the global ordering from your original file.
original_to_aggregated_dict = {
    # OUTDOORS
    17: 0,
    12: 1,
    33: 2,
    23: 3,
    49: 4,
    37: 5,

    # NATURE
    51: 6,
    47: 7,
    0: 8,
    54: 9,
    53: 10,
    52: 11,
    59: 12,
    57: 13,
    56: 14,

    # PEOPLE
    2: 15,
    11: 16,
    35: 17,

    # OBJECTS / VEHICLES
    16: 18,
    13: 19,
    10: 20,
    9: 21,
    8: 22,
    5: 23,
    58: 24,
    48: 25,
    41: 26,
    40: 27,
    39: 28,
    28: 29,
    25: 30,
    22: 31,
    20: 32,
    94: 33,
    90: 34,
    89: 35,
    87: 36,
    86: 37,
    85: 38,
    84: 39,
    81: 40,
    69: 41,
    61: 42,
}

NUM_GLOBAL_CLASSES = len(original_to_aggregated_dict)
assert NUM_GLOBAL_CLASSES == 43
assert set(original_to_aggregated_dict.values()) == set(range(NUM_GLOBAL_CLASSES))

aggregated_to_original_dict = {
    global_idx: cifar_idx
    for cifar_idx, global_idx in original_to_aggregated_dict.items()
}


def map_original_to_aggregated(
    targets_tensor: torch.Tensor,
    original_to_aggregated: Dict[int, int] = original_to_aggregated_dict,
) -> torch.Tensor:
    """Map original CIFAR-100 labels to the compact global 0..42 label space."""

    original_labels = [int(x.item()) for x in targets_tensor]
    missing = sorted(set(original_labels) - set(original_to_aggregated.keys()))

    if missing:
        raise KeyError(
            "Evaluation/calibration batch contains CIFAR labels that are not "
            f"part of the 43-class global model: {missing}"
        )

    return torch.tensor(
        [original_to_aggregated[x] for x in original_labels],
        device=targets_tensor.device,
        dtype=torch.long,
    )


# =============================================================================
# DATASETS / LOADERS
# =============================================================================


def concat_first_experiences(stream, n: int = NUM_USED_EXPERIENCES):
    """Concatenate the first n non-empty experience datasets from an Avalanche stream."""

    datasets = []
    for experience in stream:
        if len(experience.dataset) == 0:
            continue
        datasets.append(experience.dataset)
        if len(datasets) == n:
            break

    if not datasets:
        raise RuntimeError("No non-empty datasets found in stream.")

    if len(datasets) < n:
        print(
            f"WARNING: requested {n} experiences but only found "
            f"{len(datasets)} non-empty experiences."
        )

    return ConcatDataset(datasets)


# Test datasets, kept consistent with your existing first-three-experience setup.
combined_dataset_outdoors = concat_first_experiences(
    outdoors_specialised_models_test_stream_class_il
)
combined_dataset_nature = concat_first_experiences(
    nature_specialised_models_test_stream_class_il
)
combined_dataset_people = concat_first_experiences(
    people_specialised_models_test_stream_class_il
)
combined_dataset_objects_and_vehicles = concat_first_experiences(
    objects_and_vehicles_specialised_models_test_stream_class_il
)

combined_dataset = ConcatDataset(
    [
        combined_dataset_outdoors,
        combined_dataset_nature,
        combined_dataset_people,
        combined_dataset_objects_and_vehicles,
    ]
)

test_loader = DataLoader(
    combined_dataset,
    batch_size=32,
    shuffle=False,
)

# Training/calibration datasets used ONLY for BN recalibration and feature alignment.
train_dataset_by_model = {
    "outdoors": concat_first_experiences(
        outdoors_specialised_models_train_stream_class_il
    ),
    "nature": concat_first_experiences(
        nature_specialised_models_train_stream_class_il
    ),
    "people": concat_first_experiences(
        people_specialised_models_train_stream_class_il
    ),
    "objects_and_vehicles": concat_first_experiences(
        objects_and_vehicles_specialised_models_train_stream_class_il
    ),
}

alignment_loaders = {
    model_name: DataLoader(
        dataset,
        batch_size=ALIGNMENT_BATCH_SIZE,
        shuffle=False,
    )
    for model_name, dataset in train_dataset_by_model.items()
}

combined_train_dataset = ConcatDataset(list(train_dataset_by_model.values()))

bn_calibration_loader = DataLoader(
    combined_train_dataset,
    batch_size=BN_RECALIBRATION_BATCH_SIZE,
    shuffle=False,
)


# =============================================================================
# FEATURE EXTRACTOR AGGREGATION
# =============================================================================


def extract_feature_layers(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract every feature-extractor state_dict entry and exclude ONLY linear.*.

    This intentionally includes:
      - convolution weights
      - BatchNorm gamma/beta
      - BatchNorm running_mean/running_var
      - BatchNorm num_batches_tracked
      - residual shortcut state
    """

    feature_layers = OrderedDict(
        (key, value.detach().clone())
        for key, value in model.state_dict().items()
        if not key.startswith("linear.")
    )

    if VERBOSE:
        print("Feature keys:")
        for key in feature_layers:
            print("  ", key)

    return feature_layers


def average_feature_layers(features_list: list) -> Dict[str, torch.Tensor]:
    """
    Average floating-point feature state across specialists.

    Integer buffers such as BatchNorm num_batches_tracked are not meaningful to
    average numerically, so the maximum value is retained. BN running statistics
    are subsequently recalibrated on the union training data when enabled.
    """

    if not features_list:
        raise ValueError("No feature layers supplied for averaging.")

    reference_keys = list(features_list[0].keys())

    for i, features in enumerate(features_list[1:], start=1):
        if list(features.keys()) != reference_keys:
            raise RuntimeError(
                f"Feature-state keys for model {i} differ from model 0. "
                "All specialist architectures must match."
            )

    averaged_features = OrderedDict()

    for key in reference_keys:
        tensors = [features[key] for features in features_list]
        reference = tensors[0]

        if torch.is_floating_point(reference):
            stacked = torch.stack([t.float() for t in tensors], dim=0)
            averaged = stacked.mean(dim=0).to(reference.dtype)
        else:
            # Typically BN num_batches_tracked (scalar int64).
            stacked = torch.stack([t.to(torch.long) for t in tensors], dim=0)
            averaged = stacked.max(dim=0).values.to(reference.dtype)

        averaged_features[key] = averaged

    return averaged_features


def load_averaged_feature_layers(
    global_model: nn.Module,
    averaged_features: Dict[str, torch.Tensor],
) -> nn.Module:
    """Load merged feature state while leaving the global classifier untouched."""

    global_state = global_model.state_dict()

    missing = []
    for key, value in averaged_features.items():
        if key not in global_state:
            missing.append(key)
            continue
        if global_state[key].shape != value.shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: global={global_state[key].shape}, "
                f"merged={value.shape}"
            )
        global_state[key] = value.to(global_state[key].device)

    if missing:
        raise RuntimeError(f"Merged feature keys missing from global model: {missing}")

    global_model.load_state_dict(global_state)
    return global_model


@torch.no_grad()
def recalibrate_batchnorm(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> None:
    """
    Recompute BatchNorm running statistics on the union calibration dataset.

    Learned BN affine parameters (weight/bias) remain untouched. Only running
    statistics/counters are reset and rebuilt by forward passes.
    """

    bn_modules = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]

    if not bn_modules:
        print("No BatchNorm modules found; skipping BN recalibration.")
        return

    for bn in bn_modules:
        if bn.running_mean is not None:
            bn.running_mean.zero_()
        if bn.running_var is not None:
            bn.running_var.fill_(1)
        if bn.num_batches_tracked is not None:
            bn.num_batches_tracked.zero_()

    # ResNet20 normally has no dropout. train() is required for BN updates.
    model.train()

    sample_count = 0
    for batch in loader:
        x = batch[0].to(device)
        _ = model(x)
        sample_count += x.size(0)

    model.eval()
    print(
        f"  ✓ Recalibrated {len(bn_modules)} BatchNorm modules "
        f"using {sample_count} training samples."
    )


# =============================================================================
# CLASSIFIER LABEL-SPACE REMAPPING / VALIDATION
# =============================================================================


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
    model_name: str,
    model: nn.Module,
    new_to_original: Dict[int, int],
    num_subset_classes: int,
    num_classes: int = 100,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DIAGNOSTIC: copy specialist classifier rows into original CIFAR-100 positions.

    This function no longer depends on dictionary insertion order. It explicitly
    iterates local classifier rows 0..num_subset_classes-1.
    """

    mapping = validate_specialist_mapping(
        model_name,
        model,
        new_to_original,
        num_subset_classes,
    )

    classifier = model.linear
    feature_dim = classifier.weight.shape[1]

    remapped_weights = torch.zeros(
        num_classes,
        feature_dim,
        device=device,
        dtype=classifier.weight.dtype,
    )
    remapped_bias = torch.zeros(
        num_classes,
        device=device,
        dtype=classifier.bias.dtype,
    )

    print(f"\n===== CIFAR-100 REMAP DIAGNOSTIC: {model_name.upper()} =====")

    for local_idx in range(num_subset_classes):
        cifar_idx = mapping[local_idx]

        source_weight = classifier.weight[local_idx].detach().to(device)
        source_bias = classifier.bias[local_idx].detach().to(device)

        remapped_weights[cifar_idx].copy_(source_weight)
        remapped_bias[cifar_idx].copy_(source_bias)

        if not torch.allclose(remapped_weights[cifar_idx], source_weight):
            raise RuntimeError(
                f"{model_name}: weight remapping failed for local row {local_idx} "
                f"-> CIFAR row {cifar_idx}."
            )
        if not torch.allclose(remapped_bias[cifar_idx], source_bias):
            raise RuntimeError(
                f"{model_name}: bias remapping failed for local row {local_idx} "
                f"-> CIFAR row {cifar_idx}."
            )

        print(
            f"local {local_idx:2d} -> CIFAR {cifar_idx:2d} | "
            f"source norm={source_weight.norm().item():.8f} | "
            f"destination norm={remapped_weights[cifar_idx].norm().item():.8f}"
        )

    expected_rows = [mapping[i] for i in range(num_subset_classes)]
    print("Expected populated CIFAR rows:", sorted(expected_rows))

    # Important: do not infer population only from nonzero bias; a trained bias may
    # legitimately be zero. Weight integrity is verified above row-by-row.
    return remapped_weights, remapped_bias


def run_cifar100_remap_diagnostics(
    models: Dict[str, nn.Module],
    mappings: Dict[str, Dict[int, int]],
    num_subset_classes: Dict[str, int],
    device: str,
) -> None:
    """Run local -> CIFAR-100 integrity checks for every specialist."""

    for model_name, model in models.items():
        w, _ = remap_classifier_to_cifar100(
            model_name=model_name,
            model=model,
            new_to_original=mappings[model_name],
            num_subset_classes=num_subset_classes[model_name],
            num_classes=100,
            device=device,
        )

        # Display actual non-zero rows without PyTorch tensor abbreviation.
        row_norms = w.norm(p=2, dim=1)
        nonzero_rows = torch.where(row_norms > 1e-12)[0].tolist()
        print(f"{model_name} non-zero CIFAR-100 weight rows: {nonzero_rows}")


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


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_specialist_models(
    model_names: list,
    model_dir: Path,
    device: str,
):
    models = {}
    mappings = {}
    num_subset_classes = {}

    print("Step 1: Loading specialised models...")

    for model_name in model_names:
        state_path = model_dir / f"final_{model_name}_model.pt"
        metadata_path = model_dir / f"{model_name}_metadata.pkl"

        state_dict = torch.load(state_path, map_location=device)
        classifier_size = state_dict["linear.weight"].shape[0]

        model = ResNet20_DynamicModel(num_classes=classifier_size).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        mapping = normalize_mapping(metadata["new_to_original"])
        n_classes = int(metadata["num_subset_classes"])

        validate_specialist_mapping(
            model_name,
            model,
            mapping,
            n_classes,
        )

        models[model_name] = model
        mappings[model_name] = mapping
        num_subset_classes[model_name] = n_classes

        trained_labels = [mapping[i] for i in range(n_classes)]

        print(f"  ✓ Loaded {model_name}")
        print(f"    classifier size: {classifier_size}")
        print(f"    trained local rows: 0..{n_classes - 1}")
        print(f"    trained CIFAR labels: {trained_labels}")

    return models, mappings, num_subset_classes


# =============================================================================
# MAIN AGGREGATION PIPELINE
# =============================================================================


def aggregate_models_all_layers_with_feature_averaging(
    model_names: list,
    model_dir: Path,
    device: str = "cpu",
    align_classifiers: bool = ALIGN_CLASSIFIERS_TO_AVERAGED_FEATURES,
) -> Tuple[nn.Module, Dict]:
    """Build a 43-class global model with averaged features and optional head alignment."""

    print("\n" + "=" * 88)
    print("GLOBAL MODEL AGGREGATION: 43-CLASS HEAD + FEATURE-SPACE ALIGNMENT")
    print("=" * 88 + "\n")

    models, mappings, num_subset_classes = load_specialist_models(
        model_names=model_names,
        model_dir=model_dir,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Diagnostic only: prove local -> CIFAR-100 remapping preserves every row.
    # -------------------------------------------------------------------------
    if RUN_CIFAR100_REMAP_DIAGNOSTIC:
        print("\nStep 2: Verifying local -> CIFAR-100 classifier remapping...")
        run_cifar100_remap_diagnostics(
            models=models,
            mappings=mappings,
            num_subset_classes=num_subset_classes,
            device=device,
        )

    # -------------------------------------------------------------------------
    # Merge feature extractor state.
    # -------------------------------------------------------------------------
    print("\nStep 3: Extracting and averaging feature extractor state...")
    all_feature_layers = [extract_feature_layers(models[name]) for name in model_names]
    averaged_features = average_feature_layers(all_feature_layers)

    print(f"  ✓ Averaged {len(averaged_features)} non-classifier state entries.")

    # -------------------------------------------------------------------------
    # Create global 43-class architecture.
    # -------------------------------------------------------------------------
    print("\nStep 4: Creating 43-class global model...")
    global_model = ResNet20_DynamicModel(num_classes=NUM_GLOBAL_CLASSES).to(device)
    global_model = load_averaged_feature_layers(global_model, averaged_features)

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
        recalibrate_batchnorm(
            model=global_model,
            loader=bn_calibration_loader,
            device=device,
        )
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


# =============================================================================
# SAVE / RELOAD / TEST
# =============================================================================


def save_global_model(
    global_model: nn.Module,
    output_path: Path,
    report: Dict,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / GLOBAL_MODEL_FILENAME
    torch.save(global_model.state_dict(), model_path)
    print(f"✓ Saved global model to {model_path}")

    report_path = output_path / REPORT_FILENAME
    with open(report_path, "wb") as f:
        pickle.dump(report, f)
    print(f"✓ Saved aggregation report to {report_path}")


@torch.no_grad()
def test_global_model(global_model: nn.Module, device: str = "cpu"):
    print("\nTesting global model output shape...")
    global_model.eval()

    dummy_input = torch.randn(4, 3, 32, 32, device=device)
    output = global_model(dummy_input)

    expected_shape = torch.Size([4, NUM_GLOBAL_CLASSES])
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: {expected_shape}")

    if output.shape != expected_shape:
        raise RuntimeError(
            f"Global output shape mismatch: got {output.shape}, expected {expected_shape}."
        )

    print("  ✓ Output shape is correct.")
    return output


# =============================================================================
# EVALUATION
# =============================================================================


@torch.no_grad()
def evaluate_aggregated_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    label: str = "Global model",
):
    """Evaluate a 43-output global model using compact 0..42 targets."""

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    model.eval()

    for batch_data, batch_labels, batch_task_ids in loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        compact_targets = map_original_to_aggregated(batch_labels)
        logits = model(batch_data)

        if logits.shape[1] != NUM_GLOBAL_CLASSES:
            raise RuntimeError(
                f"Evaluator expected {NUM_GLOBAL_CLASSES} logits but received "
                f"shape {tuple(logits.shape)}."
            )

        loss = F.cross_entropy(logits, compact_targets, reduction="mean")

        total_loss += loss.item() * compact_targets.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == compact_targets).sum().item()
        total_samples += compact_targets.size(0)

    avg_loss = total_loss / total_samples if total_samples else 0.0
    accuracy = total_correct / total_samples if total_samples else 0.0

    print(
        f"{label}: Loss={avg_loss:.6f} | "
        f"Accuracy={accuracy * 100:.2f}% ({total_correct}/{total_samples})"
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "correct": total_correct,
        "total": total_samples,
    }


def evaluate_aggregated_model_FedAvg_code_v3(
    target_model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
):
    """Compatibility wrapper around the corrected 43-class evaluator."""

    result = evaluate_aggregated_model(
        model=target_model,
        loader=loader,
        device=device,
        label="Global model (FedAvg-code compatibility evaluator)",
    )
    print(f"Fine-tuned model accuracy: {result['accuracy']:.2%}")
    return result


# =============================================================================
# MAIN
# =============================================================================


def main():
    # Keep the same project-relative location used by your original aggregation
    # function, but now honour it through a Path object instead of overwriting the
    # function argument with a string inside the aggregation function.
    MODEL_DIR = Path("../Fourteenth_implementation_attempt/final_specialised_models")
    OUTPUT_DIR = Path("./final_specialised_models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserves the CPU default from your original aggregation script.
    # Change to "cuda" manually after the logic is validated if desired.
    DEVICE = "cpu"

    model_names = [
        "outdoors",
        "nature",
        "people",
        "objects_and_vehicles",
    ]

    print(f"Using device: {DEVICE}")

    global_model, report = aggregate_models_all_layers_with_feature_averaging(
        model_names=model_names,
        model_dir=MODEL_DIR,
        device=DEVICE,
        align_classifiers=ALIGN_CLASSIFIERS_TO_AVERAGED_FEATURES,
    )

    save_global_model(
        global_model=global_model,
        output_path=OUTPUT_DIR,
        report=report,
    )

    test_global_model(global_model, device=DEVICE)

    print("\n===== EVALUATION: IN-MEMORY GLOBAL MODEL =====")
    evaluate_aggregated_model_FedAvg_code_v3(
        global_model,
        test_loader,
        device=DEVICE,
    )
    evaluate_aggregated_model(
        global_model,
        test_loader,
        device=DEVICE,
        label="Global model",
    )

    # Reload using the CORRECT 43-output architecture.
    print("\n===== EVALUATION: RELOADED GLOBAL MODEL =====")
    model_path = OUTPUT_DIR / GLOBAL_MODEL_FILENAME
    global_state = torch.load(model_path, map_location=DEVICE)

    global_model_reload = ResNet20_DynamicModel(
        num_classes=NUM_GLOBAL_CLASSES
    ).to(DEVICE)
    global_model_reload.load_state_dict(global_state)
    global_model_reload.eval()

    evaluate_aggregated_model_FedAvg_code_v3(
        global_model_reload,
        test_loader,
        device=DEVICE,
    )
    evaluate_aggregated_model(
        global_model_reload,
        test_loader,
        device=DEVICE,
        label="Reloaded global model",
    )

    return global_model


if __name__ == "__main__":
    global_model = main()
