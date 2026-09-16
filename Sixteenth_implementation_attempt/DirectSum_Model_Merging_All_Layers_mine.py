"""
DIRECT-SUM / CONCATENATED-FEATURE MODEL MERGING
(replacement for CORRECTED_Complete_Model_Aggregation_All_Layers_mine.py)

WHAT CHANGED vs. the previous file
----------------------------------
OLD (FedAvg-style):
    theta_G = mean(theta_O, theta_N, theta_P, theta_V)     <-- feature weights averaged
    W_G rows copied from specialist heads                  <-- head attached to a DIFFERENT
                                                               feature space than it was
                                                               trained on  ==> needed adapters

NEW (direct sum / structural merge):
    z_G(x) = [ f_O(x) || f_N(x) || f_P(x) || f_V(x) ]      <-- every specialist extractor kept
                                                               EXACTLY as trained, all four are
                                                               evaluated for every input
    W_G    = block-sparse 43 x 256 head, row of class c of
             specialist i = [0 ... W_i[c_local] ... 0]

    ==> W_G z_G = W_i z_i  exactly, for every class.
        No convolution averaging. No BN averaging. No feature adapters.
        No task-id is used anywhere in forward().

The result is ONE nn.Module, ONE state_dict, ONE forward(x) -> [B, C_global], ONE argmax.

CLASS-INCREMENTAL COMPLIANCE
----------------------------
forward(x) takes only x. No task label is consumed, no expert is selected, no branch is
skipped. All 43 classes compete in a single output space. This is a class-IL model.

THE ONE REMAINING PROBLEM (it is now the ONLY problem)
------------------------------------------------------
Cross-specialist logit comparability. An Outdoors logit was never trained to compete against
a Nature logit. Kim et al., "A Theoretical Study on Solving Continual Learning" (NeurIPS 2022)
shows class-IL accuracy is bounded by within-task prediction (WP) x task-id prediction (TP).
This merge makes WP *exact* (each head sees its native features) and leaves TP entirely to
whatever OOD/confidence signal the raw logit magnitudes happen to carry.

This file therefore also provides:
  * exact functional-equivalence verification (catches label-mapping bugs immediately)
  * a WP/TP diagnostic split, so you can see WHICH of the two is failing
  * several LABEL-FREE calibration modes (no training, satisfies "no further training")
  * an OPT-IN labelled calibration (8 params) and an OPT-IN head fine-tune (~11k params),
    both OFF by default

Setup (unchanged):
  Model 1 (Outdoors):          6 classes
  Model 2 (Nature):            9 classes
  Model 3 (People):            3 classes
  Model 4 (Objects/Vehicles): 25 classes
  -> 43 trained CIFAR-100 classes total, disjoint.
"""

import copy
import pickle
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Fourteenth_implementation_attempt.ResNet20_DynamicModel_v14 import ResNet20_DynamicModel

############ Imports for Evaluating: ###############################################################
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.data import make_avalanche_dataset
from avalanche.benchmarks.utils.data_attribute import DataAttribute
from avalanche.benchmarks.utils.classification_dataset import ClassificationDataset
from torch.utils.data import DataLoader, ConcatDataset
from avalanche.benchmarks.utils import AvalancheDataset

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
from Fourteenth_implementation_attempt.Load_cifar100_v14 import avl_test
####################################################################################################

VERBOSE = False

# ==================================================================================================
# CONFIGURATION
# ==================================================================================================

# "cifar100"  -> global model outputs 100 logits, row index == CIFAR-100 class id.
#                Untrained classes are masked out, so argmax is identical to a 43-way argmax.
#                Your existing evaluation code (which compares against raw CIFAR labels) works
#                UNCHANGED with this setting. <-- default
# "compact43" -> global model outputs 43 logits in the order of original_to_aggregated_dict.
#                Requires targets to be remapped with map_original_to_aggregated().
GLOBAL_LABEL_SPACE = "cifar100"

# Mask the 57 CIFAR classes no specialist ever learned. Leave True for "cifar100".
# (A zero weight row produces a logit of exactly 0.0, which will beat any genuinely negative
#  specialist logit. That is a silent accuracy killer in the previous file too.)
MASK_UNTRAINED_CLASSES = True
MASKED_LOGIT_VALUE = -1.0e4

# Calibration applied to the specialist blocks. See fit_specialist_calibration().
#   "none"        -> raw logits concatenated. Fully training-free. Start here.
#   "standardise" -> per-specialist z-scoring of logits using UNLABELLED data statistics.
#                    No labels, no gradient steps.
#   "max_logit"   -> match the mean max-logit of every specialist. No labels.
#   "temperature" -> fit 4 scales + 4 shifts by gradient descent on LABELLED data.
#                    This is 8 trainable parameters; it violates a strict "no further
#                    training" reading, so it is opt-in.
CALIBRATION_MODE = "none"

# Optional Stage-2 (OFF by default): unfreeze ONLY the 43x256 head and train it so that classes
# can use *other* specialists' features. ~11k params. Violates "no further training".
TRAIN_GLOBAL_HEAD = False
HEAD_TRAINING_EPOCHS = 5
HEAD_TRAINING_LR = 1e-2

MODEL_NAMES = ["outdoors", "nature", "people", "objects_and_vehicles"]


# ==================================================================================================
# DATA (carried over unchanged from the aggregation script)
# ==================================================================================================

test_stream_dict = {
    "outdoors_specialised_models_test_stream_class_il_variable": outdoors_specialised_models_test_stream_class_il,
    "nature_specialised_models_test_stream_class_il_variable": nature_specialised_models_test_stream_class_il,
    "people_specialised_models_test_stream_class_il_variable": people_specialised_models_test_stream_class_il,
    "objects_and_vehicles_specialised_models_test_stream_class_il_variable": objects_and_vehicles_specialised_models_test_stream_class_il,
}

for test_stream_name, test_stream in test_stream_dict.items():
    for test_exp in test_stream:
        if len(test_exp.dataset) != 0:
            print(
                f"{test_stream_name}.Exp{test_exp.current_experience}.dataset.targets.uniques",
                test_exp.dataset.targets.uniques,
            )

combined_dataset_outdoors = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[0].dataset,
    outdoors_specialised_models_test_stream_class_il[1].dataset,
    outdoors_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_nature = ConcatDataset([
    nature_specialised_models_test_stream_class_il[0].dataset,
    nature_specialised_models_test_stream_class_il[1].dataset,
    nature_specialised_models_test_stream_class_il[2].dataset,
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
    combined_dataset_objects_and_vehicles,
])

test_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)
outdoors_loader = DataLoader(combined_dataset_outdoors, batch_size=32, shuffle=False)
nature_loader = DataLoader(combined_dataset_nature, batch_size=32, shuffle=False)
people_loader = DataLoader(combined_dataset_people, batch_size=32, shuffle=False)
objects_and_vehicles_loader = DataLoader(combined_dataset_objects_and_vehicles, batch_size=32, shuffle=False)

# Train-side loaders, used only if you opt in to calibration / head training.
combined_dataset_TRAIN = ConcatDataset([
    outdoors_specialised_models_train_stream_class_il[0].dataset,
    outdoors_specialised_models_train_stream_class_il[1].dataset,
    outdoors_specialised_models_train_stream_class_il[2].dataset,
    nature_specialised_models_train_stream_class_il[0].dataset,
    nature_specialised_models_train_stream_class_il[1].dataset,
    nature_specialised_models_train_stream_class_il[2].dataset,
    people_specialised_models_train_stream_class_il[0].dataset,
    people_specialised_models_train_stream_class_il[1].dataset,
    people_specialised_models_train_stream_class_il[2].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[0].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[1].dataset,
    objects_and_vehicles_specialised_models_train_stream_class_il[2].dataset,
])
calibration_loader = DataLoader(combined_dataset_TRAIN, batch_size=64, shuffle=True)

avl_test_loader = DataLoader(avl_test, batch_size=128, shuffle=False)

all_labels = [int(combined_dataset[i][1]) for i in range(len(combined_dataset))]
print("Label histogram:", Counter(all_labels))
print("Unique labels:", sorted(set(all_labels)))
print("Number of unique labels:", len(set(all_labels)))


# ==================================================================================================
# LABEL SPACE
# ==================================================================================================

original_to_aggregated_dict = {
    17: 0, 12: 1, 33: 2, 23: 3, 49: 4, 37: 5,                                    # OUTDOORS
    51: 6, 47: 7, 0: 8, 54: 9, 53: 10, 52: 11, 59: 12, 57: 13, 56: 14,           # NATURE
    2: 15, 11: 16, 35: 17,                                                       # PEOPLE
    16: 18, 13: 19, 10: 20, 9: 21, 8: 22, 5: 23, 58: 24, 48: 25, 41: 26, 40: 27,
    39: 28, 28: 29, 25: 30, 22: 31, 20: 32, 94: 33, 90: 34, 89: 35, 87: 36,
    86: 37, 85: 38, 84: 39, 81: 40, 69: 41, 61: 42,                              # OBJECTS_AND_VEHICLES
}


def map_original_to_aggregated(targets_tensor, original_to_aggregated=original_to_aggregated_dict):
    """CIFAR-100 label tensor -> compact 0..42 label tensor."""
    return torch.tensor(
        [original_to_aggregated[int(e.item())] for e in targets_tensor],
        device=targets_tensor.device,
        dtype=torch.long,
    )


def num_global_classes() -> int:
    return 100 if GLOBAL_LABEL_SPACE == "cifar100" else 43


def cifar_class_to_global_row(cifar_class: int) -> int:
    """Where does a CIFAR-100 class live in the global output vector?"""
    if GLOBAL_LABEL_SPACE == "cifar100":
        return int(cifar_class)
    return original_to_aggregated_dict[int(cifar_class)]


def to_global_targets(batch_labels: torch.Tensor) -> torch.Tensor:
    """Dataset labels (CIFAR-100 ids) -> targets in the global output space."""
    if GLOBAL_LABEL_SPACE == "cifar100":
        return batch_labels.long()
    return map_original_to_aggregated(batch_labels)


# ==================================================================================================
# STEP 1: FEATURE TAP  (replaces extract_feature_layers / average_feature_layers)
# ==================================================================================================

class _FeatureTap:
    """
    Captures the penultimate representation z = f_i(x) of a specialist.

    Implemented as a forward-PRE-hook on the specialist's final `linear` layer, so it works for
    ANY ResNet20_DynamicModel forward() implementation without needing to know its internals.
    Whatever tensor is fed into `linear` is by definition the feature vector its classifier was
    trained to consume, which is exactly the object we want to preserve.
    """

    def __init__(self, linear_module: nn.Module):
        self.features: Optional[torch.Tensor] = None
        self._handle = linear_module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.features = inputs[0]
        return None

    def close(self):
        self._handle.remove()


# ==================================================================================================
# STEP 2: THE GLOBAL MODEL
# ==================================================================================================

class GlobalDirectSumModel(nn.Module):
    """
    One model. Four preserved specialist feature extractors. One global block-sparse head.

        z_G(x) = [ f_O(x) || f_N(x) || f_P(x) || f_V(x) ]        (concatenated, no task-id)
        logits = W_G z_G + b_G                                    (43 or 100 rows)
        logits = logits * scale[specialist(c)] + shift[specialist(c)]   (calibration, identity by default)
        logits = mask(logits)                                     (untrained CIFAR classes -> -1e4)

    At initialisation W_G is block-sparse, so W_G z_G reproduces each specialist's own logits
    bit-for-bit. Verify this with verify_functional_equivalence().
    """

    def __init__(
        self,
        specialists: "OrderedDict[str, nn.Module]",
        feature_dims: Dict[str, int],
        class_to_specialist: torch.Tensor,
        n_global_classes: int,
        freeze_branches: bool = True,
        mask_untrained: bool = True,
        masked_logit_value: float = MASKED_LOGIT_VALUE,
    ):
        super().__init__()

        self.specialist_names: List[str] = list(specialists.keys())
        self.branches = nn.ModuleDict(specialists)
        self.freeze_branches = freeze_branches
        self.mask_untrained = mask_untrained
        self.masked_logit_value = masked_logit_value

        # Feature layout: contiguous block per specialist.
        self.feature_dims = [int(feature_dims[n]) for n in self.specialist_names]
        self.total_feature_dim = int(sum(self.feature_dims))
        offsets, running = [], 0
        for d in self.feature_dims:
            offsets.append((running, running + d))
            running += d
        self.feature_slices: Dict[str, Tuple[int, int]] = {
            n: offsets[i] for i, n in enumerate(self.specialist_names)
        }

        # The single global classifier.
        self.linear = nn.Linear(self.total_feature_dim, n_global_classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Per-specialist logit calibration. Identity by default -> exact function preservation.
        self.logit_scale = nn.Parameter(torch.ones(len(self.specialist_names)))
        self.logit_shift = nn.Parameter(torch.zeros(len(self.specialist_names)))
        self.logit_scale.requires_grad_(False)
        self.logit_shift.requires_grad_(False)

        # class_to_specialist[c] = index of the owning specialist, or -1 if untrained.
        self.register_buffer("class_to_specialist", class_to_specialist.long())
        self.register_buffer("valid_class_mask", (class_to_specialist >= 0))

        # Feature taps (not parameters, rebuilt on construction).
        self._taps = {n: _FeatureTap(self.branches[n].linear) for n in self.specialist_names}

        if freeze_branches:
            for p in self.branches.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ mode handling

    def train(self, mode: bool = True):
        """
        Branches are ALWAYS kept in eval mode.

        This matters: ResNet20 uses BatchNorm. If a branch were ever put in train mode, it would
        switch to batch statistics and update its running statistics, silently destroying the
        exact function-preservation property that the whole design rests on.
        """
        super().train(mode)
        for b in self.branches.values():
            b.eval()
        return self

    # ------------------------------------------------------------------ forward

    def branch_outputs(self, x: torch.Tensor):
        """Run all four specialists once. Returns (features dict, native logits dict)."""
        feats, logits = OrderedDict(), OrderedDict()
        for name in self.specialist_names:
            out = self.branches[name](x)
            logits[name] = out
            feats[name] = self._taps[name].features
        return feats, logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats, _ = self.branch_outputs(x)
        return torch.cat([feats[n] for n in self.specialist_names], dim=1)

    def raw_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Global logits BEFORE calibration and BEFORE masking."""
        return self.linear(self.extract_features(x))

    def apply_calibration(self, logits: torch.Tensor) -> torch.Tensor:
        idx = self.class_to_specialist.clamp(min=0)
        return logits * self.logit_scale[idx] + self.logit_shift[idx]

    def apply_mask(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.mask_untrained:
            return logits
        return logits.masked_fill(~self.valid_class_mask, self.masked_logit_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: signature is forward(self, x). No task id. This is what makes it class-IL.
        return self.apply_mask(self.apply_calibration(self.raw_logits(x)))

    # ------------------------------------------------------------------ housekeeping

    def head_parameters(self):
        return list(self.linear.parameters())

    def calibration_parameters(self):
        return [self.logit_scale, self.logit_shift]

    def config(self) -> Dict:
        return {
            "specialist_names": self.specialist_names,
            "specialist_classifier_sizes": {
                n: int(self.branches[n].linear.weight.shape[0]) for n in self.specialist_names
            },
            "feature_dims": {n: d for n, d in zip(self.specialist_names, self.feature_dims)},
            "n_global_classes": int(self.linear.weight.shape[0]),
            "global_label_space": GLOBAL_LABEL_SPACE,
            "mask_untrained": self.mask_untrained,
            "masked_logit_value": self.masked_logit_value,
        }


# ==================================================================================================
# STEP 3: BUILD  (replaces aggregate_models_all_layers_with_feature_averaging)
# ==================================================================================================

def _selected_local_to_cifar(mapping: Dict[int, int], num_subset_classes: int) -> List[Tuple[int, int]]:
    """
    Take the first `num_subset_classes` entries of new_to_original, in insertion order.

    This reproduces exactly the selection rule used by remap_classifier_to_cifar100() in the
    previous file (count-and-break over dict items), so the label mapping is unchanged from the
    runs you have already done.
    """
    pairs = []
    for count, (local_idx, cifar_class) in enumerate(mapping.items()):
        if count >= num_subset_classes:
            break
        pairs.append((int(local_idx), int(cifar_class)))
    return pairs


def load_specialists(model_names: List[str], model_dir: str, device: str = "cpu"):
    """Load the four trained specialists plus their label mappings. Unchanged logic."""
    models, mappings, num_subset_classes = OrderedDict(), {}, {}

    print("Step 1: Loading specialised models...")
    for model_name in model_names:
        state_dict = torch.load(f"{model_dir}final_{model_name}_model.pt", map_location=device)

        classifier_size = state_dict["linear.weight"].shape[0]
        model = ResNet20_DynamicModel(num_classes=classifier_size)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models[model_name] = model

        with open(f"{model_dir}{model_name}_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        mappings[model_name] = metadata["new_to_original"]
        num_subset_classes[model_name] = metadata["num_subset_classes"]

        print(f"  [ok] {model_name}: classifier size {classifier_size}, "
              f"trained classes {num_subset_classes[model_name]}, "
              f"feature dim {model.linear.weight.shape[1]}")
        print(f"       CIFAR classes: {sorted(mappings[model_name].values())}")

    return models, mappings, num_subset_classes


def build_direct_sum_global_model(
    model_names: List[str],
    model_dir: str,
    device: str = "cpu",
) -> Tuple[GlobalDirectSumModel, Dict]:
    """
    Build the merged global model.

      1. load specialists
      2. concatenate their feature spaces (NO averaging of any conv or BN tensor)
      3. build the block-sparse global head from the specialist heads
      4. return one nn.Module
    """
    print("\n" + "=" * 90)
    print("DIRECT-SUM MODEL MERGING  (feature concatenation + block-sparse global head)")
    print("=" * 90 + "\n")

    models, mappings, num_subset_classes = load_specialists(model_names, model_dir, device)

    # --------------------------------------------------------------- feature layout
    print("\nStep 2: Building concatenated feature space (no weight averaging)...")
    feature_dims = {n: int(models[n].linear.weight.shape[1]) for n in model_names}
    total_dim = sum(feature_dims.values())
    for n in model_names:
        print(f"  {n:22s} z_{n[0].upper()} in R^{feature_dims[n]}")
    print(f"  -> z_G in R^{total_dim}")

    # --------------------------------------------------------------- class ownership
    print("\nStep 3: Resolving class ownership (local row -> CIFAR class -> global row)...")
    C = num_global_classes()
    class_to_specialist = torch.full((C,), -1, dtype=torch.long)
    global_row_sources: Dict[int, Tuple[str, int, int]] = {}   # global_row -> (name, local_idx, cifar)
    class_coverage: Dict[int, List[str]] = {}

    for spec_idx, name in enumerate(model_names):
        pairs = _selected_local_to_cifar(mappings[name], num_subset_classes[name])
        for local_idx, cifar_class in pairs:
            g = cifar_class_to_global_row(cifar_class)

            if class_to_specialist[g].item() != -1:
                prev = global_row_sources[g]
                raise ValueError(
                    f"Class overlap: CIFAR class {cifar_class} (global row {g}) is claimed by "
                    f"both '{prev[0]}' and '{name}'. The block-sparse construction assumes "
                    f"disjoint specialist class sets."
                )

            class_to_specialist[g] = spec_idx
            global_row_sources[g] = (name, local_idx, cifar_class)
            class_coverage.setdefault(cifar_class, []).append(name)

        print(f"  {name:22s} {len(pairs):2d} classes -> global rows "
              f"{sorted(cifar_class_to_global_row(c) for _, c in pairs)}")

    n_covered = int((class_to_specialist >= 0).sum().item())
    print(f"  -> {n_covered} of {C} global rows populated")

    # --------------------------------------------------------------- assemble
    print("\nStep 4: Assembling the global model...")
    global_model = GlobalDirectSumModel(
        specialists=OrderedDict((n, models[n]) for n in model_names),
        feature_dims=feature_dims,
        class_to_specialist=class_to_specialist,
        n_global_classes=C,
        freeze_branches=True,
        mask_untrained=MASK_UNTRAINED_CLASSES and GLOBAL_LABEL_SPACE == "cifar100",
    ).to(device)

    print("\nStep 5: Block-sparse initialisation of the global head...")
    with torch.no_grad():
        W = torch.zeros_like(global_model.linear.weight)
        b = torch.zeros_like(global_model.linear.bias)

        for g, (name, local_idx, cifar_class) in global_row_sources.items():
            start, end = global_model.feature_slices[name]
            src = models[name].linear
            W[g, start:end] = src.weight[local_idx].detach().to(W.dtype)
            b[g] = src.bias[local_idx].detach().to(b.dtype)

        global_model.linear.weight.copy_(W)
        global_model.linear.bias.copy_(b)

    nnz = int((global_model.linear.weight != 0).sum().item())
    print(f"  head shape      : {tuple(global_model.linear.weight.shape)}")
    print(f"  non-zero entries: {nnz} / {global_model.linear.weight.numel()} "
          f"({100.0 * nnz / global_model.linear.weight.numel():.1f}% dense, block-sparse as designed)")

    total_params = sum(p.numel() for p in global_model.parameters())
    head_params = sum(p.numel() for p in global_model.linear.parameters())
    print(f"  total params    : {total_params:,}  (head: {head_params:,}, branches frozen)")

    report = {
        "method": "direct_sum_feature_concatenation",
        "feature_averaging": False,
        "feature_adapters": False,
        "models_used": model_names,
        "feature_dims": feature_dims,
        "total_feature_dim": total_dim,
        "global_label_space": GLOBAL_LABEL_SPACE,
        "n_global_classes": C,
        "trained_classes": len(class_coverage),
        "untrained_classes": C - n_covered,
        "class_coverage": class_coverage,
        "global_row_sources": global_row_sources,
        "total_params": total_params,
        "device": device,
    }

    print("\n" + "=" * 90 + "\n")
    return global_model, report


# ==================================================================================================
# STEP 4: VERIFICATION  (the most important function in this file)
# ==================================================================================================

def verify_functional_equivalence(
    global_model: GlobalDirectSumModel,
    report: Dict,
    device: str = "cpu",
    n_samples: int = 16,
    tol: float = 1e-4,
    loader: Optional[DataLoader] = None,
) -> bool:
    """
    Assert that, for every class c owned by specialist i,

        global_logits[:, global_row(c)]  ==  specialist_i_logits[:, local_row(c)]

    If this passes, the merge is exactly function-preserving and any accuracy you then measure is
    purely a cross-specialist calibration story, never a feature-mismatch story.

    Run it immediately after construction. It is EXPECTED to fail after train_global_head(),
    because that deliberately fills in the off-block weights.
    """
    print("Verification: exact function preservation")
    print("-" * 90)

    global_model.eval()

    if loader is not None:
        batch = next(iter(loader))
        x = batch[0][:n_samples].to(device)
    else:
        x = torch.randn(n_samples, 3, 32, 32, device=device)

    with torch.no_grad():
        # Pre-calibration, pre-mask: this is the quantity that must match exactly.
        g_logits = global_model.raw_logits(x)
        _, native = global_model.branch_outputs(x)

    max_err = 0.0
    n_checked = 0
    for g, (name, local_idx, cifar_class) in report["global_row_sources"].items():
        diff = (g_logits[:, g] - native[name][:, local_idx]).abs().max().item()
        max_err = max(max_err, diff)
        n_checked += 1
        if diff > tol:
            print(f"  [FAIL] global row {g:3d} (CIFAR {cifar_class:3d}) <- {name}[{local_idx}]  "
                  f"max abs error {diff:.3e}")

    ok = max_err <= tol
    print(f"  checked {n_checked} class rows across {len(global_model.specialist_names)} specialists")
    print(f"  max abs deviation: {max_err:.3e}  (tolerance {tol:.1e})")
    if ok:
        print("  result: PASS - specialists are preserved exactly")
    else:
        print("  result: FAIL - either the label mapping (new_to_original / num_subset_classes)")
        print("          is wrong, or the head is no longer block-sparse (expected after")
        print("          train_global_head).")
    print("-" * 90 + "\n")
    return ok


# ==================================================================================================
# STEP 5: CALIBRATION  (the only genuinely new technical problem)
# ==================================================================================================

@torch.no_grad()
def collect_block_statistics(
    global_model: GlobalDirectSumModel,
    loader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Per-specialist logit statistics over a loader. Uses NO labels, so this is transductive
    calibration rather than supervised training.
    """
    global_model.eval()
    n_spec = len(global_model.specialist_names)

    total = torch.zeros(n_spec, dtype=torch.float64)
    total_sq = torch.zeros(n_spec, dtype=torch.float64)
    count = torch.zeros(n_spec, dtype=torch.float64)
    max_sum = torch.zeros(n_spec, dtype=torch.float64)
    max_count = torch.zeros(n_spec, dtype=torch.float64)

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = batch[0].to(device)
        logits = global_model.raw_logits(x).double().cpu()

        for i in range(n_spec):
            cols = (global_model.class_to_specialist.cpu() == i)
            block = logits[:, cols]
            total[i] += block.sum()
            total_sq[i] += (block ** 2).sum()
            count[i] += block.numel()
            max_sum[i] += block.max(dim=1).values.sum()
            max_count[i] += block.shape[0]

    mean = total / count
    var = (total_sq / count) - mean ** 2
    std = var.clamp(min=1e-12).sqrt()
    max_mean = max_sum / max_count

    return {
        "mean": mean.float(),
        "std": std.float(),
        "max_mean": max_mean.float(),
    }


def fit_specialist_calibration(
    global_model: GlobalDirectSumModel,
    mode: str = "none",
    unlabelled_loader: Optional[DataLoader] = None,
    labelled_loader: Optional[DataLoader] = None,
    device: str = "cpu",
    epochs: int = 3,
    lr: float = 0.05,
) -> GlobalDirectSumModel:
    """
    Make the four heads' outputs numerically comparable.

    "none"        identity. Fully training-free baseline. Run this first and record it.
    "standardise" scale_i = 1/std_i, shift_i = -mean_i/std_i, from unlabelled statistics.
                  Every specialist block then has zero mean and unit variance, so no head can
                  dominate through sheer magnitude.
    "max_logit"   shift each block so its mean max-logit matches the global mean max-logit.
                  Preserves within-specialist spacing, only removes the offset.
    "temperature" fit the 8 parameters by cross-entropy on labelled data.
    """
    n_spec = len(global_model.specialist_names)
    print(f"Calibration: mode = '{mode}'")
    print("-" * 90)

    if mode == "none":
        with torch.no_grad():
            global_model.logit_scale.fill_(1.0)
            global_model.logit_shift.fill_(0.0)
        print("  identity (raw specialist logits concatenated, no parameters fitted)")
        print("-" * 90 + "\n")
        return global_model

    if mode in ("standardise", "max_logit"):
        if unlabelled_loader is None:
            raise ValueError(f"mode '{mode}' needs unlabelled_loader")
        stats = collect_block_statistics(global_model, unlabelled_loader, device)
        with torch.no_grad():
            if mode == "standardise":
                scale = 1.0 / stats["std"].clamp(min=1e-6)
                shift = -stats["mean"] * scale
            else:
                target = stats["max_mean"].mean()
                scale = torch.ones(n_spec)
                shift = target - stats["max_mean"]
            global_model.logit_scale.copy_(scale.to(global_model.logit_scale.device))
            global_model.logit_shift.copy_(shift.to(global_model.logit_shift.device))

        for i, name in enumerate(global_model.specialist_names):
            print(f"  {name:22s} mean {stats['mean'][i]:+7.3f}  std {stats['std'][i]:6.3f}  "
                  f"mean-max {stats['max_mean'][i]:+7.3f}  ->  scale {global_model.logit_scale[i]:.4f}  "
                  f"shift {global_model.logit_shift[i]:+.4f}")
        print("  (no labels used)")
        print("-" * 90 + "\n")
        return global_model

    if mode == "temperature":
        if labelled_loader is None:
            raise ValueError("mode 'temperature' needs labelled_loader")
        print("  WARNING: this fits 8 parameters by gradient descent on labelled data.")
        print("           It is NOT training-free. Report it separately from the 'none' result.")

        global_model.eval()
        global_model.logit_scale.requires_grad_(True)
        global_model.logit_shift.requires_grad_(True)
        opt = torch.optim.Adam(global_model.calibration_parameters(), lr=lr)

        for ep in range(epochs):
            running, seen = 0.0, 0
            for batch in labelled_loader:
                x, y = batch[0].to(device), to_global_targets(batch[1]).to(device)
                with torch.no_grad():
                    raw = global_model.raw_logits(x)          # branches + head are frozen
                logits = global_model.apply_mask(global_model.apply_calibration(raw))
                loss = F.cross_entropy(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * y.size(0)
                seen += y.size(0)
            print(f"  epoch {ep + 1}/{epochs}  loss {running / max(seen, 1):.4f}")

        global_model.logit_scale.requires_grad_(False)
        global_model.logit_shift.requires_grad_(False)
        for i, name in enumerate(global_model.specialist_names):
            print(f"  {name:22s} scale {global_model.logit_scale[i]:.4f}  "
                  f"shift {global_model.logit_shift[i]:+.4f}")
        print("-" * 90 + "\n")
        return global_model

    raise ValueError(f"unknown calibration mode '{mode}'")


# ==================================================================================================
# STEP 6: OPTIONAL STAGE 2 - train only the global head (OFF by default)
# ==================================================================================================

def train_global_head(
    global_model: GlobalDirectSumModel,
    loader: DataLoader,
    device: str = "cpu",
    epochs: int = HEAD_TRAINING_EPOCHS,
    lr: float = HEAD_TRAINING_LR,
) -> GlobalDirectSumModel:
    """
    Unfreeze ONLY the global head so classes can start using other specialists' features
    (the block-sparse rows fill in off-block entries). Feature extractors stay frozen and in
    eval mode, so nothing the specialists learned is overwritten.

    This is "further training" and breaks the training-free property. Opt-in only.
    """
    print("Stage 2: training the global head only (branches frozen)")
    print("-" * 90)

    global_model.train()                       # branches forced to eval() by the override
    for p in global_model.branches.parameters():
        p.requires_grad_(False)
    for p in global_model.linear.parameters():
        p.requires_grad_(True)

    opt = torch.optim.SGD(global_model.linear.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for ep in range(epochs):
        running, correct, seen = 0.0, 0, 0
        for batch in loader:
            x, y = batch[0].to(device), to_global_targets(batch[1]).to(device)
            with torch.no_grad():
                z = global_model.extract_features(x)
            logits = global_model.apply_mask(global_model.apply_calibration(global_model.linear(z)))
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        print(f"  epoch {ep + 1}/{epochs}  loss {running / max(seen, 1):.4f}  "
              f"train acc {100.0 * correct / max(seen, 1):.2f}%")

    global_model.eval()
    print("-" * 90 + "\n")
    return global_model


# ==================================================================================================
# STEP 7: EVALUATION
# ==================================================================================================

def evaluate_aggregated_model(model: nn.Module, loader: DataLoader, device: str = "cpu"):
    """Kept with the original name/signature so your existing calls still work."""
    total_loss, total_correct, total_samples = 0.0, 0, 0

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels, batch_task_ids in loader:
            batch_data = batch_data.to(device)
            targets = to_global_targets(batch_labels).to(device)

            logits = model(batch_data)          # no task id is passed anywhere
            loss = F.cross_entropy(logits, targets, reduction="mean")

            total_loss += loss.item() * targets.size(0)
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / total_samples if total_samples else 0.0
    acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    print(f"Loss = {avg_loss:.6f} | Accuracy = {acc:.2f}%  ({total_correct}/{total_samples})")
    return acc


@torch.no_grad()
def evaluate_with_wp_tp_diagnostics(
    global_model: GlobalDirectSumModel,
    loader: DataLoader,
    device: str = "cpu",
    label: str = "combined",
):
    """
    Split the class-IL result into its two causes (Kim et al., NeurIPS 2022):

      TP (task-id prediction)   : does the global argmax land in the block owned by the
                                  true class's specialist?
      WP (within-task prediction): restricted to the true specialist's own classes, is the
                                  argmax correct? This is the task-IL upper bound and, because
                                  the merge is exact, equals each specialist's native accuracy.

      CIL accuracy <= WP, and the gap between (TP x WP) and CIL tells you whether calibration
      or representation is your bottleneck.
    """
    global_model.eval()
    names = global_model.specialist_names
    n_spec = len(names)
    c2s = global_model.class_to_specialist.to(device)

    n = 0
    cil_correct = 0
    tp_correct = 0
    wp_correct = 0
    block_conf = torch.zeros(n_spec, n_spec, dtype=torch.long)   # true block x predicted block
    per_spec_total = torch.zeros(n_spec, dtype=torch.long)

    for batch_data, batch_labels, _ in loader:
        x = batch_data.to(device)
        y = to_global_targets(batch_labels).to(device)

        logits = global_model(x)
        pred = logits.argmax(dim=1)

        true_block = c2s[y]
        pred_block = c2s[pred]

        cil_correct += (pred == y).sum().item()
        tp_correct += (pred_block == true_block).sum().item()

        # Oracle-block (task-IL) accuracy: argmax restricted to the true specialist's columns.
        block_mask = (c2s.unsqueeze(0) == true_block.unsqueeze(1))
        masked = logits.masked_fill(~block_mask, float("-inf"))
        wp_correct += (masked.argmax(dim=1) == y).sum().item()

        for t, p in zip(true_block.cpu().tolist(), pred_block.cpu().tolist()):
            if t >= 0 and p >= 0:
                block_conf[t, p] += 1
            if t >= 0:
                per_spec_total[t] += 1

        n += y.size(0)

    print(f"Diagnostics [{label}]  n = {n}")
    print(f"  CIL accuracy (43-way, no task id) : {100.0 * cil_correct / max(n, 1):6.2f}%")
    print(f"  TP  specialist-block accuracy     : {100.0 * tp_correct / max(n, 1):6.2f}%")
    print(f"  WP  within-specialist accuracy    : {100.0 * wp_correct / max(n, 1):6.2f}%   "
          f"(task-IL upper bound)")
    print(f"  TP x WP (approx. attainable CIL)  : "
          f"{100.0 * (tp_correct / max(n, 1)) * (wp_correct / max(n, 1)):6.2f}%")
    print("  block confusion (rows = true specialist, cols = predicted specialist, row %):")
    header = "      " + "".join(f"{nm[:9]:>11s}" for nm in names)
    print(header)
    for i, nm in enumerate(names):
        tot = max(int(per_spec_total[i].item()), 1)
        row = "".join(f"{100.0 * block_conf[i, j].item() / tot:10.1f}%" for j in range(n_spec))
        print(f"  {nm[:9]:>9s}{row}")
    print()

    return {
        "cil": 100.0 * cil_correct / max(n, 1),
        "tp": 100.0 * tp_correct / max(n, 1),
        "wp": 100.0 * wp_correct / max(n, 1),
    }


def test_global_model(global_model: nn.Module, device: str = "cpu"):
    """Shape smoke test."""
    global_model.eval()
    dummy = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        out = global_model(dummy)
    expected = torch.Size([4, num_global_classes()])
    print(f"Smoke test: input {tuple(dummy.shape)} -> output {tuple(out.shape)} "
          f"(expected {tuple(expected)}) {'OK' if out.shape == expected else 'MISMATCH'}")
    return out


# ==================================================================================================
# STEP 8: SAVE / LOAD
# ==================================================================================================

def save_global_model(global_model: GlobalDirectSumModel, output_path: Path, report: Dict):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "global_model_direct_sum.pt"
    torch.save(
        {"state_dict": global_model.state_dict(), "config": global_model.config()},
        model_path,
    )
    print(f"[saved] global model -> {model_path}")

    report_path = output_path / "direct_sum_merge_report.pkl"
    serialisable = {k: v for k, v in report.items() if k != "global_row_sources"}
    serialisable["global_row_sources"] = {int(k): v for k, v in report["global_row_sources"].items()}
    with open(report_path, "wb") as f:
        pickle.dump(serialisable, f)
    print(f"[saved] merge report -> {report_path}")


def load_global_model(checkpoint_path: Path, device: str = "cpu") -> GlobalDirectSumModel:
    """Rebuild the merged model from a single checkpoint (architecture comes from the config)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    specialists = OrderedDict(
        (name, ResNet20_DynamicModel(num_classes=cfg["specialist_classifier_sizes"][name]))
        for name in cfg["specialist_names"]
    )
    dummy_c2s = torch.full((cfg["n_global_classes"],), -1, dtype=torch.long)

    model = GlobalDirectSumModel(
        specialists=specialists,
        feature_dims=cfg["feature_dims"],
        class_to_specialist=dummy_c2s,               # overwritten by load_state_dict (it is a buffer)
        n_global_classes=cfg["n_global_classes"],
        mask_untrained=cfg["mask_untrained"],
        masked_logit_value=cfg["masked_logit_value"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


# ==================================================================================================
# MAIN
# ==================================================================================================

def main():
    MODEL_DIR = "../Fourteenth_implementation_attempt/final_specialised_models/"
    OUTPUT_DIR = Path("./final_specialised_models")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = "cpu"

    # ---------------------------------------------------------------- build
    global_model, report = build_direct_sum_global_model(
        model_names=MODEL_NAMES, model_dir=MODEL_DIR, device=DEVICE
    )

    # ---------------------------------------------------------------- verify
    ok = verify_functional_equivalence(global_model, report, device=DEVICE, loader=test_loader)
    if not ok:
        raise RuntimeError(
            "Function preservation failed. Fix new_to_original / num_subset_classes before "
            "interpreting any accuracy number."
        )

    test_global_model(global_model, device=DEVICE)

    # ---------------------------------------------------------------- baseline: no calibration
    print("\n" + "=" * 90)
    print("RESULT A: raw logit concatenation (fully training-free)")
    print("=" * 90)
    fit_specialist_calibration(global_model, mode="none")
    print("Combined 43-class test set:")
    evaluate_aggregated_model(global_model, test_loader, DEVICE)
    evaluate_with_wp_tp_diagnostics(global_model, test_loader, DEVICE, label="combined, uncalibrated")

    print("Per-specialist test subsets (still 43-way inference, no task id):")
    for name, loader in [
        ("outdoors", outdoors_loader),
        ("nature", nature_loader),
        ("people", people_loader),
        ("objects_and_vehicles", objects_and_vehicles_loader),
    ]:
        print(f"  {name}: ", end="")
        evaluate_aggregated_model(global_model, loader, DEVICE)

    # ---------------------------------------------------------------- optional calibration
    if CALIBRATION_MODE != "none":
        print("\n" + "=" * 90)
        print(f"RESULT B: calibrated ({CALIBRATION_MODE})")
        print("=" * 90)
        fit_specialist_calibration(
            global_model,
            mode=CALIBRATION_MODE,
            unlabelled_loader=test_loader,
            labelled_loader=calibration_loader,
            device=DEVICE,
        )
        print("Combined 43-class test set:")
        evaluate_aggregated_model(global_model, test_loader, DEVICE)
        evaluate_with_wp_tp_diagnostics(global_model, test_loader, DEVICE,
                                        label=f"combined, {CALIBRATION_MODE}")

    # ---------------------------------------------------------------- optional stage 2
    if TRAIN_GLOBAL_HEAD:
        print("\n" + "=" * 90)
        print("RESULT C: global head trained (branches frozen) - NOT training-free")
        print("=" * 90)
        train_global_head(global_model, calibration_loader, device=DEVICE)
        print("Combined 43-class test set:")
        evaluate_aggregated_model(global_model, test_loader, DEVICE)
        evaluate_with_wp_tp_diagnostics(global_model, test_loader, DEVICE, label="combined, head trained")

    # ---------------------------------------------------------------- save
    print("\n" + "=" * 90)
    save_global_model(global_model, OUTPUT_DIR, report)
    print("=" * 90)
    print("MERGE COMPLETE")
    print(f"  method            : direct sum (feature concatenation), no weight averaging")
    print(f"  feature space     : {report['total_feature_dim']}-D "
          f"({' + '.join(str(d) for d in report['feature_dims'].values())})")
    print(f"  output space      : {report['n_global_classes']} logits, "
          f"{report['trained_classes']} trained classes")
    print(f"  parameters        : {report['total_params']:,}")
    print(f"  task id at test   : none")
    print("=" * 90 + "\n")

    return global_model


if __name__ == "__main__":
    global_model = main()

    # Round-trip check: the saved single checkpoint reconstructs the same model.
    reloaded = load_global_model(
        Path("./final_specialised_models/global_model_direct_sum.pt"), device="cpu"
    )
    print("Reloaded-from-disk model, combined test set:")
    evaluate_aggregated_model(reloaded, test_loader, "cpu")
