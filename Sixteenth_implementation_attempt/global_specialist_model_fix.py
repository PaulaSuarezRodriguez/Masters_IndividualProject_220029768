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

    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.log_T], lr=0.01, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        pre_nll = nll(logits, labels).item()
        post_nll = nll(scaler(logits), labels).item()
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
