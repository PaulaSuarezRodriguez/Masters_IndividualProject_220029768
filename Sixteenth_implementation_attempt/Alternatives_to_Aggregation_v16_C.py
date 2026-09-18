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



'''class GlobalSpecialistModel(nn.Module):

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

    def scale_specialised_logits_globally(
        self,
        specialised_logits_list: list[torch.Tensor],):

        # 2. Define your target range (e.g., [0, 1] or [-1, 1])
        target_min = 0.0
        target_max = 1.0

        # 3. Find the global min and max across all tensors in the list
        global_min = min(t.min() for t in specialised_logits_list)
        global_max = max(t.max() for t in specialised_logits_list)

        # 4. Scale each tensor using the same global anchor points
        scaled_specialised_logits_list = []
        for specialised_logits in specialised_logits_list:
            # Normalize to [0, 1] using global min/max
            normalized = (specialised_logits - global_min) / (global_max - global_min)
            # Scale to target range
            scaled_specialised_logits = (normalized * (target_max - target_min)) + target_min

            scaled_specialised_logits_list.append(scaled_specialised_logits)

        return scaled_specialised_logits_list

    def forward(self, x):

        outdoors_logits = self.outdoors(x)[
            :, :self.outdoors_num_classes
        ]
        prediction_outdoors_logits1 = torch.argmax(outdoors_logits, dim=0)
        prediction_outdoors_logits2 = torch.argmax(outdoors_logits, dim=1)
        prediction_outdoors_logits3 = torch.argmax(outdoors_logits, dim=-1)

        nature_logits = self.nature(x)[
            :, :self.nature_num_classes
        ]
        prediction_nature_logits1 = torch.argmax(nature_logits, dim=-0)
        prediction_nature_logits2 = torch.argmax(nature_logits, dim=1)
        prediction_nature_logits3 = torch.argmax(nature_logits, dim=-1)

        people_logits = self.people(x)[
            :, :self.people_num_classes
        ]
        prediction_people_logits1 = torch.argmax(people_logits, dim=0)
        prediction_people_logits2 = torch.argmax(people_logits, dim=1)
        prediction_people_logits3 = torch.argmax(people_logits, dim=-1)

        objects_logits = (
            self.objects_and_vehicles(x)[
                :, :self.objects_num_classes
            ]
        )
        prediction_objects_logits1 = torch.argmax(objects_logits, dim=0)
        prediction_objects_logits2 = torch.argmax(objects_logits, dim=1)
        prediction_objects_logits3 = torch.argmax(objects_logits, dim=-1)


        # ---------------------------------------
        # Calibrate each specialist's logits.
        # ---------------------------------------
        
        specialised_logits_list = [outdoors_logits, nature_logits, people_logits, objects_logits]

        scale_specialised_logits = self.scale_specialised_logits_globally(specialised_logits_list)
        
        scaled_outdoors_logits = scale_specialised_logits[0]
        prediction_scaled_outdoors_logits1 = torch.argmax(scaled_outdoors_logits, dim=0)
        prediction_scaled_outdoors_logits2 = torch.argmax(scaled_outdoors_logits, dim=1)
        prediction_scaled_outdoors_logits3 = torch.argmax(scaled_outdoors_logits, dim=-1)

        scaled_nature_logits = scale_specialised_logits[1]
        prediction_scaled_nature_logits1 = torch.argmax(scaled_nature_logits, dim=-0)
        prediction_scaled_nature_logits2 = torch.argmax(scaled_nature_logits, dim=1)
        prediction_scaled_nature_logits3 = torch.argmax(scaled_nature_logits, dim=-1)

        scaled_people_logits = scale_specialised_logits[2]
        prediction_scaled_people_logits1 = torch.argmax(scaled_people_logits, dim=0)
        prediction_scaled_people_logits2 = torch.argmax(scaled_people_logits, dim=1)
        prediction_scaled_people_logits3 = torch.argmax(scaled_people_logits, dim=-1)

        scaled_objects_logits = scale_specialised_logits[3]
        prediction_scaled_objects_logits1 = torch.argmax(scaled_objects_logits, dim=0)
        prediction_scaled_objects_logits2 = torch.argmax(scaled_objects_logits, dim=1)
        prediction_scaled_objects_logits3 = torch.argmax(scaled_objects_logits, dim=-1)

        raw_prediction_list = [
            prediction_outdoors_logits2,
            prediction_nature_logits2,
            prediction_people_logits2,
            prediction_objects_logits3
        ]

        scaled_prediction_list = [
            prediction_scaled_outdoors_logits2,
            prediction_scaled_nature_logits2,
            prediction_scaled_people_logits2,
            prediction_scaled_objects_logits3
        ]

        names = [
            "outdoors",
            "nature",
            "people",
            "objects"
        ]

        for name, raw, scaled in zip(
                names,
                raw_prediction_list,
                scaled_prediction_list
        ):
            #print(f"\n{name}:")
            #print("Raw predictions:   ", raw)
            #print("Scaled predictions:", scaled)

            # Compare prediction for each sample
            for i, (raw_pred, scaled_pred) in enumerate(
                    zip(raw, scaled)
            ):
                #print(
                 #   f"Sample {i}: "
                 #   f"raw={raw_pred.item()}, "
                 #   f"scaled={scaled_pred.item()}"
                #)

                if raw_pred.item() != scaled_pred.item():
                    print(
                        f"Predictions ARE DIFFERENT. "
                        "The prediction changed after scaling."
                    )
                    break

        ''''''global_logits = outdoors_logits.new_full(
            (x.size(0), self.num_classes),
            float("-inf"),
        )''''''
        global_logits = scaled_outdoors_logits.new_full(
            (x.size(0), self.num_classes),
            float("-inf"),
        )

        global_logits[
            :, self.outdoors_indices
        ] = scaled_outdoors_logits

        global_logits[
            :, self.nature_indices
        ] = scaled_nature_logits

        global_logits[
            :, self.people_indices
        ] = scaled_people_logits

        global_logits[
            :, self.objects_and_vehicles_indices
        ] = scaled_objects_logits

        return global_logits'''


def expert_evidence_score(
        logits: torch.Tensor,
        temperature: float = 1.0,
):
    """
    Higher score = stronger evidence that the sample belongs
    to this specialist's class distribution.

    logits:
        [batch_size, number_of_specialist_classes]
    """
    return temperature * torch.logsumexp(
        logits / temperature,
        dim=1,
    )

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
        expert_thresholds,
        expert_scales,
        num_classes=43,
    ):
        super().__init__()

        self.outdoors = outdoors_model
        self.nature = nature_model
        self.people = people_model

        self.objects_and_vehicles = (
            objects_and_vehicles_model
        )

        self.outdoors_num_classes = int(
            outdoors_num_classes
        )

        self.nature_num_classes = int(
            nature_num_classes
        )

        self.people_num_classes = int(
            people_num_classes
        )

        self.objects_num_classes = int(
            objects_num_classes
        )

        self.num_classes = num_classes

        # ----------------------------------------
        # Global classifier locations
        # ----------------------------------------

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

        # ----------------------------------------
        # Expert confidence calibration
        # ----------------------------------------

        self.register_buffer(
            "expert_thresholds",
            expert_thresholds.float()
        )

        self.register_buffer(
            "expert_scales",
            expert_scales.float()
        )

        # ----------------------------------------
        # Which expert owns each global class?
        # ----------------------------------------

        global_class_owner = torch.full(
            (num_classes,),
            -1,
            dtype=torch.long,
        )

        expert_global_indices = [
            outdoors_indices,
            nature_indices,
            people_indices,
            objects_and_vehicles_indices,
        ]

        for expert_idx, indices in enumerate(
            expert_global_indices
        ):

            if torch.any(
                global_class_owner[
                    indices
                ] != -1
            ):
                raise RuntimeError(
                    "Two experts own the same "
                    "global class."
                )

            global_class_owner[
                indices
            ] = expert_idx

        if torch.any(
            global_class_owner < 0
        ):
            raise RuntimeError(
                "Some global classes have no owner."
            )

        self.register_buffer(
            "global_class_owner",
            global_class_owner,
        )

    def get_specialist_logits(
        self,
        x,
    ):

        outdoors_logits = self.outdoors(
            x
        )[
            :,
            :self.outdoors_num_classes
        ]

        nature_logits = self.nature(
            x
        )[
            :,
            :self.nature_num_classes
        ]

        people_logits = self.people(
            x
        )[
            :,
            :self.people_num_classes
        ]

        objects_logits = (
            self.objects_and_vehicles(
                x
            )[
                :,
                :self.objects_num_classes
            ]
        )

        return [
            outdoors_logits,
            nature_logits,
            people_logits,
            objects_logits,
        ]

    def compute_expert_confidence(
        self,
        specialist_logits,
    ):
        """
        Returns:

            evidence:
                raw ID evidence [B, 4]

            normalized_evidence:
                threshold-relative scores [B, 4]

            confidence:
                convenient 0..1 representation [B, 4]

        confidence > 0.5 means:
            specialist score > its learned threshold.

        Note:
            confidence is a normalized gating confidence,
            not a guaranteed probability of correctness.
        """

        evidence = torch.stack(
            [
                expert_evidence_score(
                    logits
                )
                for logits
                in specialist_logits
            ],
            dim=1,
        )

        normalized_evidence = (
            evidence
            -
            self.expert_thresholds.unsqueeze(
                0
            )
        ) / (
            self.expert_scales.unsqueeze(
                0
            )
        )

        confidence = torch.sigmoid(
            normalized_evidence
        )

        return (
            evidence,
            normalized_evidence,
            confidence,
        )

    def select_expert(
        self,
        normalized_evidence,
    ):
        """
        Hard confidence gating.

        An expert passes its learned ID threshold when:

            normalized_evidence >= 0

        If multiple experts pass:
            choose the strongest one.

        If none pass:
            fall back to the strongest expert rather
            than producing no prediction.
        """

        passes_threshold = (
            normalized_evidence >= 0.0
        )

        gated_scores = (
            normalized_evidence.clone()
        )

        gated_scores[
            ~passes_threshold
        ] = float("-inf")

        # Samples where every expert failed threshold
        no_expert_passed = ~(
            passes_threshold.any(
                dim=1
            )
        )

        # Fall back to the strongest normalized
        # evidence for those samples.
        if no_expert_passed.any():

            gated_scores[
                no_expert_passed
            ] = normalized_evidence[
                no_expert_passed
            ]

        selected_expert = (
            gated_scores.argmax(
                dim=1
            )
        )

        return (
            selected_expert,
            passes_threshold,
        )

    def forward(
        self,
        x,
        return_gate_info=False,
    ):

        # ========================================
        # Every image goes through ALL specialists.
        # No task ID.
        # ========================================

        specialist_logits = (
            self.get_specialist_logits(x)
        )

        (
            evidence,
            normalized_evidence,
            expert_confidence,
        ) = self.compute_expert_confidence(
            specialist_logits
        )

        (
            selected_expert,
            passes_threshold,
        ) = self.select_expert(
            normalized_evidence
        )

        # ========================================
        # First construct the full 43-class vector
        # ========================================

        global_logits = (
            specialist_logits[0]
            .new_full(
                (
                    x.size(0),
                    self.num_classes,
                ),
                float("-inf"),
            )
        )

        global_logits[
            :,
            self.outdoors_indices
        ] = specialist_logits[0]

        global_logits[
            :,
            self.nature_indices
        ] = specialist_logits[1]

        global_logits[
            :,
            self.people_indices
        ] = specialist_logits[2]

        global_logits[
            :,
            self.objects_and_vehicles_indices
        ] = specialist_logits[3]

        # ========================================
        # HARD EXPERT GATE
        #
        # Only classes belonging to the selected
        # expert may now compete in argmax.
        # ========================================

        class_owner = (
            self.global_class_owner
            .unsqueeze(0)
        )

        selected = (
            selected_expert
            .unsqueeze(1)
        )

        wrong_expert_classes = (
            class_owner != selected
        )

        global_logits = (
            global_logits.masked_fill(
                wrong_expert_classes,
                float("-inf"),
            )
        )

        if return_gate_info:

            gate_info = {
                "selected_expert":
                    selected_expert,

                "evidence":
                    evidence,

                "normalized_evidence":
                    normalized_evidence,

                "confidence":
                    expert_confidence,

                "passes_threshold":
                    passes_threshold,
            }

            return (
                global_logits,
                gate_info,
            )

        return global_logits

# ==============================================================================
# STEP 2: Classifier Remapping (UNCHANGED)
# ==============================================================================

'''original_to_aggregated_dict = {17: 0, 12: 1, 33: 2, 23: 3, 49: 4, 37: 5, #OUTDOORS
                               51: 6, 47: 7, 0: 8, 54: 9, 53: 10, 52: 11, 59: 12, 57: 13, 56: 14, #NATURE
                               2: 15, 11: 16, 35: 17, #PEOPLE
                               16: 18, 13: 19, 10: 20, 9: 21, 8: 22, 5: 23, 58: 24, 48: 25, 41: 26, 40: 27, 39: 28, 28: 29, 25: 30, 22: 31, 20: 32, 94: 33, #OBJECTS_AND_VEHICLES
                               90: 34, 89: 35, 87: 36, 86: 37, 85: 38, 84: 39, 81: 40, 69: 41, 61: 42} #OBJECTS_AND_VEHICLES'''

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

EXPERT_NAMES = [
    "outdoors",
    "nature",
    "people",
    "objects_and_vehicles",
]


def build_cifar_to_expert_lookup(
    mappings,
    num_subset_classes,
    expert_names=EXPERT_NAMES,
    num_cifar_classes=100,
):
    """
    Returns a tensor such that:

        lookup[cifar_class] = expert_index

    Expert indices:
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
                    "belongs to multiple specialists."
                )

            lookup[cifar_class] = expert_idx

    return lookup

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

    remapped_local_weights = torch.zeros(
        num_classes,
        feature_dim,
        device=device,
        dtype=model.linear.weight.dtype #dtype=classifier.weight.dtype
    )

    remapped_local_bias = torch.zeros(
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
        remapped_local_weights[cifar_class] = classifier.weight[local_idx].to(device)
        remapped_local_bias[cifar_class] = classifier.bias[local_idx].to(device)

    if VERBOSE:
        print("remapped_local_weights", remapped_local_weights)
        print("remapped_local_bias", remapped_local_bias)

    return remapped_local_weights, remapped_local_bias


def remap_classifier_to_aggregated(
    #model_name,
    model: nn.Module,
    w,
    b,
    original_to_aggregated_dict: Dict[int, int],
    #num_subset_classes,
    num_classes: int = 43,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:

    classifier = model.linear
    feature_dim = classifier.weight.shape[1]

    remapped_global_weights = torch.zeros(
        num_classes,
        feature_dim,
        device=device,
        dtype=model.linear.weight.dtype #dtype=classifier.weight.dtype
    )

    remapped_global_bias = torch.zeros(
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
        remapped_global_weights[aggregated_idx] = w[cifar_idx].to(device)
        remapped_global_bias[aggregated_idx] = b[cifar_idx].to(device)

    if VERBOSE:
        print("remapped_weights", remapped_global_weights)
        print("remapped_bias", remapped_global_bias)

    return remapped_global_weights, remapped_global_bias


def find_best_expert_threshold(
    scores: torch.Tensor,
    is_in_distribution: torch.Tensor,
):
    """
    Find the threshold that maximizes balanced binary accuracy:

        ID sample  -> score >= threshold
        OOD sample -> score < threshold

    Returns:
        threshold
        scale
        balanced_accuracy
    """

    scores = scores.float().cpu()

    labels = (
        is_in_distribution
        .bool()
        .cpu()
    )

    positive_scores = scores[labels]
    negative_scores = scores[~labels]

    if len(positive_scores) == 0:
        raise RuntimeError(
            "No in-distribution validation samples."
        )

    if len(negative_scores) == 0:
        raise RuntimeError(
            "No out-of-distribution validation samples."
        )

    # Highest evidence first
    order = torch.argsort(
        scores,
        descending=True,
    )

    sorted_scores = scores[order]

    sorted_labels = (
        labels[order].float()
    )

    num_positive = sorted_labels.sum()

    num_negative = (
        len(sorted_labels)
        - num_positive
    )

    true_positive = torch.cumsum(
        sorted_labels,
        dim=0,
    )

    false_positive = torch.cumsum(
        1.0 - sorted_labels,
        dim=0,
    )

    tpr = (
        true_positive
        / num_positive.clamp_min(1)
    )

    tnr = (
        num_negative - false_positive
    ) / num_negative.clamp_min(1)

    balanced_accuracy = (
        tpr + tnr
    ) / 2.0

    best_idx = int(
        balanced_accuracy.argmax()
    )

    # Put threshold between adjacent scores
    if best_idx + 1 < len(sorted_scores):

        threshold = (
            sorted_scores[best_idx]
            +
            sorted_scores[best_idx + 1]
        ) / 2.0

    else:

        threshold = sorted_scores[
            best_idx
        ]

    # Scale used to make scores from different
    # specialists comparable.
    positive_var = positive_scores.var(
        unbiased=False
    )

    negative_var = negative_scores.var(
        unbiased=False
    )

    scale = torch.sqrt(
        (
            positive_var
            +
            negative_var
        ) / 2.0
    ).clamp_min(1e-6)

    return (
        threshold,
        scale,
        balanced_accuracy[best_idx],
    )

@torch.no_grad()
def fit_expert_confidence_gate(
    models,
    calibration_loader,
    mappings,
    num_subset_classes,
    device="cpu",
    expert_names=EXPERT_NAMES,
):
    """
    Fits one ID/OOD evidence threshold per specialist.

    NO specialist parameters are changed.

    The validation labels are used only to determine which
    expert each validation image actually belongs to.
    """

    device = torch.device(device)

    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings,
            num_subset_classes,
            expert_names,
        ).to(device)
    )

    for name in expert_names:
        models[name].to(device)
        models[name].eval()

    all_scores = []
    all_true_experts = []

    for batch_data, batch_labels, batch_task_ids in (
        calibration_loader
    ):

        batch_data = batch_data.to(device)

        batch_labels = batch_labels.to(device)

        true_experts = cifar_to_expert[
            batch_labels
        ]

        if torch.any(true_experts < 0):

            invalid = torch.unique(
                batch_labels[
                    true_experts < 0
                ]
            )

            raise RuntimeError(
                "Validation set contains classes "
                "not owned by the global specialists: "
                f"{invalid.tolist()}"
            )

        batch_expert_scores = []

        for model_name in expert_names:

            n = int(
                num_subset_classes[
                    model_name
                ]
            )

            logits = models[
                model_name
            ](batch_data)[:, :n]

            score = expert_evidence_score(
                logits
            )

            batch_expert_scores.append(
                score
            )

        # [B, 4]
        batch_expert_scores = (
            torch.stack(
                batch_expert_scores,
                dim=1,
            )
        )

        all_scores.append(
            batch_expert_scores.cpu()
        )

        all_true_experts.append(
            true_experts.cpu()
        )

    # [N, 4]
    all_scores = torch.cat(
        all_scores,
        dim=0,
    )

    # [N]
    all_true_experts = torch.cat(
        all_true_experts,
        dim=0,
    )

    thresholds = []
    scales = []

    print(
        "\n===== FITTING EXPERT CONFIDENCE GATE ====="
    )

    for expert_idx, model_name in enumerate(
        expert_names
    ):

        is_id = (
            all_true_experts
            == expert_idx
        )

        threshold, scale, bal_acc = (
            find_best_expert_threshold(
                all_scores[
                    :, expert_idx
                ],
                is_id,
            )
        )

        thresholds.append(
            threshold
        )

        scales.append(
            scale
        )

        print(
            f"{model_name:22s} | "
            f"threshold={threshold.item():.6f} | "
            f"scale={scale.item():.6f} | "
            f"ID/OOD balanced acc="
            f"{bal_acc.item() * 100:.2f}%"
        )

    thresholds = torch.stack(
        thresholds
    )

    scales = torch.stack(
        scales
    )

    # ---------------------------------------
    # Test gate accuracy on validation set
    # ---------------------------------------

    normalized_scores = (
        all_scores
        -
        thresholds.unsqueeze(0)
    ) / scales.unsqueeze(0)

    selected_expert = (
        normalized_scores.argmax(
            dim=1
        )
    )

    gate_accuracy = (
        selected_expert
        ==
        all_true_experts
    ).float().mean()

    print(
        "\nValidation expert-selection accuracy: "
        f"{gate_accuracy.item() * 100:.2f}%"
    )

    return thresholds, scales


@torch.no_grad()
def evaluate_expert_gate(
    global_model,
    loader,
    mappings,
    num_subset_classes,
    device="cpu",
    expert_names=EXPERT_NAMES,
):

    device = torch.device(device)

    global_model.eval()
    global_model.to(device)

    cifar_to_expert = (
        build_cifar_to_expert_lookup(
            mappings,
            num_subset_classes,
            expert_names,
        ).to(device)
    )

    total = 0
    correct_expert = 0

    confusion = torch.zeros(
        len(expert_names),
        len(expert_names),
        dtype=torch.long,
    )

    for (
        batch_data,
        batch_labels,
        batch_task_ids,
    ) in loader:

        batch_data = batch_data.to(
            device
        )

        batch_labels = batch_labels.to(
            device
        )

        true_expert = cifar_to_expert[
            batch_labels
        ]

        (
            global_logits,
            gate_info,
        ) = global_model(
            batch_data,
            return_gate_info=True,
        )

        selected_expert = (
            gate_info[
                "selected_expert"
            ]
        )

        correct_expert += (
            selected_expert
            ==
            true_expert
        ).sum().item()

        total += batch_labels.size(0)

        flattened = (
            true_expert
            * len(expert_names)
            +
            selected_expert
        )

        batch_confusion = torch.bincount(
            flattened,
            minlength=(
                len(expert_names)
                ** 2
            ),
        ).reshape(
            len(expert_names),
            len(expert_names),
        )

        confusion += (
            batch_confusion.cpu()
        )

    print(
        "\nExpert-selection accuracy: "
        f"{100 * correct_expert / total:.2f}%"
    )

    print(
        "\nRows = TRUE expert"
    )

    print(
        "Columns = SELECTED expert"
    )

    print(
        expert_names
    )

    print(
        confusion
    )

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

    expert_thresholds, expert_scales = (
        fit_expert_confidence_gate(
            models=models,
            calibration_loader=val_loader,
            mappings=mappings,
            num_subset_classes=(
                num_subset_classes
            ),
            device=device,
        )
    )

    global_model = GlobalSpecialistModel(
        outdoors_model=models[
            "outdoors"
        ],

        nature_model=models[
            "nature"
        ],

        people_model=models[
            "people"
        ],

        objects_and_vehicles_model=models[
            "objects_and_vehicles"
        ],

        outdoors_indices=(
            local_to_global_indices[
                "outdoors"
            ]
        ),

        nature_indices=(
            local_to_global_indices[
                "nature"
            ]
        ),

        people_indices=(
            local_to_global_indices[
                "people"
            ]
        ),

        objects_and_vehicles_indices=(
            local_to_global_indices[
                "objects_and_vehicles"
            ]
        ),

        outdoors_num_classes=(
            num_subset_classes[
                "outdoors"
            ]
        ),

        nature_num_classes=(
            num_subset_classes[
                "nature"
            ]
        ),

        people_num_classes=(
            num_subset_classes[
                "people"
            ]
        ),

        objects_num_classes=(
            num_subset_classes[
                "objects_and_vehicles"
            ]
        ),

        expert_thresholds=(
            expert_thresholds
        ),

        expert_scales=(
            expert_scales
        ),

        num_classes=43,
    )

    evaluate_expert_gate(
        global_model,
        test_loader,
        mappings,
        num_subset_classes,
        #device=DEVICE,
    )

    '''global_model = GlobalSpecialistModel(
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

        num_classes=43,
    )'''

    return global_model


# ==============================================================================
# STEP 4: Save and Test
# ==============================================================================

def save_global_model(global_model: nn.Module,
                      output_path: Path): #,
                      #report: Dict):
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
        save_global_model(global_model, OUTPUT_DIR)

        # ============================================================
        # Test
        # ============================================================
        #output = test_global_model(global_model, device=DEVICE)

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
