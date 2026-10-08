import torch
import torch.nn as nn
import pickle
from pathlib import Path
from typing import Dict, Tuple
from collections import OrderedDict
from Final_implementation_attempt.ResNet20_DynamicModel_vFinal import ResNet20_DynamicModel
from Final_implementation_attempt.ResNet20_DynamicModel_Compact_vFinal import ResNet20_Compact, ResNet20_Compact_XL, ResNet20_XL, ResNet20_Compact_S
from Final_implementation_attempt.Train_ResNet20_Outdoors_LwF_vFinal import outdoors_specialised_models_test_stream_class_il, outdoors_specialised_models_train_stream_class_il, outdoors_specialised_models_val_stream_class_il
from Final_implementation_attempt.Train_ResNet20_Nature_LwF_vFinal import nature_specialised_models_test_stream_class_il, nature_specialised_models_train_stream_class_il, nature_specialised_models_val_stream_class_il
from Final_implementation_attempt.Train_ResNet20_People_LwF_vFinal import people_specialised_models_test_stream_class_il, people_specialised_models_train_stream_class_il, people_specialised_models_val_stream_class_il
from Final_implementation_attempt.Train_ResNet20_Objects_and_Vehicles_LwF_vFinal import objects_and_vehicles_specialised_models_test_stream_class_il, objects_and_vehicles_specialised_models_train_stream_class_il, objects_and_vehicles_specialised_models_val_stream_class_il
from Final_implementation_attempt.Load_cifar100_vFinal import avl_test
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import Module, CrossEntropyLoss
import torch.nn.functional as F

VERBOSE = False


# Fixed expert ordering -- used by the lookup, training, and forward()
EXPERT_ORDER = ["outdoors", "nature", "people", "objects_and_vehicles"]
NUM_EXPERTS = len(EXPERT_ORDER)


# EVALUATION
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
    combined_dataset_people
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
])

combined_dataset_nature_train = ConcatDataset([
    nature_specialised_models_train_stream_class_il[0].dataset,
    nature_specialised_models_train_stream_class_il[1].dataset,
    nature_specialised_models_train_stream_class_il[2].dataset,
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
    shuffle=False
)
print("print train_loader", train_loader)
print("print train_loader.dataset", train_loader.dataset)



combined_dataset_outdoors_first_experience_test = ConcatDataset([
    outdoors_specialised_models_test_stream_class_il[0].dataset,
])

combined_dataset_nature_first_experience_test = ConcatDataset([
    nature_specialised_models_test_stream_class_il[0].dataset,
])

combined_dataset_people_first_experience_test = ConcatDataset([
    people_specialised_models_test_stream_class_il[0].dataset,
])

combined_dataset_objects_and_vehicles_first_experience_test = ConcatDataset([
    objects_and_vehicles_specialised_models_test_stream_class_il[0].dataset,
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
    shuffle=False
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
    outdoors_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_nature_middle_experience_test = ConcatDataset([
    nature_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_people_middle_experience_test = ConcatDataset([
    people_specialised_models_test_stream_class_il[1].dataset,
])

combined_dataset_objects_and_vehicles_middle_experience_test = ConcatDataset([
    objects_and_vehicles_specialised_models_test_stream_class_il[1].dataset,
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
    shuffle=False
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
    outdoors_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_nature_last_experience_test = ConcatDataset([
    nature_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_people_last_experience_test = ConcatDataset([
    people_specialised_models_test_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_last_experience_test = ConcatDataset([
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
    shuffle=False
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
print("print last_experience_test_loader", last_experience_test_loader)
print("print last_experience_test_loader.dataset", last_experience_test_loader.dataset)


combined_dataset_outdoors_last_experience_train = ConcatDataset([
    outdoors_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_nature_last_experience_train = ConcatDataset([
    nature_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_people_last_experience_train = ConcatDataset([
    people_specialised_models_train_stream_class_il[2].dataset,
])

combined_dataset_objects_and_vehicles_last_experience_train = ConcatDataset([
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
])

combined_dataset_nature_val = ConcatDataset([
    nature_specialised_models_val_stream_class_il[0].dataset,
    nature_specialised_models_val_stream_class_il[1].dataset,
    nature_specialised_models_val_stream_class_il[2].dataset,
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
    batch_size=128,
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



# To which expert does each CIFAR class belong mapping for creating the ground truth for the gating network
def build_cifar_to_expert_lookup(
    mappings: dict,
    expert_order: list = EXPERT_ORDER,
    num_cifar_classes: int = 100,
) -> torch.Tensor:
    lookup = torch.full((num_cifar_classes,), -1, dtype=torch.long)
    for expert_id, model_name in enumerate(expert_order):
        for local_idx, cifar_class in mappings[model_name].items():
            lookup[int(cifar_class)] = expert_id
    return lookup



# Training the gating network
def train_gating_network(
    gate: nn.Module,
    loader: DataLoader,
    cifar_to_expert_lookup: torch.Tensor,
    epochs: int = 10,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cpu",
) -> nn.Module:

    cifar_to_expert_lookup = cifar_to_expert_lookup.to(device)
    gate = gate.to(device)
    gate.train()

    optimizer = torch.optim.SGD(
        gate.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch_data, batch_labels, *_ in loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            for bl in batch_labels:
                if bl.item() not in original_to_aggregated_dict.keys():
                    raise RuntimeError("Labels used in training gate are not known by any specialist model")

            expert_targets = cifar_to_expert_lookup[batch_labels]
            if (expert_targets < 0).any():
                bad = batch_labels[expert_targets < 0].unique().tolist()
                raise RuntimeError(
                    f"train_gating_network: CIFAR classes {bad} are not "
                    f"owned by any expert in {EXPERT_ORDER} -- check "
                    f"build_cifar_to_expert_lookup / your loaders."
                )

            optimizer.zero_grad()
            gate_logits = gate(batch_data)
            loss = criterion(gate_logits, expert_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * expert_targets.size(0)
            total_correct += (gate_logits.argmax(dim=1) == expert_targets).sum().item()
            total_samples += expert_targets.size(0)

        avg_loss = total_loss / total_samples
        routing_acc = total_correct / total_samples * 100
        print(f"[Gate] Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  routing_acc={routing_acc:.2f}%")

    gate.eval()
    return gate



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
        gate: nn.Module = None,
        routing: str = "soft",  # "soft" (weighted sum) or "hard" (top-1)
        num_classes=43,
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

        self.gate = gate
        assert routing in ("soft", "hard")
        self.routing = routing

        self.register_buffer("outdoors_indices", outdoors_indices.long())
        self.register_buffer("nature_indices", nature_indices.long())
        self.register_buffer("people_indices", people_indices.long())
        self.register_buffer("objects_and_vehicles_indices", objects_and_vehicles_indices.long())

    def forward(self, x):
        outdoors_logits = self.outdoors(x)[:, :self.outdoors_num_classes]
        nature_logits = self.nature(x)[:, :self.nature_num_classes]
        people_logits = self.people(x)[:, :self.people_num_classes]
        objects_logits = self.objects_and_vehicles(x)[:, :self.objects_num_classes]

        outdoors_probs = F.softmax(outdoors_logits, dim=1)
        nature_probs = F.softmax(nature_logits, dim=1)
        people_probs = F.softmax(people_logits, dim=1)
        objects_probs = F.softmax(objects_logits, dim=1)

        global_probs = outdoors_probs.new_zeros((x.size(0), self.num_classes))

        if self.gate is None:
            global_probs[:, self.outdoors_indices] = outdoors_probs
            global_probs[:, self.nature_indices] = nature_probs
            global_probs[:, self.people_indices] = people_probs
            global_probs[:, self.objects_and_vehicles_indices] = objects_probs
            return global_probs


        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=1)

        if self.routing == "hard":
            chosen_expert = gate_probs.argmax(dim=1)
            gate_weights = F.one_hot(chosen_expert, num_classes=NUM_EXPERTS).to(gate_probs.dtype)
        else:
            gate_weights = gate_probs

        outdoors_w = gate_weights[:, EXPERT_ORDER.index("outdoors")].unsqueeze(1)
        nature_w = gate_weights[:, EXPERT_ORDER.index("nature")].unsqueeze(1)
        people_w = gate_weights[:, EXPERT_ORDER.index("people")].unsqueeze(1)
        objects_w = gate_weights[:, EXPERT_ORDER.index("objects_and_vehicles")].unsqueeze(1)

        global_probs[:, self.outdoors_indices] = outdoors_probs * outdoors_w
        global_probs[:, self.nature_indices] = nature_probs * nature_w
        global_probs[:, self.people_indices] = people_probs * people_w
        global_probs[:, self.objects_and_vehicles_indices] = objects_probs * objects_w

        return global_probs


# Classifier Remapping
original_to_aggregated_dict = {12: 0, 17: 1, 23: 2, 33: 3, 37: 4, 49: 5, #OUTDOORS
                               0: 6, 47: 7, 51: 8, 52: 9, 53: 10, 54: 11, 56: 12, 57: 13, 59: 14, #NATURE
                               2: 15, 11: 16, 35: 17, #PEOPLE
                               5: 18, 8: 19, 9: 20, 10: 21, 13: 22, 16: 23, 20: 24, 22: 25, 25: 26, 28: 27, 39: 28, 40: 29, 41: 30, 48: 31, 58: 32, 61: 33, 69: 34,#OBJECTS_AND_VEHICLES
                               81: 35, 84: 36, 85: 37, 86: 38, 87: 39, 89: 40, 90: 41, 94: 42} #OBJECTS_AND_VEHICLES


def map_original_to_aggregated(targets_tensor, original_to_aggregated=original_to_aggregated_dict):
    aggregated_targets_tensor = torch.tensor(
        [
            original_to_aggregated[int(element.item())] for element in targets_tensor
        ],
        device=targets_tensor.device,
        dtype=torch.long
    )

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


        # local specialist index -> CIFAR-100 class
        cifar_class = new_to_original[local_idx]

        if cifar_class not in original_to_aggregated_dict:
            raise RuntimeError(
                f"{model_name}: CIFAR class {cifar_class} "
                "is not present in the global 43-class mapping"
            )


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




def FORMERLY_aggregate_models_all_layers_with_feature_averaging(
    model_names: list,
    model_dir: Path,
    device: str = 'cpu'
) -> nn.Module:


    print("\n" + "="*80)
    print("COMPLETE MODEL AGGREGATION - ALL LAYERS WITH FEATURE AVERAGING")
    print("="*80 + "\n")

    model_dir = f"../Final_implementation_attempt/final_specialised_models/"
    models = {}
    mappings = {}
    num_subset_classes = {}

    
    # Load all models and their mappings
    print("Step 1: Loading specialized models...")
    local_to_global_indices = {}
    for model_name in model_names:
        try:
            state_dict = torch.load(f"{model_dir}final_{model_name}_model.pt", map_location=device)
            print("Model:", f"{model_dir}final_{model_name}_model.pt")

            # Determine number of classes from classifier weight shape
            classifier_size = state_dict['linear.weight'].shape[0]
            print(f"classifier_size for {model_name}", classifier_size)
            model = ResNet20_DynamicModel(num_classes=classifier_size)
            model.load_state_dict(state_dict)
            models[model_name] = model

            # Load mapping
            with open(f"{model_dir}{model_name}_metadata.pkl", 'rb') as f:
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

    cifar_to_expert_lookup = build_cifar_to_expert_lookup(mappings, EXPERT_ORDER)
    gate = ResNet20_DynamicModel(num_classes=NUM_EXPERTS)
    gate = train_gating_network(gate,
                                last_experience_train_loader,
                                cifar_to_expert_lookup,
                                epochs=10,
                                lr=1e-3,)

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

        gate=gate,
        routing="soft",
        num_classes=43,
    )

    return global_model



# Save and Test
def save_global_model(global_model: nn.Module,
                      output_path: Path):
    """Save the global model and aggregation Final."""

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = output_path / 'final_MoE_gating_network_global_model.pt'
    torch.save(global_model.state_dict(), model_path)
    print(f"✓ Saved global model to {model_path}")



def fine_tune_aggregated_model(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        fine_tune_epochs: int = 10,
        lr: float = 1e-4,
        momentum: float=0.9,
        weight_decay:float=5e-4,
):

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model.train()

    for epoch in range(fine_tune_epochs):

        # Reset statistics at the beginning of EACH epoch
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_data, batch_labels, batch_task_ids in loader:
            # Move minibatch to the same device as the model

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

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
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

            global_probs = global_model(batch_data)


            # -----------------------------------------
            # Loss
            # -----------------------------------------

            loss = F.nll_loss(torch.log(global_probs.clamp_min(1e-12)), global_targets)

            # -----------------------------------------
            # Prediction
            # -----------------------------------------

            predictions = global_probs.argmax(dim=1)

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
        f"{output_message} --> Loss = {avg_loss:.6f} | Accuracy = {accuracy * 100:.2f}% [correct:{correct}, total:{total}. {correct} of {total} are correct]"
    )


def evaluate_aggregated_model(model: nn.Module, loader: torch.utils.data.DataLoader, output_message:str):
    total_loss_all_experiences = 0.0
    total_correct_all_experiences = 0
    total_samples_all_experiences = 0

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels, batch_task_ids in loader:

            logits_all_experiences = model(batch_data)
            targets_all_experiences = batch_labels


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


            outputs = target_model(batch_data)
            targets_all_experiences = batch_labels

            predictions = outputs.argmax(dim=1)

            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

    accuracy = (correct / total ) if total > 0 else 0.0
    print(f"Fine-tuned model accuracy: {accuracy:.2%}")


# MAIN EXECUTION
def main():
    """
    Complete pipeline MoE with gating network.
    """

    MODEL_DIR = Path("./final_specialised_models")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = Path('./final_specialised_models')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE = 'cpu'

    model_names = ['outdoors', 'nature', 'people', 'objects_and_vehicles']

    try:

        
        global_model = FORMERLY_aggregate_models_all_layers_with_feature_averaging(
            model_names=model_names,
            model_dir=MODEL_DIR,
            device=DEVICE
        )

        
        # Save
        save_global_model(global_model, OUTPUT_DIR)

        print("="*80)
        print(f"\nModel saved to: {OUTPUT_DIR / 'final_MoE_gating_network_global_model.pt'}")
        print("="*80 + "\n")

        return global_model

    except Exception as e:
        print(f"\n✗ MoE with gating network pipeline failed: {e}")
        raise



if __name__ == '__main__':
    global_model = main()

    evaluate_global_specialist_model(global_model, test_loader, original_to_aggregated_dict,"Evaluating global model on test_loader (i.e. on the ENTIRE TEST SET)")
    evaluate_global_specialist_model(global_model, outdoors_loader, original_to_aggregated_dict,"Evaluating global model on outdoors_loader")
    evaluate_global_specialist_model(global_model, nature_loader, original_to_aggregated_dict,"Evaluating global model on nature_loader")
    evaluate_global_specialist_model(global_model, people_loader, original_to_aggregated_dict,"Evaluating global model on people_loader")
    evaluate_global_specialist_model(global_model, objects_and_vehicles_loader, original_to_aggregated_dict,"Evaluating global model on objects_and_vehicles_loader")

    print(
        "\n\n############# Evaluating global model on last_experiences -- is it the same as Evaluating global model on test_loader? If so, then Spatial-CF may be fixed by my strategy #############")
    evaluate_global_specialist_model(global_model, last_experience_test_loader, original_to_aggregated_dict,
                              "Evaluating global model on last_experience_test_loader")
    evaluate_global_specialist_model(global_model, last_experience_test_outdoors, original_to_aggregated_dict,
                              "Evaluating global model on  last_experience_test_outdoors")
    evaluate_global_specialist_model(global_model, last_experience_test_nature, original_to_aggregated_dict,
                              "Evaluating global model on last_experience_test_nature")
    evaluate_global_specialist_model(global_model, last_experience_test_people, original_to_aggregated_dict,
                              "Evaluating global model on last_experience_test_people")
    evaluate_global_specialist_model(global_model, last_experience_test_objects_and_vehicles, original_to_aggregated_dict,
                              "Evaluating global model on last_experience_test_objects_and_vehicles")

    print(
        "\n\n############# Evaluating global model on middle_experiences -- is it the same as Evaluating global model on test_loader? If so, then Temporal-CF may be fixed by my strategy too!!! #############")
    evaluate_global_specialist_model(global_model, middle_experience_test_loader, original_to_aggregated_dict,
                              "Evaluating global model on middle_experience_test_loader")
    evaluate_global_specialist_model(global_model, middle_experience_test_outdoors,
                              original_to_aggregated_dict,"Evaluating global model on  middle_experience_test_outdoors")
    evaluate_global_specialist_model(global_model, middle_experience_test_nature,
                              original_to_aggregated_dict,"Evaluating global model on middle_experience_test_nature")
    evaluate_global_specialist_model(global_model, middle_experience_test_people,
                              original_to_aggregated_dict,"Evaluating global model on middle_experience_test_people")
    evaluate_global_specialist_model(global_model, middle_experience_test_objects_and_vehicles,
                              original_to_aggregated_dict,"Evaluating global model on middle_experience_test_objects_and_vehicles")

    print(
        "\n\n############# Evaluating global model on first_experiences -- is it the same as Evaluating global model on test_loader? If so, then Temporal-CF may be fixed by my strategy too!!! #############")
    evaluate_global_specialist_model(global_model, first_experience_test_loader, original_to_aggregated_dict,
                              "Evaluating global model on first_experience_test_loader")
    evaluate_global_specialist_model(global_model, first_experience_test_outdoors,
                              original_to_aggregated_dict,"Evaluating global model on  first_experience_test_outdoors")
    evaluate_global_specialist_model(global_model, first_experience_test_nature,
                              original_to_aggregated_dict,"Evaluating global model on first_experience_test_nature")
    evaluate_global_specialist_model(global_model, first_experience_test_people,
                              original_to_aggregated_dict,"Evaluating global model on first_experience_test_people")
    evaluate_global_specialist_model(global_model, first_experience_test_objects_and_vehicles,
                              original_to_aggregated_dict,"Evaluating global model on first_experience_test_objects_and_vehicles")

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
    '''print("\n\n")
    evaluate_global_specialist_model(global_model, first_experience_test_loader, original_to_aggregated_dict,
                                     "Evaluating global model on first_experience_test_loader")
    evaluate_global_specialist_model(global_model, middle_experience_test_loader, original_to_aggregated_dict,
                                     "Evaluating global model on middle_experience_test_loader")
    evaluate_global_specialist_model(global_model, last_experience_train_loader, original_to_aggregated_dict,
                                     "Evaluating global model on last_experience_train_loader")'''
